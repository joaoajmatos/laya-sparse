"""Shared mask semantics and the dense fp32 reference (T031, research.md R8, FR-022).

Every kernel takes ``q, k, v`` shaped ``[batch, heads, L, head_dim]`` and a `MaskSpec`. The spec
names one attention pattern; `allowed(L)` materializes exactly that pattern as a boolean
``[batch, 1, L, L]`` mask, which only the reference and the dense kernels use. Sparse kernels
build their own compact masks from the same spec and must match the reference.

Patterns (key j is always also required to be inside the row's valid length, ``j < lengths[b]``;
padding sits at the end of a row):

* ``full``         every key.
* ``block_local``  queries in block ``i`` see keys in blocks ``i-1, i, i+1`` (block = ``block``).
* ``band``         exact sliding window: query i sees keys j with ``|i-j| <= half_width`` (native
                   Laya local layers, ``half_width`` 64). Not ``block_local``, which is a superset.
* ``gather``       queries in block ``i`` see keys in the blocks `gather_blocks` selects: block 0
                   (where Laya puts the question and options), blocks ``i-1, i, i+1``, and evenly
                   spaced blocks up to ``selection_blocks`` in total. Fixed, not learned (R8).

Convention: a query with no allowed key returns zeros, never NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

PATTERNS = ("full", "block_local", "band", "gather")


@dataclass
class MaskSpec:
    lengths: Sequence[int]
    pattern: str = "full"
    block: int = 128
    selection_blocks: int = 4
    half_width: int = 64

    def __post_init__(self):
        if self.pattern not in PATTERNS:
            raise ValueError("pattern must be one of %s" % (PATTERNS,))
        if self.block < 1 or self.selection_blocks < 1:
            raise ValueError("block and selection_blocks must be >= 1")
        if self.half_width < 0:
            raise ValueError("half_width must be >= 0")

    def lengths_tensor(self, device=None) -> torch.Tensor:
        return torch.as_tensor(list(self.lengths), dtype=torch.long, device=device)

    def n_blocks(self, L: int) -> int:
        return -(-L // self.block)

    def allowed(self, L: int) -> torch.Tensor:
        """Boolean ``[batch, 1, L, L]``: True where query i may attend key j."""
        lengths = self.lengths_tensor()
        pos = torch.arange(L)
        key_ok = pos[None, :] < lengths[:, None]                      # [B, L]
        if self.pattern == "full":
            pat = torch.ones(L, L, dtype=torch.bool)
        elif self.pattern == "block_local":
            blk = pos // self.block
            pat = (blk[:, None] - blk[None, :]).abs() <= 1
        elif self.pattern == "band":
            pat = (pos[:, None] - pos[None, :]).abs() <= self.half_width
        else:
            idx, ok = gather_blocks(self.n_blocks(L), self.selection_blocks)
            nb = self.n_blocks(L)
            sel = torch.zeros(nb, nb, dtype=torch.bool)
            for qb in range(nb):
                sel[qb, idx[qb][ok[qb]]] = True
            blk = pos // self.block
            pat = sel[blk][:, blk]
        return pat[None, None, :, :] & key_ok[:, None, None, :]


def gather_blocks(n_blocks: int, selection_blocks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Key-block indices ``[n_blocks, selection_blocks]`` per query block, and a validity mask.

    Duplicate or missing slots (when there are fewer distinct blocks than slots) are marked
    invalid, so no key is counted twice in a softmax.
    """
    idx = torch.zeros(n_blocks, selection_blocks, dtype=torch.long)
    ok = torch.zeros(n_blocks, selection_blocks, dtype=torch.bool)
    for qb in range(n_blocks):
        chosen: List[int] = []
        for b in [0, qb - 1, qb, qb + 1]:
            if 0 <= b < n_blocks and b not in chosen:
                chosen.append(b)
        extra = selection_blocks - len(chosen)
        if extra > 0 and n_blocks > 0:
            step = n_blocks / (extra + 1)
            for t in range(1, extra + 1):
                b = min(n_blocks - 1, int(round(t * step)))
                if b not in chosen:
                    chosen.append(b)
        chosen = chosen[:selection_blocks]
        for s, b in enumerate(chosen):
            idx[qb, s] = b
            ok[qb, s] = True
    return idx, ok


def zero_empty_rows(out: torch.Tensor, row_has_key: torch.Tensor) -> torch.Tensor:
    """Apply the fully-masked-row convention: zeros where a query has no allowed key."""
    return torch.where(row_has_key[..., None], out, torch.zeros((), dtype=out.dtype))


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: MaskSpec) -> torch.Tensor:
    """Explicit fp32 scores, boolean mask, softmax and value product. Defines the semantics."""
    q, k, v = q.float(), k.float(), v.float()
    L = q.shape[-2]
    mask = spec.allowed(L)                                           # [B, 1, L, L]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    has_key = mask.any(-1)                                           # [B, 1, L]
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(has_key[..., None], probs, torch.zeros((), dtype=probs.dtype))
    return probs @ v


def score_matrix_bytes(batch: int, heads: int, L: int, dtype_size: int = 4) -> int:
    """Analytical size of one full score matrix (research.md R9)."""
    return int(batch) * int(heads) * int(L) * int(L) * int(dtype_size)
