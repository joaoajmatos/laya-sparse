"""Exact sliding-window attention, |i-j| <= half_width (laya:007, S0).

Native Laya local layers attend keys within 64 positions of the query. ``local.py`` is block-granular
(a superset of that window), so it cannot replace them. Here queries are grouped into blocks of
``block`` (= half_width by default); every key within ``half_width`` of a query lies in the query's
own or a neighbouring key block, so each block sees a 3-block span of keys and the exact band mask
and the key-validity mask are applied inside it. Work is linear in length (about 3 x block keys per
query against the ideal 2 x half_width + 1). Blocks go into the batch dimension so SDPA takes its
fused 4-D path, as in ``local.py``. Mask building, padding and copies are inside the call (FR-021).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .reference import zero_empty_rows


def local_exact_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, half_width: int,
                          lengths: Optional[torch.Tensor] = None, block: Optional[int] = None,
                          key_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`q, k, v`: ``[B, H, L, D]``. `lengths`: valid tokens per row (padding at the end); None means all.
    `key_valid` (``[B, L]`` bool) states validity per key directly and wins over `lengths`.

    A query with no allowed key returns zeros (repo convention).
    """
    B, H, L, D = q.shape
    bs = int(block or max(1, half_width))
    if bs < half_width:
        raise ValueError("block (%d) must be >= half_width (%d) for the 3-block span to cover the window" % (bs, half_width))
    nb = -(-L // bs)
    Lp = nb * bs
    pad = Lp - L
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    kp = F.pad(k, (0, 0, bs, bs))
    vp = F.pad(v, (0, 0, bs, bs))
    kn = kp.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    vn = vp.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    qb = q.view(B, H, nb, bs, D).permute(0, 2, 1, 3, 4).reshape(B * nb, H, bs, D)
    q_pos = torch.arange(nb)[:, None] * bs + torch.arange(bs)[None, :]                  # [nb, bs]
    k_pos = (torch.arange(nb)[:, None] - 1) * bs + torch.arange(3 * bs)[None, :]         # [nb, 3bs]
    band = (q_pos[:, :, None] - k_pos[:, None, :]).abs() <= half_width                   # [nb, bs, 3bs]
    in_seq = (k_pos >= 0) & (k_pos < L)
    if key_valid is not None:
        kv = F.pad(key_valid.to(torch.bool), (0, pad))
        key_ok = F.pad(kv, (bs, bs)).unfold(1, 3 * bs, bs)                                  # [B, nb, 3bs]
    elif lengths is None:
        key_ok = in_seq[None].expand(B, -1, -1)
    else:
        key_ok = in_seq[None] & (k_pos[None] < lengths.to(k_pos.device)[:, None, None])    # [B, nb, 3bs]
    mask = (band[None] & key_ok[:, :, None, :]).reshape(B * nb, 1, bs, 3 * bs)
    out = F.scaled_dot_product_attention(qb, kn, vn, attn_mask=mask)
    out = zero_empty_rows(out, mask.any(-1))
    out = out.view(B, nb, H, bs, D).permute(0, 2, 1, 3, 4)
    return out.reshape(B, H, Lp, D)[:, :, :L]
