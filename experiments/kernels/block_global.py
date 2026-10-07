"""Block-local attention with global tokens: the CPU path of candidate A1 (laya:008).

Pattern (``block_global_allowed`` is the dense definition, used by the mask-only reference):
query i may attend key j iff j is a valid key and (``|block(i) - block(j)| <= 1`` or i is global or j is global).

CPU path: queries are grouped into blocks; each block attends its 3-block key span plus the global keys that lie
outside that span (so no key is counted twice in a softmax); global queries attend every valid key in one dense
``[|G|, L]`` product and overwrite their rows. Work is linear in L plus ``|G| x L``; the ``L x L`` product is
never formed. Mask building, padding and copies are inside the call.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import zero_empty_rows


def block_global_allowed(L: int, block: int, key_valid: torch.Tensor, is_global: torch.Tensor) -> torch.Tensor:
    """Boolean ``[B, 1, L, L]``: True where query i may attend key j (the mask-only reference pattern)."""
    pos = torch.arange(L, device=key_valid.device)
    blk = pos // block
    local = (blk[:, None] - blk[None, :]).abs() <= 1                                  # [L, L]
    g = is_global.to(torch.bool)
    pat = local[None] | g[:, :, None] | g[:, None, :]                                 # [B, L, L]
    return (pat & key_valid.to(torch.bool)[:, None, :])[:, None]


def block_global_reference(q, k, v, block: int, key_valid, is_global) -> torch.Tensor:
    """Dense SDPA with the pattern as a mask (also the ``a1_mask`` variant's attention)."""
    mask = block_global_allowed(q.shape[-2], block, key_valid, is_global)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return zero_empty_rows(out, mask.any(-1))


def block_global_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block: int,
                           key_valid: torch.Tensor, is_global: torch.Tensor) -> torch.Tensor:
    """`q, k, v`: ``[B, H, L, D]``; `key_valid`, `is_global`: ``[B, L]`` bool. Rows with no allowed key give zeros."""
    B, H, L, D = q.shape
    dev = q.device
    bs = int(block)
    key_valid = key_valid.to(torch.bool)
    is_global = is_global.to(torch.bool) & key_valid        # a padded position is never a global token
    nb = -(-L // bs)
    Lp = nb * bs
    pad = Lp - L
    if pad:
        qp, kp0, vp0 = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        kv_p = F.pad(key_valid, (0, pad))
    else:
        qp, kp0, vp0, kv_p = q, k, v, key_valid

    # ---- global key set per row, padded to the batch maximum
    counts = is_global.sum(-1)                                                          # [B]
    gmax = int(counts.max()) if B else 0
    if gmax:
        order = torch.argsort((~is_global).to(torch.int8), dim=-1, stable=True)         # globals first, in position order
        gpos = order[:, :gmax]                                                          # [B, G]
        gvalid = torch.arange(gmax, device=dev)[None, :] < counts[:, None]              # [B, G]
        gather = gpos[:, None, :, None].expand(B, H, gmax, D)
        kg, vg = k.gather(2, gather), v.gather(2, gather)
        qg = q.gather(2, gather)
    # ---- block-local queries: span keys plus out-of-span global keys
    kpad = F.pad(kp0, (0, 0, bs, bs))
    vpad = F.pad(vp0, (0, 0, bs, bs))
    kn = kpad.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    vn = vpad.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    qb = qp.view(B, H, nb, bs, D).permute(0, 2, 1, 3, 4).reshape(B * nb, H, bs, D)
    k_pos = (torch.arange(nb, device=dev)[:, None] - 1) * bs + torch.arange(3 * bs, device=dev)[None, :]   # [nb, 3bs]
    span_ok = F.pad(kv_p, (bs, bs)).unfold(1, 3 * bs, bs)                                # [B, nb, 3bs] (False outside / padding)
    mask_span = span_ok[:, :, None, :].expand(B, nb, bs, 3 * bs)
    if gmax:
        lo = (torch.arange(nb, device=dev) - 1) * bs                                     # span is [lo, lo + 3bs)
        in_span = (gpos[:, None, :] >= lo[None, :, None]) & (gpos[:, None, :] < (lo + 3 * bs)[None, :, None])   # [B, nb, G]
        g_ok = gvalid[:, None, :] & ~in_span                                             # [B, nb, G]
        kk = torch.cat([kn, kg[:, None].expand(B, nb, H, gmax, D).reshape(B * nb, H, gmax, D)], dim=2)
        vv = torch.cat([vn, vg[:, None].expand(B, nb, H, gmax, D).reshape(B * nb, H, gmax, D)], dim=2)
        mask = torch.cat([mask_span, g_ok[:, :, None, :].expand(B, nb, bs, gmax)], dim=-1)
    else:
        kk, vv, mask = kn, vn, mask_span
    mask = mask.reshape(B * nb, 1, bs, -1)
    out = F.scaled_dot_product_attention(qb, kk, vv, attn_mask=mask)
    out = zero_empty_rows(out, mask.any(-1))
    out = out.view(B, nb, H, bs, D).permute(0, 2, 1, 3, 4).reshape(B, H, Lp, D)[:, :, :L]

    # ---- global queries: all valid keys
    if gmax:
        gm = (gvalid[:, None, None, :].new_ones(1) & key_valid[:, None, None, :]).expand(B, 1, gmax, L)
        og = F.scaled_dot_product_attention(qg, k, v, attn_mask=gm)
        og = zero_empty_rows(og, gm.any(-1))
        # padded global slots point at position 0; give them an out-of-range slot so they cannot overwrite row 0
        idx = torch.where(gvalid, gpos, torch.full_like(gpos, L))[:, None, :, None].expand(B, H, gmax, D)
        ext = F.pad(out, (0, 0, 0, 1))
        out = ext.scatter(2, idx, og)[:, :, :L]
    return out
