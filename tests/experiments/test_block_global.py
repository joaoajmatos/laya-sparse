"""laya:008: A1 CPU kernel (experiments/kernels/block_global.py) against its dense mask-only reference."""
import pytest
import torch

from experiments.kernels.block_global import block_global_allowed, block_global_attention, block_global_reference
from experiments.kernels.local import local_attention
from experiments.kernels.reference import MaskSpec


def _qkv(B, L, H=2, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, L, D, generator=g) for _ in range(3))


def _inputs(lengths, L, header):
    valid = torch.arange(L)[None, :] < torch.tensor(lengths)[:, None]
    g = torch.zeros(len(lengths), L, dtype=torch.bool)
    for b, h in enumerate(header):
        g[b, :h] = True
    return valid, g


@pytest.mark.parametrize("L,block", [(40, 8), (64, 16), (65, 16), (130, 32), (200, 16)])
@pytest.mark.parametrize("header", [(0, 0), (5, 3), (9, 20), (17, 1)])
def test_cpu_path_equals_mask_only_reference(L, block, header):
    lengths = [L, max(1, L - 7)]
    q, k, v = _qkv(2, L)
    valid, g = _inputs(lengths, L, header)
    out = block_global_attention(q, k, v, block, valid, g)
    ref = block_global_reference(q, k, v, block, valid, g)
    assert torch.isfinite(out).all()
    assert float((out - ref).abs().max()) < 1e-5


def test_header_larger_than_several_blocks_and_all_global_equals_dense():
    L, block = 96, 8
    q, k, v = _qkv(2, L)
    valid, _ = _inputs([L, 70], L, (0, 0))
    g = torch.ones(2, L, dtype=torch.bool)
    full = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :].expand(2, 1, L, L))
    out = block_global_attention(q, k, v, block, valid, g)
    assert float((out[0] - full[0]).abs().max()) < 1e-5                       # all-global row 0 is native dense attention
    big = torch.zeros(2, L, dtype=torch.bool)
    big[:, :40] = True                                                         # header spans 5 blocks
    ref = block_global_reference(q, k, v, block, valid, big)
    assert float((block_global_attention(q, k, v, block, valid, big) - ref).abs().max()) < 1e-5


def test_no_global_token_is_pure_block_local():
    L, block = 64, 16
    q, k, v = _qkv(1, L)
    valid = torch.ones(1, L, dtype=torch.bool)
    out = block_global_attention(q, k, v, block, valid, torch.zeros(1, L, dtype=torch.bool))
    ref = local_attention(q, k, v, MaskSpec([L], "block_local", block=block))
    assert float((out - ref).abs().max()) < 1e-5


def test_pattern_is_neither_native_dense_nor_block_local():
    L, block = 64, 8
    q, k, v = _qkv(1, L)
    valid, g = _inputs([L], L, (4,))
    out = block_global_attention(q, k, v, block, valid, g)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    local = local_attention(q, k, v, MaskSpec([L], "block_local", block=block))
    assert float((out - dense).abs().max()) > 1e-3
    assert float((out - local).abs().max()) > 1e-3
    allowed = block_global_allowed(L, block, valid, g)[0, 0]
    assert allowed[40, 2] and allowed[2, 40] and allowed[40, 41] and not allowed[40, 20] and not allowed[20, 50]


def test_padded_positions_are_never_keys_and_padded_global_slots_do_not_overwrite_row_zero():
    L, block = 48, 8
    q, k, v = _qkv(2, L)
    valid, g = _inputs([48, 30], L, (6, 2))            # different |G| per row: row 1 has padded global slots
    out = block_global_attention(q, k, v, block, valid, g)
    k2, v2 = k.clone(), v.clone()
    k2[1, :, 30:] += 50.0                                # change padded keys of row 1 only
    v2[1, :, 30:] += 50.0
    out2 = block_global_attention(q, k2, v2, block, valid, g)
    assert float((out[:, :, :30] - out2[:, :, :30]).abs().max()) < 1e-6
    ref = block_global_reference(q, k, v, block, valid, g)
    assert float((out - ref).abs().max()) < 1e-5


def test_rows_with_no_allowed_key_are_zero_not_nan():
    L, block = 24, 8
    q, k, v = _qkv(2, L)
    valid, g = _inputs([0, 24], L, (0, 0))
    out = block_global_attention(q, k, v, block, valid, g)
    assert torch.isfinite(out).all() and torch.equal(out[0], torch.zeros_like(out[0]))
