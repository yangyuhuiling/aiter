# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import ActivationType, dtypes
from aiter.fused_moe import (
    _get_mori_sbo_metadata,
    fused_moe,
    moe_sorting,
)
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2, get_flydsl_kernel_params
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.ops.triton.moe.sbo import build_mori_sbo_route_tiles


def test_mori_sbo_rejects_fhmoe_before_dispatch():
    empty = torch.empty(0)
    with pytest.raises(ValueError, match="FHMoE"):
        fused_moe(
            empty.reshape(0, 1),
            empty.reshape(0, 1, 1),
            empty.reshape(0, 1, 1),
            empty.reshape(0, 1),
            torch.empty((0, 1), dtype=torch.int32),
            shared_expert_id=0,
            sbo_args={},
        )


def test_mori_sbo_rejects_a16w4_before_dispatch():
    empty = torch.empty(0)
    with pytest.raises(ValueError, match="A16W4"):
        flydsl_moe_stage2(
            inter_states=empty.reshape(0, 1, 1),
            w2=empty.reshape(0, 1, 1),
            sorted_token_ids=torch.empty(0, dtype=torch.int32),
            sorted_expert_ids=torch.empty(0, dtype=torch.int32),
            num_valid_ids=torch.zeros(2, dtype=torch.int32),
            a_dtype="bf16",
            b_dtype="fp4",
            sbo_tile_state=torch.empty(0, dtype=torch.int32),
        )


@pytest.mark.parametrize("token,tile_m", [(1, 32), (64, 32), (2048, 64)])
def test_mori_sbo_metadata_forces_atomic_a8w4(token, tile_m):
    metadata = _get_mori_sbo_metadata(
        token,
        3072,
        dtypes.bf16,
        ActivationType.Silu,
        0,
        0,
    )
    stage2_name = metadata.stage2.keywords["kernelName"]
    params = get_flydsl_kernel_params(stage2_name)
    assert params["a_dtype"] == "fp8"
    assert params["b_dtype"] == "fp4"
    assert params["mode"] == "atomic"
    assert params["tile_m"] == tile_m


@pytest.mark.parametrize("tile_m", [32, 64])
def test_mori_sbo_stage2_signal_and_output(tile_m):
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("MORI SBO v1 is gfx950-only")

    torch.manual_seed(7)
    token, topk, experts = 65, 1, 1
    hidden, inter = 7168, 3072
    topk_ids = torch.zeros((token, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((token, topk), dtype=torch.float32, device="cuda")
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden,
        torch.bfloat16,
        tile_m,
    )

    a2 = torch.randn((token, topk, inter), dtype=torch.bfloat16, device="cuda") * 0.01
    a2_q, a2_scale = per_1x32_f8_scale_f8_quant(
        a2, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a2_scale = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter,
    )
    w2 = (
        torch.randn((experts, hidden, inter), dtype=torch.bfloat16, device="cuda")
        * 0.01
    )
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = shuffle_weight_a16w4(w2_q.view(experts, hidden, inter // 2), 16, False)
    w2_scale = shuffle_scale_a16w4(w2_scale, experts, False)

    common = {
        "inter_states": a2_q.view(token, topk, inter),
        "w2": w2_q,
        "sorted_token_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk": topk,
        "tile_m": tile_m,
        "tile_n": 128,
        "tile_k": 256,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "mode": "atomic",
        "w2_scale": w2_scale,
        "a2_scale": a2_scale,
        "sorted_weights": sorted_weights,
    }
    baseline = torch.zeros((token, hidden), dtype=torch.bfloat16, device="cuda")
    candidate = torch.zeros_like(baseline)
    flydsl_moe_stage2(out=baseline, **common)

    tile_capacity = (sorted_ids.numel() + tile_m - 1) // tile_m
    tile_state = torch.zeros(tile_capacity, dtype=torch.int32, device="cuda")
    route_tiles = torch.empty((token, topk), dtype=torch.int32, device="cuda")
    build_mori_sbo_route_tiles(
        sorted_ids,
        num_valid_ids,
        route_tiles,
        token_num=token,
        topk=topk,
        tile_m=tile_m,
    )
    flydsl_moe_stage2(
        out=candidate,
        sbo_tile_state=tile_state,
        **common,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    valid_tiles = int(num_valid_ids[0].item()) // tile_m
    assert torch.all(tile_state[:valid_tiles] == hidden // 128)
    assert torch.all(tile_state[valid_tiles:] == 0)
    expected_route_tiles = torch.arange(token, device="cuda") // tile_m
    assert torch.equal(route_tiles[:, 0], expected_route_tiles)


@pytest.mark.parametrize(
    "sort_m,tile_m",
    [(32, 32), (64, 64), (128, 32), (128, 64)],
)
def test_mori_sbo_route_tiles_cover_per_expert_padding(sort_m, tile_m):
    token, topk, experts, hidden = 128, 6, 48, 7168
    rows = torch.arange(token, dtype=torch.int32, device="cuda")
    bases = torch.where(rows < token // 2, experts - topk, rows % (experts - topk))
    slots = torch.arange(topk, dtype=torch.int32, device="cuda")
    topk_ids = (bases[:, None] + slots[None, :]) % experts
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden,
        torch.bfloat16,
        sort_m,
    )

    padded_capacity = (sorted_ids.numel() + tile_m - 1) // tile_m
    unpadded_capacity = (token * topk + tile_m - 1) // tile_m
    assert padded_capacity > unpadded_capacity
    assert padded_capacity <= (token * topk + 31) // 32 + experts * 4
    route_tiles = torch.empty((token, topk), dtype=torch.int32, device="cuda")
    build_mori_sbo_route_tiles(
        sorted_ids,
        num_valid_ids,
        route_tiles,
        token_num=token,
        topk=topk,
        tile_m=tile_m,
    )
    torch.cuda.synchronize()

    assert torch.all(route_tiles >= 0)
    assert int(route_tiles.max()) < padded_capacity
