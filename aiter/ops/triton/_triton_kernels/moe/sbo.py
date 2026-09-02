# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _build_mori_sbo_route_tiles_kernel(
    sorted_token_ids,
    num_valid_ids,
    route_tiles,
    token_num,
    topk,
    TILE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_valid = tl.load(num_valid_ids)
    valid_row = row < num_valid
    packed = tl.load(sorted_token_ids + row, mask=valid_row, other=0)
    token = packed & 0x00FFFFFF
    slot = packed >> 24
    valid = valid_row & (token < token_num) & (slot < topk)
    tile = row // TILE_M
    tl.store(route_tiles + token * topk + slot, tile, mask=valid)
