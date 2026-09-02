# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sbo import (
    _build_mori_sbo_route_tiles_kernel,
)


def build_mori_sbo_route_tiles(
    sorted_token_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    route_tiles: torch.Tensor,
    *,
    token_num: int,
    topk: int,
    tile_m: int,
) -> None:
    """Map each valid ``(recv row, top-k slot)`` to its sorted-M tile."""
    if route_tiles.dtype != torch.int32 or not route_tiles.is_contiguous():
        raise ValueError("route_tiles must be a contiguous int32 tensor")
    if route_tiles.shape != (token_num, topk):
        raise ValueError(
            f"route_tiles must have shape {(token_num, topk)}, "
            f"got {tuple(route_tiles.shape)}"
        )
    if tile_m <= 0 or tile_m & (tile_m - 1):
        raise ValueError(f"tile_m must be a positive power of two, got {tile_m}")

    route_tiles.fill_(-1)
    block = 256
    _build_mori_sbo_route_tiles_kernel[(triton.cdiv(sorted_token_ids.numel(), block),)](
        sorted_token_ids,
        num_valid_ids,
        route_tiles,
        token_num,
        topk,
        TILE_M=tile_m,
        BLOCK=block,
    )


def supports_mori_sbo_tile_signal_v1() -> bool:
    return True
