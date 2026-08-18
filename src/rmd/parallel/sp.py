"""Sequence-parallel helpers (migrated from utils/parallel_config.py)."""

from __future__ import annotations

import torch.distributed as dist


def set_parallel(transformer, dp_size: int | None = None, sp_size: int | None = None, enable_cp: bool | None = False):
    """Enable sequence parallel on a transformer that supports it (Ascend only)."""
    if sp_size is None:
        sp_size = dist.get_world_size()
        dp_size = 1
    else:
        assert dist.get_world_size() % sp_size == 0, f"world_size {dist.get_world_size()} must be divisible by sp_size"
        dp_size = dist.get_world_size() // sp_size

    transformer.enable_parallel(dp_size, sp_size, enable_cp)
