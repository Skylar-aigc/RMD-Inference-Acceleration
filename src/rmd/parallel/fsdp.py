"""FSDP wrapping helpers (migrated from utils/parallel_config.py)."""

from __future__ import annotations

import os
from functools import partial

import torch
from torch.distributed.fsdp import (
    FullStateDictConfig,  # noqa: F401  (public API kept for compatibility)
    MixedPrecision,
    ShardingStrategy,
    StateDictType,  # noqa: F401
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=True,
    wrap_strategy="size",
    min_num_params=int(1e6),
    transformer_module=None,
    device_id=None,
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_module)
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    if device_id is None:
        try:
            device_id = next(module.parameters()).device
        except StopIteration:
            device_id = None

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=device_id,
        limit_all_gathers=True,
        sync_module_states=False,
    )
    return module
