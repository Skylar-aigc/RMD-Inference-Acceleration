"""Wan2.1 model loading adapter.

Centralizes teacher/student/real model loading for training and single-model
loading for inference. The Ascend sequence-parallel patch (``ascendx_video``)
is optional: when unavailable the code degrades to the stock diffusers
transformer with a logged warning.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _maybe_apply_sp_patch() -> bool:
    """Monkey-patch diffusers Wan modules with Ascend SP versions, if available."""
    try:
        from ascendx_video.framework.diffusers_framework.transformer.wan import (  # type: ignore
            AscendWanAttnProcessor2_0,
            AscendWanRotaryPosEmbed,
            AscendWanTransformer3DModel,
            AscendWanTransformerBlock,
            get_enhance_scores_wan,
        )
    except ImportError:
        logger.warning("ascendx_video not found; sequence-parallel patch disabled.")
        return False

    from diffusers.models.transformers import transformer_wan

    transformer_wan.WanAttnProcessor2_0 = AscendWanAttnProcessor2_0
    transformer_wan.WanAttnProcessor2_0.get_enhance_scores_wan = get_enhance_scores_wan
    transformer_wan.WanRotaryPosEmbed = AscendWanRotaryPosEmbed
    transformer_wan.WanTransformerBlock = AscendWanTransformerBlock
    transformer_wan.WanTransformer3DModel = AscendWanTransformer3DModel
    logger.info("Ascend SP patch applied.")
    return True


def load_tokenizer_text_encoder(cfg, device):
    """Load the UMT5 tokenizer + text encoder (frozen)."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
    ).to(device=device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder


def load_transformer(cfg, device=None, dtype=None, enable_sp: bool = False):
    """Load a single WanTransformer3DModel, optionally with the Ascend SP patch."""
    if enable_sp:
        from diffusers.models.transformers import transformer_wan

        model_class = transformer_wan.WanTransformer3DModel
    else:
        from diffusers import WanTransformer3DModel

        model_class = WanTransformer3DModel
    if dtype is None:
        dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    model = model_class.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=cfg.revision,
        variant=cfg.variant,
    ).to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    return model


def resolve_sequence_parallel(cfg, enable_sp: bool) -> bool:
    """Apply the optional SP patch once and update the effective config flag."""
    if enable_sp:
        enable_sp = _maybe_apply_sp_patch()
    cfg.enable_sp = enable_sp
    return enable_sp


def load_training_transformer(cfg, device=None, enable_sp: bool = False, frozen: bool = False):
    """Load one training transformer, optionally on CPU for later FSDP sharding."""
    load_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    transformer = load_transformer(cfg, device=device, dtype=load_dtype, enable_sp=enable_sp)
    if frozen:
        transformer.requires_grad_(False)
    return transformer


def load_teacher_student_real(cfg, accelerator, enable_sp: bool = False):
    """Compatibility loader for the teacher and frozen real transformer."""

    enable_sp = resolve_sequence_parallel(cfg, enable_sp)
    transformer = load_training_transformer(cfg, device=accelerator.device, enable_sp=enable_sp)
    transformer_real = load_training_transformer(cfg, device=accelerator.device, enable_sp=enable_sp, frozen=True)
    return transformer, transformer_real


def load_vae(cfg, device):
    """Load the Wan VAE (frozen, fp16)."""
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        revision=cfg.revision,
        variant=cfg.variant,
    ).to(device=device, dtype=torch.float16)
    vae.requires_grad_(False)
    return vae


def load_student_for_infer(cfg, device, enable_sp: bool = False):
    """Load a trained student transformer from ``cfg.model_dir`` for inference."""
    from accelerate import init_empty_weights, load_checkpoint_in_model

    if enable_sp:
        enable_sp = _maybe_apply_sp_patch()
        cfg.enable_sp = enable_sp
    if enable_sp:
        from diffusers.models.transformers import transformer_wan

        model_class = transformer_wan.WanTransformer3DModel
    else:
        from diffusers import WanTransformer3DModel

        model_class = WanTransformer3DModel
    load_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    # Instantiate the exact 1.3B/14B architecture from the base model config.
    # Calling ``model_class()`` would silently use Diffusers' default Wan
    # dimensions, which need not match the distilled checkpoint.
    model_config = model_class.load_config(
        cfg.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=cfg.revision,
    )
    # Build only meta tensors first, then stream either a single safetensors
    # file or an indexed multi-shard checkpoint directly onto this rank's
    # device. This avoids a second full CPU/GPU copy and remains compatible
    # with the >10 GB sharded checkpoints produced for the 14B model.
    with init_empty_weights():
        transformer = model_class.from_config(model_config)
    load_checkpoint_in_model(
        transformer,
        cfg.model_dir,
        device_map={"": device},
        dtype=load_dtype,
        strict=True,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer
