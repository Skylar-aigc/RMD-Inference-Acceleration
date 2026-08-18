"""Wan2.1 model loading adapter for training and inference."""

from __future__ import annotations

import torch


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


def load_transformer(cfg, device=None, dtype=None):
    """Load a single WanTransformer3DModel."""
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


def load_training_transformer(cfg, device=None, frozen: bool = False):
    """Load one training transformer, optionally on CPU for later FSDP sharding."""
    load_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    transformer = load_transformer(cfg, device=device, dtype=load_dtype)
    if frozen:
        transformer.requires_grad_(False)
    return transformer


def load_teacher_student_real(cfg, accelerator):
    """Compatibility loader for the teacher and frozen real transformer."""

    transformer = load_training_transformer(cfg, device=accelerator.device)
    transformer_real = load_training_transformer(cfg, device=accelerator.device, frozen=True)
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


def load_student_for_infer(cfg, device):
    """Load a trained student transformer from ``cfg.model_dir`` for inference."""
    from accelerate import init_empty_weights, load_checkpoint_in_model
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
