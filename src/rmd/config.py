"""Typed configuration for RMD (replaces ``utils/argument_config.py``).

Uses pydantic models so invalid or inconsistent settings fail fast with clear
errors instead of argparse silently accepting anything.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# (channels, temporal, H, W) latent layouts for each generation resolution.
LATENT_SIZES: dict[int, tuple[int, int, int, int]] = {
    270: (16, 21, 34, 60),
    480: (16, 21, 60, 104),
    720: (16, 21, 90, 160),
}


class TrainConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # --- model ---
    pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    revision: str | None = None
    variant: str | None = None

    # --- training ---
    device: Literal["auto", "cuda", "cpu"] = "auto"
    train_batch_size: int = 1
    max_train_steps: int = 1000
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    learning_rate_fake: float = 2e-5
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 5
    lr_num_cycles: int = 1
    adam_beta1: float = 0.0
    adam_beta2: float = 0.95
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    seed: int | None = 42
    output_dir: str = "outputs/rmd"
    checkpointing_steps: int = 50
    resume_from_checkpoint: str | None = None

    # --- distillation method ---
    k_step: int = 6
    cfg: float = 5.0
    eta: float = 0.9
    score_weighting_mode: Literal["inverse_sigma", "legacy_snr"] = "inverse_sigma"
    num_euler_timesteps: int = 1000

    # --- multi-stage upsampling ---
    flow_shift: float = 3.0
    flow_shift_trans: Literal[0, 1, 2] = 0
    low_rs_step: int = 0
    t_offset_factor: float = 0.5
    timestep_offset_mode: Literal["scaled", "zero"] = "scaled"
    split_timestep: bool = False
    warmup_step: int = 15
    upsampler_mode: str = "bilinear"
    infer_upsampler_mode: str = "bilinear"
    resolution: Literal["480", "720"] = "480"

    # --- feature flags ---
    multistage_upsample: bool = False
    gan: bool = False

    # --- GAN mode (experimental) ---
    K_step_get_fact_latents: int = 20
    weight_fact_latents: float = 0.8
    weight_gan_generator: float = 1e-2

    # --- data ---
    data_path: str = "examples/sample_prompts.txt"
    prompt_embed_batch_size: int = 8
    validation_prompt_path: str | None = None
    validation_num_prompts: int = 10
    validation_at_step_one: bool = True

    # --- logging ---
    debug_print: bool = False

    @field_validator("resolution", mode="before")
    @classmethod
    def _coerce_resolution(cls, v):
        # Accept int 480/720 from yaml and coerce to the literal string form.
        if isinstance(v, int):
            return str(v)
        return v

    @field_validator("flow_shift_trans")
    @classmethod
    def _check_fst(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"flow_shift_trans must be 0/1/2, got {v}")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> TrainConfig:
        if self.multistage_upsample:
            self.flow_shift_trans = max(self.flow_shift_trans, 1)
        if self.flow_shift_trans > 0 and self.resolution == "480":
            self.resolution = "720"
        if self.k_step < 1:
            raise ValueError(f"k_step must be >= 1, got {self.k_step}")
        if self.max_train_steps < 0:
            raise ValueError(f"max_train_steps must be >= 0, got {self.max_train_steps}")
        if self.checkpointing_steps < 1:
            raise ValueError(f"checkpointing_steps must be >= 1, got {self.checkpointing_steps}")
        if not 0 <= self.low_rs_step <= self.k_step:
            raise ValueError(f"low_rs_step must be between 0 and k_step ({self.k_step}), got {self.low_rs_step}")
        if self.prompt_embed_batch_size < 1:
            raise ValueError(f"prompt_embed_batch_size must be >= 1, got {self.prompt_embed_batch_size}")
        if self.validation_num_prompts < 1:
            raise ValueError(f"validation_num_prompts must be >= 1, got {self.validation_num_prompts}")
        if self.lr_num_cycles < 1:
            raise ValueError(f"lr_num_cycles must be >= 1, got {self.lr_num_cycles}")
        return self

    @property
    def latent_size(self) -> tuple[int, int, int, int]:
        return LATENT_SIZES[720 if self.resolution == "720" else 480]

    @property
    def effective_device(self) -> str:
        from rmd.platform import get_device

        return get_device(None if self.device == "auto" else self.device)


class InferConfig(BaseModel):
    """Configuration for running generation with a distilled student model."""

    model_config = ConfigDict(protected_namespaces=())

    pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    revision: str | None = None
    variant: str | None = None
    model_dir: str  # path to trained student weights
    device: Literal["auto", "cuda", "cpu"] = "auto"
    train_batch_size: int = 1  # used by Predictor for the unconditional prompt batch
    resolution: Literal["480", "720"] = "480"
    flow_shift: float = 3.0
    flow_shift_trans: Literal[0, 1, 2] = 0
    low_rs_step: int = 0
    infer_upsampler_mode: str = "bilinear"
    eta: float = 0.0
    k_step: int = 6
    cfg: float = 5.0
    num_euler_timesteps: int = 1000
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    seed: int = 42
    prompt_enhanced: bool = False
    output_dir: str = "outputs/rmd-infer"

    @field_validator("resolution", mode="before")
    @classmethod
    def _coerce_resolution(cls, v):
        if isinstance(v, int):
            return str(v)
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> InferConfig:
        if self.flow_shift_trans > 0 and self.resolution == "480":
            self.resolution = "720"
        if self.k_step < 1:
            raise ValueError(f"k_step must be >= 1, got {self.k_step}")
        if not 0 <= self.low_rs_step <= self.k_step:
            raise ValueError(f"low_rs_step must be between 0 and k_step ({self.k_step}), got {self.low_rs_step}")
        return self

    @property
    def source_target_latent_sizes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return source/target layouts with training-compatible upsampling semantics."""
        if self.flow_shift_trans > 0:
            return LATENT_SIZES[480], LATENT_SIZES[720]
        layout = LATENT_SIZES[720 if self.resolution == "720" else 480]
        return layout, layout

    @property
    def latent_size(self) -> tuple[int, int, int, int]:
        return LATENT_SIZES[720 if self.resolution == "720" else 480]

    @property
    def effective_device(self) -> str:
        from rmd.platform import get_device

        return get_device(None if self.device == "auto" else self.device)


def load_config(path: str) -> TrainConfig:
    """Load a TrainConfig from a yaml file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config file {path} must contain a mapping, got {type(data)}")
    return TrainConfig(**data)


def load_infer_config(model_dir: str, args=None) -> InferConfig:
    """Build an InferConfig from a model dir plus optional CLI overrides."""
    kwargs: dict = {"model_dir": model_dir}
    if args is not None:
        for key in (
            "pretrained_model_name_or_path",
            "resolution",
            "k_step",
            "flow_shift_trans",
            "low_rs_step",
            "eta",
            "seed",
            "output_dir",
            "device",
        ):
            value = getattr(args, key, None)
            if value is not None:
                kwargs[key] = value
    return InferConfig(**kwargs)
