"""Unit tests for the pydantic config system (no torch required)."""

import pytest
from pydantic import ValidationError

from rmd.config import LATENT_SIZES, InferConfig, TrainConfig, load_config
from rmd.data.dataset import load_prompt_dataset


def test_defaults():
    cfg = TrainConfig()
    assert cfg.k_step == 6
    assert cfg.gan is False
    assert cfg.flow_shift_trans == 0
    assert cfg.resolution == "480"
    assert cfg.checkpointing_steps == 50


def test_invalid_flow_shift_trans():
    with pytest.raises(ValidationError):
        TrainConfig(flow_shift_trans=5)


def test_infer_multistage_uses_480p_source_and_720p_target():
    cfg = InferConfig(model_dir="student", resolution="720", flow_shift_trans=1, low_rs_step=3, k_step=6)
    assert cfg.source_target_latent_sizes == (LATENT_SIZES[480], LATENT_SIZES[720])


def test_infer_single_resolution_keeps_requested_layout():
    cfg = InferConfig(model_dir="student", resolution="480", flow_shift_trans=0)
    assert cfg.source_target_latent_sizes == (LATENT_SIZES[480], LATENT_SIZES[480])


def test_invalid_k_step():
    with pytest.raises(ValidationError):
        TrainConfig(k_step=0)


def test_invalid_low_rs_step():
    with pytest.raises(ValidationError):
        TrainConfig(k_step=4, low_rs_step=5)


def test_zero_train_steps_is_valid():
    assert TrainConfig(max_train_steps=0).max_train_steps == 0


def test_invalid_prompt_embed_batch_size():
    with pytest.raises(ValidationError):
        TrainConfig(prompt_embed_batch_size=0)


def test_latent_size_720():
    cfg = TrainConfig(resolution="720")
    assert cfg.latent_size == (16, 21, 90, 160)


def test_latent_size_480():
    cfg = TrainConfig(resolution="480")
    assert cfg.latent_size == (16, 21, 60, 104)


def test_multistage_forces_720p():
    cfg = TrainConfig(resolution="480", multistage_upsample=True)
    assert cfg.flow_shift_trans >= 1
    assert cfg.resolution == "720"


def test_load_config_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("k_step: 4\nresolution: 480\n", encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.k_step == 4
    assert cfg.resolution == "480"


def test_infer_config_has_model_loading_fields():
    cfg = InferConfig(model_dir="checkpoint")
    assert cfg.revision is None
    assert cfg.variant is None
    assert cfg.infer_upsampler_mode == "bilinear"


def test_eight_gpu_recipes_have_expected_training_budget():
    small = load_config("configs/train_wan_1_3b.yaml")
    large = load_config("configs/train_wan_14b.yaml")

    assert small.max_train_steps == large.max_train_steps == 300
    assert small.checkpointing_steps == large.checkpointing_steps == 50
    assert small.train_batch_size == large.train_batch_size == 1
    assert small.gradient_accumulation_steps == 1
    assert large.gradient_accumulation_steps == 3
    assert small.data_path == large.data_path == "examples/vidprom_prompts_2000.txt"


def test_bundled_vidprom_subset_has_2000_unique_prompts():
    prompts = load_prompt_dataset("examples/vidprom_prompts_2000.txt")

    assert len(prompts) == 2000
    assert len(set(prompts)) == 2000
    assert all(prompt.strip() for prompt in prompts)
