from pathlib import Path

import yaml


def test_fsdp_config_targets_eight_processes():
    config = yaml.safe_load(Path("configs/accelerate_fsdp.yaml").read_text(encoding="utf-8"))

    assert config["distributed_type"] == "FSDP"
    assert config["num_machines"] == 1
    assert config["num_processes"] == 8
    assert config["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"
    assert config["fsdp_config"]["fsdp_state_dict_type"] == "SHARDED_STATE_DICT"
    assert config["fsdp_config"]["fsdp_backward_prefetch_policy"] == "BACKWARD_PRE"


def test_eight_gpu_launcher_uses_accelerate_config():
    launcher = Path("scripts/train_8gpu.sh").read_text(encoding="utf-8")

    assert "accelerate launch" in launcher
    assert "--config_file configs/accelerate_fsdp.yaml" in launcher
    assert "--num_processes 8" in launcher


def test_seven_gpu_fsdp_recipe_wraps_wan_blocks():
    config = yaml.safe_load(Path("configs/accelerate_fsdp_7.yaml").read_text(encoding="utf-8"))

    assert config["distributed_type"] == "FSDP"
    assert config["num_processes"] == 7
    assert config["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"
    assert config["fsdp_config"]["fsdp_state_dict_type"] == "SHARDED_STATE_DICT"
    assert config["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == "WanTransformerBlock"


def test_seven_gpu_training_recipe_matches_documented_settings():
    config = yaml.safe_load(Path("configs/train_wan_1_3b_7gpu.yaml").read_text(encoding="utf-8"))

    assert config["train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 4
    assert config["eta"] == 0.0
    assert config["split_timestep"] is False
    assert config["timestep_offset_mode"] == "zero"
    assert config["adam_beta1"] == 0.0
    assert config["lr_scheduler"] == "cosine_with_restarts"
    assert config["validation_prompt_path"] == "examples/sample_prompts.txt"


def test_seven_gpu_launcher_uses_fsdp_config():
    launcher = Path("scripts/train_7gpu.sh").read_text(encoding="utf-8")

    assert "--config_file configs/accelerate_fsdp_7.yaml" in launcher
    assert "--num_processes 7" in launcher
    assert "src/rmd/cli.py" in launcher
    assert "train" in launcher
