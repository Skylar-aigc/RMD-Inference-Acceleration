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
