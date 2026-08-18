"""Tests for CLI argument parsing and config override (no torch required)."""

import argparse

import pytest
from pydantic import ValidationError

from rmd.cli import _apply_overrides
from rmd.config import TrainConfig


def test_zero_step_cli_returns_before_importing_training_engine(monkeypatch, tmp_path):
    import sys

    from rmd.cli import train_main

    monkeypatch.setitem(sys.modules, "rmd.engines.train", None)
    train_main(
        [
            "--config",
            "configs/train_wan_1_3b.yaml",
            "--output_dir",
            str(tmp_path),
            "--max_train_steps",
            "0",
        ]
    )


def test_module_main_dispatches_infer_without_mutating_sys_argv(monkeypatch):
    from rmd import cli

    calls = []
    monkeypatch.setattr(cli, "infer_main", lambda argv=None: calls.append(argv))

    cli.main(["infer", "--model_dir", "student"])

    assert calls == [["--model_dir", "student"]]


def _make_args(**kwargs):
    ns = argparse.Namespace()
    for k in (
        "pretrained_model_name_or_path",
        "output_dir",
        "k_step",
        "resolution",
        "device",
        "data_path",
        "flow_shift_trans",
        "max_train_steps",
        "seed",
    ):
        setattr(ns, k, kwargs.get(k))
    return ns


def test_apply_overrides_sets_only_provided():
    cfg = TrainConfig()
    _apply_overrides(cfg, _make_args(k_step=4, output_dir="tmp_out"))
    assert cfg.k_step == 4
    assert cfg.output_dir == "tmp_out"
    # unset fields untouched
    assert cfg.resolution == "480"
    assert cfg.device == "auto"


def test_apply_overrides_ignores_none():
    cfg = TrainConfig(k_step=6)
    _apply_overrides(cfg, _make_args())
    assert cfg.k_step == 6


def test_apply_overrides_device():
    cfg = TrainConfig()
    _apply_overrides(cfg, _make_args(device="cuda"))
    assert cfg.device == "cuda"


def test_apply_overrides_multistage_and_low_rs():
    cfg = TrainConfig()
    ns = _make_args()
    ns.multistage_upsample = True
    ns.low_rs_step = 3
    ns.gradient_checkpointing = True
    ns.mixed_precision = "bf16"
    _apply_overrides(cfg, ns)
    assert cfg.multistage_upsample is True
    assert cfg.low_rs_step == 3
    assert cfg.gradient_checkpointing is True
    assert cfg.mixed_precision == "bf16"
    assert cfg.flow_shift_trans == 1
    assert cfg.resolution == "720"


def test_apply_overrides_revalidates_fields():
    cfg = TrainConfig()
    with pytest.raises(ValidationError):
        _apply_overrides(cfg, _make_args(k_step=0))
