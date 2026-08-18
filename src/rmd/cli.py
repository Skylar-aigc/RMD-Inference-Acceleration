"""Command-line entry points for RMD.

Exposes the console scripts declared in ``pyproject.toml``:
``rmd-train``, ``rmd-infer``, ``rmd-demo``, ``rmd-build-latents``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from rmd.config import TrainConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rmd.cli")


def _add_common_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--config", type=str, default=None, help="Path to a yaml config file.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "npu", "cpu"],
        help="Override the device from the config.",
    )
    return p


_TRAIN_OVERRIDE_KEYS = (
    "pretrained_model_name_or_path",
    "output_dir",
    "k_step",
    "resolution",
    "device",
    "data_path",
    "flow_shift_trans",
    "max_train_steps",
    "seed",
    "multistage_upsample",
    "low_rs_step",
    "enable_sp",
    "gradient_checkpointing",
    "mixed_precision",
    "resume_from_checkpoint",
    "checkpointing_steps",
    "prompt_embed_batch_size",
    "use_8bit_adam",
    "adam_beta1",
    "adam_beta2",
    "validation_prompt_path",
)


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply non-None CLI overrides onto a config object."""
    data = cfg.model_dump()
    for key in _TRAIN_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            data[key] = value
    # Rebuild the model so field and cross-field validators also apply to CLI
    # values (plain setattr would bypass Pydantic validation).
    validated = TrainConfig.model_validate(data)
    # Preserve the helper's existing in-place behavior for callers that keep
    # using the original object, but only copy values from a validated model.
    for key, value in validated:
        setattr(cfg, key, value)
    return cfg


def train_main(argv: Sequence[str] | None = None) -> None:
    """Train the RMD few-step student model."""
    p = _add_common_parser("rmd-train", "Train the RMD few-step Wan student model.")
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--k_step", type=int, default=None)
    p.add_argument("--resolution", choices=["480", "720"], default=None)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--flow_shift_trans", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--checkpointing_steps", type=int, default=None)
    p.add_argument("--prompt_embed_batch_size", type=int, default=None)
    p.add_argument("--use_8bit_adam", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--adam_beta1", type=float, default=None)
    p.add_argument("--adam_beta2", type=float, default=None)
    p.add_argument("--validation_prompt_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--multistage_upsample", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--enable_sp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--low_rs_step", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config) if args.config else TrainConfig()
    cfg = _apply_overrides(cfg, args)

    # Accelerate resolves the concrete local-rank device inside train(); doing
    # so here would make every distributed worker probe the default GPU first.
    logger.info("Launching RMD training: device=%s k_step=%s resolution=%s", cfg.device, cfg.k_step, cfg.resolution)
    # Configuration-only validation should not import torch/diffusers or build
    # any model when the requested training length is explicitly zero.
    if cfg.max_train_steps == 0:
        logger.info("max_train_steps is 0; configuration is valid, nothing to train.")
        return
    from rmd.engines.train import train

    train(cfg)


def infer_main(argv: Sequence[str] | None = None) -> None:
    """Generate videos with a trained student model."""

    p = _add_common_parser("rmd-infer", "Generate videos with an RMD student model.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None, help="A single prompt; overrides --prompt_file.")
    p.add_argument("--prompt_file", type=str, default=None, help="Txt file, one prompt per line.")
    p.add_argument("--resolution", choices=["480", "720"], default=None)
    p.add_argument("--k_step", type=int, default=None)
    p.add_argument("--flow_shift_trans", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--low_rs_step", type=int, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--enable_sp", action=argparse.BooleanOptionalAction, default=None)
    args = p.parse_args(argv)

    from rmd.config import load_infer_config
    from rmd.engines.infer import infer_from_cli

    cfg = load_infer_config(args.model_dir, args)
    infer_from_cli(cfg, prompt=args.prompt, prompt_file=args.prompt_file)


def demo_main(argv: Sequence[str] | None = None) -> None:
    """Launch the Gradio demo app."""
    from rmd.config import InferConfig

    p = _add_common_parser("rmd-demo", "Launch the Gradio demo for RMD.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--pretrained_model_name_or_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--resolution", choices=["480", "720"], default="720")
    p.add_argument("--k_step", type=int, default=6)
    p.add_argument("--flow_shift_trans", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--low_rs_step", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args(argv)

    cfg = InferConfig(
        model_dir=args.model_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        device=args.device or "auto",
        resolution=args.resolution,
        k_step=args.k_step,
        flow_shift_trans=args.flow_shift_trans,
        low_rs_step=args.low_rs_step,
        eta=args.eta,
        seed=args.seed,
    )
    from app import build_demo

    build_demo(cfg).launch(server_port=args.port, share=args.share)


def build_latents_main(argv: Sequence[str] | None = None) -> None:
    """Encode a folder of videos into VAE latents for training."""
    from rmd.config import InferConfig  # noqa: F401  (module import for side effects)

    p = _add_common_parser("rmd-build-latents", "Encode videos into latent training data.")
    p.add_argument("--input_video_folder", type=str, required=True)
    p.add_argument("--output_latent_folder", type=str, required=True)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--target_height", type=int, default=720)
    p.add_argument("--target_width", type=int, default=1280)
    args = p.parse_args(argv)

    from rmd.data.build_latents import build_latents

    build_latents(
        input_dir=args.input_video_folder,
        output_path=args.output_latent_folder,
        pretrained_path=args.pretrained_model_name_or_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        height=args.height,
        width=args.width,
        target=(args.target_height, args.target_width),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch ``python -m rmd.cli {train,infer,demo,build-latents}``."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "train": train_main,
        "infer": infer_main,
        "demo": demo_main,
        "build-latents": build_latents_main,
    }
    command = commands.get(args[0]) if args else None
    if command is None:
        train_main(args)
    else:
        command(args[1:])


if __name__ == "__main__":
    main()
