"""Batch VBench-style evaluation over a prompt set.

Runs the distilled student over a folder of prompts and writes one video per
prompt. Adapt the prompt source to your VBench manifest as needed.
"""

from __future__ import annotations

import argparse
import logging
import os

from rmd.config import InferConfig

logger = logging.getLogger(__name__)


def run_vbench(cfg: InferConfig, prompt_file: str, output_dir: str, iterations: int = 5) -> None:
    """Generate ``iterations`` videos per prompt into ``output_dir``."""
    from diffusers.utils import export_to_video

    from rmd.engines.infer import infer

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]

    os.makedirs(output_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        for it in range(iterations):
            videos = infer(cfg, [prompt], seed=cfg.seed + it * 100 + idx)
            # VBench's standard convention is ``<prompt>-<sample index>.mp4``.
            # Forward slashes are the only characters that cannot appear in a
            # Linux filename, so replace them without otherwise truncating the prompt.
            filename_prompt = prompt.replace("/", "_")
            name = os.path.join(output_dir, f"{filename_prompt}-{it}.mp4")
            export_to_video(videos[0], name, fps=16)
            logger.info("Saved %s", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBench-style batch generation.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Hugging Face model ID or local base Wan Diffusers directory.",
    )
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/vbench")
    parser.add_argument("--k_step", type=int, default=4)
    parser.add_argument("--resolution", choices=["480", "720"], default="480")
    parser.add_argument("--flow_shift_trans", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--low_rs_step", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = InferConfig(
        model_dir=args.model_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        k_step=args.k_step,
        resolution=args.resolution,
        flow_shift_trans=args.flow_shift_trans,
        low_rs_step=args.low_rs_step,
        eta=args.eta,
        seed=args.seed,
        device=args.device,
    )
    run_vbench(cfg, args.prompt_file, args.output_dir, iterations=args.iterations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
