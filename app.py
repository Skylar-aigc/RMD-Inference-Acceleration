"""Gradio demo: few-step Wan video generation with an RMD student model.

Run with ``rmd-demo --model_dir <ckpt>`` or ``python app.py --model_dir <ckpt>``.
"""

from __future__ import annotations

import argparse

import gradio as gr

from rmd.config import InferConfig
from rmd.engines.infer import infer


def build_demo(cfg: InferConfig) -> gr.Blocks:
    """Build the Gradio app. Model + VAE weights load lazily on first request."""

    def generate(prompt: str, k_step: int, resolution: str, flow_shift_trans: int, seed: int):
        run_cfg = InferConfig(
            model_dir=cfg.model_dir,
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            k_step=k_step,
            resolution=resolution,
            flow_shift_trans=flow_shift_trans,
            low_rs_step=min(cfg.low_rs_step, k_step),
            eta=cfg.eta,
            seed=seed,
            device=cfg.device,
        )
        videos = infer(run_cfg, [prompt], seed=seed)
        return videos[0]

    with gr.Blocks(title="RMD · Few-step Wan video generation") as demo:
        gr.Markdown(
            "# 🎬 RMD: Few-step Wan video distillation\n\n"
            "Generate videos with a distilled student model in as few as 4 steps "
            "(the base Wan2.1 model needs ~50). Tune the step count and resolution below."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool "
                "in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar.",
                lines=3,
            )
        with gr.Row():
            k_step = gr.Slider(1, 8, value=cfg.k_step, step=1, label="Sampling steps (K)")
            resolution = gr.Dropdown(["480", "720"], value=cfg.resolution, label="Resolution")
            flow_shift_trans = gr.Dropdown([0, 1, 2], value=cfg.flow_shift_trans, label="Flow-shift transition")
            seed = gr.Slider(0, 9999, value=cfg.seed, step=1, label="Seed")
        btn = gr.Button("Generate", variant="primary")
        out = gr.Video(label="Generated video", format="mp4")
        btn.click(generate, inputs=[prompt, k_step, resolution, flow_shift_trans, seed], outputs=out)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="rmd-demo", description="Launch the RMD Gradio demo.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained student weights.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "npu", "cpu"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public shareable link.")
    args = parser.parse_args()

    cfg = InferConfig(
        model_dir=args.model_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device=args.device,
    )
    build_demo(cfg).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
