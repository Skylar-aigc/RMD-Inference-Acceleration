"""Inference for a distilled RMD student model.

Ports the generation loop from the original ``infer/infer_rmd.py`` into a
reusable function (``infer``) plus a CLI wrapper (``infer_from_cli``) that
writes videos to disk. Also exposes ``infer_prompt`` used by the Gradio demo.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from rmd.config import InferConfig
from rmd.engines.predictor import EulerSolver, Predictor
from rmd.models.prompt import compute_prompt_embeddings
from rmd.models.wan import load_student_for_infer, load_tokenizer_text_encoder, load_vae
from rmd.parallel.sharding import round_robin_indices
from rmd.platform import patch_torch_for_device, set_device

logger = logging.getLogger(__name__)


def _weight_dtype(cfg: InferConfig) -> torch.dtype:
    if cfg.mixed_precision == "fp16":
        return torch.float16
    if cfg.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _vae_decode(vae, model_input):
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(model_input.device, model_input.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        model_input.device, model_input.dtype
    )
    model_input = model_input / latents_std + latents_mean
    return vae.decode(model_input.to(vae.dtype), return_dict=False)[0]


def infer(
    cfg: InferConfig,
    prompts: list[str],
    seed: int = 42,
    device: str | None = None,
    seeds: list[int] | None = None,
) -> list[np.ndarray]:
    """Generate videos for ``prompts`` using the distilled student model.

    Returns a list of numpy arrays (``[T, H, W, C]``) in 0-255 uint8 range.
    """
    device = device or cfg.effective_device
    set_device(device)
    patch_torch_for_device(device)
    torch.manual_seed(seed)

    weight_dtype = _weight_dtype(cfg)
    tokenizer, text_encoder = load_tokenizer_text_encoder(cfg, device)
    transformer = load_student_for_infer(cfg, device, enable_sp=cfg.enable_sp)
    vae = load_vae(cfg, device)

    latent_init, latent_target = cfg.source_target_latent_sizes
    temporal = latent_init[1]
    _, _, target_h, target_w = latent_target

    solver = EulerSolver(
        FlowMatchEulerDiscreteScheduler(shift=cfg.flow_shift).sigmas.numpy()[::-1],
        cfg.num_euler_timesteps,
        euler_timesteps=cfg.num_euler_timesteps,
    ).to(device)
    solver_hrs = EulerSolver(
        FlowMatchEulerDiscreteScheduler(shift=5.0).sigmas.numpy()[::-1],
        cfg.num_euler_timesteps,
        euler_timesteps=cfg.num_euler_timesteps,
    ).to(device)

    if cfg.low_rs_step:
        relusion_shift = cfg.low_rs_step - 1
    else:
        relusion_shift = (cfg.k_step - 1) // 2 - 1

    predictor = Predictor(cfg, tokenizer, text_encoder, device, weight_dtype, solver, solver_hrs)
    videoprocessor = VideoProcessor(vae_scale_factor=8.0)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=cfg.flow_shift)

    videos = []
    for index, prompt in enumerate(prompts):
        torch.manual_seed(seeds[index] if seeds is not None else seed + index)
        prompt_embeds = compute_prompt_embeddings(
            tokenizer, text_encoder, [prompt], 512, device, weight_dtype, requires_grad=False
        )
        noise = (
            torch.randn(1, latent_init[0], temporal, latent_init[2], latent_init[3]).to(device).to(dtype=weight_dtype)
        )
        with torch.no_grad():
            model_input = predictor.generate_new_upsample(
                transformer,
                noise_scheduler,
                noise,
                noise,
                prompt_embeds,
                None,
                eta=cfg.eta,
                steps=cfg.k_step,
                total_steps=cfg.num_euler_timesteps,
                flow_shift_trans=cfg.flow_shift_trans,
                relusion_shift=relusion_shift,
                target_size=(target_h, target_w),
                mode=cfg.infer_upsampler_mode,
            )
            images_t = _vae_decode(vae, model_input)
        video = videoprocessor.postprocess_video(images_t, output_type="np")[0]
        videos.append(video)
    return videos


def infer_from_cli(cfg: InferConfig, prompt: str | None = None, prompt_file: str | None = None) -> None:
    """CLI entry: generate and save videos to ``cfg.output_dir``."""
    if prompt is not None:
        prompts = [prompt]
    elif prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = [
            (
                "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a "
                "serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar."
            )
        ]
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=cfg.device == "cpu")
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    local_indices = round_robin_indices(len(prompts), rank, world_size)
    local_prompts = [prompts[index] for index in local_indices]
    local_seeds = [cfg.seed + index for index in local_indices]
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    logger.info(
        "Rank %d/%d generating %d of %d videos -> %s",
        rank,
        world_size,
        len(local_prompts),
        len(prompts),
        cfg.output_dir,
    )

    videos = (
        infer(
            cfg,
            local_prompts,
            seed=cfg.seed,
            device=str(accelerator.device),
            seeds=local_seeds,
        )
        if local_prompts
        else []
    )
    for i, p, video in zip(local_indices, local_prompts, videos):
        name = f"out_{i:03d}.mp4"
        export_to_video(video, os.path.join(cfg.output_dir, name), fps=16)
        logger.info("Saved %s (prompt: %s)", name, p[:60])
    accelerator.wait_for_everyone()
