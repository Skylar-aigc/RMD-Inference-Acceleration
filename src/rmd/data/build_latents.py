"""Encode a folder of videos into VAE latents for training.

Cleaned from the original ``infer/vae_p.py``: hardcoded server paths are
parameterized and the CLI is replaced by a callable function + ``rmd-build-latents``.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.utils import load_video
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def upsample_video(video, size: tuple[int, int]):
    N, C, T, H, W = video.shape
    video_2d = video.reshape(N * T, C, H, W)
    up = F.interpolate(video_2d, size=size, mode="bicubic", align_corners=False)
    return up.reshape(N, C, T, size[0], size[1])


def build_latents(
    input_dir: str,
    output_path: str,
    pretrained_path: str,
    height: int = 480,
    width: int = 832,
    target: tuple[int, int] = (720, 1280),
) -> None:
    """Encode all videos under ``input_dir`` into a latent ``.pt`` dataset.

    The ``.pt`` file is a list of ``{"prompt": name, "latent": tensor}`` records
    compatible with :class:`rmd.data.dataset.load_prompt_dataset`.
    """
    device = torch.cuda.current_device()
    torch.set_grad_enabled(False)

    vae = AutoencoderKLWan.from_pretrained(pretrained_path, subfolder="vae", torch_dtype=torch.float32).to(
        device=device, dtype=torch.float32
    )

    video_paths = [p for p in sorted(os.listdir(input_dir)) if not p.startswith(".")]
    os.makedirs(output_path, exist_ok=True)

    video_processor = VideoProcessor(vae_scale_factor=8)
    dataset = []

    for video_path in tqdm(video_paths, desc="Encoding videos"):
        video = load_video(os.path.join(input_dir, video_path))
        prompt = os.path.splitext(video_path)[0]
        video = video_processor.preprocess_video(video, height=height, width=width).to(device, dtype=torch.float32)
        video = upsample_video(video, size=target)

        init_latents = retrieve_latents(vae.encode(video), sample_mode="argmax")
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(init_latents.device, init_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            init_latents.device, init_latents.dtype
        )
        init_latents = init_latents.to(torch.float32)
        init_latents = (init_latents - latents_mean) * latents_std

        dataset.append({"prompt": prompt, "latent": init_latents.cpu().detach()})

    out_file = os.path.join(output_path, "latents.pt")
    torch.save(dataset, out_file)
    logger.info("Saved %d latents to %s", len(dataset), out_file)


def retrieve_latents(encoder_output, generator=None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
