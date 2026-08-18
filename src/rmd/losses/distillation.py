"""Loss terms for RMD distillation. All pure functions over tensors.

Extracted from the original training scripts so each term can be unit tested
and the numerical behavior of the paper is preserved exactly.
"""

from __future__ import annotations

import torch


def compute_score_loss(
    fake_latents: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor,
    eps: float = 1e-8,
    mode: str = "inverse_sigma",
    legacy_cap: float = 5.0,
) -> torch.Tensor:
    """Sigma-weighted MSE between the student prediction and teacher target.

    ``inverse_sigma`` is the default RMD objective. ``legacy_snr`` is retained
    for reproducing checkpoints trained with the earlier capped weighting.
    """
    # Compute the small per-sample weighting tensor in float64, then convert it
    # before broadcasting over the full latent tensor.
    sigmas_for_weight = sigmas.to(dtype=torch.float64)
    if mode == "inverse_sigma":
        weighting = ((1.0 - sigmas_for_weight) / (sigmas_for_weight + eps)) ** 2
    elif mode == "legacy_snr":
        weighting = (sigmas_for_weight / (1.0 - sigmas_for_weight + eps)) ** 2
        weighting = weighting.clamp(max=legacy_cap)
    else:
        raise ValueError(f"Unknown score weighting mode: {mode}")
    weighting = weighting.to(dtype=torch.float32)
    bsz = fake_latents.shape[0]
    loss = torch.mean((weighting * (fake_latents - target) ** 2).reshape(bsz, -1), dim=1)
    return loss.mean()


def compute_revised_latents(
    model_latents: torch.Tensor, real_latents: torch.Tensor, fake_latents: torch.Tensor
) -> torch.Tensor:
    """Adversarial revision target: ``model + real - fake``.

    Moves the teacher prediction by the difference between a real-model
    prediction and the student prediction, pushing the generator toward
    trajectories the real model agrees with.
    """
    return model_latents + real_latents - fake_latents


def compute_weighting_factor(
    model_latents: torch.Tensor, real_latents: torch.Tensor, clip_min: float = 0.2
) -> torch.Tensor:
    """Per-sample adaptive weighting used in the student Huber loss."""
    factor = torch.abs(model_latents - real_latents).mean(dim=[1, 2, 3], keepdim=True).detach()
    return factor.clip(clip_min)


def compute_huber_c(latent_shape, base: float = 1e-3, ref: int = 64 * 64 * 4) -> float:
    """Scale-aware Huber delta (matches the original resolution-dependent value)."""
    # A tensor shape commonly includes a leading batch dimension. The legacy
    # formula is resolution-dependent only (C*T*H*W), so batch size must not
    # change the loss hyperparameter.
    if len(latent_shape) == 5:
        latent_shape = latent_shape[1:]
    numel = 1
    for d in latent_shape:
        numel *= d
    return base / (ref**0.5) * (numel**0.5)


def compute_student_loss(
    model_latents: torch.Tensor,
    revised_latents: torch.Tensor,
    weighting_factor: torch.Tensor,
    huber_c: float,
) -> torch.Tensor:
    """Pseudo-Huber student loss against the revised target."""
    diff = model_latents.float() - revised_latents.detach().float()
    huber = torch.sqrt(diff**2 + huber_c**2) - huber_c
    return torch.mean(huber / weighting_factor)
