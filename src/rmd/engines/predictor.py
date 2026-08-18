"""Sampling and solver utilities for RMD.

Merged from the original ``utils/predictor.py`` and ``utils/solver.py``.
Numerical behavior is preserved exactly; only imports and device handling are
cleaned up (devices go through :mod:`rmd.platform`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rmd.models.prompt import compute_prompt_embeddings

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards"
)


def extract_into_tensor(a, t, x_shape):
    """Gather ``a`` at indices ``t`` and reshape to match ``x_shape`` (per-batch)."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gather_trajectory_batch(noisy_imgs_list, ind_t):
    """Gather a batch of trajectory states using the sampled phase resolution."""
    first_index = int(ind_t[0].item())
    batch = torch.empty_like(noisy_imgs_list[first_index])
    expected_shape = batch.shape[1:]
    for i in range(ind_t.shape[0]):
        sample = noisy_imgs_list[int(ind_t[i].item())][i]
        if sample.shape != expected_shape:
            raise ValueError(
                "A training batch mixed low- and high-resolution trajectory states: "
                f"expected {tuple(expected_shape)}, got {tuple(sample.shape)}"
            )
        batch[i] = sample
    return batch


class EulerSolver:
    """Euler-step solver over a sigma schedule."""

    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (np.arange(1, euler_timesteps + 1) * self.step_ratio).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray([0] + self.euler_timesteps[:-1].tolist())
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()
        )  # either use sigma0 or 0

        self.euler_timesteps = torch.from_numpy(self.euler_timesteps).long()
        self.euler_timesteps_prev = torch.from_numpy(self.euler_timesteps_prev).long()
        self.sigmas = torch.from_numpy(self.sigmas)
        self.sigmas_prev = torch.from_numpy(self.sigmas_prev)

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.euler_timesteps_prev = self.euler_timesteps_prev.to(device)
        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index, model_pred.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev


class Predictor:
    """Wraps a frozen score model for K-step trajectory generation and distillation sampling."""

    def __init__(
        self,
        args,
        tokenizer,
        text_encoder,
        device,
        weight_dtype,
        solver,
        solver_hrs=None,
        uncond_prompt_embeds=None,
    ):
        super().__init__()
        if uncond_prompt_embeds is None:
            if tokenizer is None or text_encoder is None:
                raise ValueError("tokenizer and text_encoder are required without precomputed embeddings")
            uncond_prompt_embeds = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                [DEFAULT_NEGATIVE_PROMPT],
                512,
                device,
                weight_dtype,
                requires_grad=False,
            )
        self.uncond_prompt_embeds = uncond_prompt_embeds
        self.weight_dtype = weight_dtype
        self.solver = solver
        self.solver_hrs = solver_hrs

    def predict(
        self,
        score_model,
        noisy_samples,
        timesteps,
        encoder_hidden_states,
        cfg=None,
        steps=1,
        return_double=False,
        timestep_cond=None,
        encoder_hidden_states_image=None,
        conditions=None,
        higher_shift=False,
    ):
        weight_dtype = self.weight_dtype
        if higher_shift and self.solver_hrs:
            sigmas = extract_into_tensor(self.solver_hrs.sigmas, timesteps, noisy_samples.shape)
        else:
            sigmas = extract_into_tensor(self.solver.sigmas, timesteps, noisy_samples.shape)
        timesteps_sd3 = (sigmas * 1000).squeeze().reshape(noisy_samples.shape[0])

        score_pred = score_model(
            hidden_states=noisy_samples.to(weight_dtype),
            timestep=timesteps_sd3.to(weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(weight_dtype),
            return_dict=False,
        )[0]
        if cfg is not None:
            uncond_prompt_embeds = self.uncond_prompt_embeds.to(noisy_samples.device, dtype=weight_dtype)
            if uncond_prompt_embeds.shape[0] == 1 and noisy_samples.shape[0] > 1:
                uncond_prompt_embeds = uncond_prompt_embeds.expand(noisy_samples.shape[0], -1, -1)
            else:
                uncond_prompt_embeds = uncond_prompt_embeds[: noisy_samples.shape[0]]
            score_uncon_pred = score_model(
                hidden_states=noisy_samples.to(weight_dtype),
                timestep=timesteps_sd3.to(weight_dtype),
                encoder_hidden_states=uncond_prompt_embeds,
                return_dict=False,
            )[0]
            pred_latents_cond = score_pred * (-sigmas) + noisy_samples
            pred_latents_uncond = score_uncon_pred * (-sigmas) + noisy_samples
            pred_latents = pred_latents_uncond + cfg * (pred_latents_cond - pred_latents_uncond)
            if return_double:
                pred_epsilon_cond = noisy_samples + (1 - sigmas) * score_pred
                pred_epsilon_uncond = noisy_samples + (1 - sigmas) * score_uncon_pred
                pred_epsilon = pred_epsilon_uncond + cfg * (pred_epsilon_cond - pred_epsilon_uncond)
                return pred_epsilon, pred_latents
        else:
            pred_latents = score_pred * (-sigmas) + noisy_samples
            if return_double:
                pred_epsilon = noisy_samples + (1 - sigmas) * score_pred
                return pred_epsilon, pred_latents
        return pred_latents

    def add_noise(self, samples, noise, t1, t2, higher_shift=False):
        """Add noise so a sample evolves from sigma(t1) to sigma(t2)."""
        if higher_shift and self.solver_hrs:
            sigmas = extract_into_tensor(self.solver_hrs.sigmas, t1, samples.shape)
            sigmas_new = extract_into_tensor(self.solver_hrs.sigmas, t2, samples.shape)
        else:
            sigmas = extract_into_tensor(self.solver.sigmas, t1, samples.shape)
            sigmas_new = extract_into_tensor(self.solver.sigmas, t2, samples.shape)
        samples = samples / (1 - sigmas) * (1 - sigmas_new)
        beta = sigmas_new**2 - (sigmas / (1 - sigmas) * (1 - sigmas_new)) ** 2
        beta = beta**0.5
        samples = samples + beta * noise
        return samples.to(self.weight_dtype)

    def generate_new(
        self,
        transformer,
        noise_scheduler,
        latent,
        noise,
        encoder_hidden_states,
        image_rotary_emb,
        steps=4,
        eta=1,
        return_mid=False,
        mid_points=None,
        encode_steps=True,
        shift=False,
        total_steps=1000,
        encoder_hidden_states_image=None,
        conditions=None,
    ):
        T_ = torch.randint(total_steps - 1, total_steps, (latent.shape[0],), device=latent.device)
        T_ = T_.long()
        zero_t = torch.zeros_like(T_)
        imgs_list = []
        pure_noisy = noise
        noisy_imgs_list = []

        for ind in range(steps):
            sigmas = extract_into_tensor(self.solver.sigmas, T_, noise.shape)
            T_sd3 = (sigmas * 1000).squeeze().reshape(noise.shape[0])
            noisy_imgs_list.append(pure_noisy)

            model_v = transformer(
                hidden_states=pure_noisy,
                encoder_hidden_states=encoder_hidden_states,
                timestep=T_sd3,
                return_dict=False,
            )[0]
            model_input = model_v * (-sigmas) + pure_noisy
            pred_epsilon = pure_noisy + (1 - sigmas) * model_v
            latent = model_input
            imgs_list.append(latent)
            if mid_points is not None:
                T_ = mid_points[ind + 1] + zero_t
            else:
                T_ = T_ - total_steps // steps
            add_eps = eta * pred_epsilon + ((1 - eta**2) ** 0.5) * torch.randn_like(pred_epsilon)
            T_[T_ < 0] = T_[T_ < 0].clone() * 0
            sigmas_new = extract_into_tensor(self.solver.sigmas, T_, noise.shape)
            pure_noisy = ((1.0 - sigmas_new) * latent + sigmas_new * add_eps).to(self.weight_dtype)
        noisy_imgs_list.append(latent)
        if return_mid:
            return imgs_list, noisy_imgs_list
        return latent

    def upsample_infer(self, latent, target_size=(90, 160), mode="bilinear"):
        B, C, T, H, W = latent.shape
        latent = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        latent_up = F.interpolate(latent, size=target_size, mode=mode)
        latent_up = latent_up.view(B, T, C, target_size[0], target_size[1]).permute(0, 2, 1, 3, 4)
        return latent_up

    def generate_new_upsample(
        self,
        transformer,
        noise_scheduler,
        latent,
        noise,
        encoder_hidden_states,
        image_rotary_emb,
        steps=4,
        eta=1,
        return_mid=False,
        mid_points=None,
        encode_steps=True,
        shift=False,
        total_steps=1000,
        encoder_hidden_states_image=None,
        conditions=None,
        flow_shift_trans=0,
        relusion_shift=4,
        target_size=(90, 160),
        mode="bilinear",
    ):
        T_ = torch.randint(total_steps - 1, total_steps, (latent.shape[0],), device=latent.device)
        T_ = T_.long()
        zero_t = torch.zeros_like(T_)
        imgs_list = []
        pure_noisy = noise
        noisy_imgs_list = []

        for ind in range(steps):
            if ind > relusion_shift and flow_shift_trans == 2:
                sigmas = extract_into_tensor(self.solver_hrs.sigmas, T_, noise.shape)
            else:
                sigmas = extract_into_tensor(self.solver.sigmas, T_, noise.shape)
            T_sd3 = (sigmas * 1000).squeeze().reshape(noise.shape[0])
            noisy_imgs_list.append(pure_noisy)

            model_v = transformer(
                hidden_states=pure_noisy,
                encoder_hidden_states=encoder_hidden_states,
                timestep=T_sd3,
                return_dict=False,
            )[0]
            model_input = model_v * (-sigmas) + pure_noisy
            pred_epsilon = pure_noisy + (1 - sigmas) * model_v
            if ind == relusion_shift and (flow_shift_trans == 1 or flow_shift_trans == 2):
                model_input = self.upsample_infer(model_input, target_size, mode)
                pred_epsilon = self.upsample_infer(pred_epsilon, target_size, mode)
            latent = model_input
            imgs_list.append(latent)
            if mid_points is not None:
                T_ = mid_points[ind + 1] + zero_t
            else:
                T_ = T_ - total_steps // steps
            add_eps = eta * pred_epsilon + ((1 - eta**2) ** 0.5) * torch.randn_like(pred_epsilon)
            T_[T_ < 0] = T_[T_ < 0].clone() * 0
            if ind >= relusion_shift and flow_shift_trans == 2:
                sigmas_new = extract_into_tensor(self.solver_hrs.sigmas, T_, noise.shape)
            else:
                sigmas_new = extract_into_tensor(self.solver.sigmas, T_, noise.shape)
            pure_noisy = ((1.0 - sigmas_new) * latent + sigmas_new * add_eps).to(self.weight_dtype)
        noisy_imgs_list.append(latent)
        if return_mid:
            return imgs_list, noisy_imgs_list
        return latent
