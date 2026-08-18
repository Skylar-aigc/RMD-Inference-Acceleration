"""Unified two-phase distillation training loop.

Replaces the three near-duplicate scripts (``train_rmd.py``,
``train_rmd_wan2_1.py``, ``train_tdm_gan.py``). Mode is selected by
:class:`rmd.config.TrainConfig` flags:

- ``flow_shift_trans`` / ``multistage_upsample``: 480p -> 720p latent upsampling
- ``gan``: optional discriminator objective (experimental)

The algorithm follows the original scripts exactly; only the loss terms were
factored into :mod:`rmd.losses.distillation` and device calls route through
:mod:`rmd.platform`.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler

from rmd.config import LATENT_SIZES, TrainConfig
from rmd.data.dataset import PromptDataset, build_dataloader, load_prompt_dataset
from rmd.engines.predictor import (
    DEFAULT_NEGATIVE_PROMPT,
    EulerSolver,
    Predictor,
    extract_into_tensor,
    gather_trajectory_batch,
)
from rmd.losses.distillation import (
    compute_huber_c,
    compute_revised_latents,
    compute_score_loss,
    compute_student_loss,
    compute_weighting_factor,
)
from rmd.models.prompt import gather_prompt_embeddings, precompute_prompt_embeddings
from rmd.models.wan import (
    load_tokenizer_text_encoder,
    load_training_transformer,
    resolve_sequence_parallel,
)
from rmd.platform import empty_cache, patch_torch_for_device, set_device

logger = get_logger(__name__)


def _weight_dtype(cfg: TrainConfig) -> torch.dtype:
    if cfg.mixed_precision == "fp16":
        return torch.float16
    if cfg.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _init_accelerator(cfg: TrainConfig) -> Accelerator:
    logging_dir = Path(cfg.output_dir, "logs")
    project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    gradient_plugin = GradientAccumulationPlugin(
        num_steps=cfg.gradient_accumulation_steps,
        # Training intentionally loops over the prompt loader until max steps.
        # Do not force a partial optimizer update at every short loader boundary.
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        cpu=cfg.device == "cpu",
        gradient_accumulation_plugin=gradient_plugin,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if accelerator.is_main_process and cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    return accelerator


def _build_optimizer(lr: float, cfg: TrainConfig, model) -> torch.optim.Optimizer:
    params = {"params": model.parameters(), "lr": lr}
    optimizer_class = torch.optim.AdamW
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("use_8bit_adam=true requires bitsandbytes; install RMD with the gpu extra") from exc
        optimizer_class = bnb.optim.AdamW8bit
    return optimizer_class(
        [params],
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )


def _sigmas(flow_shift: float, num_timesteps: int):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    return scheduler.sigmas.numpy()[::-1]


def _sample_timesteps(cfg: TrainConfig, device, ind_t, K_step, mid_step, scales, total_steps=1000):
    """Sample noise-destruction timesteps following the paper's noise-interval strategy."""
    bsz = ind_t.shape[0]
    timesteps_g = ind_t * total_steps // K_step - 1
    timesteps_mid = timesteps_g - total_steps // K_step + 1
    timesteps = timesteps_g.clone() * 0
    for ind_bw in range(bsz):
        if cfg.split_timestep:
            if ind_t[ind_bw] >= mid_step:
                top_timestep = 980
            else:
                top_timestep = (mid_step) * total_steps // K_step - 1
        else:
            top_timestep = 980
        if cfg.timestep_offset_mode == "zero":
            t_offset = 0
        else:
            t_offset = int((top_timestep - timesteps_mid[ind_bw].item()) * scales[K_step - ind_t[ind_bw]])
        timesteps[ind_bw] = torch.randint(t_offset + timesteps_mid[ind_bw], top_timestep, (1,), device=device)[0]
    return timesteps.long(), timesteps_g, timesteps_mid


def _sample_ind_t(cfg: TrainConfig, bsz, device, mid_step, K_step, global_step):
    if cfg.flow_shift_trans and (global_step < cfg.warmup_step or torch.rand(1).item() < 0.5):
        return torch.randint(mid_step, K_step + 1, (bsz,), device=device).long()
    elif cfg.flow_shift_trans:
        return torch.randint(1, mid_step, (bsz,), device=device).long()
    else:
        return torch.randint(1, K_step + 1, (bsz,), device=device).long()


def _resume_training(accelerator: Accelerator, cfg: TrainConfig) -> int:
    """Restore an Accelerator state checkpoint and return its global step."""
    checkpoint = cfg.resume_from_checkpoint
    if not checkpoint:
        return 0

    if checkpoint == "latest":
        output_dir = Path(cfg.output_dir)
        candidates = []
        if output_dir.is_dir():
            for path in output_dir.glob("checkpoint-*"):
                try:
                    candidates.append((int(path.name.rsplit("-", 1)[1]), path))
                except ValueError:
                    continue
        if not candidates:
            logger.warning("No checkpoint-* directories found under %s; starting from step 0", output_dir)
            return 0
        global_step, checkpoint_path = max(candidates)
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir() and not checkpoint_path.is_absolute():
            checkpoint_path = Path(cfg.output_dir, checkpoint_path)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
        try:
            global_step = int(checkpoint_path.name.rsplit("-", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Training checkpoint directory must end in '-<step>': {checkpoint_path}") from exc

    logger.info("Resuming training from %s (step %d)", checkpoint_path, global_step)
    accelerator.load_state(str(checkpoint_path))
    return global_step


def _decode_test_video(vae, latent):
    """Decode one latent batch to uint8 frames with the Wan VAE."""
    from diffusers.video_processor import VideoProcessor

    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    )
    latents_std = 1.0 / (
        torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    )
    decoded = vae.decode((latent / latents_std + latents_mean).to(vae.dtype), return_dict=False)[0]
    return VideoProcessor(vae_scale_factor=8.0).postprocess_video(decoded, output_type="np")[0]


def _generate_test_samples(
    cfg,
    accelerator,
    transformer,
    predictor,
    vae,
    test_prompts,
    test_prompt_embeds,
    global_step,
    weight_dtype,
    latent_init,
    latent_target,
    relusion_shift,
):
    """Run FSDP-safe validation forwards; only the main rank decodes/writes videos."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    if not test_prompts:
        return
    if accelerator.is_main_process:
        from diffusers.utils import export_to_video

        sample_dir = Path(cfg.output_dir, "test_samples", f"step-{global_step}")
        sample_dir.mkdir(parents=True, exist_ok=True)

    was_training = transformer.training
    transformer.eval()
    _, temporal, source_h, source_w = latent_init
    _, _, target_h, target_w = latent_target
    scheduler = FlowMatchEulerDiscreteScheduler(shift=cfg.flow_shift)
    for index, prompt in enumerate(test_prompts):
        # All ranks use identical inputs and participate in every FSDP/DDP
        # collective. A rank-0-only transformer forward would deadlock FSDP.
        generator = torch.Generator(device=accelerator.device).manual_seed(
            (cfg.seed or 0) + global_step * 10_000 + index
        )
        noise = torch.randn(
            1,
            latent_init[0],
            temporal,
            source_h,
            source_w,
            generator=generator,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        prompt_embeds = test_prompt_embeds[prompt].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            latent = predictor.generate_new_upsample(
                transformer,
                scheduler,
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
            if accelerator.is_main_process:
                video = _decode_test_video(vae, latent)
                export_to_video(video, sample_dir / f"out_{index:03d}.mp4", fps=16)
                logger.info(
                    "Saved validation sample %d at step %d (prompt: %s)",
                    index,
                    global_step,
                    prompt[:60],
                    main_process_only=True,
                )
    if was_training:
        transformer.train()
    accelerator.wait_for_everyone()
    empty_cache(str(accelerator.device))


def train(cfg: TrainConfig) -> None:
    """Train the RMD student model."""
    if cfg.max_train_steps == 0:
        logger.info("max_train_steps is 0; nothing to train.")
        return
    accelerator = _init_accelerator(cfg)
    # Accelerate owns local-rank placement in distributed runs. Resolving the
    # device before this point would make every worker briefly initialize GPU 0.
    device = str(accelerator.device)
    set_device(device)
    patch_torch_for_device(device)
    if cfg.seed is not None:
        # Give every worker an independent, deterministic random stream.
        set_seed(cfg.seed, device_specific=True)
    tracker_config = {key: value for key, value in cfg.model_dump(mode="json").items() if value is not None}
    accelerator.init_trackers("rmd", config=tracker_config)
    weight_dtype = _weight_dtype(cfg)

    prompts = load_prompt_dataset(cfg.data_path)
    if not prompts:
        raise ValueError(f"Training dataset is empty: {cfg.data_path}")

    # Preparing the loader is what actually partitions batches between the
    # eight Accelerate workers. Without this, every GPU repeats the same data.
    train_dataset = PromptDataset(prompts)
    train_dataloader = accelerator.prepare(build_dataloader(train_dataset, cfg.train_batch_size))
    local_prompts: list[str] = []
    for batch in train_dataloader:
        local_prompts.extend(batch[0])

    # The text encoder is only needed once. Cache unique prompt embeddings on
    # CPU before any Wan transformer is loaded, then release its ~11 GB VRAM.
    # Each rank only encodes prompts present in its DataLoader shard.
    tokenizer, text_encoder = load_tokenizer_text_encoder(cfg, accelerator.device)
    prompt_embed_cache = precompute_prompt_embeddings(
        tokenizer,
        text_encoder,
        local_prompts,
        accelerator.device,
        weight_dtype,
        batch_size=cfg.prompt_embed_batch_size,
    )
    uncond_prompt_embeds = precompute_prompt_embeddings(
        tokenizer,
        text_encoder,
        [DEFAULT_NEGATIVE_PROMPT],
        accelerator.device,
        weight_dtype,
        batch_size=1,
    )[DEFAULT_NEGATIVE_PROMPT].unsqueeze(0)
    test_prompts = None
    test_prompt_embeds = None
    if cfg.validation_prompt_path:
        test_prompts = load_prompt_dataset(cfg.validation_prompt_path)[: cfg.validation_num_prompts]
        test_prompt_embeds = precompute_prompt_embeddings(
            tokenizer,
            text_encoder,
            test_prompts,
            accelerator.device,
            weight_dtype,
            batch_size=min(cfg.prompt_embed_batch_size, 4),
        )
    cache_bytes = sum(t.numel() * t.element_size() for t in prompt_embed_cache.values())
    logger.info(
        "Rank %d cached %d unique prompt embeddings on CPU (%.2f GiB)",
        accelerator.process_index,
        len(prompt_embed_cache),
        cache_bytes / 1024**3,
    )
    del text_encoder, tokenizer
    gc.collect()
    empty_cache(device)

    use_fsdp = accelerator.distributed_type == DistributedType.FSDP
    enable_sp = resolve_sequence_parallel(cfg, cfg.enable_sp)
    model_load_device = None if use_fsdp else accelerator.device

    # In FSDP mode each full model is loaded on CPU and immediately handed to
    # Accelerate for sharding. This avoids ever materializing all three full
    # transformers on one accelerator.
    transformer = load_training_transformer(cfg, device=model_load_device, enable_sp=enable_sp)
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if cfg.enable_sp:
        from rmd.parallel.sp import set_parallel

        set_parallel(transformer)
    transformer = accelerator.prepare(transformer)

    transformer_fake = load_training_transformer(cfg, device=model_load_device, enable_sp=enable_sp)
    if cfg.gradient_checkpointing:
        transformer_fake.enable_gradient_checkpointing()
    if cfg.enable_sp:
        set_parallel(transformer_fake)
    transformer_fake = accelerator.prepare(transformer_fake)

    transformer_real = load_training_transformer(cfg, device=model_load_device, enable_sp=enable_sp, frozen=True)
    if cfg.enable_sp:
        set_parallel(transformer_real)
    transformer_real = accelerator.prepare(transformer_real)

    vae = None
    if cfg.validation_prompt_path and accelerator.is_main_process:
        from rmd.models.wan import load_vae

        vae = load_vae(cfg, accelerator.device)

    solver = EulerSolver(
        _sigmas(cfg.flow_shift, cfg.num_euler_timesteps),
        cfg.num_euler_timesteps,
        euler_timesteps=cfg.num_euler_timesteps,
    ).to(accelerator.device)
    solver_hrs = EulerSolver(
        _sigmas(5.0, cfg.num_euler_timesteps),
        cfg.num_euler_timesteps,
        euler_timesteps=cfg.num_euler_timesteps,
    ).to(accelerator.device)

    optimizer = _build_optimizer(cfg.learning_rate, cfg, transformer)
    optimizer_d = _build_optimizer(cfg.learning_rate_fake, cfg, transformer_fake)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles,
    )

    optimizer, optimizer_d, lr_scheduler = accelerator.prepare(optimizer, optimizer_d, lr_scheduler)

    model_gan = None
    optimizer_gan = None
    if cfg.gan:
        from rmd.models.gan import GAN

        model_gan = GAN().to(accelerator.device, dtype=weight_dtype)
        optimizer_gan = _build_optimizer(cfg.learning_rate, cfg, model_gan)
        model_gan, optimizer_gan = accelerator.prepare(model_gan, optimizer_gan)

    predictor = Predictor(
        cfg,
        None,
        None,
        accelerator.device,
        weight_dtype,
        solver,
        solver_hrs,
        uncond_prompt_embeds=uncond_prompt_embeds.to(accelerator.device),
    )
    accumulated_models = [transformer, transformer_fake]
    if model_gan is not None:
        accumulated_models.append(model_gan)

    K_step = cfg.k_step
    total_steps = cfg.num_euler_timesteps
    # Multi-stage upsampling always starts from 480p and targets 720p; the base
    # (single-resolution) mode trains directly at ``cfg.resolution``.
    if cfg.flow_shift_trans > 0:
        latent_init = LATENT_SIZES[480]  # (C, T, H, W) source resolution
        latent_target = LATENT_SIZES[720]  # target resolution (multistage)
    else:
        latent_init = latent_target = LATENT_SIZES[720 if cfg.resolution == "720" else 480]
    temporal = latent_init[1]
    _, _, target_h, target_w = latent_target

    if cfg.low_rs_step:
        relusion_shift = cfg.low_rs_step - 1
    else:
        relusion_shift = (K_step - 1) // 2 - 1
    mid_step = K_step - relusion_shift

    scales = [cfg.t_offset_factor] * cfg.low_rs_step + [cfg.t_offset_factor - 0.15] * (K_step - cfg.low_rs_step)

    image_rotary_emb = None
    global_step = _resume_training(accelerator, cfg)

    global_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info(
        "Running RMD training: device=%s processes=%d per_device_batch=%d "
        "gradient_accumulation=%d global_batch=%d K_step=%s resolution=%s "
        "flow_shift_trans=%s",
        device,
        accelerator.num_processes,
        cfg.train_batch_size,
        cfg.gradient_accumulation_steps,
        global_batch_size,
        K_step,
        cfg.resolution,
        cfg.flow_shift_trans,
        main_process_only=True,
    )

    while global_step < cfg.max_train_steps:
        for batch in train_dataloader:
            if global_step >= cfg.max_train_steps:
                break
            if global_step % 20 == 0:
                empty_cache(device)
                gc.collect()

            prompts_batch = list(batch[0])
            batch_size = len(prompts_batch)
            # ``Accelerator.accumulate`` takes variadic model arguments. Passing
            # the list as one object breaks ``no_sync`` on accumulation steps.
            with accelerator.accumulate(*accumulated_models):
                prompt_embeds = gather_prompt_embeddings(
                    prompt_embed_cache, prompts_batch, accelerator.device, weight_dtype
                )

                noise_init = (
                    torch.randn(batch_size, latent_init[0], temporal, latent_init[2], latent_init[3])
                    .to(accelerator.device)
                    .to(dtype=weight_dtype)
                )

                # --- Generate the teacher's K-step trajectory (frozen) ---
                with torch.no_grad():
                    transformer.eval()
                    new_noise = torch.randn_like(noise_init)
                    imgs_list, noisy_imgs_list = predictor.generate_new_upsample(
                        transformer,
                        FlowMatchEulerDiscreteScheduler(shift=cfg.flow_shift),
                        new_noise,
                        new_noise,
                        prompt_embeds,
                        image_rotary_emb,
                        eta=cfg.eta,
                        steps=K_step,
                        return_mid=True,
                        total_steps=total_steps,
                        flow_shift_trans=cfg.flow_shift_trans,
                        relusion_shift=relusion_shift,
                        target_size=(target_h, target_w),
                        mode=cfg.infer_upsampler_mode,
                    )
                    noisy_imgs_list.reverse()
                    bsz = noise_init.shape[0]
                    k_list = torch.randint(mid_step, K_step, (bsz,), device=noise_init.device).long()
                    model_input = torch.randn_like(imgs_list[k_list[0]])
                    for ii in range(model_input.shape[0]):
                        model_input[ii] = imgs_list[k_list[ii]][ii]
                # The same transformer is the trainable generator in Phase 2.
                # Trajectory construction above is inference-only, but leaving
                # the module in eval mode would also disable training behavior
                # for the subsequent gradient-bearing forward.
                transformer.train()

                # --- Phase 1: train the student to match the teacher target ---
                loss_score = _phase1(
                    cfg,
                    predictor,
                    transformer,
                    transformer_fake,
                    solver,
                    solver_hrs,
                    model_input,
                    noisy_imgs_list,
                    prompt_embeds,
                    image_rotary_emb,
                    K_step,
                    mid_step,
                    scales,
                    total_steps,
                    weight_dtype,
                    device,
                    global_step,
                    target_size=(target_h, target_w),
                )
                accelerator.backward(loss_score)
                optimizer_d.step()
                optimizer_d.zero_grad()

                # --- Optional GAN discriminator step ---
                if cfg.gan and model_gan is not None and optimizer_gan is not None:
                    loss_d = _gan_discriminator_step(
                        cfg,
                        predictor,
                        transformer_real,
                        transformer_fake,
                        model_gan,
                        noise_init,
                        prompt_embeds,
                        image_rotary_emb,
                        weight_dtype,
                    )
                    accelerator.backward(loss_d)
                    optimizer_gan.step()
                    optimizer_gan.zero_grad()

                # --- Phase 2: train the generator toward the revised target ---
                loss_student = _phase2(
                    cfg,
                    predictor,
                    transformer,
                    transformer_fake,
                    transformer_real,
                    solver,
                    solver_hrs,
                    model_input,
                    noisy_imgs_list,
                    prompt_embeds,
                    image_rotary_emb,
                    K_step,
                    mid_step,
                    scales,
                    total_steps,
                    weight_dtype,
                    device,
                    model_gan,
                    global_step,
                    target_size=(target_h, target_w),
                )
                accelerator.backward(loss_student)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                global_step += 1
                logs = {
                    "loss_score": accelerator.reduce(loss_score.detach(), reduction="mean").item(),
                    "loss_student": accelerator.reduce(loss_student.detach(), reduction="mean").item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs, step=global_step)
                logger.info("step %d: %s", global_step, logs, main_process_only=True)

                if global_step >= cfg.checkpointing_steps and global_step % cfg.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(cfg.output_dir, f"student_model-{global_step}")
                    state_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    logger.info(
                        "Saving model to %s and training state to %s",
                        save_path,
                        state_path,
                        main_process_only=True,
                    )
                    accelerator.save_model(transformer, save_path, max_shard_size="10GB", safe_serialization=True)
                    accelerator.save_state(state_path)

                should_validate = bool(
                    cfg.validation_prompt_path
                    and (
                        (cfg.validation_at_step_one and global_step == 1) or global_step % cfg.checkpointing_steps == 0
                    )
                )
                if should_validate:
                    logger.info(
                        "Generating %d validation samples at step %d",
                        len(test_prompts),
                        global_step,
                        main_process_only=True,
                    )
                    _generate_test_samples(
                        cfg,
                        accelerator,
                        transformer,
                        predictor,
                        vae,
                        test_prompts,
                        test_prompt_embeds,
                        global_step,
                        weight_dtype,
                        latent_init,
                        latent_target,
                        relusion_shift,
                    )

    accelerator.end_training()


def _phase1(
    cfg,
    predictor,
    transformer,
    transformer_fake,
    solver,
    solver_hrs,
    model_input,
    noisy_imgs_list,
    prompt_embeds,
    image_rotary_emb,
    K_step,
    mid_step,
    scales,
    total_steps,
    weight_dtype,
    device,
    global_step,
    target_size=(90, 160),
):
    bsz = model_input.shape[0]
    ind_t = _sample_ind_t(cfg, bsz, model_input.device, mid_step, K_step, global_step)
    noisy_latents_ode = gather_trajectory_batch(noisy_imgs_list, ind_t)
    timesteps, timesteps_g, timesteps_mid = _sample_timesteps(
        cfg, model_input.device, ind_t, K_step, mid_step, scales, total_steps
    )
    with torch.no_grad():
        higher_shift = cfg.flow_shift_trans == 2 and ind_t[0] < mid_step
        model_eps, model_latents = predictor.predict(
            transformer,
            noisy_latents_ode,
            timesteps_g,
            prompt_embeds,
            cfg=None,
            return_double=True,
            higher_shift=higher_shift,
        )
        if ind_t[0] >= mid_step:
            model_latents = predictor.upsample_infer(model_latents, target_size=target_size, mode=cfg.upsampler_mode)
            model_eps = predictor.upsample_infer(model_eps, target_size=target_size, mode=cfg.upsampler_mode)
    add_eps = cfg.eta * model_eps + ((1 - cfg.eta**2) ** 0.5) * torch.randn_like(model_eps)
    # Solver selection follows the original script: use the high-shift solver
    # whenever flow-shift transition is enabled.
    if cfg.flow_shift_trans == 2:
        sigmas_mid = extract_into_tensor(solver_hrs.sigmas, timesteps_mid, model_input.shape)
    else:
        sigmas_mid = extract_into_tensor(solver.sigmas, timesteps_mid, model_input.shape)
    noisy_model_latents_ode = (1.0 - sigmas_mid) * model_latents + sigmas_mid * add_eps
    noisy_model_latents = predictor.add_noise(
        noisy_model_latents_ode.detach(),
        torch.randn_like(noisy_model_latents_ode),
        timesteps_mid,
        timesteps,
        higher_shift=True,
    ).to(weight_dtype)
    fake_latents = predictor.predict(
        transformer_fake,
        noisy_model_latents,
        timesteps,
        prompt_embeds,
        encoder_hidden_states_image=None,
        conditions=None,
        higher_shift=True,
    )
    if cfg.flow_shift_trans == 2:
        sigmas = extract_into_tensor(solver_hrs.sigmas, timesteps, fake_latents.shape)
    else:
        sigmas = extract_into_tensor(solver.sigmas, timesteps, fake_latents.shape)
    return compute_score_loss(
        fake_latents,
        model_latents,
        sigmas,
        mode=cfg.score_weighting_mode,
    )


def _phase2(
    cfg,
    predictor,
    transformer,
    transformer_fake,
    transformer_real,
    solver,
    solver_hrs,
    model_input,
    noisy_imgs_list,
    prompt_embeds,
    image_rotary_emb,
    K_step,
    mid_step,
    scales,
    total_steps,
    weight_dtype,
    device,
    model_gan,
    global_step,
    target_size=(90, 160),
):
    bsz = model_input.shape[0]
    ind_t = _sample_ind_t(cfg, bsz, model_input.device, mid_step, K_step, global_step)
    noisy_latents = gather_trajectory_batch(noisy_imgs_list, ind_t)
    timesteps, timesteps_g, timesteps_mid = _sample_timesteps(
        cfg, model_input.device, ind_t, K_step, mid_step, scales, total_steps
    )
    higher_shift = cfg.flow_shift_trans == 2 and ind_t[0] < mid_step
    model_eps, model_latents = predictor.predict(
        transformer,
        noisy_latents,
        timesteps_g,
        prompt_embeds,
        cfg=None,
        return_double=True,
        higher_shift=higher_shift,
    )
    if ind_t[0] >= mid_step:
        model_latents = predictor.upsample_infer(model_latents, target_size=target_size, mode=cfg.upsampler_mode)
        model_eps = predictor.upsample_infer(model_eps, target_size=target_size, mode=cfg.upsampler_mode)
    add_eps = cfg.eta * model_eps + ((1 - cfg.eta**2) ** 0.5) * torch.randn_like(model_eps)
    sigmas_mid = extract_into_tensor(
        solver_hrs.sigmas if cfg.flow_shift_trans == 2 else solver.sigmas, timesteps_mid, model_input.shape
    )
    noisy_model_latents_ode = (1.0 - sigmas_mid) * model_latents + sigmas_mid * add_eps
    noisy_model_latents = predictor.add_noise(
        noisy_model_latents_ode.detach(),
        torch.randn_like(noisy_model_latents_ode),
        timesteps_mid,
        timesteps,
        higher_shift=(cfg.flow_shift_trans == 2),
    ).to(weight_dtype)

    with torch.no_grad():
        real_latents = predictor.predict(
            transformer_real,
            noisy_model_latents,
            timesteps,
            prompt_embeds,
            cfg=cfg.cfg,
            higher_shift=(cfg.flow_shift_trans == 2),
        )
        fake_latents = predictor.predict(
            transformer_fake,
            noisy_model_latents,
            timesteps,
            prompt_embeds,
            higher_shift=(cfg.flow_shift_trans == 2),
        )

    revised_latents = compute_revised_latents(model_latents, real_latents, fake_latents)
    huber_c = compute_huber_c(model_input.shape)
    weighting_factor = compute_weighting_factor(model_latents, real_latents)

    loss = compute_student_loss(model_latents, revised_latents, weighting_factor, huber_c)

    if cfg.gan and model_gan is not None:
        gen_cls_loss = model_gan.compute_generator_clean_cls_loss(fake_latents)
        loss = loss + cfg.weight_gan_generator * gen_cls_loss

    return loss


def _gan_discriminator_step(
    cfg,
    predictor,
    transformer_real,
    transformer_fake,
    model_gan,
    noise_init,
    prompt_embeds,
    image_rotary_emb,
    weight_dtype,
):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=cfg.flow_shift)
    with torch.no_grad():
        fact_latents = predictor.generate_new(
            transformer_real,
            scheduler,
            torch.randn_like(noise_init),
            torch.randn_like(noise_init),
            prompt_embeds,
            image_rotary_emb,
            eta=cfg.eta,
            steps=cfg.K_step_get_fact_latents,
            total_steps=cfg.num_euler_timesteps,
        )
    # fake latents produced by the student for the discriminator
    fake_latents = predictor.generate_new(
        transformer_fake,
        scheduler,
        torch.randn_like(noise_init),
        torch.randn_like(noise_init),
        prompt_embeds,
        image_rotary_emb,
        eta=cfg.eta,
        steps=cfg.k_step,
        total_steps=cfg.num_euler_timesteps,
    )
    return model_gan.compute_guidance_clean_cls_loss(
        fact_latents.to(dtype=weight_dtype), fake_latents.to(dtype=weight_dtype)
    )
