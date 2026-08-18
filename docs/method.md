# Method

RMD (Few-Step Video Diffusion Distillation with Multi-Resolution Latent Upsampling)
distills a large text-to-video diffusion model (Wan2.1) into a student that samples
in K steps (K = 4 / 6 / 8) with joint multi-resolution (480p → 720p) latent
upsampling. This document describes the algorithm exactly as implemented in
`src/rmd/`.

## Overview

Diffusion distillation trains a *student* to reproduce the sampling trajectories
of a frozen (or slowly-moving) *teacher* in far fewer steps. RMD augments the
standard trajectory-matching objective with an **adversarial-style revision
target**, and adds a **multi-stage latent upsampler** so the same distilled model
can generate at higher resolution without a separate upsampling network.

### The three models

| Symbol | Model | Frozen? | Role |
|--------|-------|---------|------|
| `G` (transformer) | teacher | generator (trained in phase 2) | produces the K-step trajectory |
| `S` (transformer_fake) | student | trained in phase 1 | the distilled few-step model |
| `R` (transformer_real) | reference | frozen | "real" prediction for the revision target |

Both `S` and `R` are initialized as copies of the pretrained teacher. `R` stays
frozen; `S` is trained to match teacher latents; `G` is fine-tuned toward the
revised target.

## Training: two losses per step

### Phase 1 — Student matches the teacher trajectory (score loss)

Given a batch, the teacher runs a **K-step trajectory** starting from noise. For a
randomly chosen intermediate point `ind_t`, we take the teacher's noisy latent
`x_noisy` and timestep `t_g`, then add noise to land at a sampled noise-destruction
level `t`. The student predicts `S(x_noisy(t))` and is supervised against the
teacher's clean target `x_target` with an **SNR-weighted MSE**:

```
w = ((1 − σ) / (σ + 1e-8))²
L_score = mean( w · (S(x) − x_target)² )
```

### Phase 2 — Generator toward the revised target (revision loss)

The generator `G` is pulled toward an **adversarially revised** latent:

```
x_rev = x_G + (R(x) − S(x))
```

where `R(x)` is the frozen reference model's prediction at the same noise level and
`S(x)` is the student's. Intuitively, `R(x) − S(x)` measures how far the student
drifted from the reference; adding it to the teacher prediction corrects the target
toward trajectories the reference model agrees with.

The loss is a **scale-aware pseudo-Huber** with per-sample adaptive weighting:

```
c     = 1e-3 / √(64·64·4) · √numel(latent)
m     = |x_G − x_R|            # adaptive weight, clipped at 0.2
L_rev = mean( ( √((x_G − x_rev)² + c²) − c ) / m )
```

### Noise-interval scheduling (multi-stage)

For multi-stage upsampling, the noise interval is split between low- and
high-resolution segments:

- **`split_timestep`** caps the noise level differently for the low- and
  high-resolution halves of the trajectory.
- **`t_offset_factor`** biases sampling toward the high-noise interval.
- **`warmup_step`** keeps training in the low-resolution regime for the first
  N steps to stabilize the upsample transition.

## Multi-resolution upsampling

At a fixed point in the trajectory (`relusion_shift`), the low-resolution latent
(480p) is **bilinearly upsampled** in spatial dimensions to the target (720p):

```
(1, 16, 21, 60, 104)  →  (1, 16, 21, 90, 160)
```

- `flow_shift_trans = 1`: upsampling only.
- `flow_shift_trans = 2`: upsampling **plus** a flow-shift transition from
  `shift=3` (480p) to `shift=5` (720p), implemented with a second sigma solver
  (`solver_hrs`).

Because both resolutions share one model, the student learns to refine details
during the high-resolution segment — no extra upsampling network is needed.

## Config → code map

| Paper concept | Config field | Code |
|---------------|--------------|------|
| K steps | `k_step` | `Predictor.generate_new*` |
| Revision target | — | `losses.distillation.compute_revised_latents` |
| Score loss | — | `losses.distillation.compute_score_loss` |
| Revision loss | — | `losses.distillation.compute_student_loss` |
| Adaptive weighting | — | `losses.distillation.compute_weighting_factor` |
| Upsampling mode | `flow_shift_trans`, `low_rs_step` | `Predictor.generate_new_upsample` |
| Noise scheduling | `split_timestep`, `t_offset_factor`, `warmup_step` | `engines.train._sample_timesteps` |
