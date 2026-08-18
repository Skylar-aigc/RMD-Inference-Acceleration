# Training

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1, and CUDA
- A Wan2.1 checkpoint reachable via `--pretrained_model_name_or_path`
  (HuggingFace repo id or local path). The `1.3B` recipe runs on a few GPUs and
  the `14B` recipe targets multi-GPU training.
- A prompt file (`.pt`, `.txt`, or `.jsonl`). A starter file ships at
  `examples/vidprom_prompts_2000.txt`. It is a filtered, text-only subset of
  VidProM under CC BY-NC 4.0; see `examples/VIDPROM_DATASET.md`. Prompts-only
  training is fully supported — the teacher generates trajectories on the fly.

## Install

```bash
pip install -e ".[dev]"
```

## Training recipes

### Wan2.1 1.3B (small footprint)

```bash
rmd-train --config configs/train_wan_1_3b.yaml
```

Equivalent flags (any of these can be overridden on the CLI):

```bash
rmd-train \
  --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --data_path examples/sample_prompts.txt \
  --k_step 6 \
  --multistage_upsample \
  --low_rs_step 3 \
  --output_dir outputs/rmd-1.3b \
  --device cuda
```

### Wan2.1 on one 8-GPU server (FSDP)

```bash
bash scripts/train_8gpu.sh configs/train_wan_1_3b.yaml
# or the memory-safer 14B recipe:
bash scripts/train_8gpu.sh configs/train_wan_14b.yaml
```

The FSDP recipes default to standard AdamW. The 8-bit implementation remains
opt-in for controlled experiments.

The launcher starts eight Accelerate processes and uses FSDP `FULL_SHARD` for
the Wan transformers. `train_batch_size` is a **per-GPU micro-batch**. Effective
global batch size is:

```text
train_batch_size * 8 * gradient_accumulation_steps
```

Consequently, the 1.3B recipe has global batch 8. The 14B recipe uses a
per-GPU batch of 1 with three accumulation steps, giving global batch 24 while
keeping activation memory lower. Both recipes run for 300 optimizer steps and
save at steps 50, 100, 150, 200, 250, and 300.

Pass server paths and other overrides after the config:

```bash
bash scripts/train_8gpu.sh configs/train_wan_14b.yaml \
  --data_path /data/prompts.txt \
  --output_dir /data/rmd-14b
```

Accelerate shards the `DataLoader`, so each process receives a different batch.
Each worker also uses a deterministic rank-specific random seed. The text
encoder only precomputes embeddings needed by that worker's prompt shard before
it is released.

## Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `k_step` | 6 | number of student sampling steps to train toward |
| `multistage_upsample` | false | enable 480p → 720p latent upsampling |
| `flow_shift_trans` | 0 | 0 = single resolution, 1 = upsample, 2 = upsample + shift |
| `low_rs_step` | 0 | steps kept in the low-resolution regime |
| `t_offset_factor` | 0.5 | bias toward the high-noise interval |
| `split_timestep` | false | split the noise interval between low/high resolution |
| `warmup_step` | 15 | low-resolution-only warmup steps |
| `cfg` | 5.0 | classifier-free guidance strength for the reference model |
| `eta` | 0.9 | stochasticity (Euler ancestral) |
| `score_weighting_mode` | `inverse_sigma` | Phase 1 sigma weighting |
| `gan` | false | experimental discriminator objective |
| `prompt_embed_batch_size` | 8 | prompts encoded per batch during one-time embedding precomputation |

## Data formats

`--data_path` accepts:

- **`.txt`** — one prompt per line
- **`.jsonl`** — `{"prompt": "..."}` per line
- **`.pt`** — list of `{"prompt": str, "latent": tensor?}` records

To build a latent dataset from real videos:

```bash
rmd-build-latents \
  --input_video_folder /path/to/videos \
  --output_latent_folder /path/to/latents \
  --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

This runs the VAE encoder to produce `latents.pt` ready for `--data_path`.

## Monitoring & checkpoints

- TensorBoard logs under `<output_dir>/logs` (loss_score, loss_student, lr),
  written only by the main Accelerate process.
- Student weights are saved every 50 steps by default (`--checkpointing_steps`)
  as `student_model-<step>`.
- Full optimizer/scheduler state is saved alongside it as `checkpoint-<step>`.
- Resume with `--resume_from_checkpoint latest` or a `checkpoint-<step>` path.

## Memory-efficient initialization

Training encodes every unique prompt once, stores the embeddings in CPU memory,
and releases the frozen text encoder before loading the Wan transformers. In
FSDP runs, each transformer is loaded on CPU and immediately passed to
Accelerate for sharding; the three complete models are never placed on one
accelerator at the same time. Both trainable transformers use gradient
checkpointing when `gradient_checkpointing: true`.

For a very large prompt set, lower `prompt_embed_batch_size` if the temporary
text-encoding batch causes an OOM. The startup log reports the final CPU cache
size.

## Tips

- Start with `examples/vidprom_prompts_2000.txt` and 100–300 steps to validate the
  pipeline on a single GPU before scaling.
- For the 14B recipe, first train a 1.3B student end-to-end to tune
  `t_offset_factor` / `low_rs_step` — they are resolution-sensitive.
