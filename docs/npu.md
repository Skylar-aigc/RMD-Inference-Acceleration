# Ascend NPU support

RMD runs on NVIDIA CUDA by default and also supports Huawei Ascend NPUs. All
device code routes through `src/rmd/platform.py`, so NPU support is **optional
and lazy** — a machine without `torch-npu` never imports it.

## Setup

```bash
# Install torch_npu matching your PyTorch / CANN versions (see torch_npu docs)
pip install torch-npu

# Optional: sequence-parallel kernels for large models
pip install ascendx-video
```

## Usage

Select the device explicitly with `--device npu`:

```bash
rmd-train --config configs/train_wan_14b.yaml --device npu
rmd-infer --model_dir /path/to/ckpt --device npu
```

Without `--device`, RMD auto-detects NPU first, then CUDA, then CPU.

## Sequence parallel (14B on NPU)

The 14B recipe benefits from sequence parallel. Enable with `--enable_sp`.

```bash
rmd-train --config configs/train_wan_14b.yaml --device npu --enable_sp
```

If the `ascendx-video` kernels are not installed, RMD logs a warning and falls
back to standard FSDP — training still works, just slower.

## Reference configuration (40 × NPU)

The published training run used 40 Ascend NPUs. See `configs/train_wan_14b.yaml`
together with `configs/accelerate_fsdp.yaml` (FSDP FULL_SHARD, bf16). The
launcher:

```bash
accelerate launch \
  --num_processes 40 \
  --config_file configs/accelerate_fsdp.yaml \
  src/rmd/cli.py \
  --config configs/train_wan_14b.yaml \
  --device npu
```

## Troubleshooting

- **`import torch_npu` not found** — you are on a non-NPU machine; this is fine,
  RMD runs on CUDA/CPU without it.
- **OOM on NPU** — enable `--gradient_checkpointing`, lower `--train_batch_size`,
  and consider `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.
