# RMD: Cross-Resolution Distribution Matching Distillation

[中文说明](README_CN.md)

[[Paper](https://arxiv.org/abs/2603.06136)]

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
[![CI](https://github.com/Skylar-aigc/RMD-Inference-Acceleration/actions/workflows/ci.yml/badge.svg)](https://github.com/Skylar-aigc/RMD-Inference-Acceleration/actions/workflows/ci.yml)

Official implementation of **Cross-Resolution Distribution Matching for Diffusion Distillation**. RMD distills a pretrained diffusion model into a few-step generator that builds global structure at low resolution and refines details at high resolution.

![RMD training framework](assets/rmd_training_framework.png)

## Highlights

- Few-step 480p-to-720p cascaded video generation.
- Distribution matching along cross-resolution trajectories.
- Predicted-noise re-injection for stable resolution transitions.
- CUDA/NPU support, FSDP training, distributed inference, and VBench-format generation.

## Results

Wan2.1-T2V-14B results reported in the paper. Speed was measured on one NVIDIA A100.

| Method | NFE | Speedup | VBench Total | Quality | Semantic | T2V-CompBench |
|---|---:|---:|---:|---:|---:|---:|
| Wan2.1 720p | 50 × 2 | 1.0× | 83.75 | 85.44 | 76.95 | 54.17 |
| DMD2 | 6 | 16.7× | 80.30 | 81.98 | 73.60 | 52.81 |
| TDM | 6 | 16.7× | 80.48 | 81.85 | 75.00 | 52.27 |
| **RMD** | **3 + 3** | **25.6×** | **82.51** | **84.37** | **75.05** | **54.00** |

![Wan2.1-14B comparison](assets/wan_video_comparison.png)

## Installation

Linux, Python 3.10/3.11, and CUDA-compatible PyTorch are recommended. Install the PyTorch build matching your server first, then install RMD:

```bash
git clone <repository-url> RMD
cd RMD

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Example only; choose the wheel matching your CUDA driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[gradio]"   # web demo
pip install -e ".[dev]"      # tests and lint
pip install -e ".[npu]"      # Ascend NPU
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
rmd-train --help
rmd-infer --help
```

## Model and data

RMD expects a Diffusers-format Wan model, either as a Hugging Face model ID or a local directory:

Distilled student weights are not stored in this source repository. Train a student with one of the recipes below, or point `--model_dir` to a compatible exported `student_model-<step>` directory.

```text
/data/models/Wan2.1-T2V-1.3B-Diffusers/
├── model_index.json
├── tokenizer/
├── text_encoder/
├── transformer/
├── scheduler/
└── vae/
```

Set `pretrained_model_name_or_path` in the training YAML, or override it from the command line. The provided recipes use:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `Wan-AI/Wan2.1-T2V-14B-Diffusers`

Training data is a UTF-8 text file with one prompt per line. The repository includes `examples/vidprom_prompts_2000.txt`, a reproducible 2,000-prompt VidProM subset. It contains text only and follows the upstream CC BY-NC 4.0 license. The paper experiments used JourneyDB prompts; the bundled subset is a convenient starter set, not the exact paper training data. See [dataset details](examples/VIDPROM_DATASET.md).

## Training

The default recipes train for 300 optimizer steps and save at steps 50, 100, 150, 200, 250, and 300.

### Single process

```bash
rmd-train \
  --config configs/train_wan_1_3b.yaml \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --data_path examples/vidprom_prompts_2000.txt \
  --output_dir /data/runs/rmd-1.3b
```

### Eight GPUs with FSDP

```bash
bash scripts/train_8gpu.sh configs/train_wan_1_3b.yaml \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --output_dir /data/runs/rmd-1.3b-8gpu
```

For Wan2.1-14B, replace the training config with `configs/train_wan_14b.yaml`. The 14B model requires substantially more GPU and host memory.

Resume the latest training state with:

```bash
bash scripts/train_8gpu.sh configs/train_wan_14b.yaml \
  --output_dir /data/runs/rmd-14b \
  --resume_from_checkpoint latest
```

When `validation_prompt_path` is set, ten samples are generated at step 1 and every checkpoint under `test_samples/step-<N>/`.

### Important options

| Option | Meaning |
|---|---|
| `k_step` | Number of student denoising steps. |
| `low_rs_step` | Number of early low-resolution steps. |
| `flow_shift_trans: 0` | No resolution transition. |
| `flow_shift_trans: 1` | Upsample from 480p to 720p during sampling. |
| `flow_shift_trans: 2` | Upsample and switch the flow shift from 3 to 5. |
| `eta` | Predicted-noise mixing coefficient; it may be tuned independently for inference. |
| `checkpointing_steps` | Checkpoint interval; defaults to 50 in the provided recipes. |

## Inference

Use the same base model family and resolution schedule as training. Inference
may use a different `eta`; the examples below use `0.9`:

```bash
rmd-infer \
  --model_dir /data/runs/rmd-1.3b/student_model-300 \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox running through a snowy forest" \
  --resolution 720 \
  --k_step 6 \
  --flow_shift_trans 1 \
  --low_rs_step 3 \
  --eta 0.9 \
  --output_dir outputs/demo
```

For a prompt file:

```bash
rmd-infer \
  --model_dir /data/runs/rmd-1.3b/student_model-300 \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --prompt_file examples/sample_prompts.txt \
  --resolution 720 --k_step 6 --flow_shift_trans 1 --low_rs_step 3 --eta 0.9 \
  --output_dir outputs/batch
```

Distribute prompts across eight GPUs:

```bash
accelerate launch --num_processes 8 -m rmd.cli infer \
  --model_dir /data/runs/rmd-1.3b/student_model-300 \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --prompt_file examples/sample_prompts.txt \
  --resolution 720 --k_step 6 --flow_shift_trans 1 --low_rs_step 3 --eta 0.9 \
  --output_dir outputs/distributed
```

Launch the web demo with:

```bash
rmd-demo \
  --model_dir /data/runs/rmd-1.3b/student_model-300 \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --resolution 720 --k_step 6 --flow_shift_trans 1 --low_rs_step 3 --eta 0.9
```

## Evaluation

Generate videos using VBench-compatible filenames:

```bash
python benchmarks/vbench.py \
  --model_dir /data/runs/rmd-1.3b/student_model-300 \
  --pretrained_model_name_or_path /data/models/Wan2.1-T2V-1.3B-Diffusers \
  --prompt_file /data/VBench/prompts.txt \
  --output_dir outputs/vbench \
  --resolution 720 --k_step 6 --flow_shift_trans 1 --low_rs_step 3 --eta 0.9 \
  --iterations 5
```

Score the generated directory with the official [VBench](https://github.com/Vchitect/VBench) toolkit.

## Testing

```bash
ruff check src tests benchmarks app.py
pytest -q
```

Before a full run, append `--max_train_steps 2 --checkpointing_steps 1` to a training command and confirm that all ranks start, losses are finite, and a checkpoint is written.

## Documentation

- [Method](docs/method.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Ascend NPU](docs/npu.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Citation

```bibtex
@article{chen2026rmd,
  title={Cross-Resolution Distribution Matching for Diffusion Distillation},
  author={Chen, Feiyang and Pan, Hongpeng and Xu, Haonan and Duan, Xinyu and Wang, Zhefeng and Yang, Yang},
  journal={arXiv preprint arXiv:2603.06136},
  year={2026},
  doi={10.48550/arXiv.2603.06136}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Prompt data retains its upstream license.
