# Inference

Run generation with a trained RMD student.

## CLI

```bash
rmd-infer \
  --model_dir /path/to/student_ckpt \
  --prompt "A panda playing a miniature guitar in a bamboo forest" \
  --resolution 720 \
  --k_step 4 \
  --output_dir outputs/rmd-infer
```

Or batch from a file (one prompt per line):

```bash
rmd-infer \
  --model_dir /path/to/student_ckpt \
  --prompt_file prompts.txt \
  --k_step 6
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model_dir` | required | path to the trained student weights |
| `--pretrained_model_name_or_path` | `Wan-AI/...1.3B-Diffusers` | VAE + text encoder source |
| `--prompt` / `--prompt_file` | — | single prompt or file of prompts |
| `--resolution` | `480` | generation resolution (`480` or `720`) |
| `--k_step` | `6` | sampling steps |
| `--flow_shift_trans` | `0` | 0 = single resolution, 1 = upsample, 2 = upsample + shift |
| `--low_rs_step` | `0` | low-resolution steps (match training value) |
| `--eta` | `0` | stochasticity; may be tuned independently from training |
| `--seed` | `42` | reproducibility |
| `--device` | `auto` | `cuda` / `cpu` |

## Gradio demo

```bash
rmd-demo --model_dir /path/to/student_ckpt --port 7860
```

Opens a browser UI with a prompt box, a sampling-steps slider (1–8), a
resolution dropdown, and the flow-shift-transition mode selector. Add
`--share` for a public link.

## Programmatic API

```python
from rmd.config import InferConfig
from rmd.engines.infer import infer

cfg = InferConfig(model_dir="/path/to/student_ckpt", k_step=4, resolution="720")
videos = infer(cfg, ["A cat walking across a sunlit kitchen floor"], seed=0)
# videos[0] is a numpy array [T, H, W, C] (0-255)
```

## Batch evaluation (VBench-style)

```bash
python benchmarks/vbench.py \
  --model_dir /path/to/student_ckpt \
  --prompt_file vbench_prompts.txt \
  --output_dir outputs/vbench \
  --iterations 5
```

For distributed batch inference, launch the module subcommand. Prompt indices
are assigned round-robin and output numbering remains global:

```bash
accelerate launch --num_processes 8 -m rmd.cli infer \
  --model_dir /path/to/student_ckpt \
  --pretrained_model_name_or_path /path/to/Wan-Diffusers \
  --prompt_file prompts.txt \
  --eta 0.9 \
  --output_dir outputs/distributed-infer
```

## Notes

- The distilled student is designed for few steps; the same checkpoint is used
  for both 480p and 720p when it was trained with `multistage_upsample`.
- Set `--flow_shift_trans` to match how the checkpoint was trained
  (see the training recipe). Using `2` requires the checkpoint to have been
  trained with flow-shift transition.
