# VidProM 2K text-only training subset

`vidprom_prompts_2000.txt` is a deterministic random subset of prompts from
[WenhaoWang/VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM).
No generated videos are included or downloaded.

- Upstream dataset: VidProM, NeurIPS 2024 Datasets and Benchmarks Track
- Upstream license: CC BY-NC 4.0 (non-commercial use)
- Subset size: 2,000 unique English prompts
- Sampling seed: `20260812`
- File SHA-256: `AB31CA1E01C24C2DFD9E4A22084798EBEBA3A1FEB1607AE2F36B50852A7C37DE`
- Safety threshold: every supplied VidProM safety score must be `<= 0.05`
- Filtering: removes empty/very short/very long prompts, URLs, unsafe terms,
  Discord message/attachment remnants, generator command tails, and exact normalized
  matches with the official VBench prompt suite

Rebuild the subset from public text metadata with:

```bash
python scripts/build_vidprom_prompts.py
```

Use this dataset only where the upstream CC BY-NC 4.0 license is acceptable.
Retain attribution to the VidProM authors in research outputs.
