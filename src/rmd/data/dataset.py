"""Prompt / latent dataset loading.

Replaces ``utils/dataset_loader.py``. Supports three formats:

- ``.pt``: a list of dicts with ``{"prompt": str}`` (optionally ``"latent"``)
- ``.txt``: one prompt per line
- ``.jsonl``: one JSON object per line (``{"prompt": ...}``)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def load_prompt_dataset(path: str) -> list[str]:
    """Load a list of prompt strings from a .pt / .txt / .jsonl file."""
    if path.endswith(".pt"):
        import torch

        data = torch.load(path, weights_only=False)
        if isinstance(data, dict) and "prompt" in data:
            # single-record dict
            return [data["prompt"]]
        if not isinstance(data, (list, tuple)):
            raise ValueError(f"Unexpected .pt structure in {path}: {type(data)}")
        prompts = []
        for item in data:
            if isinstance(item, dict):
                if "prompt" not in item:
                    raise ValueError(f"Record in {path} is missing 'prompt' key: {list(item.keys())[:5]}")
                prompts.append(item["prompt"])
            elif isinstance(item, str):
                prompts.append(item)
            else:
                raise TypeError(f"Unexpected record type in {path}: {type(item)}")
        return prompts

    if path.endswith(".jsonl"):
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prompts.append(record["prompt"])
        return prompts

    # default: plain text, one prompt per line
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


class PromptDataset:
    """Minimal text-only dataset; returns ``(text, text)`` to match the training loop.

    Intentionally does not subclass ``torch.utils.data.Dataset`` so the module
    remains importable without torch; ``DataLoader`` only needs ``__len__`` and
    ``__getitem__``.
    """

    def __init__(self, prompts: list[str]):
        self.prompts = list(prompts)
        logger.info("PromptDataset loaded %d prompts", len(self.prompts))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        return text, text


def build_dataloader(dataset, batch_size: int, num_workers: int = 0):
    import torch

    return torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=False
    )
