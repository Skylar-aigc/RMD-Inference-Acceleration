"""Tests for prompt embedding precomputation (requires torch)."""

import pytest

torch = pytest.importorskip("torch")

from rmd.models.prompt import gather_prompt_embeddings, precompute_prompt_embeddings


def test_precompute_prompt_embeddings_deduplicates_and_batches(monkeypatch):
    calls = []

    def fake_compute(tokenizer, text_encoder, prompts, *args, **kwargs):
        calls.append(list(prompts))
        return torch.stack([torch.full((2, 3), float(len(prompt))) for prompt in prompts])

    monkeypatch.setattr("rmd.models.prompt.compute_prompt_embeddings", fake_compute)
    cache = precompute_prompt_embeddings(
        object(), object(), ["a", "bb", "a", "ccc"], "cpu", torch.float32, batch_size=2
    )

    assert calls == [["a", "bb"], ["ccc"]]
    assert set(cache) == {"a", "bb", "ccc"}
    gathered = gather_prompt_embeddings(cache, ["ccc", "a"], "cpu", torch.float32)
    assert gathered.shape == (2, 2, 3)
    assert gathered[0, 0, 0] == 3
    assert gathered[1, 0, 0] == 1
