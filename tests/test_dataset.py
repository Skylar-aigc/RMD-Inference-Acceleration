"""Tests for the dataset loader (no torch required for txt/jsonl)."""

import json

import pytest

from rmd.data.dataset import PromptDataset, load_prompt_dataset


def test_load_txt(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("a panda\n\nswimming\n  a bicycle  \n", encoding="utf-8")
    assert load_prompt_dataset(str(p)) == ["a panda", "swimming", "a bicycle"]


def test_load_jsonl(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"prompt": "cat running"}) + "\n" + json.dumps({"prompt": "beach"}) + "\n",
        encoding="utf-8",
    )
    assert load_prompt_dataset(str(p)) == ["cat running", "beach"]


def test_load_pt_dict(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "data.pt"
    torch.save([{"prompt": "one"}, {"prompt": "two"}], path)
    assert load_prompt_dataset(str(path)) == ["one", "two"]


def test_prompt_dataset_len_and_getitem():
    ds = PromptDataset(["a", "b", "c"])
    assert len(ds) == 3
    assert ds[0] == ("a", "a")
    assert ds[2] == ("c", "c")


def test_unknown_extension_defaults_to_txt(tmp_path):
    p = tmp_path / "prompts.dat"
    p.write_text("hello world\n", encoding="utf-8")
    assert load_prompt_dataset(str(p)) == ["hello world"]
