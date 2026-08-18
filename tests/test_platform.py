"""Unit tests for CUDA/CPU device selection (no torch required)."""

import sys
import types

from rmd.platform import get_device


def _fake_torch(cuda_available: bool):
    fake = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake.cuda = fake_cuda
    return fake


def test_get_device_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert get_device() == "cpu"


def test_get_device_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    assert get_device() == "cuda"


def test_get_device_explicit_override():
    assert get_device("cuda") == "cuda"
    assert get_device("cpu") == "cpu"
