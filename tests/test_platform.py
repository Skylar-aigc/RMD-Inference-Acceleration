"""Unit tests for the device abstraction layer (no torch required)."""

import sys
import types

from rmd.platform import get_device, is_npu_device, npu_available


def test_npu_available_returns_bool():
    assert isinstance(npu_available(), bool)


def test_is_npu_device():
    assert is_npu_device("npu") is True
    assert is_npu_device("npu:0") is True
    assert is_npu_device("ascend") is True
    assert is_npu_device("cuda") is False
    assert is_npu_device("cpu") is False


def _fake_torch(cuda_available: bool):
    fake = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake.cuda = fake_cuda
    return fake


def test_get_device_cpu(monkeypatch):
    monkeypatch.setattr("rmd.platform.npu_available", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert get_device() == "cpu"


def test_get_device_cuda(monkeypatch):
    monkeypatch.setattr("rmd.platform.npu_available", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    assert get_device() == "cuda"


def test_get_device_explicit_override():
    assert get_device("npu") == "npu"
    assert get_device("cuda") == "cuda"
