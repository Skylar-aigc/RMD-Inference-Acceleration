"""Device helpers for CUDA and CPU execution."""

from __future__ import annotations


def _device_index(device: str) -> int:
    return int(device.split(":")[-1]) if ":" in device else 0


def set_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(_device_index(device))


def empty_cache(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def get_device(device: str | None = None) -> str:
    """Resolve an explicit device or auto-detect CUDA with a CPU fallback."""
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
