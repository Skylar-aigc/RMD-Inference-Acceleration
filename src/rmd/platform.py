"""Device abstraction for CUDA / Ascend NPU / CPU.

The entire codebase must access devices through this module. ``torch_npu`` is
imported lazily and ONLY when the user explicitly requests ``device="npu"``,
so a bare CUDA/CPU machine can always ``import rmd`` without errors.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_npu_available: bool | None = None


def npu_available() -> bool:
    """Return True if the torch_npu plugin is importable (Ascend hardware)."""
    global _npu_available
    if _npu_available is None:
        try:
            import torch_npu  # noqa: F401

            _npu_available = True
        except Exception:  # noqa: BLE001 — probing importability
            _npu_available = False
    return _npu_available


def is_npu_device(device: str) -> bool:
    """Return True for any npu device string (e.g. ``npu``, ``npu:0``, ``ascend``)."""
    return isinstance(device, str) and (device in {"npu", "ascend"} or device.startswith("npu"))


def _device_index(device: str) -> int:
    return int(device.split(":")[-1]) if ":" in device else 0


def set_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(_device_index(device))
    elif is_npu_device(device):
        import torch_npu

        torch_npu.npu.set_device(_device_index(device))
    else:
        logger.debug("No explicit device selection for %s", device)


def empty_cache(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
    elif is_npu_device(device):
        import torch_npu

        torch_npu.npu.empty_cache()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    elif is_npu_device(device):
        import torch_npu

        torch_npu.npu.synchronize()


def patch_torch_for_device(device: str) -> None:
    """Apply device-specific runtime patches (currently NPU only)."""
    if is_npu_device(device):
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

        torch_npu.config.allow_internal_format = False
        logger.info("Ascend NPU runtime patched (allow_internal_format=False)")


def get_device(device: str | None = None) -> str:
    """Resolve the requested device; auto-detect when None (prefers NPU, then CUDA)."""
    if device is not None:
        return device
    if npu_available():
        return "npu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
