from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    import torch


class DeviceManager:
    """Parse a device request and resolve one explicit PyTorch device.

    auto preserves the previous policy: CUDA:0 when available, otherwise CPU.
    MPS is supported only when explicitly requested. Explicit accelerator
    requests never silently fall back to CPU. No global default is modified.
    """

    def __init__(self, *, requested_device: str | None = "auto") -> None:
        requested = self._parse(requested_device)
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError("PyTorch is unavailable.") from exc

        if requested == "auto":
            resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            resolved = requested

        if resolved.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise BackendNotAvailableError("CUDA was requested but is unavailable.")
            index = int(resolved.split(":", 1)[1])
            count = torch.cuda.device_count()
            if index >= count:
                raise BackendNotAvailableError(
                    f"Requested CUDA device {index}, but only {count} device(s) are visible."
                )
        elif resolved == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise BackendNotAvailableError("MPS was requested but is unavailable.")

        self._requested = requested
        self._device = torch.device(resolved)

    @staticmethod
    def _parse(device: str | None) -> str:
        if device is None:
            return "auto"
        if not isinstance(device, str):
            raise TypeError("device must be a string or None.")
        value = device.strip().lower()
        if value.startswith("/"):
            value = value[1:]
        if value in {"", "auto"}:
            return "auto"
        if value in {"cpu", "cpu-only", "cpu:0"}:
            return "cpu"
        if value in {"gpu", "cuda"}:
            return "cuda:0"
        if value in {"mps", "mps:0"}:
            return "mps"
        if value.startswith(("gpu:", "cuda:")):
            index = value.split(":", 1)[1]
            if index.isascii() and index.isdigit():
                return f"cuda:{int(index)}"
        raise ValueError(
            "device must be None, 'auto', 'cpu', 'gpu', 'gpu:N', 'cuda:N' or 'mps'."
        )

    @property
    def requested(self) -> str:
        return self._requested

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def name(self) -> str:
        return str(self._device)

    @property
    def is_gpu(self) -> bool:
        return self._device.type in {"cuda", "mps"}
