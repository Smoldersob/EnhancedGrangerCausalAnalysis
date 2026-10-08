from dataclasses import dataclass
from typing import Any, ContextManager, Optional

from ...core.exceptions import BackendNotAvailableError


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    index: Optional[int] = None

    @property
    def is_gpu(self) -> bool:
        return self.kind == "gpu"


class DeviceManager:
    """
    Adapter for device control of active keras backend.

    Options: auto, cpu, gpu, gpu:N.
    """

    def __init__(
        self,
        *,
        backend: str,
        requested_device: Optional[str] = "auto",
    ):
        self.backend = backend
        self.requested = self._parse_device(requested_device)
        self.resolved = self._resolve()

    @staticmethod
    def _parse_device(device: Optional[str]) -> DeviceSpec:
        if device is None:
            return DeviceSpec("auto")

        value = str(device).strip().lower()

        if value in {"", "auto"}:
            return DeviceSpec("auto")

        if value in {"cpu", "cpu-only", "/cpu:0"}:
            return DeviceSpec("cpu", 0)

        if value in {"gpu", "cuda", "/gpu:0", "/cuda:0"}:
            return DeviceSpec("gpu", 0)

        if value.startswith(("gpu:", "cuda:", "/gpu:", "/cuda:")):
            raw_index = value.rsplit(":", 1)[-1]

            if not raw_index.isdigit():
                raise ValueError(
                    f"Nieprawidłowa specyfikacja urządzenia: {device!r}"
                )

            return DeviceSpec("gpu", int(raw_index))

        raise ValueError(
            "device must be one of: None, 'auto', 'cpu', 'gpu', "
            "'gpu:N' or 'cuda:N'."
        )

    def _resolve(self) -> DeviceSpec:
        if self.backend == "tensorflow":
            return self._resolve_tensorflow()

        if self.backend == "jax":
            return self._resolve_jax()

        if self.backend == "torch":
            return self._resolve_torch()

        raise BackendNotAvailableError(
            f"Unsupported active Keras backend: {self.backend!r}"
        )

    def _resolve_tensorflow(self) -> DeviceSpec:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if self.requested.kind == "auto":
            return DeviceSpec("gpu", 0) if gpus else DeviceSpec("cpu", 0)

        if self.requested.kind == "cpu":
            return DeviceSpec("cpu", 0)

        assert self.requested.index is not None

        if self.requested.index >= len(gpus):
            raise BackendNotAvailableError(
                f"Requested GPU {self.requested.index}, "
                f"but TensorFlow detects only {len(gpus)} GPU(s)."
            )

        return self.requested

    def _resolve_jax(self) -> DeviceSpec:
        import jax

        if self.requested.kind == "cpu":
            cpu_devices = jax.devices("cpu")
            if not cpu_devices:
                raise BackendNotAvailableError(
                    "JAX does not detect a CPU device."
                )
            return DeviceSpec("cpu", 0)

        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError as exc:
            if self.requested.kind != "auto":
                raise BackendNotAvailableError(
                    "GPU was requested, but JAX could not initialize "
                    "or discover its GPU backend."
                ) from exc

            gpu_devices = []

        if self.requested.kind == "auto":
            if gpu_devices:
                return DeviceSpec("gpu", 0)

            cpu_devices = jax.devices("cpu")
            if not cpu_devices:
                raise BackendNotAvailableError(
                    "JAX does not detect a CPU device."
                )
            return DeviceSpec("cpu", 0)

        index = self.requested.index
        if index is None:
            raise ValueError("An explicit GPU request requires an index.")

        if index >= len(gpu_devices):
            raise BackendNotAvailableError(
                f"Requested GPU {index}, "
                f"but JAX detects only {len(gpu_devices)} GPU(s)."
            )

        return self.requested

    def _resolve_torch(self) -> DeviceSpec:
        import torch

        if self.requested.kind == "auto":
            return (
                DeviceSpec("gpu", 0)
                if torch.cuda.is_available()
                else DeviceSpec("cpu", 0)
            )

        if self.requested.kind == "cpu":
            return DeviceSpec("cpu", 0)

        assert self.requested.index is not None

        if not torch.cuda.is_available():
            raise BackendNotAvailableError(
                "GPU was requested, but PyTorch does not detect CUDA."
            )

        if self.requested.index >= torch.cuda.device_count():
            raise BackendNotAvailableError(
                f"Requested GPU {self.requested.index}, but PyTorch detects "
                f"only {torch.cuda.device_count()} GPU(s)."
            )

        return self.requested

    @property
    def name(self) -> str:
        if self.resolved.kind == "cpu":
            return "cpu"

        return f"gpu:{self.resolved.index}"

    def scope(self) -> ContextManager[Any]:
        """Return a placement context for the resolved device."""
        import keras

        active_backend = keras.config.backend()
        if active_backend != self.backend:
            raise BackendNotAvailableError(
                f"DeviceManager was configured for {self.backend!r}, "
                f"but the active Keras backend is {active_backend!r}."
            )

        device_scope = getattr(keras, "device", None)
        if not callable(device_scope):
            raise BackendNotAvailableError(
                "DeviceManager requires a Keras installation "
                "that provides keras.device()."
            )

        target = f"{self.resolved.kind}:{self.resolved.index or 0}"
        return device_scope(target)