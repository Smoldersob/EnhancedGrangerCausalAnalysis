from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy
from .object_loaders.tf_object_loader import TensorFlowObjectLoader


class TensorFlowBackendStrategy(BackendStrategy):
	"""Strategy for TensorFlow/Keras backend."""

	def __init__(self, loading_verbose: bool = False) -> None:
		super().__init__(loading_verbose=loading_verbose)
		self._tf = None
		self._keras = None
		self._object_loader: Optional[TensorFlowObjectLoader] = None
		self._device_mode: str = "uninitialized"
		if self.is_available():
			import tensorflow as tf
			self._configure_tensorflow_runtime(tf)
			self._tf = tf
			self._keras = tf.keras
			self._object_loader = TensorFlowObjectLoader(tf, loading_verbose=loading_verbose)

	def _configure_tensorflow_runtime(self, tf_module: Any) -> None:
		"""Configure TensorFlow runtime device policy.

		Policy:
		- If CGA_TF_FORCE_CPU=1/true/yes/on => CPU only.
		- Else if GPU is available => enable memory growth.
		- If GPU setup fails => fallback to CPU-only mode.
		"""
		force_cpu_env = os.getenv("CGA_TF_FORCE_CPU", "").strip().lower()
		use_gpu_env = os.getenv("CGA_TF_USE_GPU", "").strip().lower()
		is_wsl = bool(os.getenv("WSL_DISTRO_NAME"))

		force_cpu = force_cpu_env in {"1", "true", "yes", "on"}
		explicit_use_gpu = use_gpu_env in {"1", "true", "yes", "on"}
		# WSL is often unstable for CUDA/cuDNN in long test runs.
		# Default to CPU there unless GPU is explicitly requested.
		prefer_cpu = force_cpu or (is_wsl and not explicit_use_gpu)

		if prefer_cpu:
			try:
				tf_module.config.set_visible_devices([], "GPU")
			except Exception:
				pass
			self._device_mode = "cpu-forced" if force_cpu else "cpu-wsl-default"
			return

		try:
			gpus = tf_module.config.list_physical_devices("GPU")
		except Exception:
			gpus = []

		if not gpus:
			self._device_mode = "cpu-no-gpu"
			return

		try:
			for gpu in gpus:
				tf_module.config.experimental.set_memory_growth(gpu, True)
			self._device_mode = "gpu"
		except Exception:
			try:
				tf_module.config.set_visible_devices([], "GPU")
				self._device_mode = "cpu-fallback"
			except Exception:
				# Last resort: keep default placement if runtime is already initialized.
				self._device_mode = "gpu-runtime-locked"

	def is_available(self) -> bool:
		try:
			import tensorflow  # noqa: F401
			return True
		except ImportError:
			return False

	def build_model(
		self,
		n_features: int,
		n_outputs: int,
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		seed: Optional[int] = None,
		**config,
	):
		if seed is not None:
			import numpy as np
			import random
			np.random.seed(seed)
			random.seed(seed)

			try:
				import tensorflow as tf
				tf.random.set_seed(seed)
			except ImportError:
				pass

			# TensorFlow's deterministic ops may require additional environment variables to be set,
			# but we won't enforce that here. Users can set TF_DETERMINISTIC_OPS=1 if they want.
		from .models.tensorflow_model import TensorFlowGrangerModel
		config = self._consume_loading_verbose(config)
		if self._object_loader is not None:
			self._object_loader.set_loading_verbose(self._loading_verbose)

		regularizer_resolved = self.build_regularizer(regularizer)
		constraint_resolved = self.build_constraint(constraint)
		callbacks_cfg = config.get("callbacks", None)
		callbacks_resolved = self.resolve_callbacks(callbacks_cfg)
		optimizer_resolved = self.resolve_optimizer(config.get("optimizer", "adam"))
		self.validate_components(
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			callbacks=callbacks_resolved,
			optimizer=optimizer_resolved,
		)

		return TensorFlowGrangerModel(
			backend="tensorflow",
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			optimizer=optimizer_resolved,
			loss=config.get("loss", "mse"),
			callbacks=callbacks_resolved,
			epochs=config.get("epochs", 100),
			batch_size=config.get("batch_size", 32),
			verbose=config.get("verbose", 0),
		)

	def resolve_callbacks(self, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
		if self._object_loader is None:
			return callbacks
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
		if self._object_loader is None:
			return optimizer
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_optimizer(optimizer)

	def build_constraint_from_relations(
		self,
		relations: Dict[tuple, Any],
		predictor_names: List[str],
		output_names: List[str],
		col_offsets: NDArray,
		n_features: int,
		base_mask=None,
	):
		if not relations:
			return None

		from .constraints import build_tensorflow_constraint_from_relations

		return build_tensorflow_constraint_from_relations(
			relations=relations,
			predictor_names=predictor_names,
			output_names=output_names,
			col_offsets=col_offsets,
			n_features=n_features,
			base_mask=base_mask,
		)

	def build_regularizer(self, regularizer_spec: Any):
		if self._object_loader is None:
			return regularizer_spec
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_regularizer(regularizer_spec)

	def build_constraint(self, constraint_spec: Any) -> Any:
		if self._object_loader is None:
			return constraint_spec
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_constraint(constraint_spec)

	def validate_components(
		self,
		regularizer: Optional[Any],
		constraint: Optional[Any],
		callbacks: Optional[List[Any]] = None,
		optimizer: Any = None,
	) -> None:
		if self._object_loader is None:
			return
		self._object_loader.set_loading_verbose(self._loading_verbose)
		resolved_callbacks = self._object_loader.resolve_callbacks(callbacks)
		_ = self._object_loader.resolve_optimizer(optimizer)
		_ = self._object_loader.resolve_regularizer(regularizer)
		_ = self._object_loader.resolve_constraint(constraint)
		self._log_loaded_component("callbacks", resolved_callbacks)


