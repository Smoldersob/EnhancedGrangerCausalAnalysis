import os
from importlib.util import find_spec
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .base_model import BaseGrangerModel
from ...core.exceptions import (
    BackendNotAvailableError,
    ConstraintConfigurationError,
    RegularizerConfigurationError,
    ModelNotFittedError,
    TrainingError,
)
from .keras_device import DeviceManager

if find_spec("keras") is not None:
    import keras
else:
    keras = None

import keras

@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="FeatureMask",
)
class FeatureMask(keras.layers.Layer):
    """Input masking layer"""

    def __init__(self, n_features: int, **kwargs: Any) -> None:
        kwargs.setdefault("trainable", False)
        super().__init__(**kwargs)

        self.n_features = int(n_features)
        self.mask = self.add_weight(
            name="mask",
            shape=(self.n_features,),
            initializer="ones",
            trainable=False,
        )

    def build(self, input_shape: Any) -> None:
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        return keras.ops.multiply(inputs, self.mask)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"n_features": self.n_features})
        return config


@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="SimpleKerasGrangerModel",
)
class SimpleKerasGrangerModel(keras.Model):
    """Linear Granger model in keras for any backend."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        constraint: Optional[Any] = None,
        regularizer: Optional[Any] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ):
        kwargs.setdefault("name", "SimpleKerasGrangerModel")
        super().__init__(**kwargs)

        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.use_bias = bool(use_bias)

        self.variable_control_layer = FeatureMask(
            n_features=self.n_features,
            name="variable_control",
        )

        self.coefficient_layer = keras.layers.Dense(
            units=self.n_outputs,
            use_bias=self.use_bias,
            kernel_constraint=constraint,
            kernel_regularizer=regularizer,
            name="coefficients",
        )

    def build(self, input_shape: Any) -> None:
        input_shape = tuple(input_shape)

        if len(input_shape) != 2:
            raise ValueError(
                "Expected input shape (n_samples, n_features)."
            )

        if input_shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} input features, "
                f"received shape {input_shape}."
            )

        if not self.variable_control_layer.built:
            self.variable_control_layer.build(input_shape)

        # FeatureMask preserves the input shape.
        if not self.coefficient_layer.built:
            self.coefficient_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        chosen = self.variable_control_layer(inputs)
        return self.coefficient_layer(chosen, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "n_outputs": self.n_outputs,
                "use_bias": self.use_bias,
                "constraint": keras.saving.serialize_keras_object(
                    self.coefficient_layer.kernel_constraint
                ),
                "regularizer": keras.saving.serialize_keras_object(
                    self.coefficient_layer.kernel_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
    ) -> "SimpleKerasGrangerModel":
        config = dict(config)

        for key in ("constraint", "regularizer"):
            value = config.get(key)
            if isinstance(value, dict):
                config[key] = keras.saving.deserialize_keras_object(value)

        return cls(**config)

class KerasGrangerModel(BaseGrangerModel):
    """Keras implementation of a Granger model with pluggable mask/regularization."""

    def __init__(
        self,
        backend: str = "Keras",
        keras_backend: Optional[str] = None,
        regularizer: Optional[Any] = None,
        constraint: Optional[Any] = None,
        optimizer: Union[str, Any] = "adam",
        loss: Union[str, Any] = "mse",
        callbacks: Optional[list] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0,
        device: Optional[str] = None,
        use_jit: Optional[bool] = None,
        **kwargs: Any,
    ):
        if backend != "keras":
            raise ValueError(
                "KerasGrangerModel requires backend='keras'. "
                "Use keras_backend to validate the computational backend."
            )

        self.keras_backend = self._validate_keras_backend(keras_backend)

        super().__init__(
            backend=backend,
            regularizer=regularizer,
            constraint=constraint,
            callbacks=callbacks if callbacks is not None else [],
            needs_reinit=False,
        )

        self.optimizer = optimizer
        self._optimizer_spec = optimizer
        self._loss_spec = loss

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.device_manager = DeviceManager(
            backend=self.keras_backend,
            requested_device=device,
        )
        self.device = self.device_manager.name

        if use_jit is not None and not isinstance(use_jit, bool):
            raise TypeError("use_jit must be a bool or None.")

        self.use_jit = False if use_jit is None else use_jit

        self.model: Optional[Any] = None
        self._initial_optimizer_state: Optional[tuple] = None

        self._n_features: Optional[int] = None
        self._n_outputs: Optional[int] = None
        self._X_train: Optional[NDArray[np.float32]] = None
        self._y_train: Optional[NDArray[np.float32]] = None
        self._history: Optional[Any] = None

        self._validate_keras_components()

    @staticmethod
    def _validate_keras_backend(expected: Optional[str]) -> str:
        """Return the active Keras backend and validate an optional expectation."""
        active = keras.config.backend()

        if expected is None:
            return active

        if not isinstance(expected, str):
            raise TypeError("keras_backend must be a string or None.")

        aliases = {
            "tf": "tensorflow",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "torch": "torch",
            "pytorch": "torch",
        }

        normalized = expected.strip().lower()

        if normalized not in aliases:
            raise ValueError(
                "keras_backend must be None, 'tensorflow', 'jax', or 'torch'. "
                "Aliases 'tf' and 'pytorch' are also accepted."
            )

        expected_backend = aliases[normalized]

        if expected_backend != active:
            raise BackendNotAvailableError(
                f"Expected Keras backend {expected_backend!r}, "
                f"but the active backend is {active!r}. "
                "Configure KERAS_BACKEND before importing this module "
                "in a fresh Python process."
            )

        return active

    def _validate_keras_components(self) -> None:
        """Validate optional regularizer/constraint against Keras base classes."""
        keras_regularizer = keras.regularizers.Regularizer
        keras_constraint = keras.constraints.Constraint

        if self.regularizer is not None and not isinstance(self.regularizer, keras_regularizer):
            raise RegularizerConfigurationError(
                "Regularizer must inherit from keras.regularizers.Regularizer "
                "for KerasGrangerModel"
            )

        if self.constraint is not None and not isinstance(self.constraint, keras_constraint):
            raise ConstraintConfigurationError(
                "Constraint must inherit from keras.constraints.Constraint "
                "for KerasGrangerModel"
            )

        keras_callback = keras.callbacks.Callback
        if not isinstance(self.callbacks, list):
            raise ConstraintConfigurationError(
                "Callbacks must be a list of keras.callbacks.Callback objects"
            )
        for callback in self.callbacks:
            if not isinstance(callback, keras_callback):
                raise ConstraintConfigurationError(
                    "All callbacks must inherit from keras.callbacks.Callback"
                )

    def _normalize_optimizer_spec(self, spec: Any) -> Any:
        """Convert generic optimizer config into a Keras-deserializable mapping."""
        if not isinstance(spec, dict):
            return spec

        if "class_name" in spec and isinstance(spec.get("config"), dict):
            return spec

        optimizer_name = spec.get("type") or spec.get("class_name") or spec.get("name") or "adam"
        config: Dict[str, Any] = {}

        params = spec.get("params")
        if isinstance(params, dict):
            config.update(params)

        native_config = spec.get("config")
        if isinstance(native_config, dict):
            config.update(native_config)

        for key, value in spec.items():
            if key not in {"type", "class_name", "name", "params", "config"}:
                config[key] = value

        if "learning_rate" in spec and "learning_rate" not in config:
            config["learning_rate"] = spec["learning_rate"]

        name = str(optimizer_name).strip()
        alias_map = {
            "adam": "Adam",
            "adadelta": "Adadelta",
            "adagrad": "Adagrad",
            "adamax": "Adamax",
            "ftrl": "Ftrl",
            "nadam": "Nadam",
            "rmsprop": "RMSprop",
            "sgd": "SGD",
        }
        canonical_name = alias_map.get(name.lower(), name)

        return {"class_name": canonical_name, "config": config}

    def _build_optimizer(self) -> Any:
        """Create a fresh Keras optimizer instance from optimizer spec."""
        keras_optimizer = keras.optimizers.Optimizer

        spec = self._optimizer_spec
        if isinstance(spec, str):
            return keras.optimizers.get(spec)

        if isinstance(spec, dict):
            return keras.optimizers.deserialize(self._normalize_optimizer_spec(spec))
        if isinstance(spec, type) and issubclass(spec, keras_optimizer):
            return spec()

        if isinstance(spec, keras_optimizer):
            return spec.__class__.from_config(spec.get_config())

        if callable(spec):
            candidate = spec()
            if isinstance(candidate, keras_optimizer):
                return candidate
            raise ConstraintConfigurationError(
                "Optimizer callable must return keras.optimizers.Optimizer"
            )

        raise ConstraintConfigurationError(
            "Optimizer must be string, keras optimizer, keras optimizer class, dict, or callable"
        )

    def _build_loss(self) -> Any:
        """Resolve loss spec to Keras-compatible loss object/callable."""
        return keras.losses.get(self._loss_spec)

    def _capture_initial_optimizer_state(self) -> None:
        """Capture optimizer state once, before any training."""
        if self.model is None or self.model.optimizer is None:
            raise TrainingError(
                "The model must be compiled before capturing optimizer state."
            )

        if not self.model.built:
            raise TrainingError(
                "Build the model before capturing optimizer state."
            )

        optimizer = self.model.optimizer

        # Create optimizer variables before taking the snapshot.
        if not optimizer.built:
            optimizer.build(self.model.trainable_variables)

        self._initial_optimizer_state = [
            np.array(
                keras.ops.convert_to_numpy(variable),
                copy=True,
            )
            for variable in optimizer.variables
        ]

    def _reset_optimizer_state(self) -> None:
        """Restore the captured optimizer state without recompiling."""
        if self.model is None or self.model.optimizer is None:
            raise TrainingError(
                "The model must be compiled before resetting optimizer state."
            )

        initial_state = self._initial_optimizer_state
        if initial_state is None:
            raise TrainingError(
                "Initial optimizer state was not captured during initialize()."
            )

        optimizer = self.model.optimizer
        variables = list(optimizer.variables)

        if len(variables) != len(initial_state):
            raise TrainingError(
                "Optimizer variable layout changed. Reinitialize the model."
            )

        # Validate all shapes before modifying any variable.
        for index, (variable, initial_value) in enumerate(
                zip(variables, initial_state)
        ):
            if tuple(variable.shape) != initial_value.shape:
                raise TrainingError(
                    f"Optimizer variable {index} has shape "
                    f"{tuple(variable.shape)}, expected {initial_value.shape}. "
                    "Reinitialize the model."
                )

        for variable, initial_value in zip(variables, initial_state):
            optimizer.assign(variable, initial_value)

    def initialize(
        self,
        data: NDArray[np.float32]|List[NDArray[np.float32]],
        lags: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize model using lagged features prepared externally (e.g. LagEngine)."""
        self._validate_keras_components()

        X = np.asarray(data, dtype=np.float32)
        y_raw = kwargs.get("targets")
        if y_raw is None:
            raise TrainingError(
                "initialize requires precomputed targets via targets=<ndarray>. "
                "Lagged features should be prepared by LagEngine."
            )

        y = np.asarray(y_raw, dtype=np.float32)
        if X.ndim != 2:
            raise TrainingError("Expected 2D lagged feature matrix with shape (n_samples, n_lagged_features)")
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if y.ndim != 2:
            raise TrainingError("targets must be 1D or 2D array")
        if X.shape[0] != y.shape[0]:
            raise TrainingError("Lagged features and targets must have the same number of rows")

        n_features = X.shape[1]
        n_outputs = y.shape[1]

        with self.device_manager.scope():
            self.model = SimpleKerasGrangerModel(
                n_features=n_features,
                n_outputs=n_outputs,
                constraint=self.constraint,
                regularizer=self.regularizer,
            )

            self.model.build((None, n_features))

            self.model.compile(
                optimizer=self._build_optimizer(),
                loss=self._build_loss(),
                jit_compile=self.use_jit,
            )

            self._capture_initial_optimizer_state()

        self._n_features = n_features
        self._n_outputs = n_outputs
        self._X_train = X
        self._y_train = y
        self._fitted = False

    def fit(self, return_history: bool = False) -> Dict[str, Any]:
        """Train using current weights and mask, with fresh optimizer state."""
        if (
            self.model is None
            or self._X_train is None
            or self._y_train is None
        ):
            raise ModelNotFittedError(
                "Model is not initialized. Call initialize(...) first."
            )

        if self._X_train.shape[0] == 0:
            raise TrainingError("Cannot train on an empty dataset.")

        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, (int, np.integer))
            or self.epochs <= 0
        ):
            raise TrainingError("epochs must be a positive integer.")

        if self.batch_size is None:
            batch_size = self._X_train.shape[0]
        else:
            if (
                isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, (int, np.integer))
                or self.batch_size <= 0
            ):
                raise TrainingError(
                    "batch_size must be a positive integer or None."
                )
            batch_size = int(self.batch_size)

        # Do not retain a successful-fit flag or history from a previous run.
        self._fitted = False
        self._history = None

        try:
            self._reset_optimizer_state()

            with self.device_manager.scope():
                self._history = self.model.fit(
                    self._X_train,
                    self._y_train,
                    epochs=int(self.epochs),
                    batch_size=batch_size,
                    callbacks=self.callbacks,
                    verbose=self.verbose,
                    shuffle=False,
                )

        except Exception as exc:
            raise TrainingError(
                f"Keras model training failed: {exc}"
            ) from exc

        self._fitted = True

        # Training-only mode: skip prediction and result collection.
        if not return_history:
            return {}

        with self.device_manager.scope():
            predictions = self.model(self._X_train, training=False)

        forecasts = np.asarray(
            keras.ops.convert_to_numpy(predictions),
            dtype=np.float32,
        )

        history_data = (
            self._history.history
            if self._history is not None
            else {}
        )

        loss_history = list(history_data.get("loss", []))
        final_loss = (
            float(loss_history[-1])
            if loss_history
            else float("nan")
        )

        epoch_indices = (
            list(getattr(self._history, "epoch", []))
            if self._history is not None
            else []
        )

        epochs_ran = len(epoch_indices) or len(loss_history)

        return {
            "test_statistic": final_loss,
            "p_value": np.nan,
            "weights": self.get_weights(),
            "forecasts": forecasts,
            "history": {
                "loss": loss_history,
                "epochs_ran": epochs_ran,
                "stop_reason": (
                    "stopped_before_max_epochs"
                    if epochs_ran < self.epochs
                    else "max_epochs_reached"
                ),
            },
        }

    def _get_coefficient_layer(self) -> keras.layers.Dense:
        """Return the initialized, built coefficient layer."""
        if self.model is None:
            raise ModelNotFittedError(
                "Model is not initialized. Call initialize(...) first."
            )

        layer = self.model.coefficient_layer

        if not layer.built:
            raise TrainingError(
                "Coefficient layer is not built. "
                "Call initialize(...) first."
            )

        return layer

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate predictions from fitted TensorFlow model."""
        if not self._fitted or self.model is None or self._n_features is None:
            raise ModelNotFittedError("Model is not fitted. Call fit(...) first.")

        if X.ndim != 2:
            raise TrainingError("X must be a 2D array")
        if X.shape[1] != self._n_features:
            raise TrainingError(
                f"X has {X.shape[1]} features, expected {self._n_features}"
            )

        X_array = np.asarray(X, dtype=np.float32)

        with self.device_manager.scope():
            predictions = self.model(X_array, training=False)

        return np.asarray(
            keras.ops.convert_to_numpy(predictions),
            dtype=np.float32,
        )

    def get_weights(self) -> List[NDArray[Any]]:
        """Return coefficient weights only; exclude the feature mask."""
        layer = self._get_coefficient_layer()

        return [
            np.array(weight, copy=True)
            for weight in layer.get_weights()
        ]

    def set_weights(
            self,
            weights: Any|
                NDArray[Any]|
                List[NDArray[Any]]|
                tuple[NDArray[Any],
            ],
    ) -> None:
        """Set coefficient weights only; preserve mask and optimizer state."""
        layer = self._get_coefficient_layer()

        if isinstance(weights, np.ndarray):
            supplied = [weights]
        elif isinstance(weights, (list, tuple)):
            supplied = list(weights)
        else:
            raise TrainingError(
                "weights must be a kernel ndarray, [kernel], "
                "or [kernel, bias]."
            )

        expected_count = 2 if layer.use_bias else 1

        if len(supplied) not in {1, expected_count}:
            description = (
                "kernel or [kernel, bias]"
                if layer.use_bias
                else "kernel or [kernel]"
            )
            raise TrainingError(
                f"Invalid number of weight arrays. Expected {description}."
            )

        variables = list(layer.weights)
        proposed: List[NDArray[Any]] = []

        # Validate all supplied arrays before modifying the layer.
        for index, (value, variable) in enumerate(
                zip(supplied, variables)
        ):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError) as exc:
                raise TrainingError(
                    f"Weight {index} cannot be converted to an array."
                ) from exc

            expected_shape = tuple(variable.shape)
            if array.shape != expected_shape:
                raise TrainingError(
                    f"Weight {index} has shape {array.shape}, "
                    f"expected {expected_shape}."
                )

            proposed.append(array)

        # Kernel-only updates preserve the current bias.
        if layer.use_bias and len(proposed) == 1:
            proposed.append(layer.get_weights()[1])

        try:
            layer.set_weights(proposed)
        except (TypeError, ValueError) as exc:
            raise TrainingError(
                f"Cannot set coefficient weights: {exc}"
            ) from exc

    def omit_variables(self, variable_indices: List[int]) -> None:
        """Replace the omitted feature set without changing coefficients."""
        if self.model is None or self._n_features is None:
            raise ModelNotFittedError(
                "Model is not initialized. Call initialize(...) first."
            )

        indices = np.asarray(variable_indices)

        if indices.ndim != 1:
            raise TrainingError(
                "variable_indices must be a one-dimensional list of integers."
            )

        if indices.size:
            # Validate before conversion: do not silently truncate floats.
            if indices.dtype.kind not in {"i", "u"}:
                raise TrainingError(
                    "Variable indices must be integers, not floats or booleans."
                )

            if np.any(indices < 0) or np.any(indices >= self._n_features):
                raise TrainingError(
                    f"Variable indices must belong to "
                    f"[0, {self._n_features - 1}]."
                )

        indices = indices.astype(np.intp, copy=False)

        variable_mask = np.ones(self._n_features, dtype=np.float32)
        variable_mask[indices] = 0.0

        self.model.variable_control_layer.mask.assign(variable_mask)

    def set_regularizer(self, regularizer: Any) -> None:
        """Set regularizer with Keras type validation."""
        self.regularizer = regularizer
        self._validate_keras_components()

    def set_constraint(self, constraint: Any) -> None:
        """Set constraint with Keras type validation."""
        self.constraint = constraint
        self._validate_keras_components()

    def reset_callbacks(
            self,
            run_name: Optional[str] = None,
            callback_updates: Optional[
                Mapping[Union[int, str, type], Mapping[str, Any]]
            ] = None,
    ) -> None:
        """
        Prepare callbacks for a new experiment without changing the model.

        Built-in callbacks supported here are reconstructed using their current
        configuration, discarding previous training state.

        Custom callbacks must implement either:
          - get_reset_config(): return constructor keyword arguments; or
          - reset_state(): clear runtime state in place.

        TensorBoard's original log_dir is remembered as its root directory.
        An explicit log_dir update changes that root.
        With run_name, output goes to <root>/<run_name>.

        Call between fit() calls, not during training.
        """
        if self.model is None:
            raise ModelNotFittedError(
                "Model is not initialized. Call initialize(...) first."
            )

        self._validate_keras_components()

        if run_name is not None:
            if not isinstance(run_name, str) or not run_name.strip():
                raise ConstraintConfigurationError(
                    "run_name must be None or a non-empty string."
                )
            if (
                    run_name in {".", ".."}
                    or "/" in run_name
                    or "\\" in run_name
                    or "\x00" in run_name
            ):
                raise ConstraintConfigurationError(
                    "run_name must be a single directory name."
                )

        if callback_updates is None:
            callback_updates = {}
        elif not isinstance(callback_updates, Mapping):
            raise ConstraintConfigurationError(
                "callback_updates must be a mapping or None."
            )

        # Resolve and validate all updates before changing any callback.
        resolved_updates = [{} for _ in self.callbacks]

        for selector, values in callback_updates.items():
            if not isinstance(values, Mapping):
                raise ConstraintConfigurationError(
                    "Each callback_updates value must be a mapping."
                )

            for name in values:
                if not isinstance(name, str) or name.startswith("_"):
                    raise ConstraintConfigurationError(
                        f"Invalid callback configuration key: {name!r}."
                    )

            matches = [
                index
                for index, callback in enumerate(self.callbacks)
                if self._callback_matches(selector, callback, index)
            ]

            if not matches:
                raise ConstraintConfigurationError(
                    f"Callback selector {selector!r} matches no callback."
                )

            for index in matches:
                resolved_updates[index].update(values)

        # Records contain (callback instance, original TensorBoard root).
        previous_roots = getattr(self, "_tensorboard_reset_roots", {})
        next_roots = {}
        fresh_callbacks = []
        in_place_resets = []

        for index, callback in enumerate(self.callbacks):
            updates = resolved_updates[index]
            config = self._callback_reset_config(callback)

            if config is None:
                # Custom callback with an explicit in-place reset protocol.
                for name in updates:
                    if not hasattr(callback, name):
                        raise ConstraintConfigurationError(
                            f"{type(callback).__name__} has no public "
                            f"configuration attribute {name!r}."
                        )

                in_place_resets.append((callback, updates))
                fresh_callbacks.append(callback)
                continue

            unknown = set(updates) - set(config)
            if unknown:
                raise ConstraintConfigurationError(
                    f"Unsupported configuration updates for "
                    f"{type(callback).__name__}: {sorted(unknown)}."
                )

            config.update(updates)

            root = None
            if isinstance(callback, keras.callbacks.TensorBoard):
                record = previous_roots.get(index)

                if "log_dir" in updates:
                    root = os.fspath(updates["log_dir"])
                elif record is not None and record[0] is callback:
                    root = record[1]
                else:
                    root = os.fspath(callback.log_dir)

                if run_name is not None:
                    config["log_dir"] = os.path.join(root, run_name)

            try:
                fresh_callback = type(callback)(**config)
            except Exception as exc:
                raise ConstraintConfigurationError(
                    f"Could not reconstruct {type(callback).__name__}. "
                    "Check its configuration or implement "
                    "get_reset_config() for a custom callback."
                ) from exc

            fresh_callbacks.append(fresh_callback)

            if root is not None:
                next_roots[index] = (fresh_callback, root)

        # These callbacks explicitly opt into mutation.
        # A failing custom reset_state() can partially mutate its callback.
        for callback, updates in in_place_resets:
            for name, value in updates.items():
                setattr(callback, name, value)
            callback.reset_state()

        self.callbacks = fresh_callbacks
        self._tensorboard_reset_roots = next_roots
        self.model.stop_training = False

    @staticmethod
    def _callback_matches(
            selector: Union[int, str, type],
            callback: Any,
            index: int,
    ) -> bool:
        """Match an index, exact class name, or callback type."""
        if isinstance(selector, int) and not isinstance(selector, bool):
            return selector == index

        if isinstance(selector, str):
            return selector == type(callback).__name__

        if (
                isinstance(selector, type)
                and issubclass(selector, keras.callbacks.Callback)
        ):
            return isinstance(callback, selector)

        raise ConstraintConfigurationError(
            f"Invalid callback selector: {selector!r}. "
            "Use an integer index, class name, or Keras callback type."
        )

    @staticmethod
    def _callback_reset_config(
            callback: Any,
    ):
        """
        Return constructor configuration, or None for an in-place reset.

        Exact type checks avoid reconstructing subclasses with incomplete
        configuration. Subclasses should implement the custom reset protocol.
        """
        provider = getattr(callback, "get_reset_config", None)
        if callable(provider):
            config = provider()
            if not isinstance(config, Mapping):
                raise ConstraintConfigurationError(
                    "get_reset_config() must return a mapping."
                )

            config = dict(config)
            if any(
                    not isinstance(name, str) or name.startswith("_")
                    for name in config
            ):
                raise ConstraintConfigurationError(
                    "get_reset_config() must contain public constructor "
                    "argument names."
                )
            return config

        callback_type = type(callback)

        if callback_type is keras.callbacks.EarlyStopping:
            return {
                "monitor": callback.monitor,
                "min_delta": abs(callback.min_delta),
                "patience": callback.patience,
                "verbose": callback.verbose,
                "mode": callback.mode,
                "baseline": callback.baseline,
                "restore_best_weights": callback.restore_best_weights,
                "start_from_epoch": callback.start_from_epoch,
            }

        if callback_type is keras.callbacks.ReduceLROnPlateau:
            return {
                "monitor": callback.monitor,
                "factor": callback.factor,
                "patience": callback.patience,
                "verbose": callback.verbose,
                "mode": callback.mode,
                "min_delta": abs(callback.min_delta),
                "cooldown": callback.cooldown,
                "min_lr": callback.min_lr,
            }

        if callback_type is keras.callbacks.TensorBoard:
            # Keras retains profile_batch as private start/stop settings.
            # Read these settings; do not modify writer or counter internals.
            if not (
                    hasattr(callback, "_start_batch")
                    and hasattr(callback, "_stop_batch")
            ):
                raise ConstraintConfigurationError(
                    "Cannot recover TensorBoard profile_batch in this "
                    "Keras version. Use a TensorBoard subclass implementing "
                    "get_reset_config()."
                )

            start = callback._start_batch
            stop = callback._stop_batch
            profile_batch = 0 if start == 0 and stop == 0 else (start, stop)

            return {
                "log_dir": callback.log_dir,
                "histogram_freq": callback.histogram_freq,
                "write_graph": callback.write_graph,
                "write_images": callback.write_images,
                "write_steps_per_second": callback.write_steps_per_second,
                "update_freq": callback.update_freq,
                "profile_batch": profile_batch,
                "embeddings_freq": callback.embeddings_freq,
                "embeddings_metadata": callback.embeddings_metadata,
            }

        if callback_type is keras.callbacks.ModelCheckpoint:
            return {
                "filepath": callback.filepath,
                "monitor": callback.monitor,
                "verbose": callback.verbose,
                "save_best_only": callback.save_best_only,
                "save_weights_only": callback.save_weights_only,
                "mode": callback.mode,
                "save_freq": callback.save_freq,
                # Do not use callback.best: it contains previous-run results.
                # Supply a non-default threshold through callback_updates.
                "initial_value_threshold": None,
            }

        if callable(getattr(callback, "reset_state", None)):
            return None

        raise ConstraintConfigurationError(
            f"No safe reset implementation for {callback_type.__name__}. "
            "Implement get_reset_config() to reconstruct it, or reset_state() "
            "to clear its runtime state in place."
        )