from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import keras
import numpy as np

from ....core.exceptions import (
    BackendNotAvailableError,
    ModelNotFittedError,
    TrainingError,
)
from ....core.models.protocol import (
    CallbackUpdates,
    FloatArray,
    ModelWeights,
    TrainingResult,
)
from ..callbacks.reset import reset_callbacks_for_run
from ..device import DeviceManager
from ..optimizers.reset import (
    OptimizerSnapshot,
    capture_optimizer_state,
    clone_optimizer,
    restore_optimizer_state,
)
from .network_parts import LinearGrangerNetwork
from .registry import register_model


@register_model(
    "conditional",
    "conditional_linear",
)
class KerasConditionalAdapter:
    """Multi-output linear model with independently controlled input masking."""

    def __init__(
        self,
        *,
        optimizer: keras.optimizers.Optimizer,
        loss: Any,
        regularizer: keras.regularizers.Regularizer | None = None,
        constraint: keras.constraints.Constraint | None = None,
        callbacks: Sequence[keras.callbacks.Callback] | None = None,
        epochs: int = 100,
        batch_size: int | None = 32,
        verbose: int = 0,
        device: str | None = None,
        keras_backend: str | None = None,
        use_jit: bool = False,
        use_bias: bool = True,
    ) -> None:
        if not isinstance(optimizer, keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer must be a ready Keras optimizer instance."
            )

        if isinstance(loss, type) or not callable(loss):
            raise TypeError(
                "loss must be a ready loss object or function."
            )

        if regularizer is not None and not isinstance(
            regularizer, keras.regularizers.Regularizer
        ):
            raise TypeError(
                "regularizer must be a Keras regularizer or None."
            )

        if constraint is not None and not isinstance(
            constraint, keras.constraints.Constraint
        ):
            raise TypeError(
                "constraint must be a Keras constraint or None."
            )

        if not isinstance(use_jit, bool):
            raise TypeError("use_jit must be a bool.")

        if not isinstance(use_bias, bool):
            raise TypeError("use_bias must be a bool.")

        self.callbacks = list(callbacks) if callbacks is not None else []
        if any(
            not isinstance(cb, keras.callbacks.Callback)
            for cb in self.callbacks
        ):
            raise TypeError(
                "All callbacks must inherit from keras.callbacks.Callback."
            )

        self.keras_backend = self._validate_keras_backend(keras_backend)
        self.device_manager = DeviceManager(
            backend=self.keras_backend,
            requested_device=device,
        )
        self.device = self.device_manager.name

        self.regularizer = regularizer
        self.constraint = constraint
        self.loss = loss

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_jit = use_jit
        self.use_bias = use_bias

        # Configuration template, never attached directly to a new network.
        self._optimizer_template = optimizer

        self.model: LinearGrangerNetwork | None = None
        self._optimizer_snapshot: OptimizerSnapshot | None = None

        self._X_train: Any = None
        self._y_train: Any = None
        self._n_features: int | None = None
        self._n_outputs: int | None = None

        self._fitted = False
        self._run_counter = 0

        self._validate_training_config()

    @property
    def backend(self) -> str:
        return "keras"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        return self._require_model().optimizer

    @staticmethod
    def _validate_keras_backend(expected: str | None) -> str:
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

        normalized = aliases.get(expected.strip().lower())
        if normalized is None:
            raise ValueError(
                "Unsupported keras_backend. Use tensorflow, torch or jax."
            )

        if normalized != active:
            raise BackendNotAvailableError(
                f"Expected Keras backend {normalized!r}, got {active!r}. "
                "Configure KERAS_BACKEND before importing Keras."
            )

        return active

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _validate_training_config(self) -> None:
        self._positive_int(self.epochs, "epochs")

        if self.batch_size is not None:
            self._positive_int(self.batch_size, "batch_size")

        if (
            isinstance(self.verbose, (bool, np.bool_))
            or not isinstance(self.verbose, (int, np.integer))
            or self.verbose not in (0, 1, 2)
        ):
            raise ValueError("verbose must be 0, 1 or 2.")

    def _require_model(self) -> LinearGrangerNetwork:
        if self.model is None:
            raise ModelNotFittedError(
                "Model is not initialized. Call initialize() first."
            )
        return self.model

    @staticmethod
    def _data_array(value: Any, name: str) -> np.ndarray:
        return real_array(
            value,
            name=name,
            dtype=np.float32,
            ndim=2,
        )

    def initialize(
        self,
        X: FloatArray,
        *,
        targets: FloatArray,
        seed: int | None = None,
    ) -> None:
        self._validate_training_config()

        x_array = self._data_array(X, "X")
        y_array = self._data_array(targets, "targets")

        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and targets must have the same sample count.")

        if (
            x_array.shape[0] == 0
            or x_array.shape[1] == 0
            or y_array.shape[1] == 0
        ):
            raise ValueError("Training data dimensions must be non-zero.")

        if seed is not None:
            if (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or seed < 0
            ):
                raise ValueError("seed must be a non-negative integer or None.")
            keras.utils.set_random_seed(int(seed))

        n_features = x_array.shape[1]
        n_outputs = y_array.shape[1]

        # Commit new state only after successful construction.
        with self.device_manager.scope():
            network = LinearGrangerNetwork(
                n_features=n_features,
                n_outputs=n_outputs,
                constraint=self.constraint,
                regularizer=self.regularizer,
                use_bias=self.use_bias,
                dtype="float32",
            )
            network.build((None, n_features))

            optimizer = clone_optimizer(self._optimizer_template)

            network.compile(
                optimizer=optimizer,
                loss=self.loss,
                jit_compile=self.use_jit,
            )

            optimizer.build(network.trainable_variables)
            snapshot = capture_optimizer_state(optimizer)

            x_tensor = keras.ops.convert_to_tensor(x_array, dtype="float32")
            y_tensor = keras.ops.convert_to_tensor(y_array, dtype="float32")

        self.model = network
        self._optimizer_snapshot = snapshot
        self._X_train = x_tensor
        self._y_train = y_tensor
        self._n_features = n_features
        self._n_outputs = n_outputs
        self._fitted = False

    def reset_components(
        self,
        *,
        reset_optimizer: bool = True,
        reset_callbacks: bool = True,
        run_name: str | None = None,
        callback_updates: CallbackUpdates | None = None,
    ) -> None:
        network = self._require_model()

        if not isinstance(reset_optimizer, bool):
            raise TypeError("reset_optimizer must be a bool.")

        if not isinstance(reset_callbacks, bool):
            raise TypeError("reset_callbacks must be a bool.")

        if not reset_callbacks and callback_updates is not None:
            raise ValueError(
                "callback_updates requires reset_callbacks=True."
            )

        next_counter = self._run_counter + 1
        name = (
            run_name
            if run_name is not None
            else f"run_{next_counter:04d}"
        )

        # Prepare replacements before changing the optimizer state.
        fresh_callbacks = None
        if reset_callbacks:
            fresh_callbacks = reset_callbacks_for_run(
                self.callbacks,
                run_name=name,
                callback_updates=callback_updates,
            )

        if reset_optimizer:
            if self._optimizer_snapshot is None:
                raise TrainingError("Initial optimizer state is unavailable.")

            with self.device_manager.scope():
                restore_optimizer_state(
                    network.optimizer,
                    self._optimizer_snapshot,
                )

        if fresh_callbacks is not None:
            self.callbacks = fresh_callbacks

        if reset_callbacks:
            self._run_counter = next_counter

    def fit(
        self,
        *,
        collect_history: bool = False,
        reset_components: bool = True,
        run_name: str | None = None,
        callback_updates: CallbackUpdates | None = None,
    ) -> TrainingResult:
        network = self._require_model()
        self._validate_training_config()

        if not isinstance(collect_history, bool):
            raise TypeError("collect_history must be a bool.")

        if not isinstance(reset_components, bool):
            raise TypeError("reset_components must be a bool.")

        if not reset_components and (
            callback_updates is not None or run_name is not None
        ):
            raise ValueError(
                "run_name and callback_updates require reset_components=True. "
                "Alternatively, call reset_components() explicitly."
            )

        if self._X_train is None or self._y_train is None:
            raise TrainingError("Training data is unavailable.")

        batch_size = (
            int(self._X_train.shape[0])
            if self.batch_size is None
            else int(self.batch_size)
        )

        self._fitted = False

        try:
            if reset_components:
                self.reset_components(
                    run_name=run_name,
                    callback_updates=callback_updates,
                )

            with self.device_manager.scope():
                history = network.fit(
                    self._X_train,
                    self._y_train,
                    epochs=int(self.epochs),
                    batch_size=batch_size,
                    callbacks=self.callbacks,
                    verbose=int(self.verbose),
                    shuffle=False,
                )

        except Exception as exc:
            raise TrainingError(
                f"Keras model training failed: {exc}"
            ) from exc

        self._fitted = True

        if not collect_history:
            return {}

        history_data = {
            name: [float(value) for value in values]
            for name, values in history.history.items()
        }
        iterations = len(history.epoch)

        return {
            "history": history_data,
            "metrics": {
                name: values[-1]
                for name, values in history_data.items()
                if values
            },
            "n_iterations": iterations,
            "stop_reason": (
                "stopped_before_max_epochs"
                if iterations < int(self.epochs)
                else "max_epochs_reached"
            ),
        }

    def predict(self, X: FloatArray) -> FloatArray:
        network = self._require_model()

        if not self._fitted:
            raise ModelNotFittedError(
                "Model is not fitted. Call fit() first."
            )

        x_array = self._data_array(X, "X")

        if x_array.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, "
                f"got {x_array.shape[1]}."
            )

        if x_array.shape[0] == 0:
            return np.empty(
                (0, int(self._n_outputs)),
                dtype=np.float32,
            )

        with self.device_manager.scope():
            predictions = network(x_array, training=False)
            result = keras.ops.convert_to_numpy(predictions)

        return np.array(result, dtype=np.float32, copy=True)

    def get_weights(self) -> ModelWeights:
        layer = self._require_model().coefficient_layer

        with self.device_manager.scope():
            return [
                np.array(
                    keras.ops.convert_to_numpy(variable),
                    copy=True,
                )
                for variable in layer.weights
            ]

    def set_weights(self, weights: ModelWeights) -> None:
        layer = self._require_model().coefficient_layer
        variables = list(layer.weights)

        if not isinstance(weights, list) or len(weights) != len(variables):
            raise ValueError(
                f"Expected a list of {len(variables)} parameter arrays."
            )

        converted = []

        for index, (value, variable) in enumerate(zip(weights, variables)):
            raw = np.asarray(value)

            if raw.dtype.kind not in "fiu":
                raise TypeError(
                    f"Parameter {index} must contain real numeric values."
                )

            if raw.shape != tuple(variable.shape):
                raise ValueError(
                    f"Parameter {index}: expected shape "
                    f"{tuple(variable.shape)}, got {raw.shape}."
                )

            with np.errstate(over="ignore", invalid="ignore"):
                array = np.array(raw, dtype=np.float32, copy=True)

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Parameter {index} contains non-finite values "
                    "after float32 conversion."
                )

            converted.append(array)

        # No validation failure can cause a partial parameter assignment.
        self._fitted = False
        with self.device_manager.scope():
            for variable, array in zip(variables, converted):
                variable.assign(array)

    def omit_variables(self, variable_indices: Sequence[int]) -> None:
        network = self._require_model()
        indices = list(variable_indices)

        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in indices
        ):
            raise TypeError("Feature indices must be integers, not booleans.")

        if any(
            index < 0 or index >= int(self._n_features)
            for index in indices
        ):
            raise ValueError("Feature index is out of range.")

        mask = np.ones(int(self._n_features), dtype=np.float32)
        if indices:
            mask[np.asarray(indices, dtype=np.intp)] = 0.0

        with self.device_manager.scope():
            network.variable_control_layer.mask.assign(mask)

    def reset_callbacks(
        self,
        run_name: str,
        callback_updates: CallbackUpdates | None = None,
    ) -> None:
        """Convenience method; optimizer state is preserved."""
        self.reset_components(
            reset_optimizer=False,
            reset_callbacks=True,
            run_name=run_name,
            callback_updates=callback_updates,
        )