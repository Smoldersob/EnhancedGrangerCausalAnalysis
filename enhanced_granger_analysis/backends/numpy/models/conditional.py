from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ....core.exceptions import ModelNotFittedError, TrainingError
from ....core.models.input_spec import ModelInputSpec
from ....core.models.protocol import (
	CallbackUpdates,
	FloatArray,
	ModelWeights,
	TrainingResult,
)
from ..callbacks.base import Callback
from ..callbacks.reset import reset_callbacks_for_run
from ..constraints.base import BaseNumpyConstraint
from ..regularizers.base import BaseNumpyRegularizer
from .network_parts import (
	LinearGrangerNetwork,
	positive_int,
	real_array,
)
from .registry import register_model


class _StepSizeControl:
	"""Compatibility bridge for existing callbacks, not an optimizer.

	'lr' is a relaxation factor for a curvature-scaled update.
	"""

	def __init__(self, learning_rate: float) -> None:
		self.param_groups = [{"lr": float(learning_rate)}]


class _ArrayView:
	"""Compatibility bridge for the existing histogram callback."""

	def __init__(self, value: np.ndarray) -> None:
		self._value = value.copy()

	def detach(self) -> _ArrayView:
		return self

	def cpu(self) -> np.ndarray:
		return self._value.copy()


class _CoefficientView:
	def __init__(self, adapter: NumpyConditionalAdapter) -> None:
		self._adapter = adapter

	@property
	def weight(self) -> _ArrayView:
		network = self._adapter._require_model()
		return _ArrayView(network.coefficient_layer.weight)


@register_model("conditional", "conditional_linear")
class NumpyConditionalAdapter:
	"""Granger protocol adapter around a separate NumPy architecture.

	The data objective is MSE averaged over samples and outputs.

	Built-in L1 regularizers use exact proximal steps. Custom regularizers
	without a prox() method use a gradient/subgradient fallback and may
	require a smaller learning_rate.

	Constraints are applied after each update. Arbitrary nonconvex
	constraints do not imply convergence to a global optimum.
	"""

	def __init__(
		self,
		*,
		input_spec: ModelInputSpec,
		solver: str = "auto",
		regularizer: BaseNumpyRegularizer | None = None,
		constraint: BaseNumpyConstraint | None = None,
		callbacks: Sequence[Callback] | None = None,
		epochs: int | None = None,
		batch_size: int | None = None,
		verbose: int = 0,
		use_bias: bool | None = None,
		learning_rate: float = 1.0,
		tol: float = 1e-8,
		seed: int | None = None,
		max_iter: int | None = None,
		fit_intercept: bool | None = None,
	) -> None:
		if not isinstance(input_spec, ModelInputSpec):
			raise TypeError("input_spec must be a ModelInputSpec.")

		if epochs is not None and max_iter is not None:
			raise ValueError(
				"Specify epochs or its legacy alias max_iter, not both."
			)

		if use_bias is not None and fit_intercept is not None:
			raise ValueError(
				"Specify use_bias or its legacy alias "
				"fit_intercept, not both."
			)

		if regularizer is not None and not isinstance(
			regularizer, BaseNumpyRegularizer
		):
			raise TypeError(
				"regularizer must be a BaseNumpyRegularizer or None."
			)

		if constraint is not None and not isinstance(
			constraint, BaseNumpyConstraint
		):
			raise TypeError(
				"constraint must be a BaseNumpyConstraint or None."
			)

		self._input_spec = input_spec
		self._n_features = input_spec.n_features
		self._n_outputs = input_spec.n_outputs

		self.epochs = (
			epochs
			if epochs is not None
			else (1000 if max_iter is None else max_iter)
		)
		self.use_bias = (
			use_bias
			if use_bias is not None
			else (True if fit_intercept is None else fit_intercept)
		)

		self.batch_size = batch_size
		self.verbose = verbose
		self.learning_rate = learning_rate
		self.tol = tol
		self.solve = solver
		self.solver_used_: str | None = None

		# Configuration templates. Runtime copies belong to the network.
		self.regularizer = regularizer
		self.constraint = constraint

		self.callbacks = (
			list(callbacks) if callbacks is not None else []
		)
		if any(
			not isinstance(callback, Callback)
			for callback in self.callbacks
		):
			raise TypeError(
				"All callbacks must inherit from scikit Callback."
			)

		self.model: LinearGrangerNetwork | None = None
		self._fitted = False
		self._run_counter = 0
		self._loss_history: list[float] = []
		self._coefficient_view = _CoefficientView(self)

		self._validate_training_config()
		self.rebuild(seed=seed)

	@property
	def backend(self) -> str:
		return "numpy"

	@property
	def input_spec(self) -> ModelInputSpec:
		return self._input_spec

	@property
	def is_fitted(self) -> bool:
		return self._fitted

	@property
	def max_iter(self) -> int:
		return self.epochs

	@max_iter.setter
	def max_iter(self, value: int) -> None:
		self.epochs = value

	@property
	def fit_intercept(self) -> bool:
		return self.use_bias

	@property
	def solver(self) -> str:
		return self.solve

	@property
	def coef_(self) -> np.ndarray:
		return self._require_model().coefficient_layer.weight.copy()

	@property
	def intercept_(self) -> np.ndarray:
		bias = self._require_model().coefficient_layer.bias

		if bias is None:
			return np.zeros(self._n_outputs, dtype=np.float64)

		return bias.copy()

	@property
	def _coefficient_layer(self) -> _CoefficientView:
		return self._coefficient_view

	@property
	def _variable_control_layer(self):
		return self._require_model().variable_control_layer

	@staticmethod
	def _real_scalar(value: Any, name: str) -> float:
		raw = np.asarray(value)

		if raw.shape != () or raw.dtype.kind not in "fiu":
			raise TypeError(
				f"{name} must be a real numeric scalar."
			)

		result = float(raw)

		if not np.isfinite(result):
			raise ValueError(f"{name} must be finite.")

		return result

	def _validate_training_config(self) -> None:
		positive_int(self.epochs, "epochs")

		if self.batch_size is not None:
			positive_int(self.batch_size, "batch_size")

		if not isinstance(self.use_bias, bool):
			raise TypeError("use_bias must be a bool.")

		if (
			isinstance(self.verbose, (bool, np.bool_))
			or not isinstance(self.verbose, (int, np.integer))
			or self.verbose not in (0, 1, 2)
		):
			raise ValueError("verbose must be 0, 1 or 2.")

		learning_rate = self._real_scalar(
			self.learning_rate, "learning_rate"
		)
		if not 0 < learning_rate <= 1:
			raise ValueError("learning_rate must be in (0, 1].")

		if self._real_scalar(self.tol, "tol") < 0:
			raise ValueError("tol must be non-negative.")

		if (
			self.model is not None
			and self.model.use_bias != self.use_bias
		):
			raise ValueError(
				"Changing use_bias requires a new adapter."
			)

		if not isinstance(self.solver, str):
			raise TypeError("solver must be a string.")

		allowed_solvers = {
			"auto",
			"restricted_lstsq",
			"coefficient",
		}

		if self.solver not in allowed_solvers:
			raise ValueError(
				"solver must be one of: "
				"'auto', 'restricted_lstsq', 'coefficient'."
			)

	def _require_model(self) -> LinearGrangerNetwork:
		if self.model is None:
			raise RuntimeError(
				"Predictor construction did not complete."
			)
		return self.model

	def rebuild(self, *, seed: int | None = None) -> None:
		self._validate_training_config()

		if seed is not None and (
			isinstance(seed, (bool, np.bool_))
			or not isinstance(seed, (int, np.integer))
			or seed < 0
		):
			raise ValueError(
				"seed must be a non-negative integer or None."
			)

		# Zero initialization is deterministic; no global RNG changes.
		regularizer = None
		if self.regularizer is not None:
			regularizer = type(self.regularizer).from_config(
				self.regularizer.get_config()
			)

		constraint = None
		if self.constraint is not None:
			constraint = type(self.constraint).from_config(
				self.constraint.get_config()
			)

		if regularizer is not None:
			layout = regularizer.data.layout

			if (
				tuple(layout.output_names) != self.input_spec.effects
				or dict(layout.cause_slices)
				!= dict(self.input_spec.cause_slices)
				or tuple(layout.feature_lags)
				!= self.input_spec.feature_lags
			):
				raise ValueError(
					"Regularizer layout does not match input_spec."
				)

		if (
			constraint is not None
			and tuple(constraint.prepared_data.output_names)
			!= self.input_spec.effects
		):
			raise ValueError(
				"Constraint output order does not match input_spec."
			)

		network = LinearGrangerNetwork(
			n_features=self._n_features,
			n_outputs=self._n_outputs,
			constraint=constraint,
			regularizer=regularizer,
			use_bias=self.use_bias,
		)

		self.model = network
		self._step_control = _StepSizeControl(
			float(self.learning_rate)
		)
		self._loss_history = []
		self._fitted = False
		self.solver_used_ = None

	def reset_components(
		self,
		*,
		reset_optimizer: bool = True,
		reset_callbacks: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> None:
		self._require_model()

		if (
			not isinstance(reset_optimizer, bool)
			or not isinstance(reset_callbacks, bool)
		):
			raise TypeError(
				"reset_optimizer and reset_callbacks must be bools."
			)

		if not reset_callbacks and (
			run_name is not None or callback_updates is not None
		):
			raise ValueError(
				"run_name and callback_updates require "
				"reset_callbacks=True."
			)

		next_counter = self._run_counter + 1
		name = (
			run_name
			if run_name is not None
			else f"run_{next_counter:04d}"
		)

		fresh_callbacks = None

		if reset_callbacks:
			fresh_callbacks = reset_callbacks_for_run(
				self.callbacks,
				run_name=name,
				callback_updates=callback_updates,
			)

		if reset_optimizer:
			# Uniform protocol flag: reset loop control, not an optimizer.
			rate = self._real_scalar(
				self.learning_rate, "learning_rate"
			)
			if not 0 < rate <= 1:
				raise ValueError(
					"learning_rate must be in (0, 1]."
				)
			self._step_control = _StepSizeControl(rate)

		if fresh_callbacks is not None:
			self.callbacks = fresh_callbacks
			self._run_counter = next_counter

	def reset_callbacks(
		self,
		run_name: str,
		callback_updates: CallbackUpdates | None = None,
	) -> None:
		self.reset_components(
			reset_optimizer=False,
			reset_callbacks=True,
			run_name=run_name,
			callback_updates=callback_updates,
		)

	def _run_callback_hook(
		self,
		name: str,
		state: dict[str, Any],
	) -> None:
		for callback in self.callbacks:
			response = getattr(callback, name)(state)

			if response is False:
				state["stop_training"] = True

				if state.get("stop_reason") is None:
					state["stop_reason"] = (
						f"{type(callback).__name__}:{name}"
					)

		# All callbacks receive the hook, including after a stop request.

	def _training_arrays(
		self,
		X: Any,
		y: Any,
	) -> tuple[np.ndarray, np.ndarray]:
		x_array = real_array(X, "X")
		y_array = real_array(y, "y")

		if x_array.ndim != 2 or y_array.ndim != 2:
			raise ValueError(
				"X and y must be two-dimensional arrays."
			)

		if (
			x_array.shape[0] == 0
			or x_array.shape[0] != y_array.shape[0]
		):
			raise ValueError(
				"X and y must have equal, non-zero sample counts."
			)

		if (
			x_array.shape[1] != self._n_features
			or y_array.shape[1] != self._n_outputs
		):
			raise ValueError(
				"X/y feature and output dimensions must "
				"match input_spec."
			)

		return x_array, y_array

	def _objective(
		self,
		X: np.ndarray,
		y: np.ndarray,
	) -> tuple[float, float, float]:
		network = self._require_model()

		with np.errstate(over="ignore", invalid="ignore"):
			mse = float(np.mean((network(X) - y) ** 2))
			penalty = network.regularization_loss()
			loss = mse + penalty

		if not np.isfinite([mse, penalty, loss]).all():
			raise ValueError(
				"Training objective is non-finite."
			)

		return loss, mse, penalty

	@staticmethod
	def _update_bias(
		network: LinearGrangerNetwork,
		x_mean: np.ndarray,
		y_mean: np.ndarray,
	) -> None:
		layer = network.coefficient_layer

		if layer.bias is not None:
			bias = y_mean - x_mean @ layer.weight.T

			if not np.isfinite(bias).all():
				raise ValueError("Analytic bias is non-finite.")

			layer.bias[...] = bias

	def _coefficient_step(
		self,
		data_gradient: np.ndarray,
		step_size: float,
	) -> None:
		network = self._require_model()
		layer = network.coefficient_layer
		regularizer = network.regularizer

		prox = (
			None
			if regularizer is None
			else getattr(regularizer, "prox", None)
		)

		if callable(prox):
			candidate = prox(
				layer.weight - step_size * data_gradient,
				step_size,
			)
		else:
			_, penalty_gradient = (
				network.regularization_value_and_grad()
			)
			candidate = layer.weight - step_size * (
				data_gradient + penalty_gradient
			)

		# Validation and constraint enforcement precede assignment.
		layer.weight[...] = network.constrain(candidate)


	def _get_linear_zero_mask(self) -> np.ndarray | None:
		network = self._require_model()
		expected_shape = (
			self._n_outputs,
			self._n_features,
		)

		if network.constraint is None:
			return np.ones(
				expected_shape,
				dtype=bool,
			)

		raw_mask = network.constraint.zero_only_mask()

		if raw_mask is None:
			return None

		mask = np.asarray(raw_mask)

		if mask.shape != expected_shape:
			raise ValueError(
				"zero_only_mask() must return a mask "
				f"with shape {expected_shape}."
			)

		if mask.dtype.kind != "b":
			raise TypeError(
				"zero_only_mask() must return a boolean mask "
				"or None."
			)

		return mask.copy()

	def _select_solver(
		self,
		*,
		n_samples: int,
		batch_size: int,
	) -> tuple[str, np.ndarray | None]:
		network = self._require_model()

		full_batch = batch_size >= n_samples
		no_regularizer = network.regularizer is None

		zero_mask = None
		if full_batch and no_regularizer:
			zero_mask = self._get_linear_zero_mask()

		direct_supported = (
			full_batch
			and no_regularizer
			and zero_mask is not None
		)

		if self.solver == "coefficient":
			return "coefficient", None

		if self.solver == "restricted_lstsq":
			if not direct_supported:
				raise ValueError(
					"restricted_lstsq requires a full batch, "
					"no regularizer, and either no constraint "
					"or a pure zero-mask constraint exposing "
					"linear_zero_mask."
				)

			return "restricted_lstsq", zero_mask

		if direct_supported:
			return "restricted_lstsq", zero_mask

		return "coefficient", None

	def _restricted_lstsq_target(
		self,
		X_centered: np.ndarray,
		y_centered: np.ndarray,
		zero_mask: np.ndarray,
	) -> np.ndarray:
		"""Solve the centered least-squares problem per output.

		Inputs have already passed through the feature-mask layer.
		Omitted features receive canonical zero coefficients.
		"""
		network = self._require_model()

		feature_mask = np.asarray(
			network.variable_control_layer.mask
		)

		if feature_mask.shape != (self._n_features,):
			raise ValueError(
				"Feature mask must have shape "
				f"({self._n_features},)."
			)

		active_features = feature_mask != 0

		target = np.zeros(
			(self._n_outputs, self._n_features),
			dtype=np.float64,
		)

		for output_index in range(self._n_outputs):
			active = (
				zero_mask[output_index]
				& active_features
			)

			if not np.any(active):
				continue

			solution = np.linalg.lstsq(
				X_centered[:, active],
				y_centered[:, output_index],
				rcond=None,
			)[0]

			target[output_index, active] = solution

		if not np.isfinite(target).all():
			raise ValueError(
				"Least-squares solution is non-finite."
			)

		constraint = network.constraint
		if (
			constraint is not None
			and not constraint.is_satisfied(target)
		):
			raise ValueError(
				"The least-squares solution violates the "
				"constraint. Check the linear_zero_mask contract."
			)

		return target

	def _restricted_lstsq_step(
		self,
		target: np.ndarray,
		zero_mask: np.ndarray,
		relaxation: float,
	) -> None:
		network = self._require_model()
		layer = network.coefficient_layer

		if relaxation == 0:
			return

		candidate = (
			(1.0 - relaxation) * layer.weight
			+ relaxation * target
		)

		# Hard zeros must also hold when damping a warm start.
		candidate[~zero_mask] = 0.0

		feature_mask = np.asarray(
			network.variable_control_layer.mask
		)
		candidate[:, feature_mask == 0] = 0.0

		if not np.isfinite(candidate).all():
			raise ValueError(
				"Least-squares update is non-finite."
			)

		constraint = network.constraint
		if (
			constraint is not None
			and not constraint.is_satisfied(candidate)
		):
			raise ValueError(
				"Least-squares update violates the constraint."
			)

		layer.weight[...] = candidate


	def fit(
		self,
		X: FloatArray,
		y: FloatArray,
		*,
		collect_history: bool = False,
		reset_components: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> TrainingResult:
		network = self._require_model()
		self._validate_training_config()

		if (
			not isinstance(collect_history, bool)
			or not isinstance(reset_components, bool)
		):
			raise TypeError(
				"collect_history and reset_components must be bools."
			)

		if not reset_components and (
			run_name is not None or callback_updates is not None
		):
			raise ValueError(
				"run_name and callback_updates require "
				"reset_components=True."
			)

		x_array, y_array = self._training_arrays(X, y)
		batch_size = (
			len(x_array)
			if self.batch_size is None
			else int(self.batch_size)
		)

		self._fitted = False
		self._loss_history = []
		iterations = 0

		state: dict[str, Any] = {
			"model": self,
			"stop_training": False,
			"stop_reason": None,
			"loss_history": self._loss_history,
		}

		try:
			if reset_components:
				self.reset_components(
					run_name=run_name,
					callback_updates=callback_updates,
				)

			# Compatibility key, not a prepared/native optimizer.
			state["optimizer"] = self._step_control
			self._run_callback_hook("on_train_beginning", state)

			initial_mask = np.array(
				network.variable_control_layer.mask,
				copy=True,
			)
			x_masked = network.variable_control_layer(x_array)

			if self.use_bias:
				x_mean = x_masked.mean(axis=0)
				y_mean = y_array.mean(axis=0)
			else:
				x_mean = np.zeros(self._n_features)
				y_mean = np.zeros(self._n_outputs)

			x_centered = x_masked - x_mean
			y_centered = y_array - y_mean

			
			# Solver selection and data-dependent information
			# remain local to this fit() call.
			selected_solver, zero_mask = self._select_solver(
				n_samples=len(x_array),
				batch_size=batch_size,
			)

			self.solver_used_ = selected_solver
			state["solver"] = selected_solver

			lstsq_target = None
			batches = []

			if selected_solver == "restricted_lstsq":
				assert zero_mask is not None

				lstsq_target = self._restricted_lstsq_target(
					x_centered,
					y_centered,
					zero_mask,
				)

			else:
				for start in range(
					0, len(x_array), batch_size
				):
					end = min(
						start + batch_size,
						len(x_array),
					)
					xb = x_centered[start:end]

					norm = float(
						np.linalg.norm(xb, ord=2)
					)
					curvature = (
						2.0
						* norm ** 2
						/ (
							(end - start)
							* self._n_outputs
						)
					)

					if not np.isfinite(curvature):
						raise ValueError(
							"Data curvature is non-finite."
						)

					curvature = (
						curvature
						if curvature > 0
						else 1.0
					)
					batches.append(
						(start, end, curvature)
					)

			previous_loss = None

			for epoch in range(int(self.epochs)):
				if state["stop_training"]:
					break

				state["epoch"] = epoch
				self._run_callback_hook(
					"on_epoch_beginning", state
				)

				if state["stop_training"]:
					break

				if not np.array_equal(
					network.variable_control_layer.mask,
					initial_mask,
				):
					raise ValueError(
						"Change input omission between fit() calls, "
						"not inside callbacks."
					)

				relaxation = self._real_scalar(
					self._step_control.param_groups[0]["lr"],
					"callback learning rate",
				)

				if not 0 <= relaxation <= 1:
					raise ValueError(
						"Callback learning rate must lie in [0, 1]."
					)

				if selected_solver == "restricted_lstsq":
					assert lstsq_target is not None
					assert zero_mask is not None

					self._restricted_lstsq_step(
						lstsq_target,
						zero_mask,
						relaxation,
					)

					self._update_bias(
						network,
						x_mean,
						y_mean,
					)

				else:
					for start, end, curvature in batches:
						xb = x_centered[start:end]
						yb = y_centered[start:end]

						weights = (
							network.coefficient_layer.weight
						)

						gradient = (
							2.0
							* (xb @ weights.T - yb).T
							@ xb
							/ (
								(end - start)
								* self._n_outputs
							)
						)

						step_size = (
							relaxation / curvature
						)

						if not np.isfinite(step_size):
							raise ValueError(
								"Coefficient step size "
								"is non-finite."
							)

						# A zero callback learning rate must
						# not trigger prox/constraint updates.
						if relaxation > 0:
							self._coefficient_step(
								gradient,
								step_size,
							)

						self._update_bias(
							network,
							x_mean,
							y_mean,
						)

				loss, mse, penalty = self._objective(
					x_array, y_array
				)
				self._loss_history.append(loss)
				iterations += 1

				state.update(
					epoch_loss=loss,
					loss=loss,
					mse=mse,
					regularization_loss=penalty,
				)

				self._run_callback_hook("on_epoch_end", state)

				if not np.array_equal(
					network.variable_control_layer.mask,
					initial_mask,
				):
					raise ValueError(
						"Change input omission between fit() calls, "
						"not inside callbacks."
					)

				if self.verbose:
					print(
						f"Epoch {epoch + 1}/{self.epochs}"
						f" - loss: {loss:.8f}"
					)

				if (
                    relaxation > 0
                    and previous_loss is not None
                    and abs(previous_loss - loss) <= float(self.tol)
                ):
					state["stop_training"] = True
					state["stop_reason"] = "tolerance"
					break

				previous_loss = loss

			# Exactly once. Preserve callback-restored kernel AND bias.
			self._run_callback_hook("on_train_end", state)

			final_loss, final_mse, final_penalty = self._objective(
				x_array, y_array
			)

		except Exception as exc:
			self._fitted = False
			raise TrainingError(
				f"Numpy model training failed: {exc}"
			) from exc

		self._fitted = iterations > 0

		if not collect_history:
			return TrainingResult()

		return TrainingResult(
			history={"loss": list(self._loss_history)},
			metrics={
				"loss": final_loss,
				"mse": final_mse,
				"regularization_loss": final_penalty,
			},
			n_iterations=iterations,
			stop_reason=state.get("stop_reason") or (
				"stopped_before_max_epochs"
				if iterations < int(self.epochs)
				else "max_epochs_reached"
			),
		)

	def predict(self, X: FloatArray) -> FloatArray:
		network = self._require_model()

		if not self._fitted:
			raise ModelNotFittedError(
				"Model is not fitted. Call fit() first."
			)

		array = real_array(X, "X")

		if array.ndim != 2 or array.shape[1] != self._n_features:
			raise ValueError(
				f"Expected X with shape "
				f"(n_samples, {self._n_features})."
			)

		result = network(array)

		if not np.isfinite(result).all():
			raise ValueError("Predictions are non-finite.")

		return np.array(result, dtype=np.float64, copy=True)

	def get_weights(self) -> ModelWeights:
		return self._require_model().get_predictor_weights()

	def set_weights(self, weights: ModelWeights) -> None:
		self._require_model().set_predictor_weights(weights)
		self._fitted = False

	def omit_variables(
		self,
		variable_indices: Sequence[int],
	) -> None:
		indices = list(variable_indices)

		if any(
			isinstance(index, (bool, np.bool_))
			or not isinstance(index, (int, np.integer))
			for index in indices
		):
			raise TypeError(
				"Feature indices must be integers, not booleans."
			)

		if any(
			index < 0 or index >= self._n_features
			for index in indices
		):
			raise ValueError("Feature index is out of range.")

		mask = np.ones(self._n_features, dtype=np.float64)

		if indices:
			mask[np.asarray(indices, dtype=np.intp)] = 0.0

		self._require_model().variable_control_layer.set_mask(mask)


NumpyConstrainedGrangerModel = NumpyConditionalAdapter