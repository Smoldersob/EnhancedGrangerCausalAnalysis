"""Native construction/contract tests, prepared by static source review only.

Reference: backends_construction_update @ 04631788582f240c2bedfb94c1743b8e1d4a5eab.
No tests were executed during preparation. Requires an installed Keras with a
configured execution backend. Missing keras is skipped; broken imports are not.
No fit/predict calls, training arrays, optimizer steps or dependency installs.

These are small component-integration tests, not dependency-isolated units.
Model construction MUST build its network and optimizer from ModelInputSpec.
DeviceManager.scope is intentionally not patched: device integration errors
must remain visible. No initialize() API is assumed or monkeypatched.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = "enhanced_granger_analysis"
BACKEND = f"{ROOT}.backends.keras"


@pytest.fixture(scope="module")
def api():
	if importlib.util.find_spec("keras") is None:
		pytest.skip("Native component tests require the optional keras package")
	keras = importlib.import_module("keras")

	def module(path):
		return importlib.import_module(f"{ROOT}.{path}")

	return SimpleNamespace(
		keras=keras,
		Strategy=module("backends.keras.keras_backend").KerasBackendStrategy,
		Factory=module("backends.backend_factory").BackendFactory,
		ModelInputSpec=module("core.models.input_spec").ModelInputSpec,
		errors=module("core.exceptions"),
	)


@pytest.fixture
def strategy(api):
	return api.Strategy()


@pytest.fixture
def input_spec(api):
	return api.ModelInputSpec(
		n_features=4, n_outputs=2,
		causes=("x", "z"), effects=("y1", "y2"),
		cause_slices={"x": slice(0, 2), "z": slice(2, 4)},
		feature_lags=(1, 2, 1, 2),
	)


@pytest.fixture
def regularizer_data(input_spec):
	from enhanced_granger_analysis.core.components.regularizer_spec import (
		PreparedRegularizerData, RegularizationLayout,
	)

	layout = RegularizationLayout(
		cause_slices=input_spec.cause_slices,
		output_names=input_spec.effects,
		feature_lags=np.asarray(input_spec.feature_lags, dtype=np.int64),
		base_mask=np.ones((input_spec.n_outputs, input_spec.n_features), dtype=bool),
	)
	return PreparedRegularizerData(layout=layout, params={"l1": 0.025})


@pytest.fixture
def constraint_data(input_spec):
	from enhanced_granger_analysis.core.components.constraint_spec import PreparedConstraintData

	return PreparedConstraintData(
		output_names=input_spec.effects,
		n_features=input_spec.n_features,
		mask=np.array([[True, False, True, True], [True, True, False, True]]),
		rules=(),
	)


@pytest.fixture
def components(strategy):
	return {
		"optimizer": strategy.build_optimizer({"type": "sgd", "params": {"learning_rate": 0.03125}}),
		"loss": strategy.build_loss("mse"),
		"callbacks": strategy.build_callbacks("terminate_on_nan"),
	}


@pytest.fixture
def model_class(api):
	# Explicitly load the implementation for decorator-based registration.
	# Bootstrap via the public path alone is tested separately in a subprocess.
	return importlib.import_module(f"{BACKEND}.models.conditional").KerasConditionalAdapter


class TestBackendAndFactory:
	def test_native_backend_available(self, strategy):
		assert strategy.name == "keras"
		assert strategy.is_available() is True

	@pytest.fixture
	def factory(self, api, monkeypatch):
		monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
		return api.Factory

	@pytest.mark.parametrize("alias", ["keras", "tensorflow", "tf", "KERAS", "TensorFlow"])
	def test_aliases(self, api, factory, alias):
		assert isinstance(factory.get_strategy(alias), api.Strategy)
		assert factory.is_backend_available(alias) is True

	def test_aliases_share_one_cached_strategy(self, factory):
		instance = factory.get_strategy("keras")
		assert factory.get_strategy("tf") is instance
		assert factory.get_strategy("tensorflow") is instance


class TestOptimizerConstruction:
	def test_default_is_fresh_adam(self, api, strategy):
		first, second = strategy.build_optimizer(), strategy.build_optimizer()
		assert isinstance(first, api.keras.optimizers.Adam)
		assert isinstance(second, api.keras.optimizers.Adam)
		assert first is not second

	@pytest.mark.parametrize("name, class_name", [("adam", "Adam"), ("sgd", "SGD"), ("rmsprop", "RMSprop")])
	def test_native_names(self, api, strategy, name, class_name):
		assert isinstance(strategy.build_optimizer(name), getattr(api.keras.optimizers, class_name))

	@pytest.mark.parametrize("selector", ["type", "name", "kind"])
	def test_mapping_params(self, api, strategy, selector):
		spec = {selector: "sgd", "params": {"learning_rate": 0.03125, "momentum": 0.5}}
		before = deepcopy(spec)
		result = strategy.build_optimizer(spec)
		assert isinstance(result, api.keras.optimizers.SGD)
		assert float(api.keras.ops.convert_to_numpy(result.learning_rate)) == pytest.approx(0.03125)
		assert result.get_config()["momentum"] == pytest.approx(0.5)
		assert spec == before

	def test_instance_class_factory_and_native_serialization(self, api, strategy):
		ready = api.keras.optimizers.SGD(learning_rate=0.03125)
		assert strategy.build_optimizer(ready) is ready
		assert isinstance(strategy.build_optimizer(api.keras.optimizers.SGD), api.keras.optimizers.SGD)
		assert strategy.build_optimizer(lambda: ready) is ready
		spec = api.keras.optimizers.serialize(ready)
		before = deepcopy(spec)
		reconstructed = strategy.build_optimizer(spec)
		assert isinstance(reconstructed, type(ready))
		assert reconstructed is not ready
		assert spec == before

	@pytest.mark.parametrize("kind", ["value", "wrong_class", "wrong_factory"])
	def test_invalid_inputs(self, strategy, kind):
		value = {"value": 123, "wrong_class": dict, "wrong_factory": lambda: object()}[kind]
		with pytest.raises(TypeError):
			strategy.build_optimizer(value)


class TestLossConstruction:
	@pytest.mark.parametrize("spec, class_name", [
		(None, "MeanSquaredError"), ("mse", "MeanSquaredError"),
		("mean_squared_error", "MeanSquaredError"), ("mae", "MeanAbsoluteError"), ("Huber", "Huber"),
	])
	def test_aliases(self, api, strategy, spec, class_name):
		assert isinstance(strategy.build_loss(spec), getattr(api.keras.losses, class_name))

	def test_mapping_params(self, api, strategy):
		spec = {"type": "Huber", "params": {"delta": 2.0, "name": "test_huber"}}
		before = deepcopy(spec)
		loss = strategy.build_loss(spec)
		assert isinstance(loss, api.keras.losses.Huber)
		assert loss._fn_kwargs["delta"] == pytest.approx(2.0)
		assert loss.name == "test_huber"
		assert spec == before

	def test_instance_class_and_function(self, api, strategy):
		ready = api.keras.losses.MeanSquaredError()
		assert strategy.build_loss(ready) is ready
		assert isinstance(strategy.build_loss(type(ready)), type(ready))

		def loss_function(y_true, y_pred):
			raise AssertionError("Construction must not evaluate loss")

		assert strategy.build_loss(loss_function) is loss_function

	@pytest.mark.parametrize("spec, error", [
		(123, TypeError), (dict, TypeError), ("unknown_loss", ValueError),
		({"type": "mse", "extra": True}, ValueError),
		({"type": "mse", "params": None}, TypeError),
	])
	def test_invalid_specs(self, strategy, spec, error):
		with pytest.raises(error):
			strategy.build_loss(spec)


class TestCallbackConstruction:
	@pytest.mark.parametrize("spec", [None, [], ()])
	def test_empty(self, strategy, spec):
		assert strategy.build_callbacks(spec) == []

	@pytest.mark.parametrize("alias, class_name", [
		("early_stopping", "EarlyStopping"),
		("reduce_lr", "ReduceLROnPlateau"), ("terminate_on_nan", "TerminateOnNaN"),
	])
	def test_aliases(self, api, strategy, alias, class_name):
		result = strategy.build_callbacks(alias)
		assert len(result) == 1
		assert isinstance(result[0], getattr(api.keras.callbacks, class_name))

	def test_mixed_specs_keep_order_and_instance_identity(self, api, strategy):
		ready = api.keras.callbacks.TerminateOnNaN()
		declaration = {"type": "early_stopping", "params": {"monitor": "loss", "patience": 7}}
		before = deepcopy(declaration)
		supplied = [ready, declaration, api.keras.callbacks.TerminateOnNaN]
		result = strategy.build_callbacks(supplied)
		assert result is not supplied
		assert result[0] is ready
		assert result[1].patience == 7 and result[1].monitor == "loss"
		assert isinstance(result[2], api.keras.callbacks.TerminateOnNaN)
		assert result[2] is not ready
		assert declaration == before

	def test_declarations_make_fresh_instances(self, strategy):
		assert strategy.build_callbacks("early_stopping")[0] is not strategy.build_callbacks("early_stopping")[0]

	@pytest.mark.parametrize("value, error", [("unknown_callback", ValueError), (123, TypeError), (dict, TypeError)])
	def test_invalid_specs(self, strategy, value, error):
		with pytest.raises(error):
			strategy.build_callbacks(value)


class TestRegularizerConstruction:
	def test_native_instance(self, api, strategy):
		ready = api.keras.regularizers.L1(0.1)
		assert strategy.build_regularizer(ready) is ready
		assert strategy.build_regularizer(None) is None

	@pytest.mark.parametrize("selector", ["name", "mapping", "class", "normalized"])
	def test_project_regularizer(self, strategy, regularizer_data, selector):
		from enhanced_granger_analysis.backends.keras.regularizers import L1Regularizer
		from enhanced_granger_analysis.core.components.regularizer_spec import RegularizerSpec

		spec = {
			"name": "l1", "mapping": {"type": "l1", "params": {"l1": 0.025}},
			"class": L1Regularizer, "normalized": RegularizerSpec("l1", {"l1": 0.025}),
		}[selector]
		result = strategy.build_regularizer(spec, data=regularizer_data)
		assert isinstance(result, L1Regularizer)
		assert result.l1 == pytest.approx(0.025)
		assert result.data.layout is regularizer_data.layout

	def test_prepared_params_are_authoritative(self, strategy, regularizer_data):
		result = strategy.build_regularizer({"type": "l1", "params": {"l1": 999.0}}, data=regularizer_data)
		assert result.l1 == pytest.approx(0.025)

	def test_lag_regularizer(self, strategy, regularizer_data):
		from enhanced_granger_analysis.backends.keras.regularizers import LagDependentL1Regularizer

		result = strategy.build_regularizer("lag_dependent_l1", data=regularizer_data)
		assert isinstance(result, LagDependentL1Regularizer)
		np.testing.assert_array_equal(result.lag_weight_indices, [1, 2, 1, 2])

	def test_data_required(self, strategy):
		with pytest.raises(ValueError, match="PreparedRegularizerData"):
			strategy.build_regularizer("l1")


class TestConstraintConstruction:
	def test_native_instance(self, api, strategy):
		ready = api.keras.constraints.NonNeg()
		assert strategy.build_constraint(ready) is ready
		assert strategy.build_constraint(None) is None

	@pytest.mark.parametrize("selector", ["name", "alias", "class", "mapping"])
	def test_project_constraint(self, strategy, constraint_data, selector):
		from enhanced_granger_analysis.backends.keras.constraints import MaskConstraint

		spec = {"name": "mask", "alias": "MaskConstraint", "class": MaskConstraint,
				"mapping": {"type": "mask", "params": {}}}[selector]
		before = constraint_data.mask.copy()
		result = strategy.build_constraint(spec, data=constraint_data)
		assert isinstance(result, MaskConstraint)
		np.testing.assert_array_equal(result.mask, before)
		np.testing.assert_array_equal(constraint_data.mask, before)
		assert result.mask.shape == (2, 4)
		assert not np.shares_memory(result.mask, constraint_data.mask)

	def test_algorithm_params(self, strategy, constraint_data):
		from enhanced_granger_analysis.backends.keras.constraints import MaskAndMinAbsSumConstraint

		result = strategy.build_constraint(
			{"type": "mask_and_min_abs_sum", "params": {"eps": 1e-5}}, data=constraint_data,
		)
		assert isinstance(result, MaskAndMinAbsSumConstraint)
		assert result.eps == pytest.approx(1e-5)

	def test_data_required(self, strategy):
		with pytest.raises(ValueError, match="PreparedConstraintData"):
			strategy.build_constraint("mask")


class TestModelConstruction:
	@pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
	def test_eager_network_construction_without_training(
		self, api, strategy, model_class, input_spec, components, monkeypatch, selector,
	):
		selected = {
			"default": None, "name": "conditional", "alias": "conditional_linear",
			"class": model_class, "mapping": {"type": "conditional", "params": {"epochs": 3}},
		}[selector]
		forbidden = Mock(side_effect=AssertionError("No training/prediction during construction"))
		monkeypatch.setattr(model_class, "fit", forbidden)
		monkeypatch.setattr(model_class, "predict", forbidden)
		# Do NOT mock rebuild: its execution in the constructor is required.
		model = strategy.build_model(
			selected, components=components,
			config={"input_spec": input_spec, "device": "cpu", "seed": 23},
		)
		forbidden.assert_not_called()
		assert isinstance(model, model_class)
		assert model.input_spec is input_spec
		assert model.backend == "keras"
		assert model.is_fitted is False
		assert model.device == "cpu"
		assert model.model is not None and model.model.built
		assert model.model.n_features == 4 and model.model.n_outputs == 2
		assert tuple(model.model.coefficient_layer.kernel.shape) == (4, 2)
		assert tuple(model.model.coefficient_layer.bias.shape) == (2,)
		assert model.optimizer.built is True
		assert model.optimizer is model.model.optimizer
		# The supplied optimizer is a configuration template, not shared runtime state.
		assert model.optimizer is not components["optimizer"]
		assert isinstance(model.optimizer, type(components["optimizer"]))
		assert float(api.keras.ops.convert_to_numpy(model.optimizer.learning_rate)) == pytest.approx(0.03125)
		assert model.loss is components["loss"]
		assert model.callbacks is not components["callbacks"]
		assert model.callbacks[0] is components["callbacks"][0]

	def test_model_construction_does_not_reprepare_components(
		self, strategy, model_class, input_spec, components, monkeypatch,
	):
		forbidden = Mock(side_effect=AssertionError("Already prepared component"))
		for name in ("optimizer", "loss", "callbacks", "regularizer", "constraint"):
			monkeypatch.setattr(strategy, f"build_{name}", forbidden)
		strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu"})
		forbidden.assert_not_called()

	def test_without_bias(self, strategy, model_class, input_spec, components):
		model = strategy.build_model(
			model_class, components=components,
			config={"input_spec": input_spec, "device": "cpu", "use_bias": False},
		)
		assert tuple(model.model.coefficient_layer.kernel.shape) == (4, 2)
		assert model.model.coefficient_layer.bias is None
		assert len(model.model.trainable_variables) == 1

	def test_models_do_not_share_predictor_or_runtime_optimizer(self, strategy, model_class, input_spec, components):
		config = {"input_spec": input_spec, "device": "cpu"}
		first = strategy.build_model(model_class, components=components, config=config)
		second = strategy.build_model(model_class, components=components, config=config)
		assert first is not second
		assert first.model is not second.model
		assert first.optimizer is not second.optimizer
		assert first.optimizer is not components["optimizer"]
		assert second.optimizer is not components["optimizer"]

	def test_ready_model_is_returned_without_rebuild(self, strategy, model_class, input_spec, components, monkeypatch):
		ready = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu"})
		predictor, optimizer = ready.model, ready.optimizer
		forbidden = Mock(side_effect=AssertionError("Ready instance must not be rebuilt"))
		monkeypatch.setattr(ready, "rebuild", forbidden)
		assert strategy.build_model(ready) is ready
		forbidden.assert_not_called()
		assert ready.model is predictor and ready.optimizer is optimizer

	def test_regularizer_and_constraint_are_attached_to_coefficient_layer(
		self,
		strategy,
		model_class,
		input_spec,
		components,
		regularizer_data,
		constraint_data,
	):
		supplied = dict(components)
		supplied["regularizer"] = strategy.build_regularizer(
			"l1",
			data=regularizer_data,
		)
		supplied["constraint"] = strategy.build_constraint(
			"mask",
			data=constraint_data,
		)

		model = strategy.build_model(
			model_class,
			components=supplied,
			config={
				"input_spec": input_spec,
				"device": "cpu",
			},
		)
		layer = model.model.coefficient_layer

		# The adapter retains the supplied configuration templates.
		assert model.regularizer is supplied["regularizer"]
		assert model.constraint is supplied["constraint"]

		# The coefficient layer owns independent runtime components.
		assert layer.kernel_regularizer is not supplied["regularizer"]
		assert layer.kernel_constraint is not supplied["constraint"]

		# Cloning must preserve component types and configurations.
		assert type(layer.kernel_regularizer) is type(supplied["regularizer"])
		assert type(layer.kernel_constraint) is type(supplied["constraint"])

		assert (
			layer.kernel_regularizer.get_config()
			== supplied["regularizer"].get_config()
		)
		assert (
			layer.kernel_constraint.get_config()
			== supplied["constraint"].get_config()
		)

	def test_input_spec_is_required(self, strategy, model_class, components):
		with pytest.raises(TypeError, match="input_spec"):
			strategy.build_model(model_class, components=components)

	def test_input_spec_type_is_checked(self, strategy, model_class, components):
		with pytest.raises(TypeError, match="ModelInputSpec"):
			strategy.build_model(model_class, components=components, config={"input_spec": object(), "device": "cpu"})

	@pytest.mark.parametrize("component, invalid, message", [
		("optimizer", "adam", "ready Keras optimizer"),
		("loss", "mse", "ready loss"),
		("regularizer", "l1", "Keras regularizer"),
		("constraint", "mask", "Keras constraint"),
		("callbacks", [object()], "All callbacks"),
	])
	def test_raw_components_are_not_resolved_implicitly(
		self, strategy, model_class, input_spec, components, component, invalid, message,
	):
		supplied = {**components, component: invalid}
		with pytest.raises(TypeError, match=message):
			strategy.build_model(model_class, components=supplied, config={"input_spec": input_spec, "device": "cpu"})


class TestPublicModelBootstrap:
	def test_default_model_in_fresh_interpreter(self, api):
		"""Detect missing built-in registration independently of test order.

		This intentionally does NOT import models.conditional before build_model.
		If public bootstrap is broken, the test fails instead of hiding the bug.
		No network/training is reached if resolution itself fails.
		"""
		package = importlib.import_module(ROOT)
		import_root = Path(package.__file__).resolve().parent.parent
		env = dict(os.environ)
		env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(import_root), env.get("PYTHONPATH", "")]))
		script = dedent("""
			from enhanced_granger_analysis.backends.keras.keras_backend import KerasBackendStrategy
			from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
			strategy = KerasBackendStrategy()
			spec = ModelInputSpec(
				n_features=2, n_outputs=1, causes=('x',), effects=('y',),
				cause_slices={'x': slice(0, 2)}, feature_lags=(1, 2),
			)
			model = strategy.build_model(
				components={'optimizer': strategy.build_optimizer(), 'loss': strategy.build_loss()},
				config={'input_spec': spec, 'device': 'cpu'},
			)
			assert model.backend == 'keras'
			assert model.model is not None and model.model.built
		""")
		completed = subprocess.run(
			[sys.executable, "-c", script], env=env, capture_output=True,
			text=True, check=False, timeout=120,
		)
		assert completed.returncode == 0, (
			"Default Keras model construction failed in a fresh interpreter.\n"
			f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
		)
