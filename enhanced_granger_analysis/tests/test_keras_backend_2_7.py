"""Isolated Keras strategy tests, prepared by static source review only.

Reference: backends_construction_update @ 04631788582f240c2bedfb94c1743b8e1d4a5eab.
No test was executed during preparation. No optional Keras install is required.

The strategy, common model builder, ModelInputSpec and model registry are real.
Component resolvers and the native Keras API are controlled test doubles.
The local adapter implements the NEW rebuild/fit(X, y) protocol; it has no
initialize method. It is not the production KerasConditionalAdapter.
"""
from __future__ import annotations

import builtins
import importlib
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from enhanced_granger_analysis.backends.keras import keras_backend as strategy_module
from enhanced_granger_analysis.backends.keras.keras_backend import KerasBackendStrategy
from enhanced_granger_analysis.core.exceptions import BackendNotAvailableError
from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
from enhanced_granger_analysis.core.models.model_registry import ModelClassRegistry

ROOT = "enhanced_granger_analysis.backends.keras"
COMPONENT_NAMES = ("optimizer", "loss", "callbacks", "regularizer", "constraint")


@pytest.fixture
def input_spec():
    return ModelInputSpec(
        n_features=4,
        n_outputs=2,
        causes=("x", "z"),
        effects=("y1", "y2"),
        cause_slices={"x": slice(0, 2), "z": slice(2, 4)},
        feature_lags=(1, 2, 1, 2),
    )


@pytest.fixture
def fake_keras():
    class Loss:
        def __call__(self, y_true, y_pred):
            raise AssertionError("Construction must not evaluate loss")

    class MeanSquaredError(Loss):
        pass

    class Regularizer:
        pass

    class Constraint:
        pass

    module = ModuleType("keras")
    module.losses = SimpleNamespace(Loss=Loss, MeanSquaredError=MeanSquaredError)
    module.regularizers = SimpleNamespace(Regularizer=Regularizer)
    module.constraints = SimpleNamespace(Constraint=Constraint)
    return module


@pytest.fixture
def strategy(monkeypatch, fake_keras):
    instance = KerasBackendStrategy()
    monkeypatch.setattr(instance, "_require_keras", Mock(return_value=fake_keras))
    return instance


@pytest.fixture
def resolvers(monkeypatch):
    """Replace lazy component dependencies, never production source files."""
    result = SimpleNamespace()
    for package, function in (
        ("optimizers", "resolve_optimizer"),
        ("losses", "resolve_loss"),
        ("callbacks", "resolve_callbacks"),
        ("regularizers", "resolve_regularizer_spec"),
        ("constraints", "resolve_constraint_spec"),
    ):
        name = f"{ROOT}.{package}"
        parent = importlib.import_module(ROOT)
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(parent, package, module, raising=False)
        if package in {"optimizers", "losses", "callbacks"}:
            resolver_name = f"{name}.resolver"
            resolver_module = ModuleType(resolver_name)
            monkeypatch.setitem(sys.modules, resolver_name, resolver_module)
            monkeypatch.setattr(module, "resolver", resolver_module, raising=False)
        else:
            resolver_module = module
        resolver = Mock()
        setattr(resolver_module, function, resolver)
        setattr(result, package, resolver)
    return result


@pytest.fixture
def model_registry(monkeypatch):
    """Use the REAL registry with a per-test registry instance and test adapter."""
    module = importlib.import_module(f"{ROOT}.models.registry")
    monkeypatch.setattr(module, "_MODELS", ModelClassRegistry(label="test Keras model"))

    class RecordingAdapter:
        constructions = []
        backend = "keras"

        def __init__(
            self, *, input_spec, optimizer, loss, callbacks=None,
            regularizer=None, constraint=None, epochs=3, batch_size=4,
            device="cpu", use_bias=True, seed=None,
        ):
            self._input_spec = input_spec
            self.optimizer = optimizer
            self.loss = loss
            self.callbacks = callbacks
            self.regularizer = regularizer
            self.constraint = constraint
            self.epochs = epochs
            self.batch_size = batch_size
            self.device = device
            self.use_bias = use_bias
            self.is_fitted = False
            self.rebuild_calls = []
            self.fit_calls = []
            type(self).constructions.append(self)
            self.rebuild(seed=seed)

        @property
        def input_spec(self):
            return self._input_spec

        def rebuild(self, *, seed=None):
            self.rebuild_calls.append(seed)
            self.is_fitted = False

        def fit(self, X, y, **kwargs):
            self.fit_calls.append((X, y, kwargs))
            raise AssertionError("Strategy must not call fit")

        def predict(self, X):
            raise AssertionError("Strategy must not call predict")

        def get_weights(self):
            return []

        def set_weights(self, weights):
            raise AssertionError("Strategy must not assign weights")

        def omit_variables(self, variable_indices):
            raise AssertionError("Strategy must not mask inputs")

        def reset_components(self, **kwargs):
            raise AssertionError("Strategy must not reset prepared components")

    module.register_model("conditional", "conditional_linear")(RecordingAdapter)
    return SimpleNamespace(module=module, Adapter=RecordingAdapter)


@pytest.fixture
def components():
    return {name: object() for name in COMPONENT_NAMES}


class TestAvailability:
    def test_name(self):
        assert KerasBackendStrategy.name == "keras"

    def test_missing_distribution(self, monkeypatch):
        lookup = Mock(return_value=None)
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", lookup)
        assert KerasBackendStrategy().is_available() is False
        lookup.assert_called_once_with("keras")

    def test_successful_import(self, monkeypatch, fake_keras):
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setitem(sys.modules, "keras", fake_keras)
        assert KerasBackendStrategy().is_available() is True
        assert KerasBackendStrategy()._require_keras() is fake_keras

    @pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
    def test_import_failure(self, monkeypatch, error_type):
        original = builtins.__import__
        cause = error_type("configured backend unavailable")

        def guarded_import(name, *args, **kwargs):
            if name == "keras":
                raise cause
            return original(name, *args, **kwargs)

        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        instance = KerasBackendStrategy()
        assert instance.is_available() is False
        with pytest.raises(BackendNotAvailableError) as error:
            instance._require_keras()
        assert error.value.__cause__ is cause

    def test_unexpected_errors_are_not_hidden(self, monkeypatch):
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(side_effect=RuntimeError("broken config")))
        with pytest.raises(RuntimeError, match="broken config"):
            KerasBackendStrategy().is_available()


class TestComponents:
    @pytest.mark.parametrize("method, expected", [
        ("build_callbacks", []), ("build_regularizer", None), ("build_constraint", None),
    ])
    def test_none_short_circuits_dependency_loading(self, strategy, method, expected):
        assert getattr(strategy, method)(None) == expected
        strategy._require_keras.assert_not_called()

    def test_disabled_callback_lists_are_independent(self, strategy):
        first, second = strategy.build_callbacks(), strategy.build_callbacks()
        assert first == second == []
        assert first is not second

    @pytest.mark.parametrize("spec", [None, "adam", {"type": "sgd", "params": {"learning_rate": 0.01}}, object()])
    def test_optimizer_delegation(self, strategy, resolvers, spec):
        ready = object()
        resolvers.optimizers.return_value = ready
        assert strategy.build_optimizer(spec) is ready
        resolvers.optimizers.assert_called_once_with(spec)
        strategy._require_keras.assert_called_once_with()

    def test_optimizer_context_rejected_before_loading(self, strategy):
        with pytest.raises(TypeError, match="does not accept context: aaa, zzz"):
            strategy.build_optimizer("adam", zzz=1, aaa=2)
        strategy._require_keras.assert_not_called()

    def test_default_loss_is_fresh_mse(self, strategy, fake_keras, resolvers):
        first, second = strategy.build_loss(), strategy.build_loss()
        assert isinstance(first, fake_keras.losses.MeanSquaredError)
        assert isinstance(second, fake_keras.losses.MeanSquaredError)
        assert first is not second
        resolvers.losses.assert_not_called()

    def test_callable_loss_is_returned_without_evaluation(self, strategy, resolvers):
        loss = Mock(side_effect=AssertionError("not a factory"))
        resolvers.losses.return_value = loss
        assert strategy.build_loss(loss) is loss
        loss.assert_not_called()
        resolvers.losses.assert_called_once_with(loss)

    @pytest.mark.parametrize("invalid", [None, 3, object(), dict])
    def test_invalid_resolved_loss(self, strategy, resolvers, invalid):
        resolvers.losses.return_value = invalid
        with pytest.raises(TypeError, match="Loss resolver must return"):
            strategy.build_loss("mse")

    @pytest.mark.parametrize("kind", ["none", "list", "tuple", "iterator"])
    def test_callbacks_are_normalized_to_fresh_list(self, strategy, resolvers, kind):
        callbacks = [object(), object()]
        resolved = {"none": None, "list": callbacks, "tuple": tuple(callbacks), "iterator": iter(callbacks)}[kind]
        resolvers.callbacks.return_value = resolved
        result = strategy.build_callbacks("early_stopping")
        assert isinstance(result, list)
        assert result == ([] if kind == "none" else callbacks)
        assert result is not callbacks
        resolvers.callbacks.assert_called_once_with("early_stopping")

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_native_ready_component_keeps_identity(self, strategy, resolvers, fake_keras, kind):
        base = getattr(getattr(fake_keras, f"{kind}s"), kind.capitalize())
        ready = base()
        assert getattr(strategy, f"build_{kind}")(ready) is ready
        getattr(resolvers, f"{kind}s").assert_not_called()

    def test_regularizer_uses_prepared_data_not_raw_params(self, strategy, resolvers, fake_keras):
        data = object()
        ready = fake_keras.regularizers.Regularizer()
        constructor = Mock(return_value=ready)
        resolvers.regularizers.return_value = (constructor, {"l1": 999.0})
        assert strategy.build_regularizer("l1", data=data) is ready
        constructor.assert_called_once_with(data=data)
        resolvers.regularizers.assert_called_once_with("l1")

    def test_regularizer_resolver_may_disable_component(self, strategy, resolvers):
        resolvers.regularizers.return_value = None
        assert strategy.build_regularizer("disabled") is None

    def test_constraint_receives_data_and_algorithm_params(self, strategy, resolvers, fake_keras):
        data = object()
        ready = fake_keras.constraints.Constraint()
        constructor = Mock(return_value=ready)
        params = {"eps": 1e-5}
        resolvers.constraints.return_value = (constructor, params)
        assert strategy.build_constraint("mask_and_min_abs_sum", data=data) is ready
        constructor.assert_called_once_with(data=data, eps=1e-5)
        assert params == {"eps": 1e-5}

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_project_component_requires_prepared_data(self, strategy, resolvers, kind):
        constructor = Mock()
        getattr(resolvers, f"{kind}s").return_value = (constructor, {})
        with pytest.raises(ValueError, match="Prepared"):
            getattr(strategy, f"build_{kind}")("project_component")
        constructor.assert_not_called()

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_invalid_constructed_component(self, strategy, resolvers, kind):
        getattr(resolvers, f"{kind}s").return_value = (Mock(return_value=object()), {})
        with pytest.raises(TypeError, match="invalid object"):
            getattr(strategy, f"build_{kind}")("project_component", data=object())

    @pytest.mark.parametrize("kind", COMPONENT_NAMES)
    def test_resolver_exceptions_propagate(self, strategy, resolvers, kind):
        getattr(resolvers, {"optimizer": "optimizers", "loss": "losses", "callbacks": "callbacks", "regularizer": "regularizers", "constraint": "constraints"}[kind]).side_effect = ValueError("invalid spec")
        kwargs = {"data": object()} if kind in {"regularizer", "constraint"} else {}
        with pytest.raises(ValueError, match="invalid spec"):
            getattr(strategy, f"build_{kind}")("bad", **kwargs)


class TestModelConstruction:
    @pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
    def test_selection_and_eager_constructor_rebuild(self, strategy, model_registry, input_spec, components, selector):
        selected = {
            "default": None, "name": "conditional", "alias": "conditional_linear",
            "class": model_registry.Adapter, "mapping": {"type": "conditional", "params": {"epochs": 5}},
        }[selector]
        model = strategy.build_model(selected, components=components, config={"input_spec": input_spec, "seed": 17})
        assert isinstance(model, model_registry.Adapter)
        assert model.input_spec is input_spec
        assert model.rebuild_calls == [17]
        assert model.fit_calls == []
        assert model.is_fitted is False
        assert len(model_registry.Adapter.constructions) == 1
        assert not hasattr(model, "initialize")
        assert model.epochs == (5 if selector == "mapping" else 3)
        for name, ready in components.items():
            assert getattr(model, name) is ready

    def test_build_does_not_resolve_components(self, strategy, model_registry, input_spec, components, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("Components must already be prepared"))
        for name in COMPONENT_NAMES:
            monkeypatch.setattr(strategy, f"build_{name}", forbidden)
        strategy.build_model(components=components, config={"input_spec": input_spec})
        forbidden.assert_not_called()
        strategy._require_keras.assert_not_called()

    def test_input_spec_can_be_supplied_through_model_params(self, strategy, model_registry, input_spec, components):
        model = strategy.build_model(
            {"type": "conditional", "params": {"input_spec": input_spec, "epochs": 7}},
            components=components,
        )
        assert model.input_spec is input_spec
        assert model.epochs == 7

    def test_missing_input_spec_is_a_constructor_error(self, strategy, model_registry, components):
        with pytest.raises(TypeError, match="input_spec"):
            strategy.build_model(components=components)

    def test_inputs_and_readonly_mappings_are_preserved(self, strategy, model_registry, input_spec, components):
        params = {"epochs": 7}
        config = {"input_spec": input_spec, "batch_size": 8}
        spec = {"type": "conditional", "params": params}
        before = components.copy()
        model = strategy.build_model(spec, components=MappingProxyType(components), config=MappingProxyType(config))
        assert model.epochs == 7 and model.batch_size == 8
        assert params == {"epochs": 7}
        assert config == {"input_spec": input_spec, "batch_size": 8}
        assert components == before

    @pytest.mark.parametrize("kwargs", [{}, {"components": {}}, {"config": {}}])
    def test_ready_model_is_not_rebuilt(self, strategy, model_registry, input_spec, components, kwargs):
        ready = model_registry.Adapter(input_spec=input_spec, **components)
        before = list(ready.rebuild_calls)
        assert strategy.build_model(ready, **kwargs) is ready
        assert ready.rebuild_calls == before
        assert len(model_registry.Adapter.constructions) == 1

    @pytest.mark.parametrize("kwargs", [{"components": {"optimizer": object()}}, {"config": {"epochs": 2}}])
    def test_ready_model_rejects_overrides(self, strategy, model_registry, input_spec, components, kwargs):
        ready = model_registry.Adapter(input_spec=input_spec, **components)
        with pytest.raises(ValueError, match="ready model cannot be combined"):
            strategy.build_model(ready, **kwargs)

    def test_wrong_backend_instance_rejected(self, strategy, model_registry, input_spec, components):
        ready = model_registry.Adapter(input_spec=input_spec, **components)
        ready.backend = "pytorch"
        with pytest.raises(ValueError, match="Expected a Keras model"):
            strategy.build_model(ready)

    def test_wrong_backend_constructor_result_rejected(self, strategy, model_registry, input_spec, components):
        class WrongBackend(model_registry.Adapter):
            backend = "pytorch"

        with pytest.raises(ValueError, match="Expected a Keras model"):
            strategy.build_model(WrongBackend, components=components, config={"input_spec": input_spec})

    @pytest.mark.parametrize("name", COMPONENT_NAMES)
    @pytest.mark.parametrize("source", ["params", "config"])
    def test_ready_components_must_use_components_argument(self, strategy, model_registry, name, source):
        spec = {"type": "conditional", "params": {name: object()}} if source == "params" else None
        config = {name: object()} if source == "config" else None
        with pytest.raises(ValueError, match="Pass ready components through 'components'"):
            strategy.build_model(spec, config=config)
        assert model_registry.Adapter.constructions == []

    def test_input_spec_is_not_a_component(self, strategy, model_registry, input_spec):
        with pytest.raises(ValueError, match="Unsupported component names: input_spec"):
            strategy.build_model(components={"input_spec": input_spec})

    def test_duplicate_input_spec_rejected(self, strategy, model_registry, input_spec):
        with pytest.raises(ValueError, match="supplied more than once: input_spec"):
            strategy.build_model(
                {"type": "conditional", "params": {"input_spec": input_spec}},
                config={"input_spec": input_spec},
            )

    @pytest.mark.parametrize("location", ["params", "config", "components"])
    def test_nonstring_names_rejected(self, strategy, model_registry, location):
        spec = {"type": "conditional", "params": {1: object()}} if location == "params" else None
        kwargs = {location: {1: object()}} if location != "params" else {}
        with pytest.raises(TypeError, match="must be strings"):
            strategy.build_model(spec, **kwargs)

    @pytest.mark.parametrize("spec, error", [
        ({"type": "conditional", "extra": 1}, ValueError),
        ({"type": "conditional", "params": []}, TypeError),
        ({"params": {}}, ValueError),
        (object(), TypeError),
    ])
    def test_invalid_model_specs(self, strategy, model_registry, spec, error):
        with pytest.raises(error):
            strategy.build_model(spec)

    def test_unknown_name_rejected(self, strategy, model_registry):
        from enhanced_granger_analysis.core.exceptions import UnknownComponentError

        with pytest.raises(UnknownComponentError):
            strategy.build_model("not_registered")


class TestLogging:
    def test_verbosity_can_be_toggled(self, strategy, resolvers, capsys):
        resolvers.optimizers.return_value = "ready"
        strategy.build_optimizer()
        assert capsys.readouterr().out == ""
        strategy.set_loading_verbose(True)
        strategy.build_optimizer()
        assert "[Backend:keras] optimizer -> 'ready'" in capsys.readouterr().out
        strategy.set_loading_verbose(False)
        strategy.build_optimizer()
        assert capsys.readouterr().out == ""
