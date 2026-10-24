"""Dependency-isolated PyTorch strategy tests; prepared, not executed.

Reference: backends_construction_update @ 04631788582f240c2bedfb94c1743b8e1d4a5eab.
Replace the old test_pytorch_backend_2_7.py rather than collecting both versions.

The actual strategy, common builder, ModelInputSpec and model registry are used.
Native component bases/resolvers are test doubles. The model registry is loaded
from its real source without running models/__init__.py (which imports torch).
Normal model-package registration is tested in test_pytorch_backend_native.py.
No training data, fit/predict calls, optimizer steps or dependency installation.
"""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

from enhanced_granger_analysis.backends.pytorch import pytorch_backend as strategy_module
from enhanced_granger_analysis.backends.pytorch.pytorch_backend import PyTorchBackendStrategy
from enhanced_granger_analysis.core.exceptions import BackendNotAvailableError, UnknownComponentError
from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
from enhanced_granger_analysis.core.models.model_registry import ModelClassRegistry

ROOT = "enhanced_granger_analysis.backends.pytorch"
COMPONENT_NAMES = ("optimizer", "loss", "callbacks", "regularizer", "constraint")


@pytest.fixture
def input_spec():
    return ModelInputSpec(
        n_features=4, n_outputs=2,
        causes=("x", "z"), effects=("y1", "y2"),
        cause_slices={"x": slice(0, 2), "z": slice(2, 4)},
        feature_lags=(1, 2, 1, 2),
    )


@pytest.fixture
def strategy(monkeypatch):
    instance = PyTorchBackendStrategy()
    monkeypatch.setattr(instance, "_require_torch", Mock(return_value=ModuleType("torch")))
    return instance


@pytest.fixture
def dependencies(monkeypatch):
    """Install controlled lazy dependencies, restoring sys.modules on teardown."""
    def install(suffix, *, package=False, **members):
        fullname = f"{ROOT}.{suffix}"
        module = ModuleType(fullname)
        if package:
            module.__path__ = []
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, fullname, module)
        parent_name, _, attr = fullname.rpartition(".")
        monkeypatch.setattr(sys.modules[parent_name], attr, module, raising=False)
        return module

    class Callback:
        pass

    class Regularizer:
        pass

    class Constraint:
        pass

    result = SimpleNamespace(
        optimizer=Mock(), loss=Mock(), callback=Mock(),
        regularizer=Mock(), constraint=Mock(),
        Callback=Callback, Regularizer=Regularizer, Constraint=Constraint,
    )
    install("optimizers", package=True, resolve_optimizer=result.optimizer)
    install("losses", package=True)
    install("losses.resolver", resolve_loss=result.loss)
    install("callbacks", package=True)
    install("callbacks.base", Callback=Callback)
    install("callbacks.registry", resolve_callback_spec=result.callback)
    install("regularizers", package=True, resolve_regularizer_spec=result.regularizer)
    install("regularizers.base", BasePyTorchRegularizer=Regularizer)
    install("constraints", package=True, resolve_constraint_spec=result.constraint)
    install("constraints.base", BasePyTorchConstraint=Constraint)
    return result


@pytest.fixture
def model_registry(monkeypatch):
    """Real registry source in an isolated package shell; no native adapter import."""
    folder = Path(strategy_module.__file__).resolve().parent / "models"
    package = ModuleType(f"{ROOT}.models")
    package.__path__ = [str(folder)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(sys.modules[ROOT], "models", package, raising=False)
    name = f"{ROOT}.models.registry"
    spec = importlib.util.spec_from_file_location(name, folder / "registry.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(package, "registry", module, raising=False)
    spec.loader.exec_module(module)
    module._MODELS = ModelClassRegistry(label="test PyTorch model")

    class RecordingAdapter:
        constructions = []
        backend = "pytorch"

        def __init__(
            self, *, input_spec, optimizer, loss, callbacks=None,
            regularizer=None, constraint=None, epochs=3, batch_size=4,
            device="cpu", use_bias=True, seed=None, gradient_accumulation_steps=1,
        ):
            self.input_spec = input_spec
            self.optimizer = optimizer
            self.loss = loss
            self.callbacks = callbacks
            self.regularizer = regularizer
            self.constraint = constraint
            self.epochs = epochs
            self.batch_size = batch_size
            self.device = device
            self.use_bias = use_bias
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.is_fitted = False
            self.rebuild_calls = []
            type(self).constructions.append(self)
            self.rebuild(seed=seed)

        def rebuild(self, *, seed=None):
            self.rebuild_calls.append(seed)
            self.is_fitted = False

        def fit(self, X, y, **kwargs):
            raise AssertionError("Strategy must not fit a model")

        def predict(self, X):
            raise AssertionError("Strategy must not predict")

        def get_weights(self):
            raise AssertionError("Strategy must not read weights")

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
        assert PyTorchBackendStrategy.name == "pytorch"

    def test_missing_distribution(self, monkeypatch):
        lookup = Mock(return_value=None)
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", lookup)
        assert PyTorchBackendStrategy().is_available() is False
        lookup.assert_called_once_with("torch")

    def test_successful_import(self, monkeypatch):
        module = ModuleType("torch")
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setitem(sys.modules, "torch", module)
        instance = PyTorchBackendStrategy()
        assert instance.is_available() is True
        assert instance._require_torch() is module

    @pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
    def test_import_failures_preserve_cause(self, monkeypatch, error_type):
        original = builtins.__import__
        cause = error_type("missing torch dependency")

        def guarded_import(name, *args, **kwargs):
            if name == "torch":
                raise cause
            return original(name, *args, **kwargs)

        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        instance = PyTorchBackendStrategy()
        assert instance.is_available() is False
        with pytest.raises(BackendNotAvailableError, match="PyTorch is unavailable") as error:
            instance._require_torch()
        assert error.value.__cause__ is cause

    def test_unexpected_errors_are_not_hidden(self, monkeypatch):
        monkeypatch.setattr(strategy_module.importlib.util, "find_spec", Mock(side_effect=RuntimeError("bad runtime")))
        with pytest.raises(RuntimeError, match="bad runtime"):
            PyTorchBackendStrategy().is_available()


class TestOptimizerAndLoss:
    @pytest.mark.parametrize("spec", [None, "adam", {"type": "sgd", "params": {"lr": 0.01}}, object()])
    def test_optimizer_is_delegated_once(self, strategy, dependencies, spec):
        ready = object()
        dependencies.optimizer.return_value = ready
        assert strategy.build_optimizer(spec) is ready
        dependencies.optimizer.assert_called_once_with(spec)
        strategy._require_torch.assert_called_once_with()

    def test_optimizer_rejects_context_before_loading(self, strategy, dependencies):
        with pytest.raises(TypeError, match="does not accept context: aaa, zzz"):
            strategy.build_optimizer("adam", zzz=2, aaa=1)
        strategy._require_torch.assert_not_called()
        dependencies.optimizer.assert_not_called()

    @pytest.mark.parametrize("spec", [None, "mae", {"type": "HuberLoss", "params": {"delta": 2.0}}])
    def test_loss_delegation_and_default_mse(self, strategy, dependencies, spec):
        loss = Mock()
        dependencies.loss.return_value = loss
        assert strategy.build_loss(spec) is loss
        dependencies.loss.assert_called_once_with("mse" if spec is None else spec)
        loss.assert_not_called()

    @pytest.mark.parametrize("invalid", [None, object(), 3, dict])
    def test_invalid_resolved_loss(self, strategy, dependencies, invalid):
        dependencies.loss.return_value = invalid
        with pytest.raises(TypeError, match="Loss resolver must return"):
            strategy.build_loss("mse")


class TestCallbacks:
    def test_none_short_circuits_dependency_loading(self, strategy):
        first, second = strategy.build_callbacks(), strategy.build_callbacks()
        assert first == second == []
        assert first is not second
        strategy._require_torch.assert_not_called()

    @pytest.mark.parametrize("kind", ["single", "tuple", "list"])
    def test_callback_constructor_receives_params(self, strategy, dependencies, kind):
        ready = dependencies.Callback()
        constructor = Mock(return_value=ready)
        dependencies.callback.return_value = (constructor, {"patience": 9})
        spec = {"single": "early_stopping", "tuple": ("early_stopping",), "list": ["early_stopping"]}[kind]
        assert strategy.build_callbacks(spec) == [ready]
        dependencies.callback.assert_called_once_with("early_stopping")
        constructor.assert_called_once_with(patience=9)

    def test_mapping_is_one_spec_not_an_iterable_of_keys(self, strategy, dependencies):
        declaration = {"type": "early_stopping", "params": {"patience": 4}}
        ready = dependencies.Callback()
        dependencies.callback.return_value = ready
        assert strategy.build_callbacks(declaration) == [ready]
        dependencies.callback.assert_called_once_with(declaration)

    def test_mixed_resolutions_preserve_order_and_identity(self, strategy, dependencies):
        first, second = dependencies.Callback(), dependencies.Callback()
        constructor = Mock(return_value=second)
        dependencies.callback.side_effect = [None, first, (constructor, {})]
        declarations = ["disabled", first, "early_stopping"]
        assert strategy.build_callbacks(declarations) == [first, second]
        assert dependencies.callback.call_args_list == [call(item) for item in declarations]
        constructor.assert_called_once_with()

    def test_ready_callback_is_not_called_as_factory(self, strategy, dependencies):
        ready = dependencies.Callback()
        dependencies.callback.return_value = ready
        assert strategy.build_callbacks(ready)[0] is ready

    def test_invalid_constructed_callback(self, strategy, dependencies):
        dependencies.callback.return_value = (Mock(return_value=object()), {})
        with pytest.raises(TypeError, match="invalid object"):
            strategy.build_callbacks("bad")

    def test_empty_sequence_returns_empty_list(self, strategy, dependencies):
        assert strategy.build_callbacks([]) == []
        dependencies.callback.assert_not_called()
        strategy._require_torch.assert_called_once_with()


class TestRegularizersAndConstraints:
    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_none_short_circuits_import(self, strategy, kind):
        assert getattr(strategy, f"build_{kind}")(None) is None
        strategy._require_torch.assert_not_called()

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_ready_component_identity(self, strategy, dependencies, kind):
        ready = getattr(dependencies, kind.capitalize())()
        assert getattr(strategy, f"build_{kind}")(ready) is ready
        getattr(dependencies, kind).assert_not_called()

    def test_regularizer_params_come_from_prepared_data(self, strategy, dependencies):
        data = object()
        ready = dependencies.Regularizer()
        constructor = Mock(return_value=ready)
        dependencies.regularizer.return_value = (constructor, {"l1": 999.0})
        assert strategy.build_regularizer("l1", data=data) is ready
        constructor.assert_called_once_with(data=data)
        dependencies.regularizer.assert_called_once_with("l1")

    def test_regularizer_resolver_can_disable_component(self, strategy, dependencies):
        dependencies.regularizer.return_value = None
        assert strategy.build_regularizer("disabled") is None

    def test_constraint_algorithm_params_are_forwarded(self, strategy, dependencies):
        data = object()
        params = {"eps": 1e-5}
        ready = dependencies.Constraint()
        constructor = Mock(return_value=ready)
        dependencies.constraint.return_value = (constructor, params)
        assert strategy.build_constraint("mask_and_min_abs_sum", data=data) is ready
        constructor.assert_called_once_with(data=data, eps=1e-5)
        assert params == {"eps": 1e-5}

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_project_component_requires_prepared_data(self, strategy, dependencies, kind):
        constructor = Mock()
        getattr(dependencies, kind).return_value = (constructor, {})
        with pytest.raises(ValueError, match="Prepared"):
            getattr(strategy, f"build_{kind}")("project_component")
        constructor.assert_not_called()

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_invalid_constructor_result(self, strategy, dependencies, kind):
        getattr(dependencies, kind).return_value = (Mock(return_value=object()), {})
        with pytest.raises(TypeError, match="invalid object"):
            getattr(strategy, f"build_{kind}")("project_component", data=object())


class TestErrorPropagationAndLogging:
    @pytest.mark.parametrize("kind", COMPONENT_NAMES)
    def test_resolver_exceptions_are_not_hidden(self, strategy, dependencies, kind):
        resolver_name = "callback" if kind == "callbacks" else kind
        getattr(dependencies, resolver_name).side_effect = ValueError("bad spec")
        kwargs = {"data": object()} if kind in {"regularizer", "constraint"} else {}
        with pytest.raises(ValueError, match="bad spec"):
            getattr(strategy, f"build_{kind}")("bad", **kwargs)

    def test_verbosity_can_be_toggled(self, strategy, dependencies, capsys):
        dependencies.optimizer.return_value = "ready"
        strategy.build_optimizer()
        assert capsys.readouterr().out == ""
        strategy.set_loading_verbose(True)
        strategy.build_optimizer()
        assert "[Backend:pytorch] optimizer -> 'ready'" in capsys.readouterr().out
        strategy.set_loading_verbose(False)
        strategy.build_optimizer()
        assert capsys.readouterr().out == ""


class TestModelConstruction:
    @pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
    def test_selector_and_constructor_contract(self, strategy, model_registry, input_spec, components, selector):
        selected = {
            "default": None, "name": "conditional", "alias": "conditional_linear",
            "class": model_registry.Adapter,
            "mapping": {"type": "conditional", "params": {"epochs": 5}},
        }[selector]
        model = strategy.build_model(selected, components=components, config={"input_spec": input_spec, "seed": 17})
        assert isinstance(model, model_registry.Adapter)
        assert model.input_spec is input_spec
        assert model.rebuild_calls == [17]
        assert model.is_fitted is False
        assert not hasattr(model, "initialize")
        assert model.epochs == (5 if selector == "mapping" else 3)
        assert len(model_registry.Adapter.constructions) == 1
        for name, ready in components.items():
            assert getattr(model, name) is ready
        strategy._require_torch.assert_called_once_with()

    def test_build_never_reprepares_components(self, strategy, model_registry, input_spec, components, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("Use prepared components"))
        for name in COMPONENT_NAMES:
            monkeypatch.setattr(strategy, f"build_{name}", forbidden)
        strategy.build_model(components=components, config={"input_spec": input_spec})
        forbidden.assert_not_called()

    def test_input_spec_can_be_in_model_params(self, strategy, model_registry, input_spec, components):
        model = strategy.build_model(
            {"type": "conditional", "params": {"input_spec": input_spec, "epochs": 7}},
            components=components,
        )
        assert model.input_spec is input_spec
        assert model.epochs == 7

    def test_input_spec_required(self, strategy, model_registry, components):
        with pytest.raises(TypeError, match="input_spec"):
            strategy.build_model(components=components)

    def test_mapping_inputs_are_not_mutated(self, strategy, model_registry, input_spec, components):
        params = {"epochs": 7}
        config = {"input_spec": input_spec, "batch_size": 8}
        before = components.copy()
        model = strategy.build_model(
            {"type": "conditional", "params": MappingProxyType(params)},
            config=MappingProxyType(config), components=MappingProxyType(components),
        )
        assert model.epochs == 7 and model.batch_size == 8
        assert params == {"epochs": 7}
        assert config == {"input_spec": input_spec, "batch_size": 8}
        assert components == before

    @pytest.mark.parametrize("backend_name", ["pytorch", "torch"])
    def test_both_backend_identifiers_accepted(self, strategy, model_registry, input_spec, components, backend_name):
        ready = model_registry.Adapter(input_spec=input_spec, **components)
        ready.backend = backend_name
        before = list(ready.rebuild_calls)
        assert strategy.build_model(ready) is ready
        assert ready.rebuild_calls == before

    @pytest.mark.parametrize("kwargs", [{}, {"components": {}}, {"config": {}}])
    def test_ready_model_no_rebuild(self, strategy, model_registry, input_spec, components, kwargs):
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

    def test_wrong_backend_ready_model(self, strategy, model_registry, input_spec, components):
        ready = model_registry.Adapter(input_spec=input_spec, **components)
        ready.backend = "keras"
        with pytest.raises(ValueError, match="Expected a PyTorch model"):
            strategy.build_model(ready)

    def test_wrong_backend_constructed_model(self, strategy, model_registry, input_spec, components):
        class WrongBackend(model_registry.Adapter):
            backend = "keras"

        with pytest.raises(ValueError, match="Expected a PyTorch model"):
            strategy.build_model(WrongBackend, components=components, config={"input_spec": input_spec})

    @pytest.mark.parametrize("name", COMPONENT_NAMES)
    @pytest.mark.parametrize("location", ["params", "config"])
    def test_components_must_use_components_argument(self, strategy, model_registry, name, location):
        spec = {"type": "conditional", "params": {name: object()}} if location == "params" else None
        config = {name: object()} if location == "config" else None
        with pytest.raises(ValueError, match="Pass ready components through 'components'"):
            strategy.build_model(spec, config=config)
        assert model_registry.Adapter.constructions == []

    def test_input_spec_is_not_a_component(self, strategy, model_registry, input_spec):
        with pytest.raises(ValueError, match="Unsupported component names: input_spec"):
            strategy.build_model(components={"input_spec": input_spec})

    def test_duplicate_input_spec(self, strategy, model_registry, input_spec):
        with pytest.raises(ValueError, match="supplied more than once: input_spec"):
            strategy.build_model(
                {"type": "conditional", "params": {"input_spec": input_spec}},
                config={"input_spec": input_spec},
            )

    @pytest.mark.parametrize("location", ["params", "config", "components"])
    def test_nonstring_argument_names(self, strategy, model_registry, location):
        spec = {"type": "conditional", "params": {1: object()}} if location == "params" else None
        kwargs = {location: {1: object()}} if location != "params" else {}
        with pytest.raises(TypeError, match="must be strings"):
            strategy.build_model(spec, **kwargs)

    @pytest.mark.parametrize("spec, error", [
        ({"type": "conditional", "extra": 1}, ValueError),
        ({"type": "conditional", "params": []}, TypeError),
        ({"params": {}}, ValueError), (object(), TypeError),
    ])
    def test_invalid_specs(self, strategy, model_registry, spec, error):
        with pytest.raises(error):
            strategy.build_model(spec)

    def test_unknown_model_name(self, strategy, model_registry):
        with pytest.raises(UnknownComponentError):
            strategy.build_model("not_registered")

    def test_constructor_errors_propagate(self, strategy, model_registry, input_spec, components):
        class FailingAdapter(model_registry.Adapter):
            def __init__(self, **kwargs):
                raise RuntimeError("constructor failed")

        with pytest.raises(RuntimeError, match="constructor failed"):
            strategy.build_model(FailingAdapter, components=components, config={"input_spec": input_spec})

    def test_unavailable_torch_stops_before_registry_resolution(self, strategy, model_registry, monkeypatch):
        monkeypatch.setattr(strategy, "_require_torch", Mock(side_effect=BackendNotAvailableError("not installed")))
        resolver = Mock(side_effect=AssertionError("Should not resolve model"))
        monkeypatch.setattr(model_registry.module, "resolve_model_spec", resolver)
        with pytest.raises(BackendNotAvailableError, match="not installed"):
            strategy.build_model()
        resolver.assert_not_called()
