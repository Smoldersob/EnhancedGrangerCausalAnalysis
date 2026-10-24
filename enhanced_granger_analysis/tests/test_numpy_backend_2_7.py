"""Isolated NumpyBackendStrategy units. Prepared, NOT executed.

Source: backends_construction_update @ 597d22b398d18275a9b62967fcf43446dd784e1b.
Native component resolvers/bases are controlled test doubles. The strategy,
common backend builder, ModelInputSpec and model-registry source are real.
Normal package imports may still load installed optional backend packages.
"""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

from enhanced_granger_analysis.backends.numpy import numpy_backend as backend_module
from enhanced_granger_analysis.backends.numpy.numpy_backend import NumpyBackendStrategy
from enhanced_granger_analysis.core.exceptions import BackendNotAvailableError, UnknownComponentError
from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
from enhanced_granger_analysis.core.models.model_registry import ModelClassRegistry

ROOT = "enhanced_granger_analysis.backends.numpy"
SUPPORTED = ("callbacks", "regularizer", "constraint")
IGNORED = ("optimizer", "loss")


@pytest.fixture
def strategy(monkeypatch):
    instance = NumpyBackendStrategy()
    monkeypatch.setattr(instance, "_require_numpy", Mock(return_value=ModuleType("numpy")))
    return instance


@pytest.fixture
def input_spec():
    return ModelInputSpec(
        n_features=4, n_outputs=2,
        causes=("x", "z"), effects=("y1", "y2"),
        cause_slices={"x": slice(0, 2), "z": slice(2, 4)},
        feature_lags=(1, 2, 1, 2),
    )


@pytest.fixture
def dependencies(monkeypatch):
    def install(suffix, *, package=False, **members):
        name = f"{ROOT}.{suffix}"
        module = ModuleType(name)
        if package:
            module.__path__ = []
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent], attr, module, raising=False)
        return module

    class Callback:
        pass

    class Regularizer:
        pass

    class Constraint:
        pass

    deps = SimpleNamespace(
        optimizer=Mock(return_value=None), loss=Mock(return_value=None),
        callback=Mock(), regularizer=Mock(), constraint=Mock(),
        Callback=Callback, Regularizer=Regularizer, Constraint=Constraint,
    )
    install("optimizers", package=True)
    install("optimizers.resolver", resolve_optimizer=deps.optimizer)
    install("losses", package=True)
    install("losses.resolver", resolve_loss=deps.loss)
    install("callbacks", package=True)
    install("callbacks.base", Callback=Callback)
    install("callbacks.registry", resolve_callback_spec=deps.callback)
    install("regularizers", package=True, resolve_regularizer_spec=deps.regularizer)
    install("regularizers.base", BaseNumpyRegularizer=Regularizer)
    install("constraints", package=True, resolve_constraint_spec=deps.constraint)
    install("constraints.base", BaseNumpyConstraint=Constraint)
    return deps


@pytest.fixture
def models(monkeypatch):
    """Load the real registry without executing models/__init__.py."""
    folder = Path(backend_module.__file__).resolve().parent / "models"
    package = ModuleType(f"{ROOT}.models")
    package.__path__ = [str(folder)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(sys.modules[ROOT], "models", package, raising=False)
    name = f"{ROOT}.models.registry"
    spec = importlib.util.spec_from_file_location(name, folder / "registry.py")
    assert spec is not None and spec.loader is not None
    registry = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, registry)
    monkeypatch.setattr(package, "registry", registry, raising=False)
    spec.loader.exec_module(registry)
    registry._MODELS = ModelClassRegistry(label="isolated numpy test model")

    class RecordingAdapter:
        backend = "numpy"
        constructions = []

        def __init__(
            self, *, input_spec, callbacks=None, regularizer=None,
            constraint=None, epochs=3, learning_rate=1.0,
            use_bias=True, batch_size=None, seed=None,
        ):
            # Deliberately NO optimizer/loss keyword or **kwargs: proves those
            # ignored None components are removed, not forwarded by the builder.
            self.input_spec = input_spec
            self.callbacks = callbacks
            self.regularizer = regularizer
            self.constraint = constraint
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.use_bias = use_bias
            self.batch_size = batch_size
            self.is_fitted = False
            self.rebuild_seeds = []
            type(self).constructions.append(self)
            self.rebuild(seed=seed)

        def rebuild(self, *, seed=None):
            self.rebuild_seeds.append(seed)
            self.is_fitted = False

        def fit(self, X, y, **kwargs):
            raise AssertionError("Strategy must not fit")

        def predict(self, X):
            raise AssertionError("Strategy must not predict")

        def get_weights(self):
            raise AssertionError("Strategy must not read weights")

        def set_weights(self, weights):
            raise AssertionError("Strategy must not assign weights")

        def omit_variables(self, variable_indices):
            raise AssertionError("Strategy must not mask inputs")

        def reset_components(self, **kwargs):
            raise AssertionError("Strategy must not reset components")

    registry.register_model("conditional", "conditional_linear")(RecordingAdapter)
    return SimpleNamespace(registry=registry, Adapter=RecordingAdapter)


@pytest.fixture
def components():
    return {"callbacks": [], "regularizer": object(), "constraint": object()}


class TestAvailability:
    def test_strategy_identifier(self):
        assert NumpyBackendStrategy.name == "numpy"

    def test_missing_distribution(self, monkeypatch):
        probe = Mock(return_value=None)
        monkeypatch.setattr(backend_module.importlib.util, "find_spec", probe)
        assert NumpyBackendStrategy().is_available() is False
        probe.assert_called_once_with("numpy")

    def test_successful_import_and_require(self, monkeypatch):
        numpy = ModuleType("numpy")
        monkeypatch.setattr(backend_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setitem(sys.modules, "numpy", numpy)
        instance = NumpyBackendStrategy()
        assert instance.is_available() is True
        assert instance._require_numpy() is numpy

    @pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
    def test_import_error_translated_with_cause(self, monkeypatch, error_type):
        original = builtins.__import__
        cause = error_type("missing numpy dependency")

        def guarded_import(name, *args, **kwargs):
            if name == "numpy":
                raise cause
            return original(name, *args, **kwargs)

        monkeypatch.setattr(backend_module.importlib.util, "find_spec", Mock(return_value=object()))
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        instance = NumpyBackendStrategy()
        assert instance.is_available() is False
        with pytest.raises(BackendNotAvailableError, match="numpy is unavailable") as error:
            instance._require_numpy()
        assert error.value.__cause__ is cause

    def test_unexpected_probe_error_not_suppressed(self, monkeypatch):
        monkeypatch.setattr(backend_module.importlib.util, "find_spec", Mock(side_effect=RuntimeError("probe failed")))
        with pytest.raises(RuntimeError, match="probe failed"):
            NumpyBackendStrategy().is_available()


class TestIgnoredOptimizerAndLoss:
    @pytest.mark.parametrize("kind", IGNORED)
    @pytest.mark.parametrize("spec", [None, "unused", {"type": "unused"}, 5, object()])
    def test_delegates_spec_but_returns_none_without_dependency_requirement(self, strategy, dependencies, kind, spec):
        assert getattr(strategy, f"build_{kind}")(spec) is None
        getattr(dependencies, kind).assert_called_once_with(spec)
        strategy._require_numpy.assert_not_called()

    @pytest.mark.parametrize("kind", IGNORED)
    @pytest.mark.parametrize("invalid", [False, 0, "", [], object()])
    def test_resolver_must_return_exactly_none(self, strategy, dependencies, kind, invalid):
        getattr(dependencies, kind).return_value = invalid
        with pytest.raises(TypeError, match="resolver must return None"):
            getattr(strategy, f"build_{kind}")("unused")

    def test_optimizer_context_rejected_before_resolution(self, strategy, dependencies):
        with pytest.raises(TypeError, match="does not accept context: aaa, zzz"):
            strategy.build_optimizer(None, zzz=1, aaa=2)
        dependencies.optimizer.assert_not_called()
        strategy._require_numpy.assert_not_called()


class TestCallbackConstruction:
    def test_none_returns_new_lists_without_dependency_loading(self, strategy):
        first, second = strategy.build_callbacks(), strategy.build_callbacks()
        assert first == second == [] and first is not second
        strategy._require_numpy.assert_not_called()

    @pytest.mark.parametrize("kind", ["single", "tuple", "list"])
    def test_single_and_sequence_declarations(self, strategy, dependencies, kind):
        ready = dependencies.Callback()
        constructor = Mock(return_value=ready)
        dependencies.callback.return_value = (constructor, {"patience": 7})
        spec = {"single": "early_stopping", "tuple": ("early_stopping",), "list": ["early_stopping"]}[kind]
        assert strategy.build_callbacks(spec) == [ready]
        dependencies.callback.assert_called_once_with("early_stopping")
        constructor.assert_called_once_with(patience=7)
        strategy._require_numpy.assert_called_once_with()

    @pytest.mark.parametrize("spec", [{"type": "early_stopping", "params": {}}, b"not-a-sequence-of-callbacks"])
    def test_mapping_and_bytes_not_split_into_keys_or_bytes(self, strategy, dependencies, spec):
        ready = dependencies.Callback()
        dependencies.callback.return_value = ready
        assert strategy.build_callbacks(spec) == [ready]
        dependencies.callback.assert_called_once_with(spec)

    def test_mixed_results_preserve_order_and_skip_none(self, strategy, dependencies):
        first, second = dependencies.Callback(), dependencies.Callback()
        constructor = Mock(return_value=second)
        dependencies.callback.side_effect = [None, first, (constructor, {})]
        declaration = [None, first, "early_stopping"]
        assert strategy.build_callbacks(declaration) == [first, second]
        assert declaration == [None, first, "early_stopping"]
        assert dependencies.callback.call_args_list == [call(value) for value in declaration]
        constructor.assert_called_once_with()

    def test_ready_callback_not_reconstructed(self, strategy, dependencies):
        ready = dependencies.Callback()
        dependencies.callback.return_value = ready
        assert strategy.build_callbacks(ready)[0] is ready

    def test_invalid_constructed_object(self, strategy, dependencies):
        dependencies.callback.return_value = (Mock(return_value=object()), {})
        with pytest.raises(TypeError, match="invalid object"):
            strategy.build_callbacks("bad")

    def test_empty_sequence_still_requires_backend(self, strategy, dependencies):
        assert strategy.build_callbacks([]) == []
        dependencies.callback.assert_not_called()
        strategy._require_numpy.assert_called_once_with()


class TestRegularizerAndConstraintConstruction:
    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_none_does_not_require_backend(self, strategy, kind):
        assert getattr(strategy, f"build_{kind}")(None) is None
        strategy._require_numpy.assert_not_called()

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_ready_instance_identity_without_prepared_data(self, strategy, dependencies, kind):
        ready = getattr(dependencies, kind.capitalize())()
        assert getattr(strategy, f"build_{kind}")(ready) is ready
        getattr(dependencies, kind).assert_not_called()
        strategy._require_numpy.assert_called_once_with()

    def test_prepared_regularizer_data_not_raw_params(self, strategy, dependencies):
        data = object()
        ready = dependencies.Regularizer()
        constructor = Mock(return_value=ready)
        dependencies.regularizer.return_value = (constructor, {"l1": 999.0})
        assert strategy.build_regularizer("l1", data=data) is ready
        dependencies.regularizer.assert_called_once_with("l1")
        constructor.assert_called_once_with(data=data)

    def test_regularizer_resolver_can_disable(self, strategy, dependencies):
        dependencies.regularizer.return_value = None
        assert strategy.build_regularizer("disabled") is None

    def test_constraint_algorithm_params_are_forwarded(self, strategy, dependencies):
        data = object()
        ready = dependencies.Constraint()
        params = {"eps": 1e-5}
        constructor = Mock(return_value=ready)
        dependencies.constraint.return_value = (constructor, params)
        assert strategy.build_constraint("mask_and_min_abs_sum", data=data) is ready
        constructor.assert_called_once_with(data=data, eps=1e-5)
        assert params == {"eps": 1e-5}

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_missing_data_rejected_before_constructor(self, strategy, dependencies, kind):
        constructor = Mock()
        getattr(dependencies, kind).return_value = (constructor, {})
        with pytest.raises(ValueError, match="Prepared"):
            getattr(strategy, f"build_{kind}")("component")
        constructor.assert_not_called()

    @pytest.mark.parametrize("kind", ["regularizer", "constraint"])
    def test_invalid_component_constructor_result(self, strategy, dependencies, kind):
        getattr(dependencies, kind).return_value = (Mock(return_value=object()), {})
        with pytest.raises(TypeError, match="invalid object"):
            getattr(strategy, f"build_{kind}")("component", data=object())


class TestErrorsAndLogging:
    @pytest.mark.parametrize("kind", (*IGNORED, *SUPPORTED))
    def test_resolver_errors_propagate(self, strategy, dependencies, kind):
        resolver = getattr(dependencies, "callback" if kind == "callbacks" else kind)
        resolver.side_effect = ValueError("invalid declaration")
        kwargs = {"data": object()} if kind in ("regularizer", "constraint") else {}
        with pytest.raises(ValueError, match="invalid declaration"):
            getattr(strategy, f"build_{kind}")("bad", **kwargs)

    @pytest.mark.parametrize("kind", SUPPORTED)
    def test_unavailable_backend_stops_before_resolution(self, strategy, dependencies, monkeypatch, kind):
        monkeypatch.setattr(strategy, "_require_numpy", Mock(side_effect=BackendNotAvailableError("unavailable")))
        kwargs = {"data": object()} if kind in ("regularizer", "constraint") else {}
        with pytest.raises(BackendNotAvailableError, match="unavailable"):
            getattr(strategy, f"build_{kind}")("component", **kwargs)
        getattr(dependencies, "callback" if kind == "callbacks" else kind).assert_not_called()

    def test_loading_verbosity_can_be_toggled(self, strategy, dependencies, capsys):
        strategy.build_optimizer()
        assert capsys.readouterr().out == ""
        strategy.set_loading_verbose(True)
        strategy.build_optimizer()
        assert "[Backend:numpy] optimizer -> None" in capsys.readouterr().out
        strategy.set_loading_verbose(False)
        strategy.build_loss()
        assert capsys.readouterr().out == ""


class TestRegisteredModelConstruction:
    @pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
    def test_selection_and_constructor_owned_rebuild(self, strategy, models, input_spec, components, selector):
        selected = {
            "default": None, "name": "conditional", "alias": "conditional_linear",
            "class": models.Adapter, "mapping": {"type": "conditional", "params": {"epochs": 7}},
        }[selector]
        model = strategy.build_model(selected, components=components, config={"input_spec": input_spec, "seed": 19})
        assert isinstance(model, models.Adapter)
        assert model.input_spec is input_spec and model.is_fitted is False
        assert model.rebuild_seeds == [19]
        assert model.epochs == (7 if selector == "mapping" else 3)
        assert len(models.Adapter.constructions) == 1
        for name, component in components.items():
            assert getattr(model, name) is component
        strategy._require_numpy.assert_called_once_with()

    @pytest.mark.parametrize("ignored", [{}, {"optimizer": None}, {"loss": None}, {"optimizer": None, "loss": None}])
    def test_none_optimizer_and_loss_removed_before_constructor(self, strategy, models, input_spec, components, ignored):
        model = strategy.build_model(components={**components, **ignored}, config={"input_spec": input_spec})
        assert isinstance(model, models.Adapter)
        assert not hasattr(model, "optimizer") and not hasattr(model, "loss")

    @pytest.mark.parametrize("name", IGNORED)
    @pytest.mark.parametrize("value", [False, 0, "", [], object()])
    def test_non_none_ignored_component_rejected_even_if_falsey(self, strategy, models, input_spec, name, value):
        with pytest.raises(ValueError, match=f"non-None '{name}' component"):
            strategy.build_model(components={name: value}, config={"input_spec": input_spec})
        assert models.Adapter.constructions == []

    @pytest.mark.parametrize("name", (*SUPPORTED, *IGNORED))
    @pytest.mark.parametrize("location", ["params", "config"])
    def test_component_names_cannot_be_in_params_or_config(self, strategy, models, name, location):
        selected = {"type": "conditional", "params": {name: None}} if location == "params" else None
        config = {name: None} if location == "config" else None
        with pytest.raises(ValueError, match="Pass ready components through 'components'"):
            strategy.build_model(selected, config=config)
        assert models.Adapter.constructions == []

    def test_no_component_repreparation(self, strategy, models, input_spec, components, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("Components already prepared"))
        for name in (*SUPPORTED, *IGNORED):
            monkeypatch.setattr(strategy, f"build_{name}", forbidden)
        strategy.build_model(components=components, config={"input_spec": input_spec})
        forbidden.assert_not_called()

    def test_read_only_mappings_are_not_mutated(self, strategy, models, input_spec, components):
        original_components = {**components, "optimizer": None, "loss": None}
        before = original_components.copy()
        params = {"epochs": 7}
        config = {"input_spec": input_spec, "batch_size": 4}
        model = strategy.build_model(
            {"type": "conditional", "params": MappingProxyType(params)},
            components=MappingProxyType(original_components), config=MappingProxyType(config),
        )
        assert original_components == before and params == {"epochs": 7}
        assert config == {"input_spec": input_spec, "batch_size": 4}
        assert model.epochs == 7 and model.batch_size == 4

    def test_input_spec_may_be_in_model_params(self, strategy, models, input_spec):
        model = strategy.build_model({"type": "conditional", "params": {"input_spec": input_spec}})
        assert model.input_spec is input_spec

    def test_input_spec_is_required(self, strategy, models):
        with pytest.raises(TypeError, match="input_spec"):
            strategy.build_model()

    def test_input_spec_is_not_a_component(self, strategy, models, input_spec):
        with pytest.raises(ValueError, match="Unsupported component names: input_spec"):
            strategy.build_model(components={"input_spec": input_spec})

    @pytest.mark.parametrize("name", ["input_spec", "epochs"])
    def test_duplicate_constructor_arguments(self, strategy, models, input_spec, name):
        value = input_spec if name == "input_spec" else 5
        with pytest.raises(ValueError, match=f"supplied more than once: {name}"):
            strategy.build_model({"type": "conditional", "params": {name: value}}, config={name: value})

    @pytest.mark.parametrize("location", ["params", "config", "components"])
    def test_nonstring_argument_names(self, strategy, models, location):
        selected = {"type": "conditional", "params": {1: object()}} if location == "params" else None
        kwargs = {location: {1: object()}} if location != "params" else {}
        with pytest.raises(TypeError, match="must be strings"):
            strategy.build_model(selected, **kwargs)

    @pytest.mark.parametrize("selected, error", [
        ({"type": "conditional", "extra": 1}, ValueError),
        ({"type": "conditional", "params": []}, TypeError),
        ({"params": {}}, ValueError), (dict, TypeError), (object(), TypeError),
    ])
    def test_invalid_model_specs(self, strategy, models, selected, error):
        with pytest.raises(error):
            strategy.build_model(selected)

    def test_unknown_model(self, strategy, models):
        with pytest.raises(UnknownComponentError):
            strategy.build_model("not_registered")

    def test_unknown_component(self, strategy, models):
        with pytest.raises(ValueError, match="Unsupported component names: device"):
            strategy.build_model(components={"device": "cpu"})

    def test_no_selected_model_guard(self, strategy, models, monkeypatch):
        monkeypatch.setattr(models.registry, "resolve_model_spec", Mock(return_value=(None, {})))
        with pytest.raises(ValueError, match="No numpy model was selected"):
            strategy.build_model()

    def test_constructor_errors_not_hidden(self, strategy, models, input_spec):
        class BrokenAdapter(models.Adapter):
            def __init__(self, **kwargs):
                raise RuntimeError("constructor failed")

        with pytest.raises(RuntimeError, match="constructor failed"):
            strategy.build_model(BrokenAdapter, config={"input_spec": input_spec})

    def test_backend_required_before_model_resolution(self, strategy, models, monkeypatch):
        monkeypatch.setattr(strategy, "_require_numpy", Mock(side_effect=BackendNotAvailableError("missing numpy")))
        resolver = Mock()
        monkeypatch.setattr(models.registry, "resolve_model_spec", resolver)
        with pytest.raises(BackendNotAvailableError, match="missing numpy"):
            strategy.build_model()
        resolver.assert_not_called()


class TestReadyModelValidation:
    @pytest.mark.parametrize("backend", ["numpy", "numpy"])
    def test_accepted_model_backend_identifiers(self, strategy, models, input_spec, backend):
        ready = models.Adapter(input_spec=input_spec)
        ready.backend = backend
        before = list(ready.rebuild_seeds)
        assert strategy.build_model(ready) is ready
        assert ready.rebuild_seeds == before and len(models.Adapter.constructions) == 1

    @pytest.mark.parametrize("kwargs", [{}, {"components": {}}, {"config": {}}])
    def test_no_overrides_means_identity_no_rebuild(self, strategy, models, input_spec, kwargs):
        ready = models.Adapter(input_spec=input_spec)
        before = list(ready.rebuild_seeds)
        assert strategy.build_model(ready, **kwargs) is ready
        assert ready.rebuild_seeds == before

    @pytest.mark.parametrize("kwargs", [
        {"config": {"epochs": 2}}, {"components": {"callbacks": []}},
        {"components": {"optimizer": None}}, {"components": {"loss": None}},
    ])
    def test_ready_model_rejects_even_nonempty_none_component_mapping(self, strategy, models, input_spec, kwargs):
        ready = models.Adapter(input_spec=input_spec)
        with pytest.raises(ValueError, match="ready model cannot be combined"):
            strategy.build_model(ready, **kwargs)

    @pytest.mark.parametrize("backend", ["scikit", "pytorch", "keras"])
    def test_wrong_model_backend_rejected(self, strategy, models, input_spec, backend):
        ready = models.Adapter(input_spec=input_spec)
        ready.backend = backend
        with pytest.raises(ValueError, match="Expected a Numpy model"):
            strategy.build_model(ready)

    def test_constructed_model_backend_also_validated(self, strategy, models, input_spec):
        class ForeignAdapter(models.Adapter):
            backend = "keras"

        with pytest.raises(ValueError, match="Expected a Numpy model"):
            strategy.build_model(ForeignAdapter, config={"input_spec": input_spec})

    def test_fitted_flag_must_be_bool(self, strategy, models, input_spec):
        ready = models.Adapter(input_spec=input_spec)
        ready.is_fitted = 1
        with pytest.raises(TypeError, match="boolean is_fitted"):
            strategy.build_model(ready)


class TestConstructorBoundaryErrors:
    @pytest.mark.parametrize("kind", SUPPORTED)
    def test_component_constructor_error_not_wrapped(self, strategy, dependencies, kind):
        constructor = Mock(side_effect=RuntimeError("component constructor failed"))
        resolver = getattr(dependencies, "callback" if kind == "callbacks" else kind)
        resolver.return_value = (constructor, {})
        kwargs = {"data": object()} if kind in ("regularizer", "constraint") else {}
        with pytest.raises(RuntimeError, match="component constructor failed"):
            getattr(strategy, f"build_{kind}")("component", **kwargs)

    def test_registered_aliases_are_available_without_constructor(self, models):
        names = models.registry.available_models(include_aliases=True)
        assert "conditional" in names and "conditional_linear" in names
        assert models.registry.get_model_class("conditional_linear") is models.Adapter
        assert models.Adapter.constructions == []

    def test_result_protocol_checked_after_constructor(self, strategy, models, input_spec):
        class MalformedAdapter(models.Adapter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.is_fitted = "false"

        with pytest.raises(TypeError, match="boolean is_fitted"):
            strategy.build_model(MalformedAdapter, config={"input_spec": input_spec})
