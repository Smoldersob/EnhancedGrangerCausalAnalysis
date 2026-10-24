"""Tests for loading and instantiating PyTorch backend objects.

Target location: tests/test_pytorch_backend.py.
Context: `backends_construction_update` branch, following the integration of the model adapter described in previous responses. This is not intended for the `version_2.3` variant.

Scope: imports, backend availability, factory, resolvers, and constructors.
Excludes `initialize()`, `fit()`, `predict()`, optimizer steps, and GPU requirements.
Import errors originating from the project code itself are NOT treated as skips. If the optional `torch` package is missing, tests requiring actual components will be skipped.

Code prepared via source code review; tests have not been executed.
"""

from __future__ import annotations
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import builtins
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = "enhanced_granger_analysis"
TORCH_BACKEND = f"{ROOT}.backends.pytorch"


@pytest.fixture(scope="module")
def backend_module():
    # Nie przechwytujemy ImportError z projektu: uszkodzony import ma być błędem.
    return importlib.import_module(f"{TORCH_BACKEND}.pytorch_backend")


@pytest.fixture(scope="module")
def api(backend_module):
    # Pomijamy tylko rzeczywisty brak torch. Nie maskujemy uszkodzonej instalacji
    # torch ani jej brakujących zależności przez szeroki except ImportError.
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        pytest.skip("Testy konstrukcji komponentów wymagają pakietu torch.")

    def module(path):
        return importlib.import_module(f"{ROOT}.{path}")

    callbacks = module("backends.pytorch.callbacks")
    regularizers = module("backends.pytorch.regularizers.implementations")
    constraints = module("backends.pytorch.constraints.implementations")
    models = module("backends.pytorch.models.registry")
    regularizer_spec = module("core.components.regularizer_spec")

    return SimpleNamespace(
        torch=torch,
        Strategy=backend_module.PyTorchBackendStrategy,
        Factory=module("backends.backend_factory").BackendFactory,
        OptimizerAdapter=module("backends.pytorch.optimizers.adapter").PyTorchOptimizerAdapter,
        EarlyStopping=callbacks.EarlyStopping,
        ReduceLearningRate=callbacks.ReduceLearningRate,
        ConvergenceCheck=callbacks.ConvergenceCheck,
        L1Regularizer=regularizers.L1Regularizer,
        LagL1Regularizer=regularizers.LagDependentL1Regularizer,
        MaskConstraint=constraints.MaskConstraint,
        MaskAndMinAbsSumConstraint=constraints.MaskAndMinAbsSumConstraint,
        PreparedRegularizerData=regularizer_spec.PreparedRegularizerData,
        RegularizationLayout=regularizer_spec.RegularizationLayout,
        PreparedConstraintData=module("core.components.constraint_spec").PreparedConstraintData,
        errors=module("core.exceptions"),
        models=models,
    )


@pytest.fixture
def backend(api):
    return api.Strategy()


@pytest.fixture
def regularizer_data(api):
    # Wyłącznie metadane konstrukcyjne, nie dane treningowe.
    layout = api.RegularizationLayout(
        cause_slices={"x": slice(0, 2), "z": slice(2, 4)},
        output_names=("y1", "y2"),
        feature_lags=np.array([1, 2, 1, 2], dtype=np.int64),
        base_mask=np.ones((2, 4), dtype=bool),
    )
    return api.PreparedRegularizerData(layout=layout, params={"l1": 0.025})


@pytest.fixture
def constraint_data(api):
    return api.PreparedConstraintData(
        output_names=("y1", "y2"),
        n_features=4,
        mask=np.array([[True, False, True, True], [True, True, False, True]]),
        rules=(),
    )


@pytest.fixture
def components(backend):
    return {
        "optimizer": backend.build_optimizer({"type": "adam", "params": {"lr": 0.002}}),
        "loss": backend.build_loss("mse"),
        "callbacks": backend.build_callbacks(None),
    }


class TestBackendLoading:
    def test_import_exposes_strategy(self, backend_module):
        assert isinstance(backend_module.PyTorchBackendStrategy, type)
        assert backend_module.PyTorchBackendStrategy.name == "pytorch"

    def test_available_with_importable_torch(self, backend):
        assert backend.is_available() is True

    def test_unavailable_when_torch_cannot_be_found(self, backend_module, monkeypatch):
        original = backend_module.importlib.util.find_spec

        def find_spec(name, *args, **kwargs):
            if name == "torch":
                return None
            return original(name, *args, **kwargs)

        monkeypatch.setattr(backend_module.importlib.util, "find_spec", find_spec)
        assert backend_module.PyTorchBackendStrategy().is_available() is False

    def test_import_failure_becomes_domain_error(self, backend_module, monkeypatch):
        errors = importlib.import_module(f"{ROOT}.core.exceptions")
        original = builtins.__import__
        error = ImportError("torch deliberately unavailable")

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise error
            return original(name, *args, **kwargs)

        strategy = backend_module.PyTorchBackendStrategy()
        # Patch obejmuje wyłącznie kontrolowane wywołanie strażnika importu.
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_without_torch)
            with pytest.raises(errors.BackendNotAvailableError) as caught:
                strategy._require_torch()
        assert caught.value.__cause__ is error

    @pytest.mark.parametrize("name", ["pytorch", "torch", " PyTorch "])
    def test_factory_resolves_aliases(self, api, monkeypatch, name):
        monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
        assert isinstance(api.Factory.get_strategy(name), api.Strategy)

    def test_factory_reuses_canonical_backend(self, api, monkeypatch):
        monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
        first = api.Factory.get_strategy("pytorch")
        assert api.Factory.get_strategy("torch") is first


class TestOptimizerConstruction:
    @pytest.mark.parametrize("name", [None, "adam", "Adam", " Adam "])
    def test_adam_is_prepared_but_not_bound(self, backend, api, name):
        adapter = backend.build_optimizer(name)
        assert isinstance(adapter, api.OptimizerAdapter)
        assert adapter.optimizer_class is api.torch.optim.Adam
        assert adapter.is_initialized is False
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = adapter.native

    def test_accepts_optimizer_class(self, backend, api):
        adapter = backend.build_optimizer(api.torch.optim.SGD)
        assert adapter.optimizer_class is api.torch.optim.SGD
        assert adapter.is_initialized is False

    def test_config_options_are_copied(self, backend, api):
        spec = {"type": "sgd", "params": {"lr": 0.03, "momentum": 0.8}}
        adapter = backend.build_optimizer(spec)
        assert adapter.optimizer_class is api.torch.optim.SGD
        assert adapter.options == {"lr": 0.03, "momentum": 0.8}
        spec["params"]["lr"] = 9.0
        options = adapter.options
        options["momentum"] = 0.0
        assert adapter.options == {"lr": 0.03, "momentum": 0.8}

    def test_each_call_returns_fresh_adapter(self, backend):
        assert backend.build_optimizer("adam") is not backend.build_optimizer("adam")

    def test_rejects_context(self, backend):
        with pytest.raises(TypeError, match="context"):
            backend.build_optimizer("adam", unsupported_context=True)

    @pytest.mark.parametrize("spec,error", [
        ("not_a_real_optimizer", ValueError),
        ({"name": "adam"}, ValueError),
        ({"params": {}}, ValueError),
        ({"type": "adam", "params": []}, TypeError),
        ({"type": "adam", "params": {"gradient_accumulation_steps": 2}}, ValueError),
        (42, TypeError),
    ])
    def test_rejects_invalid_specs(self, backend, spec, error):
        with pytest.raises(error):
            backend.build_optimizer(spec)

    def test_rejects_prepared_adapter_as_raw_spec(self, backend):
        adapter = backend.build_optimizer("adam")
        with pytest.raises(TypeError):
            backend.build_optimizer(adapter)

    def test_rejects_native_optimizer_as_raw_spec(self, backend, api):
        # Tworzymy obiekty, ale nie wykonujemy initialize/step/backward.
        parameter = api.torch.nn.Parameter(api.torch.zeros(1))
        native = api.torch.optim.Adam([parameter])
        with pytest.raises(TypeError):
            backend.build_optimizer(native)


class TestLossConstruction:
    @pytest.mark.parametrize("name,class_name", [
        (None, "MSELoss"), ("mse", "MSELoss"),
        ("mean_squared_error", "MSELoss"), ("mae", "L1Loss"),
        ("huber", "HuberLoss"), ("smooth_l1", "SmoothL1Loss"),
    ])
    def test_resolves_names(self, backend, api, name, class_name):
        assert isinstance(backend.build_loss(name), getattr(api.torch.nn, class_name))

    def test_accepts_class(self, backend, api):
        assert isinstance(backend.build_loss(api.torch.nn.MSELoss), api.torch.nn.MSELoss)

    def test_applies_constructor_options(self, backend, api):
        loss = backend.build_loss({"type": "huber", "params": {"delta": 0.25, "reduction": "sum"}})
        assert isinstance(loss, api.torch.nn.HuberLoss)
        assert loss.delta == pytest.approx(0.25)
        assert loss.reduction == "sum"

    def test_preserves_ready_instance(self, backend, api):
        loss = api.torch.nn.MSELoss(reduction="sum")
        assert backend.build_loss(loss) is loss

    def test_preserves_callable_without_invoking_it(self, backend):
        loss = Mock(side_effect=AssertionError("Loss must not run during construction"))
        assert backend.build_loss(loss) is loss
        loss.assert_not_called()

    @pytest.mark.parametrize("spec,error", [
        ("not_a_real_loss", ValueError),
        ({"name": "mse"}, ValueError),
        ({"type": "mse", "params": []}, TypeError),
        (42, TypeError),
    ])
    def test_rejects_invalid_specs(self, backend, spec, error):
        with pytest.raises(error):
            backend.build_loss(spec)


class TestCallbackConstruction:
    def test_none_returns_independent_empty_lists(self, backend):
        first = backend.build_callbacks(None)
        second = backend.build_callbacks(None)
        assert first == second == []
        assert first is not second

    @pytest.mark.parametrize("name,class_attr", [
        ("early_stopping", "EarlyStopping"),
        ("reduce_lr", "ReduceLearningRate"),
        ("convergence_check", "ConvergenceCheck"),
    ])
    def test_resolves_single_name(self, backend, api, name, class_attr):
        callbacks = backend.build_callbacks(name)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], getattr(api, class_attr))

    def test_mapping_passes_options(self, backend, api):
        callback, = backend.build_callbacks({
            "type": "early_stopping", "params": {"patience": 7, "min_delta": 0.01},
        })
        assert isinstance(callback, api.EarlyStopping)
        assert callback.patience == 7
        assert callback.min_delta == pytest.approx(0.01)

    def test_class_and_ready_instance(self, backend, api):
        instance = api.EarlyStopping(patience=3)
        assert backend.build_callbacks(instance)[0] is instance
        assert isinstance(backend.build_callbacks(api.EarlyStopping)[0], api.EarlyStopping)

    def test_sequence_preserves_order_and_omits_none(self, backend, api):
        callbacks = backend.build_callbacks((None, "reduce_lr", "early_stopping", None))
        assert [type(item) for item in callbacks] == [api.ReduceLearningRate, api.EarlyStopping]

    def test_repeated_spec_creates_independent_instances(self, backend):
        first, second = backend.build_callbacks(["early_stopping", "early_stopping"])
        assert first is not second

    @pytest.mark.parametrize("spec,error", [
        ({"name": "early_stopping"}, ValueError),
        ({"type": "early_stopping", "params": []}, TypeError),
        (42, TypeError),
    ])
    def test_rejects_invalid_specs(self, backend, spec, error):
        with pytest.raises(error):
            backend.build_callbacks(spec)

    def test_unknown_name(self, backend, api):
        with pytest.raises(api.errors.UnknownComponentError):
            backend.build_callbacks("not_a_real_callback")


class TestRegularizerAndConstraintConstruction:
    def test_none_disables_optional_components(self, backend):
        assert backend.build_regularizer(None) is None
        assert backend.build_constraint(None) is None

    @pytest.mark.parametrize("name,class_attr", [
        ("l1", "L1Regularizer"), ("lag_dependent_l1", "LagL1Regularizer"),
    ])
    def test_regularizer_uses_prepared_data(self, backend, api, regularizer_data, name, class_attr):
        regularizer = backend.build_regularizer(name, data=regularizer_data)
        assert isinstance(regularizer, getattr(api, class_attr))
        assert regularizer.l1 == pytest.approx(0.025)
        assert regularizer.data.layout.coefficient_shape == (2, 4)

    def test_prepared_regularizer_data_is_authoritative(self, backend, regularizer_data):
        regularizer = backend.build_regularizer(
            {"type": "l1", "params": {"l1": 99.0}}, data=regularizer_data,
        )
        # Strategia nie przygotowuje metadanych ponownie z surowego spec.
        assert regularizer.l1 == pytest.approx(0.025)

    def test_ready_regularizer_is_returned_unchanged(self, backend, regularizer_data):
        regularizer = backend.build_regularizer("l1", data=regularizer_data)
        assert backend.build_regularizer(regularizer) is regularizer

    def test_missing_regularizer_data(self, backend):
        with pytest.raises(ValueError, match="PreparedRegularizerData"):
            backend.build_regularizer("l1")

    @pytest.mark.parametrize("name", ["mask", "mask_constraint", "MaskConstraint"])
    def test_constraint_aliases(self, backend, api, constraint_data, name):
        constraint = backend.build_constraint(name, data=constraint_data)
        assert isinstance(constraint, api.MaskConstraint)
        np.testing.assert_array_equal(constraint.mask, constraint_data.mask)

    def test_constraint_constructor_options(self, backend, api, constraint_data):
        constraint = backend.build_constraint(
            {"type": "mask_and_min_abs_sum", "params": {"eps": 1e-5}},
            data=constraint_data,
        )
        assert isinstance(constraint, api.MaskAndMinAbsSumConstraint)
        assert constraint.eps == pytest.approx(1e-5)

    def test_ready_constraint_is_returned_unchanged(self, backend, constraint_data):
        constraint = backend.build_constraint("mask", data=constraint_data)
        assert backend.build_constraint(constraint) is constraint

    def test_missing_constraint_data(self, backend):
        with pytest.raises(ValueError, match="PreparedConstraintData"):
            backend.build_constraint("mask")

    @pytest.mark.parametrize("builder", ["build_regularizer", "build_constraint"])
    def test_unknown_component_name(self, backend, api, builder):
        with pytest.raises(api.errors.UnknownComponentError):
            getattr(backend, builder)("not_a_real_component")


class TestModelConstruction:
    @pytest.fixture(autouse=True)
    def forbid_initialization_and_training(self, api, monkeypatch):
        model_class = api.models.get_model_class("conditional_linear")
        initialize = Mock(side_effect=AssertionError("build_model must not initialize"))
        fit = Mock(side_effect=AssertionError("build_model must not train"))
        monkeypatch.setattr(model_class, "initialize", initialize)
        monkeypatch.setattr(model_class, "fit", fit)
        yield
        initialize.assert_not_called()
        fit.assert_not_called()

    @pytest.mark.parametrize("spec", [None, "conditional", "conditional_linear"])
    def test_default_and_registered_names(self, backend, api, components, spec):
        model = backend.build_model(spec, components=components, config={"device": "cpu"})
        assert isinstance(model, api.models.get_model_class("conditional_linear"))
        assert model.backend in {"pytorch", "torch"}
        assert model.is_fitted is False
        assert components["optimizer"].is_initialized is False

    def test_accepts_model_class(self, backend, api, components):
        model_class = api.models.get_model_class("conditional_linear")
        model = backend.build_model(model_class, components=components, config={"device": "cpu"})
        assert isinstance(model, model_class)

    def test_combines_spec_parameters_and_config(self, backend, components):
        model = backend.build_model(
            {"type": "conditional_linear", "params": {"epochs": 11}},
            components=components,
            config={"batch_size": 8, "use_bias": False, "device": "cpu"},
        )
        assert model.epochs == 11
        assert model.batch_size == 8
        assert model.use_bias is False
        assert model.is_fitted is False

    def test_passes_ready_components_without_rebuilding_them(
        self, backend, components, regularizer_data, constraint_data, monkeypatch,
    ):
        regularizer = backend.build_regularizer("l1", data=regularizer_data)
        constraint = backend.build_constraint("mask", data=constraint_data)
        callbacks = backend.build_callbacks("early_stopping")
        components.update(regularizer=regularizer, constraint=constraint, callbacks=callbacks)
        for name in ("optimizer", "loss", "callbacks", "regularizer", "constraint"):
            monkeypatch.setattr(
                backend, f"build_{name}",
                Mock(side_effect=AssertionError("build_model must not resolve components")),
            )
        model = backend.build_model(components=components, config={"device": "cpu"})
        assert model.loss is components["loss"]
        assert model.regularizer is regularizer
        assert model.constraint is constraint
        assert model.callbacks is not callbacks
        assert model.callbacks[0] is callbacks[0]
        for name in ("optimizer", "loss", "callbacks", "regularizer", "constraint"):
            getattr(backend, f"build_{name}").assert_not_called()

    def test_returns_ready_model_without_reconstruction(self, backend, components):
        model = backend.build_model(components=components, config={"device": "cpu"})
        assert backend.build_model(model) is model

    @pytest.mark.parametrize("override", [
        {"config": {"epochs": 2}}, {"components": {"loss": "mse"}},
    ])
    def test_ready_model_rejects_constructor_overrides(self, backend, components, override):
        model = backend.build_model(components=components, config={"device": "cpu"})
        with pytest.raises(ValueError, match="ready model"):
            backend.build_model(model, **override)

    def test_rejects_missing_optimizer(self, backend):
        with pytest.raises(TypeError):
            backend.build_model(config={"device": "cpu"})

    def test_rejects_optimizer_raw_spec(self, backend, components):
        components["optimizer"] = "adam"
        with pytest.raises(TypeError):
            backend.build_model(components=components, config={"device": "cpu"})

    def test_rejects_unknown_component_key(self, backend, components):
        components["unknown_component"] = object()
        with pytest.raises(ValueError, match="Unsupported component"):
            backend.build_model(components=components)

    def test_rejects_component_in_config(self, backend, components):
        with pytest.raises(ValueError, match="components"):
            backend.build_model(components=components, config={"loss": "mse"})

    def test_rejects_duplicate_constructor_option(self, backend, components):
        with pytest.raises(ValueError, match="more than once"):
            backend.build_model(
                {"type": "conditional_linear", "params": {"epochs": 5}},
                components=components, config={"epochs": 10},
            )

    def test_rejects_non_string_argument_key(self, backend, components):
        with pytest.raises(TypeError, match="names must be strings"):
            backend.build_model(components=components, config={1: "invalid"})

    def test_rejects_unknown_model(self, backend, api, components):
        with pytest.raises(api.errors.UnknownComponentError):
            backend.build_model("not_a_real_model", components=components)


class TestConstructorValidation:
    def test_invalid_callback_options_propagate_domain_error(self, backend, api):
        with pytest.raises(api.errors.TrainingConfigurationError):
            backend.build_callbacks({"type": "early_stopping", "params": {"patience": 0}})

    def test_invalid_regularizer_options_propagate_domain_error(self, backend, api, regularizer_data):
        invalid_data = api.PreparedRegularizerData(
            layout=regularizer_data.layout, params={"l1": -1.0},
        )
        with pytest.raises(api.errors.RegularizerConfigurationError):
            backend.build_regularizer("l1", data=invalid_data)

    def test_invalid_constraint_options_propagate_domain_error(self, backend, api, constraint_data):
        with pytest.raises(api.errors.ConstraintConfigurationError):
            backend.build_constraint(
                {"type": "mask_and_min_abs_sum", "params": {"eps": 0.0}},
                data=constraint_data,
            )

    def test_component_creation_is_silent_by_default(self, backend, capsys):
        capsys.readouterr()
        backend.build_loss("mse")
        assert capsys.readouterr().out == ""

    def test_verbose_loading_reports_component(self, api, capsys):
        backend = api.Strategy(loading_verbose=True)
        capsys.readouterr()
        backend.build_loss("mse")
        assert "[Backend:pytorch] loss" in capsys.readouterr().out
