"""Native PyTorch construction tests, prepared by source review, NOT executed.

Reference: backends_construction_update @ 04631788582f240c2bedfb94c1743b8e1d4a5eab.
Component integration tests, not fully dependency-isolated units. Only a missing
optional torch distribution is skipped; project and native import errors fail.

No fit, predict, forward, backward, optimizer step or training arrays. Native
optimizer initialize() IS required during construction; the MODEL uses rebuild()
and has no initialize() lifecycle. Tests request CPU, never GPU training.
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
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = "enhanced_granger_analysis"
BACKEND = f"{ROOT}.backends.pytorch"


@pytest.fixture(scope="module")
def api():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Native component tests require the optional torch package")
    torch = importlib.import_module("torch")

    def module(path):
        return importlib.import_module(f"{ROOT}.{path}")

    return SimpleNamespace(
        torch=torch,
        Strategy=module("backends.pytorch.pytorch_backend").PyTorchBackendStrategy,
        Factory=module("backends.backend_factory").BackendFactory,
        OptimizerAdapter=module("backends.pytorch.optimizers.adapter").PyTorchOptimizerAdapter,
        ModelInputSpec=module("core.models.input_spec").ModelInputSpec,
        callbacks=module("backends.pytorch.callbacks"),
        errors=module("core.exceptions"),
    )


@pytest.fixture(autouse=True)
def preserve_cpu_rng(api):
    state = api.torch.random.get_rng_state()
    yield
    api.torch.random.set_rng_state(state)


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

    return PreparedRegularizerData(
        layout=RegularizationLayout(
            cause_slices=input_spec.cause_slices,
            output_names=input_spec.effects,
            feature_lags=np.asarray(input_spec.feature_lags, dtype=np.int64),
            base_mask=np.ones((input_spec.n_outputs, input_spec.n_features), dtype=bool),
        ),
        params={"l1": 0.025},
    )


@pytest.fixture
def constraint_data(input_spec):
    from enhanced_granger_analysis.core.components.constraint_spec import PreparedConstraintData

    return PreparedConstraintData(
        output_names=input_spec.effects, n_features=input_spec.n_features,
        mask=np.array([[True, False, True, True], [True, True, False, True]]), rules=(),
    )


@pytest.fixture
def components(strategy):
    return {
        "optimizer": strategy.build_optimizer({"type": "sgd", "params": {"lr": 0.03125, "momentum": 0.5}}),
        "loss": strategy.build_loss("mse"),
        "callbacks": strategy.build_callbacks("early_stopping"),
    }


@pytest.fixture
def model_class(api):
    return importlib.import_module(f"{BACKEND}.models.conditional").PyTorchConditionalAdapter


class TestBackendAndFactory:
    def test_available(self, strategy):
        assert strategy.name == "pytorch"
        assert strategy.is_available() is True

    @pytest.fixture
    def factory(self, api, monkeypatch):
        monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
        return api.Factory

    @pytest.mark.parametrize("alias", ["pytorch", "torch", "PyTorch", "TORCH"])
    def test_aliases(self, api, factory, alias):
        assert isinstance(factory.get_strategy(alias), api.Strategy)
        assert factory.is_backend_available(alias) is True

    def test_aliases_share_cache(self, factory):
        assert factory.get_strategy("torch") is factory.get_strategy("pytorch")

    def test_factory_updates_loading_verbosity(self, factory):
        first = factory.get_strategy("torch", loading_verbose=True)
        assert first._loading_verbose is True
        assert factory.get_strategy("pytorch", loading_verbose=False) is first
        assert first._loading_verbose is False


class TestOptimizerPreparation:
    def test_default_is_new_unbound_adam_adapter(self, api, strategy):
        first, second = strategy.build_optimizer(), strategy.build_optimizer()
        assert isinstance(first, api.OptimizerAdapter)
        assert first.optimizer_class is api.torch.optim.Adam
        assert first.options == {}
        assert first is not second
        assert first.is_initialized is False
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = first.native

    @pytest.mark.parametrize("name, class_name", [("adam", "Adam"), (" SGD ", "SGD"), ("rmsprop", "RMSprop"), ("LBFGS", "LBFGS")])
    def test_native_names(self, api, strategy, name, class_name):
        adapter = strategy.build_optimizer(name)
        assert adapter.optimizer_class is getattr(api.torch.optim, class_name)
        assert adapter.is_initialized is False

    def test_class_and_mapping_selectors(self, api, strategy):
        assert strategy.build_optimizer(api.torch.optim.SGD).optimizer_class is api.torch.optim.SGD
        declaration = {"type": api.torch.optim.SGD, "params": {"lr": 0.03125, "momentum": 0.5}}
        adapter = strategy.build_optimizer(declaration)
        assert adapter.optimizer_class is api.torch.optim.SGD
        assert adapter.options == {"lr": 0.03125, "momentum": 0.5}
        assert declaration == {"type": api.torch.optim.SGD, "params": {"lr": 0.03125, "momentum": 0.5}}

    def test_options_are_defensively_copied(self, strategy):
        declaration = {"type": "adam", "params": {"lr": 0.01, "betas": [0.8, 0.95]}}
        before = deepcopy(declaration)
        adapter = strategy.build_optimizer(MappingProxyType(declaration))
        assert declaration == before
        declaration["params"]["betas"][0] = 0.1
        exposed = adapter.options
        exposed["betas"][1] = 0.2
        assert adapter.options == {"lr": 0.01, "betas": [0.8, 0.95]}

    def test_native_instance_adapter_instance_and_factory_rejected(self, api, strategy):
        layer = api.torch.nn.Linear(4, 2, device="cpu")
        native = api.torch.optim.SGD(layer.parameters(), lr=0.01)
        adapter = strategy.build_optimizer("sgd")
        factory = Mock(return_value=adapter)
        for value in (native, adapter, factory):
            with pytest.raises(TypeError):
                strategy.build_optimizer(value)
        factory.assert_not_called()

    @pytest.mark.parametrize("spec, error", [
        (123, TypeError), (dict, TypeError), ("unknown_optimizer", ValueError), (" ", ValueError),
        ({"params": {}}, ValueError), ({"type": None}, ValueError),
        ({"type": "adam", "extra": 1}, ValueError),
        ({"type": "adam", "params": None}, TypeError),
        ({"type": "adam", "params": {1: 3}}, TypeError),
    ])
    def test_invalid_specs(self, strategy, spec, error):
        with pytest.raises(error):
            strategy.build_optimizer(spec)

    @pytest.mark.parametrize("option", ["params", "closure", "gradient_accumulation_steps", "gradient_accumulation", "accumulation_steps"])
    def test_training_context_is_not_optimizer_configuration(self, strategy, option):
        with pytest.raises(ValueError, match="not supported optimizer constructor options"):
            strategy.build_optimizer({"type": "adam", "params": {option: 2}})

    def test_constructor_context_rejected(self, strategy):
        with pytest.raises(TypeError, match="does not accept context"):
            strategy.build_optimizer("sgd", parameters=[])


class TestOptimizerAdapterConstruction:
    def test_binds_exactly_supplied_model_parameters_without_step(self, api, strategy):
        layer = api.torch.nn.Linear(4, 2, device="cpu")
        parameters = list(layer.parameters())
        adapter = strategy.build_optimizer({"type": "sgd", "params": {"lr": 0.03125}})
        native = adapter.initialize(iter(parameters))
        assert adapter.is_initialized is True and adapter.native is native
        assert isinstance(native, api.torch.optim.SGD)
        bound = [p for group in native.param_groups for p in group["params"]]
        assert [id(p) for p in bound] == [id(p) for p in parameters]
        assert native.param_groups[0]["lr"] == pytest.approx(0.03125)
        assert len(native.state) == 0
        assert all(p.grad is None for p in parameters)

    @pytest.mark.parametrize("kind, error", [("empty", ValueError), ("duplicate", ValueError), ("not_tensor", TypeError)])
    def test_invalid_bindings_do_not_replace_existing_optimizer(self, api, strategy, kind, error):
        layer = api.torch.nn.Linear(4, 2, device="cpu")
        parameters = list(layer.parameters())
        adapter = strategy.build_optimizer("sgd")
        original = adapter.initialize(parameters)
        invalid = {"empty": [], "duplicate": [parameters[0], parameters[0]], "not_tensor": [object()]}[kind]
        with pytest.raises(error):
            adapter.initialize(invalid)
        assert adapter.native is original

    def test_rebinding_creates_new_native_optimizer(self, api, strategy):
        first = api.torch.nn.Linear(4, 2, device="cpu")
        second = api.torch.nn.Linear(4, 2, device="cpu")
        adapter = strategy.build_optimizer("adam")
        old = adapter.initialize(first.parameters())
        new = adapter.initialize(second.parameters())
        assert old is not new and adapter.native is new
        bound = [p for group in new.param_groups for p in group["params"]]
        assert [id(p) for p in bound] == [id(p) for p in second.parameters()]

    def test_base_optimizer_class_rejected(self, api, strategy):
        with pytest.raises(TypeError, match="concrete"):
            strategy.build_optimizer(api.torch.optim.Optimizer)


class TestLossConstruction:
    @pytest.mark.parametrize("spec, class_name", [
        (None, "MSELoss"), ("mse", "MSELoss"), ("mean_squared_error", "MSELoss"),
        ("mae", "L1Loss"), ("HuberLoss", "HuberLoss"), ("smooth_l1", "SmoothL1Loss"),
        ("bce_with_logits", "BCEWithLogitsLoss"), ("cross_entropy", "CrossEntropyLoss"),
    ])
    def test_aliases(self, api, strategy, spec, class_name):
        assert isinstance(strategy.build_loss(spec), getattr(api.torch.nn, class_name))

    def test_mappings_and_fresh_defaults(self, api, strategy):
        declaration = {"type": "huber", "params": {"delta": 2.0, "reduction": "sum"}}
        before = deepcopy(declaration)
        loss = strategy.build_loss(declaration)
        assert isinstance(loss, api.torch.nn.HuberLoss)
        assert loss.delta == pytest.approx(2.0) and loss.reduction == "sum"
        assert declaration == before
        assert strategy.build_loss() is not strategy.build_loss()

    def test_instance_class_function_and_callable_identity(self, api, strategy):
        ready = api.torch.nn.MSELoss()
        assert strategy.build_loss(ready) is ready
        assert isinstance(strategy.build_loss(type(ready)), type(ready))

        def loss_function(prediction, target):
            raise AssertionError("Construction must not evaluate loss")

        callable_object = Mock(side_effect=AssertionError("Not a factory"))
        assert strategy.build_loss(loss_function) is loss_function
        assert strategy.build_loss(callable_object) is callable_object
        callable_object.assert_not_called()

    @pytest.mark.parametrize("spec, error", [
        (123, TypeError), (dict, TypeError), ("unknown_loss", ValueError),
        ({"type": "mse", "extra": 1}, ValueError),
        ({"type": "mse", "params": None}, TypeError),
        ({"type": "mse", "params": {1: 1}}, TypeError),
        ({"params": {}}, ValueError),
    ])
    def test_invalid_specs(self, strategy, spec, error):
        with pytest.raises(error):
            strategy.build_loss(spec)

    def test_base_module_class_rejected(self, api, strategy):
        with pytest.raises(TypeError, match="concrete subclass"):
            strategy.build_loss(api.torch.nn.Module)


class TestCallbackConstruction:
    @pytest.mark.parametrize("spec", [None, [], (), [None]])
    def test_empty(self, strategy, spec):
        assert strategy.build_callbacks(spec) == []

    @pytest.mark.parametrize("alias, class_name", [
        ("early_stopping", "EarlyStopping"), ("early_stoping", "EarlyStopping"),
        ("reduce_lr", "ReduceLearningRate"), ("reduce_lr_on_plateau", "ReduceLearningRate"),
        ("convergence", "ConvergenceCheck"),
    ])
    def test_names_and_aliases(self, api, strategy, alias, class_name):
        result = strategy.build_callbacks(alias)
        assert len(result) == 1
        assert isinstance(result[0], getattr(api.callbacks, class_name))

    def test_mixed_specs_preserve_order_params_and_identity(self, api, strategy):
        ready = api.callbacks.ConvergenceCheck()
        declaration = {"type": "early_stopping", "params": {"patience": 7, "monitor": "epoch_loss"}}
        before = deepcopy(declaration)
        original = [None, ready, declaration, api.callbacks.ReduceLearningRate]
        result = strategy.build_callbacks(original)
        assert result is not original and len(result) == 3
        assert result[0] is ready
        assert isinstance(result[1], api.callbacks.EarlyStopping)
        assert result[1].patience == 7 and result[1].monitor == "epoch_loss"
        assert isinstance(result[2], api.callbacks.ReduceLearningRate)
        assert declaration == before and original[0] is None

    def test_declarations_make_fresh_callbacks(self, strategy):
        assert strategy.build_callbacks("early_stopping")[0] is not strategy.build_callbacks("early_stopping")[0]

    @pytest.mark.parametrize("spec, error", [
        (123, TypeError), (dict, TypeError), ("unknown_callback", "unknown"),
        ({"type": "early_stopping", "extra": 1}, ValueError),
        ({"type": "early_stopping", "params": None}, TypeError),
        ({"type": "early_stopping", "params": {1: 1}}, TypeError),
    ])
    def test_invalid_specs(self, api, strategy, spec, error):
        expected = api.errors.UnknownComponentError if error == "unknown" else error
        with pytest.raises(expected):
            strategy.build_callbacks(spec)


class TestRegularizersAndConstraints:
    @pytest.mark.parametrize("selector", ["name", "mapping", "class", "normalized"])
    def test_project_regularizer(self, strategy, regularizer_data, selector):
        from enhanced_granger_analysis.backends.pytorch.regularizers import L1Regularizer
        from enhanced_granger_analysis.core.components.regularizer_spec import RegularizerSpec

        spec = {
            "name": "l1", "mapping": {"type": "l1", "params": {"l1": 0.025}},
            "class": L1Regularizer, "normalized": RegularizerSpec("l1", {"l1": 0.025}),
        }[selector]
        result = strategy.build_regularizer(spec, data=regularizer_data)
        assert isinstance(result, L1Regularizer)
        assert result.l1 == pytest.approx(0.025)
        assert result.data.layout is regularizer_data.layout
        assert strategy.build_regularizer(result) is result

    def test_prepared_regularizer_params_take_precedence(self, strategy, regularizer_data):
        result = strategy.build_regularizer({"type": "l1", "params": {"l1": 999.0}}, data=regularizer_data)
        assert result.l1 == pytest.approx(0.025)

    def test_lag_regularizer_metadata_and_buffers(self, api, strategy, regularizer_data):
        from enhanced_granger_analysis.backends.pytorch.regularizers import LagDependentL1Regularizer

        result = strategy.build_regularizer("lag_dependent_l1", data=regularizer_data)
        assert isinstance(result, LagDependentL1Regularizer)
        assert isinstance(result, api.torch.nn.Module)
        assert result.lag_weight_indices.tolist() == [1, 2, 1, 2]
        assert result.feature_weights.tolist() == [2.0, 3.0, 2.0, 3.0]
        assert result._base_mask.dtype is api.torch.bool

    @pytest.mark.parametrize("selector", ["name", "alias", "class", "mapping"])
    def test_project_constraint(self, strategy, constraint_data, selector):
        from enhanced_granger_analysis.backends.pytorch.constraints import MaskConstraint

        spec = {
            "name": "mask", "alias": "MaskConstraint", "class": MaskConstraint,
            "mapping": {"type": "mask", "params": {}},
        }[selector]
        before = constraint_data.mask.copy()
        result = strategy.build_constraint(spec, data=constraint_data)
        assert isinstance(result, MaskConstraint)
        assert result.mask.shape == (2, 4)
        np.testing.assert_array_equal(result.mask, before)
        np.testing.assert_array_equal(constraint_data.mask, before)
        assert not np.shares_memory(result.mask, constraint_data.mask)
        assert strategy.build_constraint(result) is result

    def test_constraint_algorithm_params(self, strategy, constraint_data):
        from enhanced_granger_analysis.backends.pytorch.constraints import MaskAndMinAbsSumConstraint

        result = strategy.build_constraint(
            {"type": "mask_and_min_abs_sum", "params": {"eps": 1e-5}}, data=constraint_data,
        )
        assert isinstance(result, MaskAndMinAbsSumConstraint)
        assert result.eps == pytest.approx(1e-5)

    @pytest.mark.parametrize("kind, spec, message", [
        ("regularizer", "l1", "PreparedRegularizerData"),
        ("constraint", "mask", "PreparedConstraintData"),
    ])
    def test_none_and_missing_prepared_data(self, strategy, kind, spec, message):
        builder = getattr(strategy, f"build_{kind}")
        assert builder(None) is None
        with pytest.raises(ValueError, match=message):
            builder(spec)


class TestNativeModelConstruction:
    @pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
    def test_eager_build_without_training(self, api, strategy, model_class, input_spec, components, monkeypatch, selector):
        selected = {
            "default": None, "name": "conditional", "alias": "conditional_linear",
            "class": model_class, "mapping": {"type": "conditional", "params": {"epochs": 3}},
        }[selector]
        forbidden = Mock(side_effect=AssertionError("No training or prediction during construction"))
        monkeypatch.setattr(model_class, "fit", forbidden)
        monkeypatch.setattr(model_class, "predict", forbidden)
        network_class = importlib.import_module(f"{BACKEND}.models.network_parts").LinearGrangerNetwork
        monkeypatch.setattr(network_class, "forward", forbidden)
        model = strategy.build_model(selected, components=components, config={"input_spec": input_spec, "device": "cpu", "seed": 23})
        forbidden.assert_not_called()
        assert isinstance(model, model_class)
        assert model.backend == "pytorch" and model.input_spec is input_spec
        assert model.is_fitted is False and model.device.type == "cpu"
        assert isinstance(model.model, api.torch.nn.Module)
        assert model.model.n_features == 4 and model.model.n_outputs == 2
        assert tuple(model.model.coefficient_layer.weight.shape) == (2, 4)
        assert tuple(model.model.coefficient_layer.bias.shape) == (2,)
        assert all(p.device.type == "cpu" and p.dtype is api.torch.float32 for p in model.model.parameters())
        assert isinstance(model.optimizer, api.torch.optim.SGD)
        assert model.optimizer.param_groups[0]["lr"] == pytest.approx(0.03125)
        assert model.optimizer.param_groups[0]["momentum"] == pytest.approx(0.5)
        bound = [p for group in model.optimizer.param_groups for p in group["params"]]
        assert [id(p) for p in bound] == [id(p) for p in model.model.coefficient_layer.parameters()]
        assert len(model.optimizer.state) == 0
        assert all(p.grad is None for p in bound)
        assert components["optimizer"].is_initialized is False
        assert model.loss is components["loss"]
        assert model._runtime_loss is not components["loss"]
        assert isinstance(model._runtime_loss, type(components["loss"]))
        assert model.callbacks is not components["callbacks"]
        assert model.callbacks[0] is components["callbacks"][0]

    def test_default_model_loss_is_mse(self, api, strategy, model_class, input_spec, components):
        supplied = {"optimizer": components["optimizer"]}
        model = strategy.build_model(model_class, components=supplied, config={"input_spec": input_spec, "device": "cpu"})
        assert isinstance(model.loss, api.torch.nn.MSELoss)
        assert isinstance(model._runtime_loss, api.torch.nn.MSELoss)

    def test_prepared_components_are_not_resolved_again(self, strategy, model_class, input_spec, components, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("Already prepared component"))
        for name in ("optimizer", "loss", "callbacks", "regularizer", "constraint"):
            monkeypatch.setattr(strategy, f"build_{name}", forbidden)
        strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu"})
        forbidden.assert_not_called()

    def test_without_bias(self, strategy, model_class, input_spec, components):
        model = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu", "use_bias": False})
        assert model.model.coefficient_layer.bias is None
        assert len(list(model.model.parameters())) == 1
        assert len(model.optimizer.param_groups[0]["params"]) == 1

    def test_shared_templates_do_not_share_runtime_state(self, strategy, model_class, input_spec, components):
        config = {"input_spec": input_spec, "device": "cpu"}
        first = strategy.build_model(model_class, components=components, config=config)
        second = strategy.build_model(model_class, components=components, config=config)
        assert first is not second and first.model is not second.model
        assert first.optimizer is not second.optimizer
        assert first._runtime_loss is not second._runtime_loss
        first_ids = {id(p) for group in first.optimizer.param_groups for p in group["params"]}
        second_ids = {id(p) for group in second.optimizer.param_groups for p in group["params"]}
        assert first_ids.isdisjoint(second_ids)
        assert components["optimizer"].is_initialized is False

    def test_initialized_supplied_optimizer_is_not_rebound(self, api, strategy, model_class, input_spec, components):
        external_layer = api.torch.nn.Linear(4, 2, device="cpu")
        template = components["optimizer"]
        external_native = template.initialize(external_layer.parameters())
        model = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu"})
        assert template.native is external_native
        assert model.optimizer is not external_native
        assert [id(p) for p in external_native.param_groups[0]["params"]] == [id(p) for p in external_layer.parameters()]

    def test_ready_model_does_not_rebuild(self, strategy, model_class, input_spec, components, monkeypatch):
        ready = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu"})
        predictor, optimizer = ready.model, ready.optimizer
        forbidden = Mock(side_effect=AssertionError("Ready model must not be rebuilt"))
        monkeypatch.setattr(ready, "rebuild", forbidden)
        assert strategy.build_model(ready) is ready
        forbidden.assert_not_called()
        assert ready.model is predictor and ready.optimizer is optimizer

    def test_regularizer_constraint_templates_are_cloned_into_network(
        self, strategy, model_class, input_spec, components, regularizer_data, constraint_data,
    ):
        supplied = dict(components)
        supplied["regularizer"] = strategy.build_regularizer("l1", data=regularizer_data)
        supplied["constraint"] = strategy.build_constraint("mask", data=constraint_data)
        model = strategy.build_model(model_class, components=supplied, config={"input_spec": input_spec, "device": "cpu"})
        assert model.regularizer is supplied["regularizer"]
        assert model.constraint is supplied["constraint"]
        assert model.model.regularizer is not supplied["regularizer"]
        assert isinstance(model.model.regularizer, type(supplied["regularizer"]))
        assert model.model.regularizer.l1 == pytest.approx(supplied["regularizer"].l1)
        assert model.model.constraint is not supplied["constraint"]
        np.testing.assert_array_equal(model.model.constraint.mask, supplied["constraint"].mask)
        assert not np.shares_memory(model.model.constraint.mask, supplied["constraint"].mask)
        assert all(buffer.device.type == "cpu" for buffer in model.model.regularizer.buffers())

    def test_function_loss_keeps_identity(self, strategy, model_class, input_spec, components):
        def loss_function(prediction, target):
            raise AssertionError("Do not evaluate construction-time loss")

        model = strategy.build_model(
            model_class, components={**components, "loss": loss_function},
            config={"input_spec": input_spec, "device": "cpu"},
        )
        assert model.loss is loss_function and model._runtime_loss is loss_function

    def test_input_spec_required_and_validated(self, strategy, model_class, components):
        with pytest.raises(TypeError, match="input_spec"):
            strategy.build_model(model_class, components=components)
        with pytest.raises(TypeError, match="ModelInputSpec"):
            strategy.build_model(model_class, components=components, config={"input_spec": object(), "device": "cpu"})

    @pytest.mark.parametrize("component, invalid, message", [
        ("optimizer", "adam", "prepared PyTorchOptimizerAdapter"),
        ("loss", "mse", "ready loss"),
        ("regularizer", "l1", "BasePyTorchRegularizer"),
        ("constraint", "mask", "BasePyTorchConstraint"),
        ("callbacks", [object()], "All callbacks"),
    ])
    def test_component_specs_are_not_resolved_implicitly(self, strategy, model_class, input_spec, components, component, invalid, message):
        with pytest.raises(TypeError, match=message):
            strategy.build_model(model_class, components={**components, component: invalid}, config={"input_spec": input_spec, "device": "cpu"})

    @pytest.mark.parametrize("config, error, message", [
        ({"epochs": 0}, ValueError, "epochs"),
        ({"epochs": True}, ValueError, "epochs"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"verbose": True}, ValueError, "verbose"),
        ({"verbose": 3}, ValueError, "verbose"),
        ({"use_bias": 1}, TypeError, "use_bias"),
        ({"gradient_accumulation_steps": 0}, ValueError, "gradient_accumulation_steps"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": True}, ValueError, "seed"),
    ])
    def test_constructor_configuration_validation(self, strategy, model_class, input_spec, components, config, error, message):
        with pytest.raises(error, match=message):
            strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu", **config})

    def test_mean_loss_required_for_gradient_accumulation(self, strategy, model_class, input_spec, components):
        supplied = {**components, "loss": strategy.build_loss({"type": "mse", "params": {"reduction": "sum"}})}
        with pytest.raises(ValueError, match="mean-reduced loss"):
            strategy.build_model(model_class, components=supplied, config={"input_spec": input_spec, "device": "cpu", "gradient_accumulation_steps": 2})

    def test_lbfgs_rejects_accumulation(self, strategy, model_class, input_spec, components):
        supplied = {**components, "optimizer": strategy.build_optimizer("lbfgs")}
        with pytest.raises(ValueError, match="LBFGS requires"):
            strategy.build_model(model_class, components=supplied, config={"input_spec": input_spec, "device": "cpu", "gradient_accumulation_steps": 2})

    def test_valid_accumulation_is_model_configuration(self, strategy, model_class, input_spec, components):
        model = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cpu", "batch_size": None, "gradient_accumulation_steps": 2})
        assert model.gradient_accumulation_steps == 2 and model.batch_size is None
        assert "gradient_accumulation_steps" not in model.optimizer.defaults


class TestDeviceConstruction:
    @pytest.mark.parametrize("alias", ["cpu", "cpu-only", "cpu:0", "/cpu:0", " CPU "])
    def test_cpu_aliases(self, strategy, model_class, input_spec, components, alias):
        model = strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": alias})
        assert model.device.type == "cpu"
        assert all(parameter.device.type == "cpu" for parameter in model.model.parameters())

    def test_auto_falls_back_to_cpu_when_cuda_unavailable(self, api, monkeypatch):
        manager = importlib.import_module(f"{BACKEND}.device").DeviceManager
        monkeypatch.setattr(api.torch.cuda, "is_available", Mock(return_value=False))
        assert manager(requested_device="auto").name == "cpu"

    def test_explicit_cuda_does_not_fall_back(self, api, strategy, model_class, input_spec, components, monkeypatch):
        monkeypatch.setattr(api.torch.cuda, "is_available", Mock(return_value=False))
        with pytest.raises(api.errors.BackendNotAvailableError, match="CUDA was requested"):
            strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "cuda:0"})

    def test_invalid_device_does_not_allocate_model(self, strategy, model_class, input_spec, components):
        with pytest.raises(ValueError, match="device must be"):
            strategy.build_model(model_class, components=components, config={"input_spec": input_spec, "device": "gpu:bad"})


class TestPublicModelBootstrap:
    def test_default_model_in_fresh_interpreter(self, api):
        """No manual import of models.conditional; catches test-order dependence."""
        package = importlib.import_module(ROOT)
        import_root = Path(package.__file__).resolve().parent.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(import_root), env.get("PYTHONPATH", "")]))
        script = dedent("""
            from enhanced_granger_analysis.backends.pytorch.pytorch_backend import PyTorchBackendStrategy
            from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
            strategy = PyTorchBackendStrategy()
            spec = ModelInputSpec(
                n_features=2, n_outputs=1, causes=('x',), effects=('y',),
                cause_slices={'x': slice(0, 2)}, feature_lags=(1, 2),
            )
            model = strategy.build_model(
                components={'optimizer': strategy.build_optimizer(), 'loss': strategy.build_loss()},
                config={'input_spec': spec, 'device': 'cpu'},
            )
            assert model.backend == 'pytorch'
            assert model.input_spec is spec
            assert model.model is not None
            assert not model.is_fitted
        """)
        completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False, timeout=120)
        assert completed.returncode == 0, (
            "Public PyTorch model construction failed in a fresh interpreter.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


class TestAdditionalConstructorGuards:
    def test_get_strategy_accepts_surrounding_whitespace(self, api, monkeypatch):
        monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
        assert isinstance(api.Factory.get_strategy(" TORCH "), api.Strategy)

    def test_invalid_optimizer_constructor_options_are_deferred_until_binding(self, api, strategy):
        adapter = strategy.build_optimizer({"type": "sgd", "params": {"not_a_native_option": 1}})
        assert adapter.is_initialized is False
        layer = api.torch.nn.Linear(4, 2, device="cpu")
        with pytest.raises(TypeError, match="not_a_native_option"):
            adapter.initialize(layer.parameters())
        assert adapter.is_initialized is False
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = adapter.native

    def test_constraint_prepared_data_fields_cannot_be_redeclared(self, api, strategy, constraint_data):
        with pytest.raises(api.errors.ConstraintConfigurationError, match="reserved for prepared data"):
            strategy.build_constraint(
                {"type": "mask", "params": {"mask": [[True]]}}, data=constraint_data,
            )

    def test_callback_constructor_error_is_preserved(self, api, strategy):
        with pytest.raises(api.errors.TrainingConfigurationError, match="patience"):
            strategy.build_callbacks({"type": "early_stopping", "params": {"patience": 0}})
