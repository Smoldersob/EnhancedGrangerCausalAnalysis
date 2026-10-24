"""Optional real-component/construction checks; prepared, NOT executed.

Source: backends_construction_update @ 597d22b398d18275a9b62967fcf43446dd784e1b.
These are integration tests complementing test_numpy_backend.py units.
No fit/predict, network evaluation, solver updates or training observations.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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
BACKEND = f"{ROOT}.backends.numpy"


@pytest.fixture(scope="module")
def api():
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("Optional real-component checks require numpy")
    # Broken installed distributions and project import errors must fail, not skip.
    importlib.import_module("numpy")

    def module(name):
        return importlib.import_module(f"{ROOT}.{name}")

    return SimpleNamespace(
        Strategy=module("backends.numpy.numpy_backend").NumpyBackendStrategy,
        Factory=module("backends.backend_factory").BackendFactory,
        ModelInputSpec=module("core.models.input_spec").ModelInputSpec,
        callbacks=module("backends.numpy.callbacks"),
        regularizers=module("backends.numpy.regularizers"),
        constraints=module("backends.numpy.constraints"),
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
    from enhanced_granger_analysis.core.components.regularizer_spec import PreparedRegularizerData, RegularizationLayout

    return PreparedRegularizerData(
        layout=RegularizationLayout(
            cause_slices=input_spec.cause_slices,
            output_names=input_spec.effects,
            feature_lags=np.asarray(input_spec.feature_lags, dtype=np.int64),
            base_mask=np.ones((2, 4), dtype=bool),
        ),
        params={"l1": 0.025},
    )


@pytest.fixture
def constraint_data(input_spec):
    from enhanced_granger_analysis.core.components.constraint_spec import PreparedConstraintData

    return PreparedConstraintData(
        output_names=input_spec.effects, n_features=4,
        mask=np.array([[True, False, True, True], [True, True, False, True]]),
        rules=(),
    )


@pytest.fixture
def components(strategy):
    return {
        "optimizer": strategy.build_optimizer(),
        "loss": strategy.build_loss(),
        "callbacks": strategy.build_callbacks("early_stopping"),
    }


@pytest.fixture
def model_class(api):
    return importlib.import_module(f"{BACKEND}.models.conditional").NumpyConditionalAdapter


class TestBackendFactory:
    @pytest.fixture
    def factory(self, api, monkeypatch):
        monkeypatch.setattr(api.Factory, "_STRATEGY_CACHE", {})
        return api.Factory

    @pytest.mark.parametrize("alias", ["numpy", "scikit", " Numpy", "SKLEARN"])
    def test_aliases(self, api, strategy, factory, alias):
        assert strategy.name == "numpy" and strategy.is_available() is True
        assert isinstance(factory.get_strategy(alias), api.Strategy)
        assert factory.is_backend_available(alias) is True

    def test_alias_cache_and_verbosity(self, factory):
        first = factory.get_strategy("numpy", loading_verbose=True)
        assert first._loading_verbose is True
        assert factory.get_strategy("numpy", loading_verbose=False) is first
        assert factory.get_strategy("numpy") is first
        assert first._loading_verbose is False

    def test_get_strategy_trims_whitespace(self, api, factory):
        assert isinstance(factory.get_strategy(" SCIKIT "), api.Strategy)


class TestIgnoredComponentResolvers:
    @pytest.mark.parametrize("kind", ["optimizer", "loss"])
    @pytest.mark.parametrize("spec", [None, "adam", "mse", 42, False, {"type": "anything"}, object()])
    def test_current_resolvers_ignore_all_specs_and_return_none(self, strategy, kind, spec):
        # Characterization of CURRENT permissive no-op resolvers, not a future
        # requirement to support actual custom losses or optimizer algorithms.
        assert getattr(strategy, f"build_{kind}")(spec) is None

    def test_callable_specs_are_not_invoked(self, strategy):
        factory = Mock(side_effect=AssertionError("Numpy has no component factory here"))
        assert strategy.build_optimizer(factory) is None
        assert strategy.build_loss(factory) is None
        factory.assert_not_called()

    def test_optimizer_context_is_not_silently_ignored(self, strategy):
        with pytest.raises(TypeError, match="does not accept context"):
            strategy.build_optimizer(None, parameters=[])


class TestRealCallbacks:
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
        assert len(result) == 1 and isinstance(result[0], getattr(api.callbacks, class_name))

    def test_mixed_specs_preserve_order_identity_and_parameters(self, api, strategy):
        ready = api.callbacks.ConvergenceCheck()
        declaration = {"type": "early_stopping", "params": {"patience": 7, "monitor": "epoch_loss"}}
        before = deepcopy(declaration)
        source = [None, ready, declaration, api.callbacks.ReduceLearningRate]
        result = strategy.build_callbacks(source)
        assert result is not source and len(result) == 3
        assert result[0] is ready
        assert isinstance(result[1], api.callbacks.EarlyStopping)
        assert result[1].patience == 7 and result[1].monitor == "epoch_loss"
        assert isinstance(result[2], api.callbacks.ReduceLearningRate)
        assert declaration == before and source[0] is None

    def test_declarations_make_new_instances(self, strategy):
        assert strategy.build_callbacks("early_stopping")[0] is not strategy.build_callbacks("early_stopping")[0]

    @pytest.mark.parametrize("spec, error", [
        (123, TypeError), (dict, TypeError), (b"early_stopping", TypeError),
        ({"type": "early_stopping", "extra": 1}, ValueError),
        ({"type": "early_stopping", "params": None}, TypeError),
        ({"type": "early_stopping", "params": {1: 1}}, TypeError),
    ])
    def test_invalid_specs(self, strategy, spec, error):
        with pytest.raises(error):
            strategy.build_callbacks(spec)

    def test_unknown_name(self, api, strategy):
        with pytest.raises(api.errors.UnknownComponentError):
            strategy.build_callbacks("not_registered")

    def test_callback_constructor_errors_propagate(self, api, strategy):
        with pytest.raises(api.errors.TrainingConfigurationError, match="patience"):
            strategy.build_callbacks({"type": "early_stopping", "params": {"patience": 0}})


class TestRealRegularizersAndConstraints:
    @pytest.mark.parametrize("selector", ["name", "class", "mapping", "normalized"])
    def test_regularizer_forms(self, api, strategy, regularizer_data, selector):
        from enhanced_granger_analysis.core.components.regularizer_spec import RegularizerSpec

        spec = {
            "name": "l1", "class": api.regularizers.L1Regularizer,
            "mapping": {"type": "l1", "params": {"l1": 0.025}},
            "normalized": RegularizerSpec("l1", {"l1": 0.025}),
        }[selector]
        result = strategy.build_regularizer(spec, data=regularizer_data)
        assert isinstance(result, api.regularizers.L1Regularizer)
        assert result.l1 == pytest.approx(0.025)
        assert result.data.layout is regularizer_data.layout
        assert strategy.build_regularizer(result) is result

    def test_prepared_data_not_raw_spec_params_is_effective(self, strategy, regularizer_data):
        result = strategy.build_regularizer({"type": "l1", "params": {"l1": 999.0}}, data=regularizer_data)
        assert result.l1 == pytest.approx(regularizer_data.params["l1"])

    def test_lag_weights_come_from_prepared_metadata(self, api, strategy, regularizer_data):
        prepared = replace(regularizer_data, params={"l1": 0.025, "lag_weights": [1.0, 2.0, 4.0]})
        result = strategy.build_regularizer("lag_dependent_l1", data=prepared)
        assert isinstance(result, api.regularizers.LagDependentL1Regularizer)
        np.testing.assert_array_equal(result.lag_weight_indices, [1, 2, 1, 2])
        np.testing.assert_array_equal(result.feature_weights, [2.0, 4.0, 2.0, 4.0])
        exposed = result.feature_weights
        exposed[:] = 99
        np.testing.assert_array_equal(result.feature_weights, [2.0, 4.0, 2.0, 4.0])

    @pytest.mark.parametrize("selector", ["name", "alias", "class", "mapping"])
    def test_constraint_forms_and_mask_ownership(self, api, strategy, constraint_data, selector):
        spec = {
            "name": "mask", "alias": "MaskConstraint", "class": api.constraints.MaskConstraint,
            "mapping": {"type": "mask", "params": {}},
        }[selector]
        before = constraint_data.mask.copy()
        result = strategy.build_constraint(spec, data=constraint_data)
        assert isinstance(result, api.constraints.MaskConstraint)
        assert result.mask.shape == (2, 4)
        np.testing.assert_array_equal(result.mask, before)
        assert not np.shares_memory(result.mask, constraint_data.mask)
        result.mask[0, 0] = 0
        np.testing.assert_array_equal(constraint_data.mask, before)
        assert strategy.build_constraint(result) is result

    def test_constraint_algorithm_parameters(self, api, strategy, constraint_data):
        result = strategy.build_constraint(
            {"type": "mask_and_min_abs_sum", "params": {"eps": 1e-5}}, data=constraint_data,
        )
        assert isinstance(result, api.constraints.MaskAndMinAbsSumConstraint)
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

    def test_constraint_reserved_fields_cannot_be_redeclared(self, api, strategy, constraint_data):
        with pytest.raises(api.errors.ConstraintConfigurationError, match="reserved for prepared data"):
            strategy.build_constraint({"type": "mask", "params": {"mask": [[True]]}}, data=constraint_data)


class TestRealModelConstruction:
    @pytest.mark.parametrize("selector", ["default", "name", "alias", "class", "mapping"])
    def test_eager_zero_initialized_predictor_without_training(self, strategy, model_class, input_spec, components, monkeypatch, selector):
        selected = {
            "default": None, "name": "conditional", "alias": "conditional_linear",
            "class": model_class, "mapping": {"type": "conditional", "params": {"epochs": 7}},
        }[selector]
        forbidden = Mock(side_effect=AssertionError("No training or inference during construction"))
        monkeypatch.setattr(model_class, "fit", forbidden)
        monkeypatch.setattr(model_class, "predict", forbidden)
        network_class = importlib.import_module(f"{BACKEND}.models.network_parts").LinearGrangerNetwork
        monkeypatch.setattr(network_class, "__call__", forbidden)
        model = strategy.build_model(selected, components=components, config={"input_spec": input_spec, "seed": 23})
        forbidden.assert_not_called()
        assert isinstance(model, model_class)
        assert model.backend == "numpy" and model.input_spec is input_spec
        assert model.is_fitted is False
        assert model.model.n_features == 4 and model.model.n_outputs == 2
        layer = model.model.coefficient_layer
        assert layer.weight.shape == (2, 4) and layer.weight.dtype == np.float64
        assert layer.bias.shape == (2,) and layer.bias.dtype == np.float64
        np.testing.assert_array_equal(layer.weight, np.zeros((2, 4)))
        np.testing.assert_array_equal(layer.bias, np.zeros(2))
        np.testing.assert_array_equal(model.model.variable_control_layer.mask, np.ones(4))
        assert model.callbacks is not components["callbacks"]
        assert model.callbacks[0] is components["callbacks"][0]
        assert not hasattr(model, "initialize")

    def test_defaults(self, strategy, model_class, input_spec):
        model = strategy.build_model(model_class, config={"input_spec": input_spec})
        assert model.epochs == model.max_iter == 1000
        assert model.use_bias is model.fit_intercept is True
        assert model.batch_size is None and model.verbose == 0
        assert model.learning_rate == 1.0 and model.tol == 1e-8
        assert model.regularizer is None and model.constraint is None
        assert model.callbacks == []

    def test_legacy_aliases_map_to_current_configuration(self, strategy, model_class, input_spec):
        model = strategy.build_model(model_class, config={"input_spec": input_spec, "max_iter": 12, "fit_intercept": False})
        assert model.epochs == model.max_iter == 12
        assert model.use_bias is model.fit_intercept is False
        assert model.model.coefficient_layer.bias is None

    @pytest.mark.parametrize("config, message", [
        ({"epochs": 10, "max_iter": 10}, "Specify epochs or its legacy alias"),
        ({"use_bias": True, "fit_intercept": True}, "Specify use_bias or its legacy alias"),
    ])
    def test_alias_conflicts_rejected(self, strategy, model_class, input_spec, config, message):
        with pytest.raises(ValueError, match=message):
            strategy.build_model(model_class, config={"input_spec": input_spec, **config})

    def test_no_global_numpy_rng_change(self, strategy, model_class, input_spec):
        before = np.random.get_state()
        strategy.build_model(model_class, config={"input_spec": input_spec, "seed": 71})
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])

    def test_independent_predictors_from_same_configuration(self, strategy, model_class, input_spec, components):
        first = strategy.build_model(model_class, components=components, config={"input_spec": input_spec})
        second = strategy.build_model(model_class, components=components, config={"input_spec": input_spec})
        assert first.model is not second.model
        assert not np.shares_memory(first.model.coefficient_layer.weight, second.model.coefficient_layer.weight)
        assert not np.shares_memory(first.model.coefficient_layer.bias, second.model.coefficient_layer.bias)
        assert not np.shares_memory(first.model.variable_control_layer.mask, second.model.variable_control_layer.mask)

    def test_runtime_regularizer_constraint_are_clones_of_templates(self, strategy, model_class, input_spec, regularizer_data, constraint_data):
        regularizer = strategy.build_regularizer("l1", data=regularizer_data)
        constraint = strategy.build_constraint("mask", data=constraint_data)
        model = strategy.build_model(model_class, components={"regularizer": regularizer, "constraint": constraint}, config={"input_spec": input_spec})
        assert model.regularizer is regularizer and model.constraint is constraint
        runtime_regularizer, runtime_constraint = model.model.regularizer, model.model.constraint
        assert runtime_regularizer is not regularizer and isinstance(runtime_regularizer, type(regularizer))
        assert runtime_regularizer.l1 == pytest.approx(regularizer.l1)
        assert runtime_constraint is not constraint and isinstance(runtime_constraint, type(constraint))
        np.testing.assert_array_equal(runtime_constraint.mask, constraint.mask)
        assert not np.shares_memory(runtime_constraint.mask, constraint.mask)
        assert not np.shares_memory(runtime_regularizer.data.layout.base_mask, regularizer.data.layout.base_mask)

    def test_ready_model_returned_without_rebuild(self, strategy, model_class, input_spec, monkeypatch):
        ready = strategy.build_model(model_class, config={"input_spec": input_spec})
        predictor = ready.model
        forbidden = Mock(side_effect=AssertionError("Do not rebuild a ready model"))
        monkeypatch.setattr(ready, "rebuild", forbidden)
        assert strategy.build_model(ready) is ready and ready.model is predictor
        forbidden.assert_not_called()

    def test_input_spec_required_and_type_checked(self, strategy, model_class):
        with pytest.raises(TypeError, match="input_spec"):
            strategy.build_model(model_class)
        with pytest.raises(TypeError, match="ModelInputSpec"):
            strategy.build_model(model_class, config={"input_spec": object()})

    @pytest.mark.parametrize("component, value, message", [
        ("regularizer", "l1", "BaseNumpyRegularizer"),
        ("constraint", "mask", "BaseNumpyConstraint"),
        ("callbacks", [object()], "All callbacks"),
    ])
    def test_ready_components_not_specifications_required(self, strategy, model_class, input_spec, component, value, message):
        with pytest.raises(TypeError, match=message):
            strategy.build_model(model_class, components={component: value}, config={"input_spec": input_spec})

    @pytest.mark.parametrize("name", ["optimizer", "loss"])
    def test_non_none_optimizer_loss_rejected(self, strategy, model_class, input_spec, name):
        with pytest.raises(ValueError, match=f"non-None '{name}' component"):
            strategy.build_model(model_class, components={name: object()}, config={"input_spec": input_spec})

    @pytest.mark.parametrize("config, error, message", [
        ({"epochs": 0}, ValueError, "epochs"), ({"epochs": True}, ValueError, "epochs"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"verbose": True}, ValueError, "verbose"), ({"verbose": 3}, ValueError, "verbose"),
        ({"use_bias": 1}, TypeError, "use_bias"),
        ({"learning_rate": 0}, ValueError, "learning_rate"),
        ({"learning_rate": 1.1}, ValueError, "learning_rate"),
        ({"learning_rate": np.nan}, ValueError, "learning_rate"),
        ({"learning_rate": True}, TypeError, "learning_rate"),
        ({"tol": -1}, ValueError, "tol"), ({"tol": np.inf}, ValueError, "tol"),
        ({"tol": [0.0]}, TypeError, "tol"),
        ({"seed": -1}, ValueError, "seed"), ({"seed": True}, ValueError, "seed"),
    ])
    def test_configuration_validation(self, strategy, model_class, input_spec, config, error, message):
        with pytest.raises(error, match=message):
            strategy.build_model(model_class, config={"input_spec": input_spec, **config})

    @pytest.mark.parametrize("config", [
        {"epochs": np.int64(3), "batch_size": np.int64(2)},
        {"learning_rate": 0.5, "tol": 0.0}, {"use_bias": False, "seed": 0},
    ])
    def test_valid_boundary_configuration(self, strategy, model_class, input_spec, config):
        model = strategy.build_model(model_class, config={"input_spec": input_spec, **config})
        assert model.model is not None and model.is_fitted is False

    def test_regularizer_output_order_must_match_input_spec(self, strategy, model_class, input_spec, regularizer_data):
        mismatched = replace(input_spec, effects=tuple(reversed(input_spec.effects)))
        regularizer = strategy.build_regularizer("l1", data=regularizer_data)
        with pytest.raises(ValueError, match="Regularizer layout does not match input_spec"):
            strategy.build_model(model_class, components={"regularizer": regularizer}, config={"input_spec": mismatched})

    def test_constraint_output_order_must_match_input_spec(self, strategy, model_class, input_spec, constraint_data):
        mismatched = replace(input_spec, effects=tuple(reversed(input_spec.effects)))
        constraint = strategy.build_constraint("mask", data=constraint_data)
        with pytest.raises(ValueError, match="Constraint output order does not match input_spec"):
            strategy.build_model(model_class, components={"constraint": constraint}, config={"input_spec": mismatched})

    def test_constraint_feature_dimension_must_match_input_spec(self, strategy, model_class, input_spec, constraint_data):
        bad_data = replace(constraint_data, n_features=3, mask=np.ones((2, 3), dtype=bool))
        constraint = strategy.build_constraint("mask", data=bad_data)
        with pytest.raises(ValueError, match="Constraint must have coefficient shape"):
            strategy.build_model(model_class, components={"constraint": constraint}, config={"input_spec": input_spec})

    def test_model_backward_compatibility_alias(self, api, model_class):
        package = importlib.import_module(f"{BACKEND}.models")
        assert package.NumpyConstrainedGrangerModel is model_class


class TestPublicModelBootstrap:
    def test_default_model_from_clean_interpreter(self, api):
        """No manual conditional implementation import: exposes registry bootstrap bugs."""
        package = importlib.import_module(ROOT)
        import_root = Path(package.__file__).resolve().parent.parent
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(import_root), env.get("PYTHONPATH", "")]))
        script = dedent("""
            from enhanced_granger_analysis.backends.numpy.numpy_backend import NumpyBackendStrategy
            from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec
            strategy = NumpyBackendStrategy()
            spec = ModelInputSpec(
                n_features=2, n_outputs=1, causes=('x',), effects=('y',),
                cause_slices={'x': slice(0, 2)}, feature_lags=(1, 2),
            )
            model = strategy.build_model(
                components={'optimizer': strategy.build_optimizer(), 'loss': strategy.build_loss()},
                config={'input_spec': spec},
            )
            assert model.backend == 'numpy'
            assert model.input_spec is spec
            assert model.model is not None
            assert not model.is_fitted
        """)
        completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120, check=False)
        assert completed.returncode == 0, (
            "Public numpy construction failed in a fresh interpreter.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


class TestExtraConstructionBoundaries:
    def test_strategy_never_calls_callback_lifecycle_during_build(self, api, strategy, model_class, input_spec, monkeypatch):
        callback = api.callbacks.ConvergenceCheck()
        forbidden = Mock(side_effect=AssertionError("No callback lifecycle at construction"))
        for hook in ("on_train_beginning", "on_epoch_beginning", "on_epoch_end", "on_train_end", "clone_for_run"):
            monkeypatch.setattr(callback, hook, forbidden)
        model = strategy.build_model(model_class, components={"callbacks": [callback]}, config={"input_spec": input_spec})
        assert model.callbacks[0] is callback
        forbidden.assert_not_called()

    def test_regularizer_feature_lags_must_match_input_spec(self, strategy, model_class, input_spec, regularizer_data):
        mismatched = replace(input_spec, feature_lags=(1, 3, 1, 2))
        regularizer = strategy.build_regularizer("l1", data=regularizer_data)
        with pytest.raises(ValueError, match="Regularizer layout does not match input_spec"):
            strategy.build_model(model_class, components={"regularizer": regularizer}, config={"input_spec": mismatched})

    def test_regularizer_cause_blocks_must_match_input_spec(self, strategy, model_class, input_spec, regularizer_data):
        mismatched = replace(input_spec, cause_slices={"x": slice(0, 1), "z": slice(1, 4)})
        regularizer = strategy.build_regularizer("l1", data=regularizer_data)
        with pytest.raises(ValueError, match="Regularizer layout does not match input_spec"):
            strategy.build_model(model_class, components={"regularizer": regularizer}, config={"input_spec": mismatched})

    def test_read_only_constructor_mappings_preserved(self, strategy, model_class, input_spec, components):
        original = dict(components)
        config = {"input_spec": input_spec, "epochs": 7}
        model = strategy.build_model(model_class, components=MappingProxyType(components), config=MappingProxyType(config))
        assert components == original and config == {"input_spec": input_spec, "epochs": 7}
        assert model.epochs == 7

    def test_no_implicit_component_resolution(self, strategy, model_class, input_spec, components, monkeypatch):
        forbidden = Mock(side_effect=AssertionError("Only ready components expected"))
        for name in ("optimizer", "loss", "callbacks", "regularizer", "constraint"):
            monkeypatch.setattr(strategy, f"build_{name}", forbidden)
        model = strategy.build_model(model_class, components=components, config={"input_spec": input_spec})
        assert model.is_fitted is False
        forbidden.assert_not_called()
