from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from time import perf_counter

from ..backends import BackendFactory
from ..core.exceptions import DataValidationError, RegistryError
from ..core.initialization import InitializationContext
from ..initializers.registry import resolve_initializer_spec
from ..preprocessing.prepare_data import PreparedData, prepare_data as run_prepare_data
from ..preprocessing.stationarity import StationarityTransformer
from ..results.granger_results import GrangerAnalysisResults, PreparationContext
from .config_loader import BuilderConfigLoader


def _normalize_optional_name_list(
    values: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """
    Normalize a sequence of strings to a list of strings, or return None if the input is None.
    Args:
        values: A sequence of strings or None
    Returns:
        A list of strings or None
    """

    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    items = list(values)
    return items or None


class MultiTaskGrangerBuilder:
    """Fluent builder for a multitask Granger causality analysis pipeline.

    The builder stores the analysis configuration and exposes a chainable API for
    selecting the data, variables, lag settings, preprocessing, and model
    configuration before executing the full pipeline via ``fit()``.

    Example:
        builder = MultiTaskGrangerBuilder()
        results = (
            builder.data(data)
            .variables(causes=["A"], effects=["B"])
            .fit()
        )
    """

    def __init__(self, backend: Optional[str] = None, reuse_data: bool = False) -> None:
        """Create a configured builder instance.

        Args:
            backend: Optional backend name used when preparing the execution
                strategy for the analysis.
            reuse_data: Reuse policy passed to the pipeline configuration.
        """
        self._backend = backend
        self._data: Optional[pd.DataFrame | Sequence[pd.DataFrame]] = None
        self._reuse_data = bool(reuse_data)
        self._fit_kwargs: Dict[str, Any] = {
            "causes": None, "effects": None, "tested_causes": None,
            "relations": None, "lag_config": None, "lag_selector": None,
            "stationarity_transformer": StationarityTransformer(),
            "missing_value_imputer": None, "schema_policy": "union",
            "backend_sample_fraction": 1.0, "backend_max_samples": None,
            "random_state": None, "sampling_strategy": None,
            "x_scaler": "standard", "y_scaler": "standard",
            "regularizer": None, "regularizer_spec": None, "callbacks": None,
            "hiperoptimalization_state": None, "hiperoptimalization_conf": None,
            "initializer": None, "model_config": None,
        }
        self._prepared_data: Optional[PreparedData] = None
        self._strategy: Any = None
        self._constraint: Any = None
        self._regularizer: Any = None
        self._callbacks: Optional[List[Any]] = None
        self._model_config: Dict[str, Any] = {}
        self._model: Any = None
        self._initial_weights: Optional[List[np.ndarray]] = None
        self._base_weights: Optional[List[np.ndarray]] = None
        self._results: Optional[GrangerAnalysisResults] = None
        self._hyperopt_result: Any = None

    # Configuration methods remain deliberately side-effect free.
    def backend(self, backend_name: Optional[str]) -> "MultiTaskGrangerBuilder":
        """
        Configure the backend for the analysis.

        Args:
            backend_name: Name of the backend to use

        Returns:
            Self for method chaining
        """
        self._backend = backend_name
        return self

    def data(self, data: pd.DataFrame | Sequence[pd.DataFrame]) -> "MultiTaskGrangerBuilder":
        """
        Configure input data for the analysis.

        Args:
            data: Input data (pandas DataFrame or list of DataFrames)

        Returns:
            Self for method chaining
        """
        self._data = data
        self._invalidate_from("data")
        return self

    def variables(self, causes=None, effects=None, tested_causes=None):
        """
        Configure causal relationships in the analysis.

        Args:
            causes: List of variable names to consider as causes
            effects: List of variable names to consider as effects
            tested_causes: List of variable names to test as causes

        Returns:
            Self for method chaining
        """
        self._fit_kwargs.update({
            "causes": _normalize_optional_name_list(causes),
            "effects": _normalize_optional_name_list(effects),
            "tested_causes": _normalize_optional_name_list(tested_causes),
        })
        self._invalidate_from("data")
        return self

    def relations(self, relations):
        """
        Configure explicit causal relations for the analysis.

        Args:
            relations: A dictionary mapping cause variables to lists of effect variables

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["relations"] = dict(relations) if relations is not None else None
        self._invalidate_from("components")
        return self

    def lag(self, lag_config=None, lag_selector=None):
        """
        Configure the lag structure for the analysis.

        Args:
            lag_config: Configuration for the lag structure
            lag_selector: Selector for the lag structure

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["lag_config"] = lag_config
        self._fit_kwargs["lag_selector"] = lag_selector
        self._invalidate_from("data")
        return self

    def stationarity(self, transformer=StationarityTransformer()):
        """
        Configure the stationarity transformation for the analysis.

        Args:
            transformer: The transformer to use for stationarity adjustment

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["stationarity_transformer"] = transformer
        self._invalidate_from("data")
        return self

    def reuse_data(self, value=True):
        """
        Configure whether to reuse the input data after preparation.

        Args:
            value: Boolean indicating whether to reuse data

        Returns:
            Self for method chaining
        """
        self._reuse_data = bool(value)
        return self

    def scaling(self, x_scaler="standard", y_scaler="standard"):
        """
        Configure the scaling for input and output variables.
        Args:
            x_scaler: Scaler for input variables (e.g., 'standard', 'minmax')
            y_scaler: Scaler for output variables (e.g., 'standard', 'minmax')
        Returns:
            Self for method chaining
        """
        self._fit_kwargs["x_scaler"] = x_scaler
        self._fit_kwargs["y_scaler"] = y_scaler
        self._invalidate_from("data")
        return self

    def backend_load(self, backend_sample_fraction=1.0, backend_max_samples=None):
        """
        Configure the backend data loading.

        Args:
            backend_sample_fraction: The fraction of the backend data to use
            backend_max_samples: The maximum number of samples to use from the backend

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["backend_sample_fraction"] = float(backend_sample_fraction)
        self._fit_kwargs["backend_max_samples"] = backend_max_samples
        self._invalidate_from("data")
        return self
        
    def regularization(self, regularizer=None, regularizer_spec=None):
        """
        Configure the regularization for the analysis.

        Args:
            regularizer: The regularizer to use
            regularizer_spec: The specification for the regularizer

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["regularizer"] = regularizer
        self._fit_kwargs["regularizer_spec"] = copy.deepcopy(regularizer_spec)
        self._invalidate_from("components")
        return self

    def callbacks(self, callbacks=None):
        """
        Configure the callbacks for the analysis.

        Args:
            callbacks: A list of callbacks to use

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["callbacks"] = list(callbacks) if callbacks is not None else None
        self._invalidate_from("model")
        return self

    def hyperoptimization(self, state=None, config=None):
        """
        Configure hyperparameter optimization for the analysis.

        Args:
            state: The state of the hyperparameter optimization
            config: The configuration for the hyperparameter optimization
        Returns:
            Self for method chaining
        """
        self._fit_kwargs["hiperoptimalization_state"] = state
        self._fit_kwargs["hiperoptimalization_conf"] = copy.deepcopy(config)
        return self

    def initializer(self, initializer):
        """
        Store an initializer name, specification, class, or instance.
        
        Args:
           initializer: The initializer name, specification, class
                or instance to use for weight initialization.        
        Returns:
            Self for method chaining
        """
        self._fit_kwargs["initializer"] = (
            copy.deepcopy(dict(initializer))
            if isinstance(initializer, Mapping)
            else initializer
        )
        self._invalidate_from("model")
        return self

    def prepared_data(self, prepared_data):
        """
        Set the prepared data for the analysis.
        Args:
            prepared_data: The prepared data to use

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["prepared_data"] = prepared_data
        self._invalidate_from("data")
        return self

    def model(self, model_config=None):
        """
        Configure the model for the analysis.

        Args:
            model_config: The configuration for the model

        Returns:
            Self for method chaining
        """
        self._fit_kwargs["model_config"] = dict(model_config) if model_config is not None else None
        self._invalidate_from("model")
        return self

    def from_config(self, config: Mapping[str, Any]) -> "MultiTaskGrangerBuilder":
        """
        Configure the builder from a configuration dictionary.
        Args:
            config: The configuration dictionary
        Returns:
            Self for method chaining
        """
        if "backend" in config:
            self._backend = config["backend"]
        if "reuse_data" in config:
            self._reuse_data = bool(config["reuse_data"])
        if "data" in config:
            self._data = config["data"]
        for key in self._fit_kwargs:
            if key not in config:
                continue
            value = config[key]
            if key == "initializer" and isinstance(value, Mapping):
                value = copy.deepcopy(dict(value))
            elif key in {
                "regularizer_spec",
                "hiperoptimalization_conf",
                "model_config",
            } and value is not None:
                value = dict(value)
            elif key in {"causes", "effects", "tested_causes"}:
                value = _normalize_optional_name_list(value)
            elif key == "relations" and value is not None:
                value = dict(value)

            self._fit_kwargs[key] = value
        self._invalidate_from("data")
        return self

    def from_file(self, path: str | Path) -> "MultiTaskGrangerBuilder":
        """
        Configure the builder from a configuration file.
        Args:
           path (str | Path): Path to the configuration file (JSON, YAML, etc.)
        Returns:
            Self for method chaining
        """
        return self.from_config(BuilderConfigLoader.load_file(path))

    # Executable stages.
    def prepare_data(self, prepared_data: Optional[PreparedData] = None) -> PreparedData:
        """
        Prepare the data for the analysis.
        Args:
            prepared_data: The prepared data, if available.

        Returns:
            The prepared data.
        """
        if prepared_data is None:
            prepared_data = self._fit_kwargs.get("prepared_data")
        if prepared_data is None:
            if self._data is None:
                raise DataValidationError("Builder requires data(...) before prepare_data()")
            prepared_data = run_prepare_data(data=self._data, **{
                k: v for k, v in self._fit_kwargs.items()
                if k in {"causes", "effects", "tested_causes", "lag_config", "lag_selector",
                         "stationarity_transformer", "missing_value_imputer", "schema_policy",
                         "backend_sample_fraction", "backend_max_samples", "random_state",
                         "sampling_strategy", "x_scaler", "y_scaler"}
            })
        self._prepared_data = prepared_data
        self._results = GrangerAnalysisResults(
            effects=prepared_data.effects,
            causes=prepared_data.tested_causes,
            mask=prepared_data.base_mask,
            cause_slices=prepared_data.cause_slices,
            y_true=prepared_data.y_raw,
            X_true=prepared_data.X_raw,
        )
        self._results.set_preparation_context(PreparationContext(
            prepared_data=prepared_data,
            metadata={"requested_roles": prepared_data.requested_roles,
                      "effective_roles": prepared_data.effective_roles,
                      "removed_variables": prepared_data.removed_variables},
        ))
        return prepared_data

    def prepare_backend(self):
        """
        Prepare the backend for the analysis.
        Returns:
            The prepared backend.
        """
        self._require_prepared()
        self._strategy = BackendFactory.get_strategy(self._backend)
        return self._strategy

    def prepare_constraint(self):
        """
        Prepare the constraint for the analysis.
        Returns:
            The prepared constraint.
        """
        self._require_components_data()
        prepared = self._prepared_data
        self._constraint = self._strategy.build_constraint_from_relations(
            relations=dict(self._fit_kwargs["relations"] or {}),
            predictor_names=list(prepared.causes),
            output_names=list(prepared.effects),
            col_offsets=np.asarray(
                [prepared.cause_slices[name].start for name in prepared.causes],
                dtype=int,
            ),
            n_features=prepared.X_scaled.shape[1],
            base_mask=prepared.base_mask,
        )
        if self._results is not None and hasattr(self._constraint, "mask"):
            self._results.mask = np.asarray(self._constraint.mask, dtype=bool).copy()
        return self._constraint

    def prepare_regularizer(self):
        """
        Prepare the regularizer for the analysis.
        Returns:
            The prepared regularizer.
        """
        self._require_components_data()
        spec = dict(self._fit_kwargs.get("regularizer_spec") or {})
        if str(spec.get("type", "")).lower() == "lag_dependent_l1":
            selection = getattr(self._prepared_data.lag_engine, "selection_result_", None)
            if selection is not None:
                spec.setdefault("max_lags_per_pred", list(selection.max_lags_per_pred))
                spec.setdefault("col_offsets", list(selection.col_offsets))
        if spec:
            self._regularizer = self._strategy.build_regularizer(spec)
        else:
            self._regularizer = self._fit_kwargs.get("regularizer")
        return self._regularizer

    def prepare_callbacks(self):
        """
        Prepare the callbacks for the analysis.
        returns:
            The list of prepared callbacks.
        """
        callbacks = self._fit_kwargs.get("callbacks")
        self._callbacks = list(callbacks) if callbacks is not None else None
        return self._callbacks

    def prepare_components(self):
        """Prepare all components for the analysis."""
        self.prepare_backend()
        self.prepare_constraint()
        self.prepare_regularizer()
        self.prepare_callbacks()
        return self

    def build_model(self, model_config: Optional[Mapping[str, Any]] = None):
        """
        Build the Granger model for the analysis.
        Args:
            model_config: Optional configuration for the model
        Returns:
            The built model.
        """
        self._require_components_data()
        config = dict(self._fit_kwargs.get("model_config") or {})
        if model_config is not None:
            config.update(model_config)
        if self._callbacks is not None:
            config["callbacks"] = list(self._callbacks)
        self._model_config = config
        self._model = self._strategy.build_model(
            n_features=self._prepared_data.X_scaled.shape[1],
            n_outputs=self._prepared_data.y_scaled.shape[1],
            regularizer=self._regularizer,
            constraint=self._constraint,
            **config,
        )
        self._model.initialize(self._prepared_data.X_scaled, targets=self._prepared_data.y_scaled)
        self._initial_weights = None
        self._base_weights = None
        if self._results is not None:
            self._results.base_snapshot = None
            self._results.reference_snapshots.clear()
        return self._model

    def initialize_weights(self, initializer=None):
        """Resolve, construct, and execute an initializer.

        If initializer is None, use the stored specification.

        An initializer instance must accept the keyword arguments:
            Y, X_lagged, mask

        It must return:
            A: shape (n_targets, n_features)
            B: shape (n_targets,)
        """
        self._require_model()

        spec = (
            initializer
            if initializer is not None
            else self._fit_kwargs.get("initializer")
        )

        try:
            initializer_class, params = resolve_initializer_spec(spec)
        except (RegistryError, TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Invalid initializer specification: {exc}"
            ) from exc

        if initializer_class is None:
            return None

        init_obj = initializer_class(**params)

        context = InitializationContext(
            X=self._prepared_data.X_scaled,
            y=self._prepared_data.y_scaled,
            weights=tuple(
                np.asarray(weight).copy()
                for weight in self._model.get_weights()
            ),
            mask=self._prepared_data.base_mask,
        )

        weights = init_obj(context)

        self.set_weights(weights)

        # Store the representation accepted and exposed by the model.
        self._initial_weights = [
            np.asarray(weight).copy()
            for weight in self._model.get_weights()
        ]

        return self._initial_weights

    def set_weights(self, weights):
        """
        Delegate weight validation and assignment to the model.
        Args:
            weights: The weights to set.
        """
        self._require_model()
        self._model.set_weights([
            np.asarray(weight).copy()
            for weight in weights
        ])

    def reset_callbacks(self, run_name: str, callback_updates=None):
        """
        Reset the callbacks of the model for a new run.
        Args:
            run_name: The name of the run (e.g., 'base_model', 'reference_cause_X')            
            callback_updates: Optional updates to the callbacks.
        """
        self._require_model()
        self._model.reset_callbacks(run_name=run_name, callback_updates=callback_updates)

    def fit_base(self):
        """
        Fit the base model for the analysis.
        Returns:
            The weights of the fitted base model.
        """
        self._require_model()
        self._model.omit_variables([])
        timings = {}
        t0 = perf_counter()
        if self._initial_weights is not None:
            self.set_weights(self._initial_weights)
        timings["restore_weights"] = perf_counter() - t0
        
        t0 = perf_counter()
        self.reset_callbacks("base_model")
        timings["reset_callbacks"] = perf_counter() - t0

        t0 = perf_counter()
        self._model.fit()
        timings["fit_model"] = perf_counter() - t0
        self._save_snapshot(None, additional_metadata={"timings": timings})
        self._base_weights = [
                np.asarray(weight).copy()
                for weight in self._model.get_weights()
        ]
        return self._base_weights

    def fit_restricted(self, cause: str, start_with_base: bool = True, start_with_initial: bool = False):
        """
        Fit a restricted model for a specific cause.
        Args:
            cause: The cause for which to fit the restricted model.
            use_hot_start: Whether to use the base model weights as a starting point.
        """
        if self._base_weights is None:
            raise DataValidationError("fit_base() must be completed before fit_restricted()")
        if cause not in self._prepared_data.cause_slices:
            raise DataValidationError(f"Unknown tested cause: {cause!r}")
        self._require_model()
        timings = {}
        t0 = perf_counter()
        if start_with_initial and self._initial_weights is not None:
            self.set_weights(self._initial_weights)
        elif start_with_base and self._base_weights is not None:
            self.set_weights(self._base_weights)
        timings["restore_weights"] = perf_counter() - t0

        t0 = perf_counter()
        block = self._prepared_data.cause_slices[cause]
        self._model.omit_variables(list(range(block.start, block.stop)))
        timings["update_mask"] = perf_counter() - t0
        
        t0 = perf_counter()
        self.reset_callbacks(f"reference_cause_{cause}")
        timings["reset_callbacks"] = perf_counter() - t0

        t0 = perf_counter()
        self._model.fit()
        timings["fit_model"] = perf_counter() - t0
        self._save_snapshot(cause, additional_metadata={"timings": timings})

    def fit_restricted_models(self):
        """Fit restricted models for all tested causes."""
        self._require_model()
        for cause in self._prepared_data.tested_causes:
            self.fit_restricted(cause)
        return self._results

    def _save_snapshot(self, cause: Optional[str], additional_metadata: Optional[Mapping[str, Any]] = None):
        """
        Save the predictions and weights of the model for a specific cause.
        Args:
            cause: The cause for which to save the snapshot. If None, saves the base model
        """
        predictions = self._prepared_data.y_scaler.inverse_transform(
            np.asarray(self._model.predict(self._prepared_data.X_scaled), dtype=float))
        weights = np.asarray(self._model.get_weights()[0], dtype=float).copy()
        additional_metadata = dict(additional_metadata or {})
        additional_metadata["backend"] = self._backend  
        if cause is None:
            self._results.set_base_snapshot(predictions, weights, additional_metadata)
        else:
            self._results.set_reference_snapshot(cause, predictions, weights, additional_metadata)

    @property
    def results(self):
        """
        Get the results of the Granger causality analysis.
        """
        return self.get_results()

    def get_results(self):
        """
        Get the results of the Granger causality analysis.
        Returns:
            The results of the analysis.
        """
        if self._results is None:
            raise DataValidationError("No results are available; prepare_data() has not run")
        return self._results

    def fit(self, *, fit_restricted: bool = True):
        """
        Execute the pipeline, optionally skipping restricted-model fits.
        Args:
            fit_restricted: If True, fit restricted models for all tested causes after fitting the base
        """
        self.prepare_data()
        self.prepare_components()
        self.build_model()
        self.initialize_weights()
        self.fit_base()
        if fit_restricted:
            self.fit_restricted_models()

        return self.get_results()

    run = fit

    def _require_prepared(self):
        """
        Ensure that the data has been prepared before proceeding.
        Raises:    
            DataValidationError: If the data has not been prepared yet.
        """
        if self._prepared_data is None:
            raise DataValidationError("prepare_data() must be completed first")

    def _require_components_data(self):
        """
        Ensure that the components (backend, constraint, regularizer) have been prepared before proceeding.
        Raises:
            DataValidationError: If the components have not been prepared yet.
        """
        self._require_prepared()
        if self._strategy is None:
            raise DataValidationError("prepare_backend() must be completed first")

    def _require_model(self):
        """
        Ensure that the model has been built before proceeding.
        Raises:
            DataValidationError: If the model has not been built yet.
        """
        self._require_components_data()
        if self._model is None:
            raise DataValidationError("build_model() must be completed first")

    def _invalidate_from(self, stage: str, preserve_results: bool = False):
        """
        Invalidate the builder from a specific stage.
        Args:
            stage: The stage from which to invalidate the builder.
            preserve_results: If True, preserve the results.
        """
        if stage == "data":
            self._prepared_data = None
            self._strategy = self._constraint = self._regularizer = None
            self._model = None
        elif stage == "components":
            self._constraint = self._regularizer = None
            self._model = None
        elif stage == "model":
            self._model = None
        self._initial_weights = self._base_weights = None
        if not preserve_results:
            self._results = None


MultitaskGrangerBuilder = MultiTaskGrangerBuilder
