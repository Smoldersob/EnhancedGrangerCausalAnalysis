from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..configuration.analysis_spec import AnalysisSpec
from ..configuration.loader import AnalysisConfigLoader

from ..modeling.assembler import ModelAssembler
from ..modeling.initialization import ModelInitializer
from ..modeling.executor import FitExecutor
from ..modeling.artifacts import FitOutcome

from ..execution.granger_trainer import GrangerTrainer
from ..execution.runner import (
    AnalysisRunner,
    AnalysisRunProgress,
)

from ..core.exceptions import DataValidationError, ConstraintConfigurationError
from ..core.components.constraint_parsing import (
    RelationsInput,
    parse_relations_spec,
)
from ..core.components.constraint_spec import (
    PreparedConstraintData,
    RelationConstraintsSpec,
)
from ..core.components.regularizer_parsing import (
    parse_regularizer_spec,
)
from ..core.components.regularizer_spec import (
    PreparedRegularizerData,
)

from ..backends import BackendFactory
from ..backends.base_backend import BackendStrategy
from ..preprocessing.prepare_data import PreparedData
from ..results.granger_results import GrangerAnalysisResults


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
            "schema_policy": "union",
            "constraint_spec": {
                            "type": "mask_and_min_abs_sum",
                            "params": {},
                        },
            "relations": RelationConstraintsSpec(),
            "lag_config": None,
            "lag_selector": None,
            "lag_selector_params": {},
            "lag_n_jobs": 1,
            "stationarity": "adf",
            "missing_value_imputer": None, 
            "sampling_strategy": None, "backend_sample_fraction": 1.0, "backend_max_samples": None,
            "random_state": None, 
            "x_scaler": "standard", "y_scaler": "standard",
            "regularizer_spec": None,
            "callbacks": None,
            "hiperoptimalization_state": None, "hiperoptimalization_conf": None,
            "initializer": None, 
            "model_config": None,
            "prepared_data": None,
        }
        self._model_assembler = ModelAssembler()
        self._model_initializer = ModelInitializer()
        self._fit_executor = FitExecutor()
        self._granger_trainer = GrangerTrainer(
            executor=self._fit_executor,
        )
        self._analysis_runner = AnalysisRunner(
            trainer=self._granger_trainer,
        )

        self._device_config: Dict[str, Any] = {}

        self._prepared_data: Optional[PreparedData] = None
        self._strategy: BackendStrategy | None = None
        self._constraint: Any = None
        self._constraint_data: Optional[PreparedConstraintData] = None
        self._regularizer: Any = None
        self._regularizer_data: Optional[PreparedRegularizerData] = None
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
        self._strategy = None

        self._invalidate_from(
            "components",
            preserve_results=True,
        )
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
    
    def relations(
        self,
        relations: RelationsInput,
    ) -> "MultiTaskGrangerBuilder":
        """Store a RelationConstraintsSpec or a list of canonical declarations.

        Accepts only:
            - RelationConstraintsSpec instance;
            - list of relation rules [...].

        File references are resolved only by the configuration loader.
        """
        spec = parse_relations_spec(relations)

        self._fit_kwargs["relations"] = spec
        self._invalidate_from("components")
        return self

    def constraint(
        self,
        constraint_spec: Any,
    ) -> "MultiTaskGrangerBuilder":
        """Configure constraint selection.

        Accepted by the backend resolver:
            - registered name;
            - {"type": name, "params": {...}};
            - concrete backend constraint class.

        Instances and None are not supported.
        """
        if constraint_spec is None:
            raise ConstraintConfigurationError(
                "constraint_spec cannot be None. "
                "Use the default constraint to preserve the base mask."
            )

        self._fit_kwargs["constraint_spec"] = (
            copy.deepcopy(dict(constraint_spec))
            if isinstance(constraint_spec, Mapping)
            else constraint_spec
        )

        self._invalidate_from("components")
        return self

    def lag(
        self,
        lag_config=None,
        lag_selector=None,
        *,
        lag_selector_params: Optional[Mapping[str, Any]] = None,
        n_jobs: int = 1,
    ) -> "MultiTaskGrangerBuilder":
        """Store lag specifications for later resolution by the pipeline.

        Calling this method replaces the complete stored lag setup.
        No selector or engine is constructed here.
        """
        if (
            lag_selector_params is not None
            and not isinstance(lag_selector_params, Mapping)
        ):
            raise TypeError(
                "lag_selector_params must be a mapping or None."
            )

        self._fit_kwargs.update({
            "lag_config": copy.deepcopy(lag_config),
            "lag_selector": (
                copy.deepcopy(dict(lag_selector))
                if isinstance(lag_selector, Mapping)
                else lag_selector
            ),
            "lag_selector_params": (
                {}
                if lag_selector_params is None
                else copy.deepcopy(dict(lag_selector_params))
            ),
            "lag_n_jobs": n_jobs,
        })

        self._invalidate_from("data")
        return self
    
    def stationarity(
        self,
        spec: Any = None,
    ) -> "MultiTaskGrangerBuilder":
        """Store a stationarity selection for later pipeline resolution.

        None disables stationarity processing.

        Registered names, mapping specifications, classes, and instances
        are interpreted by the stationarity resolver during preparation.
        """
        self._fit_kwargs["stationarity"] = (
            copy.deepcopy(dict(spec))
            if isinstance(spec, Mapping)
            else spec
        )
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
        
    def regularization(
            self,
            regularizer_spec: Any = None,
    ) -> "MultiTaskGrangerBuilder":
        """
        Store a parsed specification; None disables regularization.
        Args:
            regularizer_spec: Specification for the regularizer, which can be a registered name, a dictionary with type and parameters, or a concrete backend regularizer class.

        Returns:
            Self for method chaining
        """
        spec = parse_regularizer_spec(regularizer_spec)

        self._fit_kwargs["regularizer_spec"] = spec

        self._invalidate_from(
            "components",
            preserve_results=True,
        )

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
        """Store reserved hyperoptimization settings.

        Hyperoptimization is disabled in the standard builder pipeline.
        These settings do not trigger optimization or modify model selection.
        """
        self._fit_kwargs["hiperoptimalization_state"] = state
        self._fit_kwargs["hiperoptimalization_conf"] = copy.deepcopy(config)
        return self
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

    def from_config(
        self,
        config: Mapping[str, Any],
    ) -> "MultiTaskGrangerBuilder":
        """Update supplied configuration fields without resolving components."""

        if "regularizer" in config:
            raise DataValidationError(
                "The 'regularizer' configuration field is not supported. "
                "Use 'regularizer_spec'."
            )

        if "constraint_spec" in config and config["constraint_spec"] is None:
            raise ConstraintConfigurationError(
                "constraint_spec cannot be None."
            )

        if "backend" in config:
            self._backend = config["backend"]

        if "reuse_data" in config:
            self._reuse_data = bool(config["reuse_data"])

        if "data" in config:
            self._data = config["data"]

        for key in self._fit_kwargs:
            if key in config:
                self._fit_kwargs[key] = config[key]

        if "compute_device" in config:
            self._device_config = {
                "compute_device": config["compute_device"]
            }
        elif "device" in config:
            self._device_config = {
                "device": config["device"]
            }

        self._invalidate_from("data")

        return self

    def from_file(
        self,
        path: str | Path,
    ) -> "MultiTaskGrangerBuilder":
        """Update configuration from a document with resolved file references."""

        config = AnalysisConfigLoader.load_file(path)
        return self.from_config(config)

    def replace_config(
        self,
        config: Mapping[str, Any],
    ) -> "MultiTaskGrangerBuilder":
        """Replace configuration and discard previous execution artifacts."""

        replacement = MultiTaskGrangerBuilder()
        replacement.from_config(config)

        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)

        return self

    def to_spec(self) -> AnalysisSpec:
        """Create an analysis specification without execution artifacts."""

        config = dict(self._fit_kwargs)
        config["backend"] = self._backend
        config["reuse_data"] = self._reuse_data
        config.update(self._device_config)

        return AnalysisSpec.from_mapping(config)

    # Executable stages.
    def prepare_constraint_data(self) -> PreparedConstraintData:
        """Prepare constraint data using the final prepared feature layout."""

        self._require_prepared()

        compiled = self._model_assembler.prepare_constraint_data(
            prepared=self._prepared_data,
            relations=self._fit_kwargs["relations"],
        )

        self._constraint_data = compiled
        return compiled

    def prepare_regularizer_data(
        self,
    ) -> Optional[PreparedRegularizerData]:
        """Prepare backend-independent input for optional regularization."""

        self._require_prepared()

        spec = parse_regularizer_spec(
            self._fit_kwargs.get("regularizer_spec")
        )

        self._invalidate_from(
            "model",
            preserve_results=True,
        )
        self._regularizer = None
        self._regularizer_data = None

        if spec is None:
            return None

        self._regularizer_data = (
            self._model_assembler.prepare_regularizer_data(
                prepared=self._prepared_data,
                regularizer_spec=spec,
            )
        )

        return self._regularizer_data
    
    
    def prepare_data(
        self,
        prepared_data: Optional[PreparedData] = None,
    ) -> PreparedData:
        """Prepare data or accept an explicitly supplied preparation artifact."""

        if prepared_data is None:
            prepared_data = self._fit_kwargs.get("prepared_data")

        if prepared_data is None:
            if self._data is None:
                raise DataValidationError(
                    "Builder requires data(...) before prepare_data()"
                )

            prepared_data = self._analysis_runner.prepare_data(
                spec=self.to_spec(),
                data=self._data,
            )

        self._invalidate_from("components")
        self._prepared_data = prepared_data

        self._results = self._analysis_runner.create_results(
            prepared_data
        )

        return prepared_data

        return prepared_data

    def prepare_backend(self):
        """Select the backend using the shared analysis specification."""

        self._require_prepared()

        modeling = self.to_spec().modeling

        self._strategy = BackendFactory.get_strategy(
            modeling.backend
        )

        return self._strategy

    def prepare_constraint(self):
        """Prepare constraint data and construct a backend implementation."""

        self._require_components_data()

        data = self.prepare_constraint_data()

        constraint = self._model_assembler.build_constraint(
            strategy=self._strategy,
            constraint_spec=self._fit_kwargs["constraint_spec"],
            data=data,
        )

        self._constraint = constraint

        if self._results is not None:
            self._results.mask = data.mask.copy()

        return constraint

    def prepare_regularizer(self) -> Any:
        """Prepare and construct optional backend regularization."""

        self._require_components_data()

        spec = parse_regularizer_spec(
            self._fit_kwargs.get("regularizer_spec")
        )

        self._invalidate_from(
            "model",
            preserve_results=True,
        )
        self._regularizer = None

        if spec is None:
            self._regularizer_data = None
            return None

        if self._regularizer_data is None:
            self.prepare_regularizer_data()

        self._regularizer = self._model_assembler.build_regularizer(
            strategy=self._strategy,
            regularizer_spec=spec,
            data=self._regularizer_data,
        )

        return self._regularizer
    
    def prepare_callbacks(self):
        """Construct callbacks using the selected backend strategy."""

        self._require_components_data()

        self._callbacks = self._model_assembler.build_callbacks(
            strategy=self._strategy,
            callbacks=self._fit_kwargs.get("callbacks"),
        )

        return self._callbacks
    
    def prepare_components(self):
        """Prepare all components for the analysis."""
        self.prepare_backend()

        self.prepare_constraint()

        self.prepare_regularizer_data()
        self.prepare_regularizer()

        self.prepare_callbacks()

        return self


    def build_model(
        self,
        model_config: Optional[Mapping[str, Any]] = None,
    ):
        """Construct a model from prepared data and ready components.

        Configuration overrides apply only to this build operation.
        Weight initialization, fitting, and hyperoptimization are not
        performed during model assembly.
        """

        self._require_components_data()

        if self._constraint is None:
            self.prepare_constraint()

        if (
            self._fit_kwargs.get("regularizer_spec") is not None
            and self._regularizer is None
        ):
            self.prepare_regularizer()

        config = dict(self.to_spec().modeling.model_config)
        if model_config is not None:
            config.update(model_config)

        assembly = self._model_assembler.assemble(
            strategy=self._strategy,
            prepared=self._prepared_data,
            model_config=config,
            callbacks=self._fit_kwargs.get("callbacks"),
            constraint=self._constraint,
            regularizer=self._regularizer,
        )

        self._callbacks = assembly.callbacks
        self._model_config = assembly.model_config
        self._model = assembly.model

        self._initial_weights = None
        self._base_weights = None

        if self._results is not None:
            self._results.base_snapshot = None
            self._results.reference_snapshots.clear()

        return self._model

    def initialize_weights(self, initializer=None):
        """Compute and assign initial weights using the configured initializer.

        When initializer is None, use the stored initializer specification.
        Weight assignment follows the existing builder and model contract.
        """

        self._require_model()

        spec = (
            initializer
            if initializer is not None
            else self._fit_kwargs.get("initializer")
        )

        weights = self._model_initializer.compute_weights(
            model=self._model,
            prepared=self._prepared_data,
            initializer_spec=spec,
        )

        if weights is None:
            return None

        self.set_weights(weights)

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
        """Reset callbacks without changing optimizer state or model weights."""
        self._require_model()

        self._model.reset_components(
            reset_optimizer=False,
            reset_callbacks=True,
            run_name=run_name,
            callback_updates=callback_updates,
        )

    def fit_base(self):
        """Fit the unrestricted model and store its fitted weight state."""

        self._require_model()
        self._require_prepared()

        self._base_weights = self._analysis_runner.fit_base(
            model=self._model,
            prepared=self._prepared_data,
            results=self._results,
            initial_weights=self._initial_weights,
            backend=self._backend,
        )

        return self._base_weights

    def fit_restricted(
        self,
        cause: str,
        start_with_base: bool = True,
        start_with_initial: bool = False,
    ):
        """Fit and store one restricted model through the analysis runner."""

        self._require_model()
        self._require_prepared()

        self._analysis_runner.fit_restricted(
            model=self._model,
            prepared=self._prepared_data,
            results=self._results,
            cause=cause,
            base_weights=self._base_weights,
            initial_weights=self._initial_weights,
            start_with_base=start_with_base,
            start_with_initial=start_with_initial,
            backend=self._backend,
        )

    def fit_restricted_models(self):
        """Fit and store restricted models for all tested causes."""

        self._require_model()
        self._require_prepared()

        return self._analysis_runner.fit_restricted_models(
            model=self._model,
            prepared=self._prepared_data,
            results=self._results,
            base_weights=self._base_weights,
            initial_weights=self._initial_weights,
            backend=self._backend,
        )

    def _store_fit_outcome(
        self,
        cause: Optional[str],
        outcome: FitOutcome,
    ) -> None:
        """Store fit artifacts through the analysis runner."""

        self._analysis_runner.store_fit_outcome(
            prepared=self._prepared_data,
            results=self._results,
            cause=cause,
            outcome=outcome,
            backend=self._backend,
        )

    def _save_snapshot(
        self,
        cause: Optional[str],
        additional_metadata: Optional[Mapping[str, Any]] = None,
    ):
        """Store the current model state through the analysis runner."""

        self._analysis_runner.save_snapshot(
            model=self._model,
            prepared=self._prepared_data,
            results=self._results,
            cause=cause,
            backend=self._backend,
            additional_metadata=additional_metadata,
        )

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
        """Prepare the analysis and execute the standard fitting workflow."""

        self.prepare_data()
        self.prepare_components()
        self.build_model()
        self.initialize_weights()

        progress = AnalysisRunProgress(
            results=self._results,
            base_weights=self._base_weights,
        )

        try:
            return self._analysis_runner.run(
                model=self._model,
                prepared=self._prepared_data,
                progress=progress,
                initial_weights=self._initial_weights,
                backend=self._backend,
                fit_restricted=fit_restricted,
            )
        finally:
            self._base_weights = progress.base_weights

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
            self._regularizer_data = None
            self._prepared_data = None
            self._constraint_data = None
            self._strategy = None
            self._constraint = None
            self._regularizer = None
            self._model = None

        elif stage == "components":
            self._regularizer_data = None
            self._constraint_data = None
            self._constraint = None
            self._regularizer = None
            self._model = None

        elif stage == "model":
            self._model = None

        self._initial_weights = None
        self._base_weights = None

        if not preserve_results:
            self._results = None
        elif self._results is not None:
            # Keep preparation context, but discard results of the old model.
            self._results.base_snapshot = None
            self._results.reference_snapshots.clear()

MultitaskGrangerBuilder = MultiTaskGrangerBuilder
