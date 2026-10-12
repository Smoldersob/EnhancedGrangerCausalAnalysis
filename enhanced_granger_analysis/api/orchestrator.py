from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..core.lag_config import LagConfiguration
from ..preprocessing.lag import BaseLagSelector
from ..preprocessing.missing import BaseImputer
from ..preprocessing.sampling import SamplingStrategy
from ..preprocessing.stationarity import StationarityTransformer
from ..results.granger_results import GrangerAnalysisResults
from .builder import MultiTaskGrangerBuilder


class MultiTaskGrangerAPI:
    """Thin orchestration layer over the fluent builder API.

    This class is a convenience wrapper that configures a
    :class:`MultiTaskGrangerBuilder` and delegates the actual analysis work to it.
    It keeps the selected backend and reuse policy, then executes the full
    preparation, model-building, and restricted-model fitting workflow in one call.

    Example:
        api = MultiTaskGrangerAPI(backend="keras", reuse_data=True)
        results = api.fit(
            data=my_data,
            causes=["cause1", "cause2"],
            effects=["effect1"],
            lag_config=LagConfiguration(max_lag=12),
        )
        print(results)
    """

    def __init__(self, backend: Optional[str] = None, reuse_data: bool = False) -> None:
        """Create an API wrapper around the builder pipeline.

        Args:
            backend: Backend name used when the underlying builder prepares the
                execution strategy.
            reuse_data: Reuse policy passed to the underlying builder.
        """
        self.backend = backend
        self._reuse_data = bool(reuse_data)
        self._stationarity_transformer = StationarityTransformer()

    def fit(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
        tested_causes: Optional[Sequence[str]] = None,
        relations: Optional[Mapping[Tuple[str, str], Any]] = None,
        lag_config: Optional[LagConfiguration] = None,
        lag_selector: Optional[BaseLagSelector] = None,
        stationarity_transformer: Optional[StationarityTransformer] = None,
        missing_value_imputer: Optional[BaseImputer] = None,
        schema_policy: str = "error",
        prepared_data: Any = None,
        backend_sample_fraction: float = 1.0,
        backend_max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        x_scaler: Optional[Any] = "standard",
        y_scaler: Optional[Any] = "standard",
        regularizer: Optional[Any] = None,
        regularizer_spec: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Sequence[Any]] = None,
        hiperoptimalization_state: Optional[str] = None,
        hiperoptimalization_conf: Optional[Dict[str, Any]] = None,
        initializer: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> GrangerAnalysisResults:
        """Execute the full Granger analysis pipeline for the provided data.

        The method builds a ``MultiTaskGrangerBuilder`` configured with the API
        settings, prepares the data, instantiates the strategy and model,
        estimates the unrestricted model, and then fits the restricted models for
        each tested cause before returning the aggregated results.
        """
        builder = MultiTaskGrangerBuilder(
            backend=self.backend, reuse_data=self._reuse_data
        ).data(data).variables(causes, effects, tested_causes).relations(relations)
        builder.lag(lag_config, lag_selector).stationarity(stationarity_transformer)
        builder._fit_kwargs.update({
            "missing_value_imputer": missing_value_imputer,
            "schema_policy": schema_policy,
            "prepared_data": prepared_data,
            "backend_sample_fraction": backend_sample_fraction,
            "backend_max_samples": backend_max_samples,
            "random_state": random_state,
            "sampling_strategy": sampling_strategy,
            "x_scaler": x_scaler, "y_scaler": y_scaler,
            "regularizer": regularizer,
            "regularizer_spec": regularizer_spec,
            "callbacks": list(callbacks) if callbacks is not None else None,
            "hiperoptimalization_state": hiperoptimalization_state,
            "hiperoptimalization_conf": hiperoptimalization_conf,
            "initializer": initializer, "model_config": model_config,
        })
        builder.prepare_data(prepared_data)
        builder.prepare_components()
        builder.build_model()
        builder.initialize_weights()
        builder.fit_base()
        builder.fit_restricted_models()
        return builder.get_results()
