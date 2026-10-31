from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from ..configuration.analysis_spec import AnalysisSpec
from ..core.components.constraint_parsing import RelationsInput
from ..core.components.constraint_spec import RelationConstraintsSpec
from ..execution.runner import AnalysisRunner
from ..preprocessing.missing import BaseImputer
from ..preprocessing.sampling import SamplingStrategy
from ..results.granger_results import GrangerAnalysisResults


class MultiTaskGrangerAPI:
    """Convenience interface for the standard Granger analysis workflow.

    Preprocessing declarations are passed unchanged to the analysis
    configuration and resolved by the existing preparation pipeline.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        reuse_data: bool = False,
    ) -> None:
        """Configure the backend and data reuse preference."""

        self.backend = backend
        self._reuse_data = bool(reuse_data)
        self._analysis_runner = AnalysisRunner()

    def fit(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
        tested_causes: Optional[Sequence[str]] = None,
        constraint_spec: Any = "mask_and_min_abs_sum",
        relations: RelationsInput = RelationConstraintsSpec(),
        lag_config: Any = None,
        lag_selector: Any = None,
        lag_selector_params: Optional[Mapping[str, Any]] = None,
        lag_n_jobs: int = 1,
        stationarity: Any = "adf",
        missing_value_imputer: Optional[BaseImputer] = None,
        schema_policy: str = "union",
        prepared_data: Any = None,
        backend_sample_fraction: float = 1.0,
        backend_max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        x_scaler: Any = "standard",
        y_scaler: Any = "standard",
        regularizer_spec: Any = None,
        callbacks: Optional[Sequence[Any]] = None,
        hiperoptimalization_state: Optional[str] = None,
        hiperoptimalization_conf: Optional[Dict[str, Any]] = None,
        initializer: Any = None,
        model_config: Optional[Dict[str, Any]] = None,
        fit_restricted: bool = True,
    ) -> GrangerAnalysisResults:
        """Execute an analysis using declarative preprocessing settings.

        Supplied prepared data takes precedence over source observations.
        Component resolution remains in the preparation and backend layers.
        """

        config = {
            "backend": self.backend,
            "reuse_data": self._reuse_data,
            "causes": causes,
            "effects": effects,
            "tested_causes": tested_causes,
            "constraint_spec": constraint_spec,
            "relations": relations,
            "lag_config": lag_config,
            "lag_selector": lag_selector,
            "lag_selector_params": (
                {}
                if lag_selector_params is None
                else dict(lag_selector_params)
            ),
            "lag_n_jobs": lag_n_jobs,
            "stationarity": stationarity,
            "missing_value_imputer": missing_value_imputer,
            "schema_policy": schema_policy,
            "backend_sample_fraction": backend_sample_fraction,
            "backend_max_samples": backend_max_samples,
            "random_state": random_state,
            "sampling_strategy": sampling_strategy,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "regularizer_spec": regularizer_spec,
            "callbacks": callbacks,
            "hiperoptimalization_state": hiperoptimalization_state,
            "hiperoptimalization_conf": hiperoptimalization_conf,
            "initializer": initializer,
            "model_config": model_config,
        }

        spec = AnalysisSpec.from_mapping(config)

        return self._analysis_runner.run_analysis(
            spec=spec,
            data=data,
            prepared_data=prepared_data,
            fit_restricted=fit_restricted,
        )