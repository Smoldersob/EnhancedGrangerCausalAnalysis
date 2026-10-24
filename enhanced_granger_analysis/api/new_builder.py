# enhanced_granger_analysis/api/builder.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .analysis_config import AnalysisConfig
from .config_reader import ConfigReader
from ..core.execution.executor import AnalysisExecutor, StartPolicy
from ..core.execution.session import AnalysisSession, Stage
from ..preprocessing.prepare_data import PreparedData
from ..results.granger_results import GrangerAnalysisResults


class MultiTaskGrangerBuilder:
    """Expose a fluent facade over configuration and execution services."""

    def __init__(
        self,
        backend: str | None = None,
        reuse_data: bool = False,
        *,
        executor: AnalysisExecutor | None = None,
    ) -> None:
        """Create an analysis facade without preparing or fitting anything.

        Args:
            backend: Optional backend identifier resolved later by its factory.
            reuse_data: Existing project-level data reuse policy.
            executor: Optional execution dependency for customization or tests.
        """
        self._executor = (
            executor if executor is not None else AnalysisExecutor()
        )
        self._config = AnalysisConfig(
            backend=backend,
            reuse_data=reuse_data,
        )
        self._session = self._executor.create_session(self._config)

    @property
    def session(self) -> AnalysisSession:
        """Expose current execution artifacts for inspection.

        Treat the returned session as read-only from client code. Use facade
        methods for mutations so dependency invalidation cannot be bypassed.
        """
        return self._session

    @property
    def config(self) -> AnalysisConfig:
        """Return a configuration snapshot without exposing owned containers."""
        return self._config.snapshot()

    def _configure(
        self,
        values: Mapping[str, Any],
        *,
        invalidate_from: Stage,
    ) -> MultiTaskGrangerBuilder:
        """Apply a partial configuration edit and synchronize execution state.

        Args:
            values: Public configuration fields to replace.
            invalidate_from: Earliest execution stage affected by this edit.

        Validate the new configuration before replacing the previous one.
        Delegate all artifact invalidation to the executor.
        """
        config = self._config.updated(values)
        self._executor.reconfigure(
            self._session,
            config,
            invalidate_from=invalidate_from,
        )
        self._config = config
        return self

    def from_config(
        self,
        config: Mapping[str, Any],
    ) -> MultiTaskGrangerBuilder:
        """Apply public configuration fields to the current specification.

        Args:
            config: Partial mapping using the same schema as AnalysisConfig.

        Conservatively invalidate data and downstream stages. Do not read files
        or instantiate backend components.
        """
        return self._configure(config, invalidate_from="data")

    def from_file(self, path: str | Path) -> MultiTaskGrangerBuilder:
        """Read a configuration file and apply its resolved raw settings.

        Args:
            path: JSON or YAML analysis configuration.
        """
        return self.from_config(ConfigReader.read(path))

    def backend(self, name: str | None) -> MultiTaskGrangerBuilder:
        """Set the backend identifier and invalidate backend-dependent artifacts.

        Args:
            name: Backend name or alias accepted by BackendFactory.
        """
        return self._configure(
            {"backend": name}, invalidate_from="components"
        )

    def data(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
    ) -> MultiTaskGrangerBuilder:
        """Set raw input data and remove any configured prepared-data override.

        Args:
            data: One time series table or a sequence of separate series tables.

        Preserve series boundaries; actual preparation belongs to preprocessing.
        """
        return self._configure(
            {"data": data, "prepared_data": None},
            invalidate_from="data",
        )

    def prepared_data(
        self,
        data: PreparedData,
    ) -> MultiTaskGrangerBuilder:
        """Set an explicit prepared-data override for future preparation.

        Args:
            data: Prepared data satisfying the project's preprocessing contract.

        Keep raw data as an optional source, but give this override precedence.
        """
        return self._configure(
            {"prepared_data": data}, invalidate_from="data"
        )

    def variables(
        self,
        causes: Sequence[str] | str | None = None,
        effects: Sequence[str] | str | None = None,
        tested_causes: Sequence[str] | str | None = None,
    ) -> MultiTaskGrangerBuilder:
        """Set variable roles; configuration parsing canonicalizes name lists.

        Args:
            causes: Predictor-variable names, or None for project defaults.
            effects: Jointly modeled output names, or None for project defaults.
            tested_causes: Cause names to omit in restricted-model comparisons.
        """
        return self._configure(
            {
                "causes": causes,
                "effects": effects,
                "tested_causes": tested_causes,
            },
            invalidate_from="data",
        )

    def lag(
        self,
        lag_config: Any = None,
        lag_selector: Any = None,
    ) -> MultiTaskGrangerBuilder:
        """Set lag construction and selection specifications.

        Args:
            lag_config: Lag configuration object or supported mapping.
            lag_selector: Selector specification or supplied selector object.

        Resolve selector details within the preprocessing subsystem.
        """
        return self._configure(
            {"lag_config": lag_config, "lag_selector": lag_selector},
            invalidate_from="data",
        )

    def stationarity(self, transformer: Any) -> MultiTaskGrangerBuilder:
        """Set stationarity preprocessing, including explicit disabling if valid.

        Args:
            transformer: Transformer specification, object, or supported None.
        """
        return self._configure(
            {"stationarity_transformer": transformer},
            invalidate_from="data",
        )

    def scaling(
        self,
        x_scaler: Any = "standard",
        y_scaler: Any = "standard",
    ) -> MultiTaskGrangerBuilder:
        """Set predictor and target scaling specifications.

        Args:
            x_scaler: Predictor scaler name, specification, or object.
            y_scaler: Target scaler name, specification, or object.

        Do not fit scalers or inspect data in this method.
        """
        return self._configure(
            {"x_scaler": x_scaler, "y_scaler": y_scaler},
            invalidate_from="data",
        )

    def relations(self, relations: Any) -> MultiTaskGrangerBuilder:
        """Set backend-independent relation declarations.

        Args:
            relations: Parsed relation specification or supported inline rules.

        File references are handled by ConfigReader, not by this method.
        """
        return self._configure(
            {"relations": relations}, invalidate_from="components"
        )

    def constraint(self, spec: Any) -> MultiTaskGrangerBuilder:
        """Set a constraint specification accepted by the backend contract.

        Args:
            spec: Constraint selector/specification or supported runtime object.

        Preserve the mandatory base-mask invariant. Do not universally accept
        None merely because other component categories allow it.
        """
        return self._configure(
            {"constraint_spec": spec}, invalidate_from="components"
        )

    def regularization(self, spec: Any = None) -> MultiTaskGrangerBuilder:
        """Set regularization; None explicitly disables this optional component.

        Args:
            spec: Supported regularizer specification or runtime object.
        """
        return self._configure(
            {"regularizer_spec": spec}, invalidate_from="components"
        )

    def optimizer(self, spec: Any) -> MultiTaskGrangerBuilder:
        """Set an optimizer specification for backend-side resolution.

        Args:
            spec: Backend-supported name, mapping, object, or default sentinel.
        """
        return self._configure(
            {"optimizer_spec": spec}, invalidate_from="components"
        )

    def loss(self, spec: Any) -> MultiTaskGrangerBuilder:
        """Set the training loss, not the Granger comparison statistic.

        Args:
            spec: Backend-supported loss specification or runtime object.
        """
        return self._configure(
            {"loss_spec": spec}, invalidate_from="components"
        )

    def callbacks(self, callbacks: Any = None) -> MultiTaskGrangerBuilder:
        """Set callback specifications without constructing backend objects.

        Args:
            callbacks: One callback specification/object, a supported sequence,
                or None to disable explicitly configured callbacks.

        AnalysisConfig must distinguish one mapping/string specification from a
        sequence; do not blindly apply list() to arbitrary input.
        """
        return self._configure(
            {"callbacks": callbacks}, invalidate_from="components"
        )

    def initializer(self, spec: Any) -> MultiTaskGrangerBuilder:
        """Set the project-level weight initialization specification.

        Args:
            spec: Initializer name, mapping, class, object, or supported None.

        Rebuild the model conservatively so initialization cannot reuse mutated
        adapter state from a previous fit.
        """
        return self._configure(
            {"initializer": spec}, invalidate_from="model"
        )

    def model(
        self,
        model_config: Mapping[str, Any] | None = None,
    ) -> MultiTaskGrangerBuilder:
        """Replace model-construction settings; None means an empty mapping.

        Args:
            model_config: Adapter/model parameters excluding separately managed
                component specifications and reserved configuration fields.
        """
        return self._configure(
            {"model_config": dict(model_config or {})},
            invalidate_from="model",
        )

    def prepare_data(self) -> PreparedData:
        """Delegate data preparation and return the canonical prepared data."""
        return self._executor.prepare_data(self._session)

    def prepare_components(self) -> MultiTaskGrangerBuilder:
        """Delegate component preparation after data has been prepared."""
        self._executor.prepare_components(self._session)
        return self

    def build_model(self) -> Any:
        """Delegate adapter construction without initialization or fitting."""
        return self._executor.build_model(self._session)

    def initialize_model(self) -> Any:
        """Delegate adapter initialization and initial-weight capture."""
        return self._executor.initialize_model(self._session)

    def fit_base(self) -> Any:
        """Delegate full-model fitting after successful initialization."""
        return self._executor.fit_base(self._session)

    def fit_restricted(
        self,
        cause: str,
        *,
        start: StartPolicy = "base",
    ) -> MultiTaskGrangerBuilder:
        """Delegate fitting of one cause-restricted multivariate model.

        Args:
            cause: Tested cause to omit.
            start: Explicit starting-weight policy for this restricted fit.
        """
        self._executor.fit_restricted(
            self._session,
            cause,
            start=start,
        )
        return self

    def fit(
        self,
        *,
        restricted_start: StartPolicy = "base",
    ) -> GrangerAnalysisResults:
        """Run the complete configured analysis through the executor.

        Args:
            restricted_start: Starting-weight policy shared by restricted fits.

        Reuse only valid prepared artifacts; execute a fresh fit phase.
        """
        return self._executor.run(
            self._session,
            restricted_start=restricted_start,
        )