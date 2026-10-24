# enhanced_granger_analysis/core/execution/executor.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from .component_preparer import ComponentPreparer
from .session import AnalysisSession, Stage
from ...preprocessing.prepare_data import PreparedData
from ...results.granger_results import GrangerAnalysisResults

if TYPE_CHECKING:
    from ...api.analysis_config import AnalysisConfig


StartPolicy = Literal["initial", "base", "current"]


class AnalysisExecutor:
    """Coordinate analysis stages without implementing backend factories."""

    def __init__(
        self,
        component_preparer: ComponentPreparer | None = None,
    ) -> None:
        """Create an executor with an optional domain-data preparator.

        Args:
            component_preparer: Dependency used to compile constraint and
                regularizer data. Create the default preparator when omitted.

        Keep per-run artifacts in AnalysisSession, not on this executor.
        """
        self.component_preparer = (
            component_preparer
            if component_preparer is not None
            else ComponentPreparer()
        )

    def create_session(self, config: AnalysisConfig) -> AnalysisSession:
        """Create a fresh session from a validated configuration snapshot.

        Args:
            config: User-facing configuration, possibly still missing input
                data if the caller intends to configure it later.

        Snapshot configuration containers without deepcopying runtime objects.
        Do not prepare data, create backend objects, or start training.
        """
        raise NotImplementedError

    def reconfigure(
        self,
        session: AnalysisSession,
        config: AnalysisConfig,
        *,
        invalidate_from: Stage,
    ) -> None:
        """Replace session configuration and invalidate dependent artifacts.

        Args:
            session: Existing execution state owned by the caller.
            config: Updated configuration to snapshot into this session.
            invalidate_from: Earliest stage affected by the configuration edit.

        Preserve unaffected upstream artifacts. Clear invalidated artifacts,
        their completion markers, and stale downstream results. Do not claim
        that preserved training outputs belong to the new configuration.
        """
        raise NotImplementedError

    def invalidate(
        self,
        session: AnalysisSession,
        *,
        from_stage: Stage,
    ) -> None:
        """Clear a stage and every artifact depending on it.

        Args:
            session: Session whose artifacts must be invalidated.
            from_stage: Earliest invalid stage.

        Apply the dependency order:
        data -> components -> model -> initialization -> fit.

        A model invalidation also clears initial/base weights and fit snapshots.
        A data invalidation also clears preparation metadata. Do not mutate
        user-supplied objects or dispose shared objects without an ownership
        contract. Preserve only results that remain valid for retained stages.
        """
        raise NotImplementedError

    def prepare_data(
        self,
        session: AnalysisSession,
        prepared_data: PreparedData | None = None,
    ) -> PreparedData:
        """Prepare or accept the canonical data for this session.

        Args:
            session: Session providing configuration and receiving data.
            prepared_data: Explicit prepared data overriding the configured
                prepared_data for this call.

        Prefer the explicit argument, then configured prepared_data; otherwise
        call the existing preprocessing pipeline with configured raw input.
        Validate externally prepared data against required layout invariants.

        Invalidate downstream artifacts before replacing data. Initialize the
        result container and preparation context using the existing result API.
        Mark completion only after successful preparation. Never independently
        prepare different sample sets for full and restricted models.
        """
        raise NotImplementedError

    def prepare_backend(self, session: AnalysisSession) -> Any:
        """Resolve the backend strategy using the existing BackendFactory.

        Args:
            session: Session containing the requested backend.

        Let BackendFactory own backend selection and backend-name aliases.
        Store and return the selected strategy. Do not instantiate model
        components here or make backend selection depend on fitted data unless
        the existing factory contract explicitly requires it.
        """
        raise NotImplementedError

    def prepare_components(self, session: AnalysisSession) -> None:
        """Prepare domain construction data and obtain backend components.

        Args:
            session: Session with successfully prepared data.

        Select the backend if needed. Use ComponentPreparer exactly once for
        each required domain-data artifact. Delegate creation or acceptance of
        constraints, regularizers, optimizers, losses, and callbacks to strategy
        methods supported by the current backend contract.

        Do not duplicate backend resolvers or component registries. Respect
        category-specific None semantics; an optional regularizer and a
        mandatory mask-preserving constraint need not follow the same rule.

        Store successful outputs, synchronize the result mask with the effective
        constraint mask, and invalidate the model. Mark components complete only
        when every required component has been prepared successfully.
        """
        raise NotImplementedError

    def build_model(self, session: AnalysisSession) -> Any:
        """Construct an adapter from prepared components and model settings.

        Args:
            session: Session with completed data and component stages.

        Call BackendStrategy with final input/output dimensions and already
        prepared components. Adapt argument names to the project's strategy
        contract; do not re-resolve component specifications inside the adapter.

        Store and return the adapter. Do not fit or initialize it here.
        Clear downstream weights and training snapshots.
        """
        raise NotImplementedError

    def initialize_model(self, session: AnalysisSession) -> Any:
        """Initialize the adapter and establish canonical initial weights.

        Args:
            session: Session containing a constructed model and prepared data.

        Initialize the adapter using prepared predictors/targets and the seed
        through its supported protocol. Resolve any project-level initializer
        through the existing initializer subsystem, not a backend component
        loader.

        Build InitializationContext with the effective mask, apply resulting
        weights through the adapter, and capture independent initial-weight
        copies. Capture initial weights even when no custom initializer is used.
        """
        raise NotImplementedError

    def set_weights(
        self,
        session: AnalysisSession,
        weights: list[np.ndarray],
    ) -> None:
        """Assign copied weights through the adapter's validation mechanism.

        Args:
            session: Session containing an initialized adapter.
            weights: Complete weight representation accepted by that adapter.

        Do not infer validity solely from array shapes in the executor.
        Invalidate fit outputs when this public operation changes model weights.
        Internal restoration during fitting must not erase unrelated snapshots.
        """
        raise NotImplementedError

    def reset_run_state(
        self,
        session: AnalysisSession,
        *,
        run_name: str,
        callback_updates: Mapping[str, Any] | None = None,
    ) -> None:
        """Prepare mutable adapter state for one training run.

        Args:
            session: Session containing an initialized adapter and components.
            run_name: Stable identifier such as 'base_model' or a cause-specific
                restricted-model identifier.
            callback_updates: Optional run-local callback settings.

        Delegate callback reset and optimizer/training-state handling to the
        adapter's documented lifecycle API. Reset each supported state exactly
        once per run. Do not equate weight restoration with optimizer reset or
        invent backend-specific reset operations inside this executor.
        """
        raise NotImplementedError

    def fit_base(self, session: AnalysisSession) -> list[np.ndarray]:
        """Train the full multivariate model and record its snapshot.

        Args:
            session: Session with completed model initialization.

        Restore canonical initial weights, clear temporary variable omissions
        while preserving permanent constraints, and reset run-local state.
        Fit through the adapter, capture copied base weights, and record the
        full-model predictions, metadata, and timings using the result API.

        Do not create one independent model per output variable.
        """
        raise NotImplementedError

    def fit_restricted(
        self,
        session: AnalysisSession,
        cause: str,
        *,
        start: StartPolicy = "base",
    ) -> None:
        """Train a multivariate model excluding one tested cause block.

        Args:
            session: Session with a successfully fitted full model.
            cause: Cause name present in the prepared cause-slice mapping and
                selected for testing.
            start: Restore initial weights, restore full-model weights, or
                explicitly retain current weights.

        Replace the previous temporary omission with the requested cause block;
        do not accumulate exclusions across causes. Preserve permanent masks
        and use the same prepared samples and targets as the full model.

        Reset run-local training state, fit, and record a cause-keyed snapshot.
        Do not overwrite canonical initial weights or full-model base weights.
        """
        raise NotImplementedError

    def evaluate(self, session: AnalysisSession) -> GrangerAnalysisResults:
        """Evaluate recorded full/restricted fits using existing project logic.

        Args:
            session: Session containing the required fitted-model snapshots.

        Delegate statistical evaluation to the project's current result or
        analysis subsystem. Preserve its configured statistic, degrees of
        freedom, sample-count conventions, and error handling.

        Do not introduce an unconditional F-test or infer valid inferential
        assumptions merely because the adapter exposes predictions.
        Return the session result object.
        """
        raise NotImplementedError

    def run(
        self,
        session: AnalysisSession,
        *,
        restricted_start: StartPolicy = "base",
    ) -> GrangerAnalysisResults:
        """Execute missing preparation stages and a complete analysis run.

        Args:
            session: Session holding configuration and optional valid artifacts.
            restricted_start: Starting-weight policy for restricted fits.

        Prepare data, components, model, and initialization in dependency order,
        skipping only stages explicitly marked complete and still valid.

        For every call, start a new fit phase: clear previous fit outputs, fit
        the full model, fit each configured tested cause, then evaluate.
        Return results only after successful completion.

        Do not silently run hyperoptimization here; an outer optimization loop
        should create or reconfigure sessions explicitly.
        """
        raise NotImplementedError

    def require_stage(
        self,
        session: AnalysisSession,
        stage: Stage,
    ) -> None:
        """Raise a project validation error when a prerequisite is incomplete.

        Args:
            session: Session whose completion markers must be inspected.
            stage: Required successfully completed stage.

        Report the missing stage and the public operation needed to prepare it.
        Do not treat a valid optional None component as a missing stage.
        """
        raise NotImplementedError