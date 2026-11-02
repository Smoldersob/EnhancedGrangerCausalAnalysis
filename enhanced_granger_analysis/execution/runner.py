from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .granger_trainer import GrangerTrainer

from ..backends import BackendFactory

from ..configuration.analysis_spec import AnalysisSpec
from ..configuration.modeling_spec import ModelingSpec

from ..core.models.protocol import GrangerModelProtocol
from ..core.exceptions import (
	ConstraintConfigurationError,
	DataValidationError,
)

from ..modeling.artifacts import FitOutcome
from ..modeling.assembler import ModelAssembler
from ..modeling.initialization import ModelInitializer

from ..preprocessing.pipeline import DataPreparer
from ..preprocessing.search_preparation import resolve_search_preparation
from ..preprocessing.prepare_data import PreparedData

from ..results.granger_results import PreparationContext
from ..results.granger_results import GrangerAnalysisResults
from .model_search_runner import ModelSearchRunner

@dataclass
class AnalysisRunProgress:
	"""Mutable progress of one standard analysis execution.

	This object is internal execution bookkeeping. It does not own
	configuration, evaluate results, or provide a session interface.
	"""

	results: GrangerAnalysisResults
	base_weights: list[np.ndarray] | None = None


class AnalysisRunner:
	"""Coordinate preparation, model construction, and Granger fitting.

	Execution artifacts remain local to each call. Component
	specifications are resolved by existing domain services and
	backend strategies.
	"""

	def __init__(
		self,
		trainer: GrangerTrainer | None = None,
		*,
		assembler: ModelAssembler | None = None,
		initializer: ModelInitializer | None = None,
		search_runner: ModelSearchRunner | None = None,
	) -> None:
		self._trainer = (
			trainer if trainer is not None else GrangerTrainer()
		)

		self._assembler = (
			assembler if assembler is not None else ModelAssembler()
		)

		self._initializer = (
			initializer if initializer is not None else ModelInitializer()
		)
		self._search_runner = (
			search_runner if search_runner is not None else ModelSearchRunner()
		)

	def run_model_search(
		self,
		*,
		spec: AnalysisSpec,
		data: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
		search_prepared_data: PreparedData | None = None,
	) -> tuple[ModelingSpec, Any]:
		"""Run optional HPO in its own preparation context."""
		if spec.search is None:
			if search_prepared_data is not None:
				raise DataValidationError(
					"search_prepared_data cannot be supplied when search is disabled."
				)
			return spec.modeling, None

		if search_prepared_data is None:
			if data is None:
				raise DataValidationError(
					"Enabled model search requires source data or search_prepared_data."
				)
			search_config = resolve_search_preparation(
				spec.preparation,
				spec.search.preparation,
			)
			search_prepared_data = DataPreparer(
				config=search_config,
			).fit(data, roles=spec.roles).prepared_data_
			preparation_metadata = {
				"config": search_config,
				"split": search_prepared_data.split_snapshot(),
				"index_space": "prepared_rows",
				"n_samples": int(search_prepared_data.X_raw.shape[0]),
				"output_names": tuple(search_prepared_data.effects),
			}
		else:
			preparation_metadata = {
				"index_space": "prepared_rows",
				"n_samples": int(search_prepared_data.X_raw.shape[0]),
				"split": search_prepared_data.split_snapshot(),
				"output_names": tuple(search_prepared_data.effects),
			}

		split = search_prepared_data.resolved_split()
		split.indices_for("training", require_nonempty=True)
		split.indices_for(
			spec.search.validation_part,
			require_nonempty=True,
		)

		result = self._search_runner.run(
			spec=spec.search,
			modeling=spec.modeling,
			prepared=search_prepared_data,
			preparation=preparation_metadata,
		)
		return result.best_modeling, result

	def prepare_data(
		self,
		*,
		spec: AnalysisSpec,
		data: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
		prepared_data: PreparedData | None = None,
	) -> PreparedData:
		"""Return supplied prepared data or prepare the source observations."""

		if prepared_data is not None:
			return prepared_data

		if data is None:
			raise DataValidationError(
				"Source data or prepared_data must be supplied."
			)

		preparer = DataPreparer(
			config=spec.preparation,
		)

		return preparer.fit(
			data,
			roles=spec.roles,
		).prepared_data_

	def run(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		progress: AnalysisRunProgress,
		initial_weights: Sequence[np.ndarray] | None = None,
		backend: str | None = None,
		fit_restricted: bool = True,
	) -> GrangerAnalysisResults:
		"""Run the sequence selected by the adapter's baseline role."""
		role = getattr(model, "base_model_role", "unrestricted")
		if role is None:
			if fit_restricted:
				for cause in prepared.tested_causes:
					self.fit_unrestricted(
						model=model, prepared=prepared, results=progress.results,
						cause=cause, initial_weights=initial_weights, backend=backend,
					)
				for cause in prepared.tested_causes:
					self.fit_restricted(
						model=model, prepared=prepared, results=progress.results,
						cause=cause, initial_weights=initial_weights, backend=backend,
					)
			return progress.results

		base_outcome = self.fit_base(
			model=model, prepared=prepared, results=progress.results,
			initial_weights=initial_weights, backend=backend,
		)
		progress.base_weights = [np.asarray(w).copy() for w in base_outcome]
		if not fit_restricted:
			return progress.results

		for cause in prepared.tested_causes:
			if role == "unrestricted":
				self.fit_restricted(
					model=model, prepared=prepared, results=progress.results,
					cause=cause, base_weights=progress.base_weights,
					backend=backend,
				)
			else:
				self.fit_unrestricted(
					model=model, prepared=prepared, results=progress.results,
					cause=cause, base_weights=progress.base_weights,
					backend=backend,
				)

		return progress.results

	def fit_base(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		results: GrangerAnalysisResults,
		initial_weights: Sequence[np.ndarray] | None = None,
		backend: str | None = None,
	) -> list[np.ndarray]:
		"""Fit and store the base model, then return its weight state."""

		outcome = self._trainer.fit_base(
			model=model,
			prepared=prepared,
			initial_weights=initial_weights,
		)

		self.store_fit_outcome(
			prepared=prepared,
			results=results,
			cause=None,
			outcome=outcome,
			backend=backend,
			role=getattr(model, "base_model_role", "unrestricted"),
		)

		return [np.asarray(weight).copy() for weight in outcome.weights]

	def fit_restricted(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		results: GrangerAnalysisResults,
		cause: str,
		base_weights: Sequence[np.ndarray] | None,
		initial_weights: Sequence[np.ndarray] | None = None,
		start_with_base: bool = True,
		start_with_initial: bool = False,
		backend: str | None = None,
	) -> None:
		"""Fit and store one restricted model."""

		outcome = self._trainer.fit_restricted(
			model=model,
			prepared=prepared,
			cause=cause,
			base_weights=base_weights,
			initial_weights=initial_weights,
		)
		self.store_fit_outcome(
			prepared=prepared,
			results=results,
			cause=cause,
			outcome=outcome,
			backend=backend,
			role="restricted",
		)

	def fit_unrestricted(self, *, model, prepared, results, cause,
		base_weights=None, initial_weights=None, backend=None) -> None:
		outcome = self._trainer.fit_unrestricted(
			model=model, prepared=prepared, cause=cause,
			base_weights=base_weights, initial_weights=initial_weights,
		)
		self.store_fit_outcome(
			prepared=prepared, results=results, cause=cause,
			outcome=outcome, backend=backend, role="unrestricted",
		)

	def fit_restricted_models(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		results: GrangerAnalysisResults,
		base_weights: Sequence[np.ndarray] | None,
		initial_weights: Sequence[np.ndarray] | None = None,
		backend: str | None = None,
	) -> GrangerAnalysisResults:
		"""Fit and store restricted models in their configured order."""

		for cause in prepared.tested_causes:
			self.fit_restricted(
				model=model,
				prepared=prepared,
				results=results,
				cause=cause,
				base_weights=base_weights,
				initial_weights=initial_weights,
				backend=backend,
			)

		return results

	@staticmethod
	def store_fit_outcome(
		*,
		prepared: PreparedData,
		results: GrangerAnalysisResults,
		cause: str | None,
		outcome: FitOutcome,
		backend: str | None = None,
		role: str | None = None,
	) -> None:
		"""Convert model-scale artifacts to the analysis snapshot format."""

		predictions = prepared.y_scaler.inverse_transform(
			outcome.predictions
		)

		raw_weights = np.asarray(outcome.weights[0], dtype=float)
		weights = raw_weights.T.copy()

		metadata = {
			"timings": dict(outcome.timings),
			"backend": backend,
		}

		if cause is None:
			results.set_base_snapshot(
				predictions,
				weights,
				metadata,
				active_slope_mask=outcome.active_slope_mask,
				bias_enabled=outcome.bias_enabled,
				role=role,
			)
		elif role == "unrestricted":
			results.set_unrestricted_snapshot(
				cause, predictions, weights, metadata,
				active_slope_mask=outcome.active_slope_mask,
				bias_enabled=outcome.bias_enabled,
			)
		else:
			results.set_reference_snapshot(
				cause,
				predictions,
				weights,
				metadata,
				active_slope_mask=outcome.active_slope_mask,
				bias_enabled=outcome.bias_enabled,
			)

	@staticmethod
	def save_snapshot(
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		results: GrangerAnalysisResults,
		cause: str | None,
		backend: str | None = None,
		additional_metadata: Mapping[str, Any] | None = None,
	) -> None:
		"""Store the current model state without performing a new fit."""

		predictions = prepared.y_scaler.inverse_transform(
			np.asarray(
				model.predict(prepared.X_scaled),
				dtype=float,
			)
		)

		raw_weights = np.asarray(model.get_weights()[0], dtype=float)
		weights = raw_weights.T.copy()

		metadata = dict(additional_metadata or {})
		metadata["backend"] = backend

		if cause is None:
			results.set_base_snapshot(
				predictions,
				weights,
				metadata,
				active_slope_mask=model.get_active_mask(),
				bias_enabled=model.use_bias,
			)
		else:
			results.set_reference_snapshot(
				cause,
				predictions,
				weights,
				metadata,
				active_slope_mask=model.get_active_mask(),
				bias_enabled=model.use_bias,
			)

	@staticmethod
	def create_results(
		prepared: PreparedData,
	) -> GrangerAnalysisResults:
		"""Create an analysis result store with preparation context."""

		results = GrangerAnalysisResults(
			effects=prepared.effects,
			causes=prepared.tested_causes,
			mask=prepared.base_mask,
			cause_slices=prepared.cause_slices,
			y_true=prepared.y_raw,
			X_true=prepared.X_raw,
		)

		results.set_preparation_context(
			PreparationContext(
				prepared_data=prepared,
				metadata={
					"requested_roles": prepared.requested_roles,
					"effective_roles": prepared.effective_roles,
					"removed_variables": prepared.removed_variables,
                    "split": prepared.split_snapshot(),
                },
			)
		)

		return results

	def run_analysis(
		self,
		*,
		spec: AnalysisSpec,
		data: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
		prepared_data: PreparedData | None = None,
		search_prepared_data: PreparedData | None = None,
		fit_restricted: bool = True,
	) -> GrangerAnalysisResults:
		"""Execute a complete analysis from configuration and observations.

		Supplied prepared data takes precedence over source observations.

		"""
		modeling, search_result = self.run_model_search(
			spec=spec,
			data=data,
			search_prepared_data=search_prepared_data,
		)

		prepared = self.prepare_data(
			spec=spec,
			data=data,
			prepared_data=prepared_data,
		)

		results = self.create_results(prepared)

		strategy = BackendFactory.get_strategy(
			modeling.backend
		)

		if modeling.constraint_spec is None:
			raise ConstraintConfigurationError(
				"constraint_spec cannot be None. "
				"Use the default constraint to preserve the base mask."
			)

		constraint_data = self._assembler.prepare_constraint_data(
			prepared=prepared,
			relations=modeling.relations,
		)

		constraint = self._assembler.build_constraint(
			strategy=strategy,
			constraint_spec=modeling.constraint_spec,
			data=constraint_data,
		)

		results.mask = constraint_data.mask.copy()

		regularizer_data = self._assembler.prepare_regularizer_data(
			prepared=prepared,
			regularizer_spec=modeling.regularizer_spec,
		)

		regularizer = self._assembler.build_regularizer(
			strategy=strategy,
			regularizer_spec=modeling.regularizer_spec,
			data=regularizer_data,
		)

		assembly = self._assembler.assemble(
			strategy=strategy,
			prepared=prepared,
			model_config=modeling.model_config,
			callbacks=modeling.callbacks,
			constraint=constraint,
			regularizer=regularizer,
		)

		model = assembly.model

		initial_weights = self._initializer.compute_weights(
			model=model,
			prepared=prepared,
			initializer_spec=modeling.initializer,
		)

		if initial_weights is not None:
			model.set_weights([
				np.asarray(weight).copy()
				for weight in initial_weights
			])

			initial_weights = [
				np.asarray(weight).copy()
				for weight in model.get_weights()
			]

		progress = AnalysisRunProgress(
			results=results,
		)

		results = self.run(
			model=model,
			prepared=prepared,
			progress=progress,
			initial_weights=initial_weights,
			backend=modeling.backend,
			fit_restricted=fit_restricted,
		)
		results.model_search_result = search_result
		results.effective_modeling = modeling
		return results