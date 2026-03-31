"""Domain-specific exceptions for the complex_granger_analysis package.

The hierarchy is intentionally granular so calling code can either:
1) catch a broad library-level error (``ExtendedGrangerError``), or
2) catch focused categories (backend/preprocessing/configuration).

Most concrete exceptions inherit from either ``ExtendedGrangerValueError``
or ``ExtendedGrangerRuntimeError`` for compatibility with existing code that
already handles ``ValueError`` / ``RuntimeError``.
"""

from __future__ import annotations


class ExtendedGrangerError(Exception):
	"""Base class for all library-specific exceptions."""


class ExtendedGrangerValueError(ExtendedGrangerError, ValueError):
	"""Library-specific value error (invalid argument/value/state)."""


class ExtendedGrangerRuntimeError(ExtendedGrangerError, RuntimeError):
	"""Library-specific runtime error (invalid runtime lifecycle/state)."""


# ---------------------------------------------------------------------------
# Configuration and validation
# ---------------------------------------------------------------------------

class ConfigurationError(ExtendedGrangerValueError):
	"""Base error for invalid or inconsistent user configuration."""


class LagConfigurationError(ConfigurationError):
	"""Lag configuration is invalid or internally inconsistent."""


class TrainingConfigurationError(ConfigurationError):
	"""Training configuration contains invalid hyperparameters."""


class ConstraintConfigurationError(ConfigurationError):
	"""Constraint/regularization configuration is invalid."""


class ValidationError(ExtendedGrangerValueError):
	"""Base error for user input validation failures."""


class DataValidationError(ValidationError):
	"""Input data did not pass schema/type/shape checks."""


class EmptyDataError(DataValidationError):
	"""Input dataset/list is empty when at least one item is required."""


class DataShapeError(DataValidationError):
	"""Input data shape/dimensionality is incompatible with operation."""


class ColumnMismatchError(DataValidationError):
	"""DataFrames in a collection do not share the same ordered columns."""


class MissingColumnsError(DataValidationError):
	"""Required columns are missing from input data."""


# ---------------------------------------------------------------------------
# Backend-related
# ---------------------------------------------------------------------------

class BackendError(ExtendedGrangerError):
	"""Base class for backend selection, availability, and execution issues."""


class BackendSelectionError(BackendError, ExtendedGrangerValueError):
	"""Unknown/unsupported backend requested by user or config."""


class BackendNotAvailableError(BackendError, ExtendedGrangerRuntimeError):
	"""Requested backend cannot be used (missing dependency/device)."""


class BackendCompatibilityError(BackendError, ExtendedGrangerRuntimeError):
	"""Chosen backend cannot work with requested model/data/options."""


class BackendExecutionError(BackendError, ExtendedGrangerRuntimeError):
	"""Backend failed during fit/predict/optimization execution."""


class ModelNotFittedError(BackendExecutionError):
	"""Operation requires a fitted model, but fit has not been performed yet."""


# ---------------------------------------------------------------------------
# Preprocessing-related
# ---------------------------------------------------------------------------

class PreprocessingError(ExtendedGrangerError):
	"""Base class for preprocessing pipeline errors."""


class ScalingError(PreprocessingError, ExtendedGrangerRuntimeError):
	"""General scaling failure."""


class InvalidFeatureRangeError(ScalingError, ExtendedGrangerValueError):
	"""Provided feature range is invalid (e.g. max <= min)."""


class ScalerNotFittedError(ScalingError):
	"""Scaler transform/inverse_transform called before fit_transform."""


class StationarityError(PreprocessingError, ExtendedGrangerRuntimeError):
	"""General stationarity-analysis failure."""


class StationarityTestError(StationarityError):
	"""Statistical stationarity test failed or produced invalid output."""


class StationarityNotFittedError(StationarityError):
	"""Stationarity transformer used before fit_stationarity."""


class DifferencingError(StationarityError):
	"""Differencing could not be applied as requested."""


class LagPreprocessingError(PreprocessingError):
	"""General lag preparation/selection failure."""


class LagSelectionError(LagPreprocessingError, ExtendedGrangerRuntimeError):
	"""Lag selection algorithm failed or returned inconsistent result."""


class LagPreparationError(LagPreprocessingError, ExtendedGrangerRuntimeError):
	"""Construction of lagged matrices failed."""


# ---------------------------------------------------------------------------
# Training / optimization
# ---------------------------------------------------------------------------

class TrainingError(ExtendedGrangerRuntimeError):
	"""General training or optimization failure."""


class ConvergenceError(TrainingError):
	"""Optimization did not converge within configured limits."""


class NumericalStabilityError(TrainingError):
	"""Numerical instability detected (NaN/Inf/singular behavior)."""


# ---------------------------------------------------------------------------
# Results / post-processing
# ---------------------------------------------------------------------------

class ResultsError(ExtendedGrangerRuntimeError):
	"""Invalid or inconsistent result object/state."""


class CausalityMatrixError(ResultsError):
	"""Causality matrix cannot be built or interpreted."""


__all__ = [
	"BackendCompatibilityError",
	"BackendError",
	"BackendExecutionError",
	"BackendNotAvailableError",
	"BackendSelectionError",
	"CausalityMatrixError",
	"ColumnMismatchError",
	"ExtendedGrangerError",
	"ExtendedGrangerRuntimeError",
	"ExtendedGrangerValueError",
	"ConfigurationError",
	"ConstraintConfigurationError",
	"ConvergenceError",
	"DataShapeError",
	"DataValidationError",
	"DifferencingError",
	"EmptyDataError",
	"InvalidFeatureRangeError",
	"LagConfigurationError",
	"LagPreparationError",
	"LagPreprocessingError",
	"LagSelectionError",
	"MissingColumnsError",
	"ModelNotFittedError",
	"NumericalStabilityError",
	"PreprocessingError",
	"ResultsError",
	"ScalerNotFittedError",
	"ScalingError",
	"StationarityError",
	"StationarityNotFittedError",
	"StationarityTestError",
	"TrainingConfigurationError",
	"TrainingError",
	"ValidationError",
]
