# enhanced_granger_analysis/api/analysis_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ..core.components.constraint_spec import RelationConstraintsSpec
from ..core.lag_config import LagConfiguration
from ..preprocessing.prepare_data import PreparedData
from ..preprocessing.stationarity import StationarityTransformer


@dataclass
class AnalysisConfig:
    """Store the user-defined analysis specification, not execution state."""

    backend: str | None = None

    data: pd.DataFrame | Sequence[pd.DataFrame] | None = None
    prepared_data: PreparedData | None = None
    reuse_data: bool = False

    causes: list[str] | None = None
    effects: list[str] | None = None
    tested_causes: list[str] | None = None
    schema_policy: str = "union"

    lag_config: LagConfiguration | Mapping[str, Any] | None = None
    lag_selector: Any = None

    stationarity_transformer: Any = field(
        default_factory=StationarityTransformer
    )
    missing_value_imputer: Any = None
    sampling_strategy: Any = None

    backend_sample_fraction: float = 1.0
    backend_max_samples: int | None = None
    random_state: int | None = None

    x_scaler: Any = "standard"
    y_scaler: Any = "standard"

    relations: RelationConstraintsSpec = field(
        default_factory=RelationConstraintsSpec
    )
    constraint_spec: Any = field(
        default_factory=lambda: {
            "type": "mask_and_min_abs_sum",
            "params": {},
        }
    )
    regularizer_spec: Any = None

    optimizer_spec: Any = None
    loss_spec: Any = None
    callback_specs: list[Any] | None = None
    initializer_spec: Any = None

    model_config: dict[str, Any] = field(default_factory=dict)

    # Keep these generic until the existing hyperoptimization API is integrated.
    hyperoptimization_state: Any = None
    hyperoptimization_config: Any = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AnalysisConfig:
        """Create a configuration from the public mapping representation.

        Args:
            values: Analysis settings received from Python code or ConfigReader.

        Canonicalize public field aliases in one place, including existing names
        such as 'callbacks', 'initializer', and legacy hyperoptimization keys.
        Reject unknown fields and conflicting aliases rather than silently
        discarding them.

        Parse backend-independent domain specifications through existing parsers.
        Do not instantiate backend-native components, fit transformers, read
        referenced files, or resolve backend component names.
        """
        raise NotImplementedError

    def updated(self, values: Mapping[str, Any]) -> AnalysisConfig:
        """Return a configuration with explicitly supplied fields replaced.

        Args:
            values: Partial public configuration using the same accepted keys
                as from_mapping().

        Apply replacement semantics to fields; do not recursively merge arbitrary
        component parameter dictionaries. Preserve omitted fields and distinguish
        an omitted field from an explicitly supplied None.

        Copy configuration-owned containers. Preserve references to input data
        and user-provided runtime objects; do not deepcopy framework objects.
        Validate the resulting configuration before returning it.
        """
        raise NotImplementedError

    def snapshot(self) -> AnalysisConfig:
        """Return a structurally independent specification for one execution.

        Copy configuration-owned mappings, lists, and domain specifications as
        appropriate. Preserve references to input data and user-supplied runtime
        objects.

        The returned object isolates subsequent configuration edits; it does not
        guarantee isolation from external mutation of shared runtime objects.
        Do not create backend components or fitted preprocessing objects.
        """
        raise NotImplementedError

    def validate(self) -> None:
        """Validate backend-independent configuration consistency.

        Check field types, sampling limits, nonempty variable names, reserved
        model_config keys, and required project-level invariants.

        Do not inspect learned data layouts, instantiate components, or reject
        backend component names based on a hardcoded global catalog. Leave
        backend-native validation to the selected strategy.

        Allow incomplete configuration during fluent setup; execution readiness
        is checked separately by the executor.
        """
        raise NotImplementedError

    def preprocessing_kwargs(self) -> dict[str, Any]:
        """Return only arguments accepted by the existing prepare_data function.

        Include variable roles, lag settings, preprocessing settings, sampling
        settings, and random_state. Exclude data itself, model settings,
        component specifications, execution state, and result objects.

        Delegate preprocessing-spec resolution to the relevant preprocessing
        subsystem; never route preprocessing components through BackendStrategy.
        """
        raise NotImplementedError