from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from .modeling_spec import ModelingSpec

from ..core.exceptions import DataValidationError

from ..preprocessing.pipeline import DataPreparationConfig
from ..preprocessing.prepare_data import VariableRoles

from ..backends import BackendFactory


def _role_names(
    values: Sequence[str] | str | None,
) -> tuple[str, ...]:
    """Represent requested names without resolving their effective roles."""

    if values is None:
        return ()

    if isinstance(values, str):
        return (values,)

    return tuple(values)


def _default_preparation() -> DataPreparationConfig:
    """Preserve the public analysis API preparation defaults."""

    return DataPreparationConfig(
        schema_policy="union",
        stationarity="adf",
    )


def _empty_roles() -> VariableRoles:
    """Leave role defaults to the existing preparation pipeline."""

    return VariableRoles(
        effects=(),
        causes=(),
        tested_causes=(),
    )


@dataclass(frozen=True)
class AnalysisSpec:
    """Configuration of one analysis without execution artifacts.

    Source data, prepared data, models, weights, and analysis results
    are supplied or produced separately.
    """

    preparation: DataPreparationConfig = field(
        default_factory=_default_preparation
    )
    roles: VariableRoles = field(
        default_factory=_empty_roles
    )
    modeling: ModelingSpec = field(
        default_factory=ModelingSpec
    )

    reuse_data: bool = False

    hiperoptimalization_state: Any = None
    hiperoptimalization_conf: Any = None

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> AnalysisSpec:
        """Read the flat analysis configuration without creating components."""

        if "stationarity_transformer" in config:
            raise DataValidationError(
                "stationarity_transformer is not supported. "
                "Use stationarity with an existing resolver specification."
            )

        if "regularizer" in config:
            raise DataValidationError(
                "The 'regularizer' configuration field is not supported. "
                "Use 'regularizer_spec'."
            )

        preparation_values = {
            item.name: config[item.name]
            for item in fields(DataPreparationConfig)
            if item.name in config
        }

        preparation_values.setdefault(
            "schema_policy",
            "union",
        )

        if "stationarity_transformer" not in preparation_values:
            preparation_values.setdefault(
                "schema_policy",
                "union",
            )
            preparation_values.setdefault(
                "stationarity",
                "adf",
            )

        if isinstance(
            preparation_values.get("lag_selector_params"),
            Mapping,
        ):
            preparation_values["lag_selector_params"] = dict(
                preparation_values["lag_selector_params"]
            )

        backend, model_config = BackendFactory.resolve_configuration(
            config.get("backend"),
            model_config=config.get("model_config"),
        )

        if "compute_device" in config:
            device = config["compute_device"]
        else:
            device = config.get("device")

        if device is not None:
            model_config.setdefault("device", device)

        modeling_values: dict[str, Any] = {
            "backend": backend,
            "model_config": model_config,
        }

        for name in (
            "constraint_spec",
            "relations",
            "regularizer_spec",
            "callbacks",
            "initializer",
        ):
            if name in config:
                modeling_values[name] = config[name]

        return cls(
            preparation=DataPreparationConfig(
                **preparation_values
            ),
            roles=VariableRoles(
                effects=_role_names(config.get("effects")),
                causes=_role_names(config.get("causes")),
                tested_causes=_role_names(
                    config.get("tested_causes")
                ),
            ),
            modeling=ModelingSpec(**modeling_values),
            reuse_data=bool(config.get("reuse_data", False)),
            hiperoptimalization_state=config.get(
                "hiperoptimalization_state"
            ),
            hiperoptimalization_conf=config.get(
                "hiperoptimalization_conf"
            ),
        )