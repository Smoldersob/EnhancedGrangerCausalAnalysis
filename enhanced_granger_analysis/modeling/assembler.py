from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from ..preprocessing.prepare_data import PreparedData

from ..backends.base_backend import BackendStrategy

from ..core.models.input_spec import ModelInputSpec
from ..core.models.protocol import GrangerModelProtocol

from ..core.components.constraint_parsing import (
    RelationsInput,
    parse_relations_spec,
)
from ..core.components.constraint_preparation import (
    prepare_constraint_data as compile_constraint_data,
)
from ..core.components.constraint_spec import PreparedConstraintData

from ..core.components.regularizer_parsing import parse_regularizer_spec
from ..core.components.regularizer_preparation import (
    prepare_regularization_layout,
)
from ..core.components.regularizer_spec import PreparedRegularizerData

from ..core.exceptions import (
    ConstraintConfigurationError,
    DataValidationError,
)


@dataclass(frozen=True)
class ModelAssembly:
    """Artifacts returned after successful model construction."""

    model: GrangerModelProtocol
    model_config: dict[str, Any]
    callbacks: list[Any]


class ModelAssembler:
    """Prepare model construction inputs without retaining execution state.

    Component parsing and layout preparation are delegated to the
    existing domain functions. Backend component resolution remains
    the responsibility of backend strategies and their resolvers.
    """

    def prepare_constraint_data(
        self,
        prepared: PreparedData,
        relations: RelationsInput,
    ) -> PreparedConstraintData:
        """Parse relation declarations and compile them against the feature layout."""

        spec = parse_relations_spec(relations)

        return compile_constraint_data(
            spec,
            cause_slices=prepared.cause_slices,
            output_names=prepared.effects,
            n_features=prepared.X_scaled.shape[1],
            base_mask=prepared.base_mask,
        )

    def prepare_regularizer_data(
        self,
        prepared: PreparedData,
        regularizer_spec: Any,
    ) -> PreparedRegularizerData | None:
        """Prepare backend-independent regularizer construction data."""

        spec = parse_regularizer_spec(regularizer_spec)

        if spec is None:
            return None

        layout = prepare_regularization_layout(
            cause_slices=prepared.cause_slices,
            output_names=prepared.effects,
            feature_lags=prepared.feature_lags,
            base_mask=prepared.base_mask,
        )

        if layout.n_features != prepared.X_scaled.shape[1]:
            raise DataValidationError(
                "Regularizer layout does not match "
                "the number of columns in X_scaled."
            )

        if layout.n_outputs != prepared.y_scaled.shape[1]:
            raise DataValidationError(
                "Regularizer layout does not match "
                "the number of columns in y_scaled."
            )

        return PreparedRegularizerData(
            layout=layout,
            params=spec.params,
        )

    def build_constraint(
        self,
        *,
        strategy: BackendStrategy,
        constraint_spec: Any,
        data: PreparedConstraintData,
    ) -> Any:
        """Construct a constraint through the backend strategy."""

        return strategy.build_constraint(
            constraint_spec,
            data=data,
        )

    def build_regularizer(
        self,
        *,
        strategy: BackendStrategy,
        regularizer_spec: Any,
        data: PreparedRegularizerData | None,
    ) -> Any:
        """Construct an optional regularizer through the backend strategy."""

        spec = parse_regularizer_spec(regularizer_spec)

        if spec is None:
            return None

        return strategy.build_regularizer(
            spec.selector,
            data=data,
        )

    def build_callbacks(
        self,
        *,
        strategy: BackendStrategy,
        callbacks: Any = None,
    ) -> list[Any]:
        """Construct callbacks through the backend strategy."""

        return strategy.build_callbacks(callbacks)

    
    def assemble(
        self,
        *,
        strategy: BackendStrategy,
        prepared: PreparedData,
        model_config: Mapping[str, Any] | None = None,
        callbacks: Any = None,
        constraint: Any = None,
        regularizer: Any = None,
    ) -> ModelAssembly:
        """Build a model using ready constraints and regularization.

        Model, optimizer, loss, and callback specifications are passed
        to the existing backend factories without local resolution.
        """

        config = dict(model_config or {})
        self._validate_model_config(config)

        if "model" in config:
            model_spec = config.pop("model")
        else:
            model_spec = config.pop("model_type", None)

        optimizer_spec = config.pop("optimizer", None)
        loss_spec = config.pop("loss", None)

        resolved_callbacks = self.build_callbacks(
            strategy=strategy,
            callbacks=callbacks,
        )

        optimizer = strategy.build_optimizer(optimizer_spec)
        loss = strategy.build_loss(loss_spec)

        input_spec = ModelInputSpec(
            n_features=prepared.X_scaled.shape[1],
            n_outputs=prepared.y_scaled.shape[1],
            causes=tuple(prepared.causes),
            effects=tuple(prepared.effects),
            cause_slices=prepared.cause_slices,
            feature_lags=tuple(prepared.feature_lags),
        )

        constructor_config = dict(config)
        constructor_config["input_spec"] = input_spec

        components = {
            "optimizer": optimizer,
            "loss": loss,
            "callbacks": resolved_callbacks,
            "regularizer": regularizer,
            "constraint": constraint,
        }

        model = strategy.build_model(
            model_spec,
            components=components,
            config=constructor_config,
        )

        return ModelAssembly(
            model=model,
            model_config=dict(config),
            callbacks=resolved_callbacks,
        )

    @staticmethod
    def _validate_model_config(config: Mapping[str, Any]) -> None:
        """Validate fields owned by the model assembly pipeline.

        Component-specific validation remains in the existing
        backend factories and resolvers.
        """

        for reserved in ("regularizer", "regularizer_spec"):
            if reserved in config:
                raise DataValidationError(
                    f"{reserved!r} must not be placed in model_config. "
                    "Use regularizer_spec or builder.regularization(...)."
                )

        for reserved in ("constraint", "constraint_spec"):
            if reserved in config:
                raise ConstraintConfigurationError(
                    f"{reserved!r} must not be placed in model_config. "
                    "Use constraint_spec or builder.constraint(...)."
                )

        for reserved in (
            "callbacks",
            "components",
            "input_spec",
            "n_features",
            "n_outputs",
        ):
            if reserved in config:
                raise DataValidationError(
                    f"{reserved!r} is managed by the builder and must "
                    "not be placed in model_config."
                )

        if "model" in config and "model_type" in config:
            raise DataValidationError(
                "Specify either 'model' or 'model_type', not both."
            )