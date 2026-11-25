import numpy as np
import pandas as pd
import pytest

from ..core.data import PreparedData, ProcessRole, ValueType, VariableDescription, VariableRoles
from ..core.exceptions import DataValidationError
from ..preprocessing.configuration.preparation import DataPreparationConfig
from ..preprocessing.fitted import FittedPreprocessor
from ..preprocessing.missing import MeanImputer
from ..preprocessing.pipeline import DataPreparer
from ..preprocessing.splitting.scope import FitScope, FitSegment


def _frame(offset=0.0, *, index=None):
    x = np.arange(30, dtype=float) + offset
    return pd.DataFrame(
        {"x": x, "y": 2.0 * x + np.sin(x)},
        index=index,
    )


def _roles():
    return VariableRoles(effects=("y",), causes=("x",), tested_causes=("x",))


def _config(**overrides):
    values = {
        "lag_config": {"max_lag": 2},
        "stationarity": None,
        "x_scaler": "standard",
        "y_scaler": "standard",
    }
    values.update(overrides)
    return DataPreparationConfig(**values)


class _CountingScaler:
    fit_calls = 0

    def fit(self, data):
        type(self).fit_calls += 1
        self.mean_ = np.mean(data, axis=0)
        return self

    def transform(self, data):
        return np.asarray(data) - self.mean_

    def inverse_transform(self, data):
        return np.asarray(data) + self.mean_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


def test_fit_publishes_prepared_data_and_transform_reuses_the_fitted_layout():
    preparer = DataPreparer(config=_config())
    fitted = preparer.fit(_frame(), roles=_roles())

    assert fitted is preparer
    assert isinstance(preparer.fitted_preprocessor_, FittedPreprocessor)
    assert isinstance(preparer.prepared_data_, PreparedData)
    assert not hasattr(preparer.fitted_preprocessor_, "training_data")

    new_data = _frame(100.0).loc[:, ["y", "x"]]
    new_data["unplanned"] = np.arange(len(new_data))
    transformed = preparer.transform(new_data)

    assert isinstance(transformed, PreparedData)
    assert transformed.feature_layout == preparer.prepared_data_.feature_layout
    assert transformed.X_raw.shape[1] == preparer.prepared_data_.X_raw.shape[1]
    assert not np.array_equal(transformed.X_raw, preparer.prepared_data_.X_raw)
    assert transformed.variable_descriptions["x"].value_type is ValueType.CONTINUOUS


def test_transform_does_not_refit_scalers():
    _CountingScaler.fit_calls = 0
    preparer = DataPreparer(config=_config(
        x_scaler=_CountingScaler,
        y_scaler=_CountingScaler,
    ))
    preparer.fit(_frame(), roles=_roles())
    fit_calls = _CountingScaler.fit_calls

    preparer.transform(_frame(50.0))

    assert fit_calls == 2
    assert _CountingScaler.fit_calls == fit_calls


def test_transform_requires_fit_and_missing_required_schema_is_explicit():
    preparer = DataPreparer(config=_config())
    with pytest.raises(RuntimeError, match="fit"):
        preparer.transform(_frame())
    with pytest.raises(RuntimeError, match="no PreparedData"):
        _ = preparer.prepared_data_

    preparer.fit(_frame(), roles=_roles())
    with pytest.raises(DataValidationError, match="missing required columns"):
        preparer.transform(_frame().drop(columns="x"))


def test_union_transform_uses_the_fitted_imputer_for_missing_schema_columns():
    preparer = DataPreparer(config=_config(
        schema_policy="union",
        missing_value_imputer=MeanImputer(),
    ))
    preparer.fit(_frame(), roles=_roles())

    transformed = preparer.transform(_frame().drop(columns="x"))

    assert transformed.feature_layout == preparer.prepared_data_.feature_layout
    assert transformed.X_raw.shape[1] == preparer.prepared_data_.X_raw.shape[1]


def test_union_fit_rejects_missing_schema_columns_without_an_imputer():
    frames = [_frame(), _frame(1.0).drop(columns="x")]

    with pytest.raises(DataValidationError, match="fitted imputer is required"):
        DataPreparer(config=_config(schema_policy="union")).fit(
            frames,
            roles=_roles(),
        )


def test_failed_refit_does_not_replace_the_previous_fitted_state():
    preparer = DataPreparer(config=_config())
    preparer.fit(_frame(), roles=_roles())
    prior_result = preparer.prepared_data_
    prior_fitted = preparer.fitted_preprocessor_

    with pytest.raises(DataValidationError, match="Unknown variables"):
        preparer.fit(_frame(), roles=VariableRoles(
            effects=("missing",), causes=("x",), tested_causes=("x",)
        ))

    assert preparer.prepared_data_ is prior_result
    assert preparer.fitted_preprocessor_ is prior_fitted
    assert preparer.transform(_frame(5.0)).feature_layout == prior_result.feature_layout


def test_value_type_strategy_is_explicit_and_types_reach_prepared_data():
    descriptions = {
        "x": VariableDescription(
            "x", ValueType.CONTINUOUS, (ProcessRole.SETPOINT,)
        ),
        "y": VariableDescription("y", ValueType.CONTINUOUS),
    }
    prepared = DataPreparer(config=_config(variable_descriptions=descriptions)).fit(
        _frame(), roles=_roles()
    ).prepared_data_
    assert prepared.variable_descriptions["x"].process_roles == (ProcessRole.SETPOINT,)

    unsupported = DataPreparer(config=_config(
        variable_descriptions={
            "x": VariableDescription("x", ValueType.CATEGORICAL),
            "y": VariableDescription("y", ValueType.CONTINUOUS),
        }
    ))
    with pytest.raises(DataValidationError, match="No preprocessing strategy"):
        unsupported.fit(_frame(), roles=_roles())


def test_configuration_mutation_after_fit_does_not_change_fitted_layout():
    lag_config = {"max_lag": 2}
    preparer = DataPreparer(config=_config(lag_config=lag_config))
    preparer.fit(_frame(), roles=_roles())
    fitted_layout = preparer.prepared_data_.feature_layout

    lag_config["max_lag"] = 1
    transformed = preparer.transform(_frame(20.0))

    assert transformed.feature_layout == fitted_layout
    assert transformed.feature_lags == (1, 2)


def test_source_split_scope_keeps_validation_values_out_of_fit():
    splitter = {
        "type": "temporal_blocks",
        "params": {"parts": {"training": 0.6, "validation": 0.4}},
    }
    original = _frame()
    changed = original.copy()
    changed.iloc[18:, :] = changed.iloc[18:, :] * 1000.0

    first = DataPreparer(config=_config(splitter=splitter)).fit(
        original, roles=_roles()
    )
    second = DataPreparer(config=_config(splitter=splitter)).fit(
        changed, roles=_roles()
    )

    first_training = first.prepared_data_.indices_for("training")
    second_training = second.prepared_data_.indices_for("training")
    np.testing.assert_array_equal(first_training, second_training)
    np.testing.assert_allclose(
        first.prepared_data_.X_scaled[first_training],
        second.prepared_data_.X_scaled[second_training],
    )
    assert first.fitted_preprocessor_.scope.segments[0].row_stop == 18


def test_prepared_split_cannot_authorize_its_own_preprocessing_fit():
    preparer = DataPreparer(config=_config(splitter={
        "type": "random_rows",
        "params": {"parts": {"training": 0.7, "validation": 0.3}},
    }))
    with pytest.raises(DataValidationError, match="cannot authorize fitting"):
        preparer.fit(_frame(), roles=_roles())


def test_restricted_fit_scope_rejects_ambiguous_duplicate_source_labels():
    duplicated = _frame(index=pd.Index([0] * 30))
    scope = FitScope(segments=(FitSegment(0, 0, 18),))

    with pytest.raises(DataValidationError, match="unique source index labels"):
        DataPreparer(config=_config()).fit(
            duplicated,
            roles=_roles(),
            scope=scope,
        )
