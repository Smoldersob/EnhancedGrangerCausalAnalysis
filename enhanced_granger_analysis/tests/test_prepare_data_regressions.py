import numpy as np
import pandas as pd
import pytest

from enhanced_granger_analysis.core.exceptions import DataValidationError
from enhanced_granger_analysis.preprocessing.missing import (
    ConstantImputer,
    ForwardFillImputer,
    MeanImputer,
)
from enhanced_granger_analysis.preprocessing.prepare_data import prepare_data
from enhanced_granger_analysis.preprocessing.sampling import StridedSampling
from enhanced_granger_analysis.preprocessing.pipeline import DataPreparer
from enhanced_granger_analysis.preprocessing.scaling import StandardScaler


def _dataframe() -> pd.DataFrame:
    values = np.arange(40, dtype=float).reshape(20, 2)
    return pd.DataFrame(values, columns=["x", "y"])


def test_prepare_data_skips_stationarity_when_transformer_is_none() -> None:
    prepared = prepare_data(
        _dataframe(),
        effects=["y"],
        lag_config=None,
        stationarity_transformer=None,
    )

    assert prepared.stationarity_transformer is None
    assert prepared.X_raw.shape[0] == prepared.y_raw.shape[0]


def test_prepare_data_maps_cause_names_to_lagged_feature_blocks() -> None:
    prepared = prepare_data(
        _dataframe(),
        causes=["y", "x"],
        tested_causes=["x"],
        effects=["y"],
        stationarity_transformer=None,
    )

    assert prepared.causes == ["y", "x"]
    assert prepared.tested_causes == ["x"]
    assert prepared.predictor_slices["x"] == slice(0, 12)
    assert prepared.predictor_slices["y"] == slice(12, 24)


def test_prepare_data_rejects_mismatched_segment_schema_by_default() -> None:
    first = _dataframe()
    second = first.rename(columns={"y": "target"})

    with pytest.raises(DataValidationError, match="expected"):
        prepare_data([first, second], stationarity_transformer=None)


def test_union_schema_is_opt_in() -> None:
    first = _dataframe()
    second = first.rename(columns={"y": "target"})

    prepared = prepare_data(
        [first, second],
        stationarity_transformer=None,
        schema_policy="union",
    )

    assert prepared.causes == ["x", "y", "target"]


def test_imputers_preserve_segment_boundaries() -> None:
    first = pd.DataFrame({"x": [1.0, None, 3.0]})
    second = pd.DataFrame({"x": [10.0, None, 30.0]})

    filled = ForwardFillImputer().fit_transform([first, second])
    assert filled[0]["x"].tolist() == [1.0, 1.0, 3.0]
    assert filled[1]["x"].tolist() == [10.0, 10.0, 30.0]


def test_imputers_support_constant_and_fitted_mean_strategies() -> None:
    data = [pd.DataFrame({"x": [1.0, None]}), pd.DataFrame({"x": [3.0, None]})]

    constant = ConstantImputer(value=-1.0).fit_transform(data)
    mean = MeanImputer().fit_transform(data)

    assert constant[0]["x"].tolist() == [1.0, -1.0]
    assert mean[0]["x"].tolist() == [1.0, 2.0]


def test_prepare_data_applies_explicit_imputer_before_preprocessing() -> None:
    data = _dataframe().astype(float)
    data.loc[3, "x"] = np.nan

    prepared = prepare_data(
        data,
        effects=["y"],
        stationarity_transformer=None,
        missing_value_imputer=MeanImputer(),
    )

    assert not np.isnan(prepared.X_raw).any()
    assert not np.isnan(prepared.y_raw).any()


def test_prepare_data_exposes_requested_and_effective_roles() -> None:
    data = _dataframe()
    data["duplicate"] = data["x"]
    data["constant"] = 1.0

    prepared = prepare_data(
        data,
        causes=["x", "duplicate", "constant"],
        tested_causes=["x", "duplicate", "constant"],
        effects=["y"],
        stationarity_transformer=None,
    )

    assert prepared.requested_roles.causes == ("x", "duplicate", "constant")
    assert prepared.requested_roles.tested_causes == ("x", "duplicate", "constant")
    assert prepared.effective_roles.causes == ("x",)
    assert prepared.effective_roles.tested_causes == ("x",)
    assert prepared.removed_variables["duplicate"].reason == "duplicate"
    assert prepared.removed_variables["duplicate"].retained_variable == "x"
    assert "cause" in prepared.removed_variables["duplicate"].removed_roles
    assert prepared.cause_slices == {"x": slice(0, 12)}


def test_prepare_data_rejects_tested_cause_outside_causes() -> None:
    with pytest.raises(DataValidationError, match="subset of causes"):
        prepare_data(
            _dataframe(),
            causes=["x"],
            tested_causes=["y"],
            effects=["y"],
            stationarity_transformer=None,
        )


def test_sampling_is_reproducible_and_preserves_selected_order() -> None:
    first = prepare_data(
        _dataframe(),
        effects=["y"],
        stationarity_transformer=None,
        backend_sample_fraction=0.5,
        random_state=17,
    )
    second = prepare_data(
        _dataframe(),
        effects=["y"],
        stationarity_transformer=None,
        backend_sample_fraction=0.5,
        random_state=17,
    )

    assert np.array_equal(first.sample_indices, second.sample_indices)
    assert np.all(first.sample_indices[:-1] <= first.sample_indices[1:])
    assert first.X_raw.shape[0] == first.sample_indices.size


def test_strided_sampling_and_sampling_parameter_validation() -> None:
    prepared = prepare_data(
        _dataframe(),
        effects=["y"],
        stationarity_transformer=None,
        backend_max_samples=4,
        sampling_strategy=StridedSampling(),
    )
    assert np.array_equal(prepared.sample_indices, np.array([0, 2, 4, 7]))
    with pytest.raises(DataValidationError, match="sample_fraction"):
        prepare_data(
            _dataframe(),
            effects=["y"],
            stationarity_transformer=None,
            backend_sample_fraction=0,
        )


def test_scalers_expose_fit_without_breaking_fit_transform() -> None:
    data = np.array([[1.0], [2.0], [3.0]])
    scaler = StandardScaler()

    assert scaler.fit(data) is scaler
    transformed = scaler.transform(data)
    assert np.allclose(scaler.inverse_transform(transformed), data)
    assert np.allclose(scaler.fit_transform(data), transformed)
    assert scaler.get_params()["eps"] == 1e-12


def test_data_preparer_supports_staged_pipeline_builder() -> None:
    prepared = (
        DataPreparer(stationarity_transformer=None)
        .set_data(_dataframe())
        .set_roles(causes=["x"], tested_causes=["x"], effects=["y"])
        .align()
        .imput()
        .remove_redundant()
        .force_stationarity()
        .scale()
        .reduce_workload()
        .lag()
    )

    assert prepared.effective_roles.causes == ("x",)
    assert prepared.effects == ["y"]
