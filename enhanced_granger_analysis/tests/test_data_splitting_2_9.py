"""Tests for data splitting and the fit/transform lag-engine contract (2.9)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ..core.exceptions import DataValidationError
from ..core.lag_config import (
    LagEngineConfig,
    LagFittedConfig,
    LaggedData,
    LagPreparationResult,
)
from ..preprocessing.lag import LagEngine
from ..preprocessing.pipeline import (
    DataPreparationConfig,
    DataPreparer,
)
from ..preprocessing.prepare_data import VariableRoles
from ..preprocessing.splitting import (
    DataSplit,
    RandomRowSplitter,
    StratifiedRandomRowSplitter,
    available_data_splitters,
    resolve_data_splitter,
)

ROLES = VariableRoles(
    causes=("a", "b", "c"),
    effects=("y1", "y2"),
    tested_causes=("a", "b"),
)


def make_datasets(seed: int = 0, n_rows: int = 60, n_datasets: int = 3):
    rng = np.random.default_rng(seed)
    frames = []

    for dataset_index in range(n_datasets):
        index = pd.RangeIndex(dataset_index * 1000, dataset_index * 1000 + n_rows)
        frame = pd.DataFrame(
            {
                "a": rng.normal(size=n_rows),
                "b": rng.normal(size=n_rows),
                "c": rng.normal(size=n_rows),
            },
            index=index,
        )
        frame["y1"] = frame["a"].shift(1).fillna(0.0) + 0.5 * rng.normal(size=n_rows)
        frame["y2"] = frame["b"].shift(2).fillna(0.0) + 0.5 * rng.normal(size=n_rows)
        frames.append(frame)

    return frames


def make_preparer(splitter, **config) -> DataPreparer:
    config = {"lag_config": {"max_lag": 2}, "splitter": splitter, **config}
    return DataPreparer(config=DataPreparationConfig(**config))


def fit_preparer(preparer: DataPreparer, data=None) -> DataPreparer:
    return preparer.fit(
        data if data is not None else make_datasets(),
        roles=ROLES,
    )


# ---------------------------------------------------------------------------
# Registry and resolution
# ---------------------------------------------------------------------------


def test_resolve_data_splitter_supports_none_name_mapping_class_and_instance():
    assert resolve_data_splitter(None) is None

    by_name = resolve_data_splitter("random_rows", parts={"training": 0.8})
    assert isinstance(by_name, RandomRowSplitter)

    by_mapping = resolve_data_splitter(
        {"type": "temporal_blocks", "params": {"parts": {"training": 0.5, "validation": 0.5}}}
    )
    assert by_mapping.parts == {"training": 0.5, "validation": 0.5}

    by_class = resolve_data_splitter(StratifiedRandomRowSplitter, parts={"training": 1.0})
    assert isinstance(by_class, StratifiedRandomRowSplitter)

    instance = RandomRowSplitter(parts={"training": 1.0})
    resolved = resolve_data_splitter(instance)
    assert resolved is not instance  # independent deepcopy


def test_resolve_data_splitter_rejects_unknown_names_and_duplicate_params():
    with pytest.raises(Exception):
        resolve_data_splitter("no_such_splitter")

    with pytest.raises(ValueError):
        resolve_data_splitter(
            {"type": "random_rows", "params": {"parts": {"training": 0.5}}},
            parts={"validation": 0.5},
        )


def test_available_splitters_lists_all_registered_names():
    names = available_data_splitters()
    for expected in (
        "dataset_level",
        "temporal_blocks",
        "random_rows",
        "stratified_random_rows",
    ):
        assert expected in names


def test_no_splitter_preserves_legacy_behavior():
    """No splitter assigns all prepared rows to training."""
    preparer = fit_preparer(make_preparer(None))
    prepared = preparer.prepared_data_
    split = prepared.split

    assert split is not None
    assert split.stage == "prepared"
    assert split.splitter_name is None
    assert split.part_names == ("training", "validation", "test")

    n_rows = prepared.X_raw.shape[0]
    assert n_rows > 0

    np.testing.assert_array_equal(
        split.training_indices,
        np.arange(n_rows, dtype=np.intp),
    )

    for part in ("validation", "test"):
        indices = split.indices_for(part)
        assert indices.shape == (0,)
        assert indices.dtype.kind in {"i", "u"}

    assert prepared.X_scaled.shape == prepared.X_raw.shape
    assert prepared.y_scaled.shape == prepared.y_raw.shape

    # Default standard scaling must still use all prepared rows.
    for raw, scaled in (
        (prepared.X_raw, prepared.X_scaled),
        (prepared.y_raw, prepared.y_scaled),
    ):
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)

        np.testing.assert_allclose(
            scaled,
            (raw - mean) / scale,
            rtol=1e-10,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Source-stage splitters
# ---------------------------------------------------------------------------


def test_temporal_blocks_assigns_prefix_and_suffix_per_dataset():
    """Assign each source prefix to training and suffix to validation."""
    data = make_datasets()

    preparer = fit_preparer(
        make_preparer({
            "type": "temporal_blocks",
            "params": {
                "parts": {
                    "training": 0.6,
                    "validation": 0.4,
                },
            },
        }),
        data=data,
    )

    prepared = preparer.prepared_data_
    split = prepared.split

    assert split is not None
    assert split.stage == "source"
    assert split.part_names == ("training", "validation", "test")
    assert split.indices_for("test").size == 0

    lagged = prepared.lag_engine.transform(data)
    expected_training = []
    expected_validation = []

    for segment in lagged.segments:
        frame = data[segment.dataset_index]

        source_positions = frame.index.get_indexer(
            segment.retained_index
        )
        assert np.all(source_positions >= 0)

        prepared_rows = np.arange(
            segment.prepared_row_slice.start,
            segment.prepared_row_slice.stop,
            dtype=np.intp,
        )

        assert prepared_rows.size == source_positions.size

        # The fixture has 60 rows: the 60% boundary is position 36.
        boundary = int(len(frame) * 0.6)
        training_mask = source_positions < boundary

        assert np.any(training_mask)
        assert np.any(~training_mask)

        expected_training.append(prepared_rows[training_mask])
        expected_validation.append(prepared_rows[~training_mask])

    np.testing.assert_array_equal(
        split.indices_for("training"),
        np.sort(np.concatenate(expected_training)),
    )
    np.testing.assert_array_equal(
        split.indices_for("validation"),
        np.sort(np.concatenate(expected_validation)),
    )

    np.testing.assert_array_equal(
        np.sort(np.concatenate([
            split.indices_for("training"),
            split.indices_for("validation"),
        ])),
        np.arange(prepared.X_raw.shape[0], dtype=np.intp),
    )


def test_temporal_blocks_fits_preprocessing_on_training_rows_only():
    # Validation rows carry a strong trend; training rows are stationary.
    # The fitted differencing orders must come from the training prefix.
    data = make_datasets()
    for frame in data:
        frame.loc[frame.index[36:], "a"] = np.cumsum(
            np.abs(np.random.default_rng(1).normal(size=24)) * 5
        ) + frame.loc[frame.index[36:], "a"]

    preparer = make_preparer(
        {"type": "temporal_blocks", "params": {"parts": {"training": 0.6, "validation": 0.4}}},
        stationarity="adf",
        lag_config={"max_lag": 2},
    )
    fit_preparer(preparer, data=data)

    enforcer = preparer.prepared_data_.stationarity_enforcer
    assert enforcer is not None
    assert enforcer.differencing_orders_["a"] == 0  # training prefix is stationary


def test_temporal_blocks_requires_fractions_summing_to_one():
    with pytest.raises(DataValidationError):
        preparer = make_preparer(
            {"type": "temporal_blocks", "params": {"parts": {"training": 0.6}}}
        )
        fit_preparer(preparer)


def test_dataset_level_assigns_whole_datasets():
    """Assign whole datasets without splitting their prepared rows."""
    data = make_datasets()

    preparer = fit_preparer(
        make_preparer({
            "type": "dataset_level",
            "params": {
                "parts": {
                    "training": 2,
                    "validation": 1,
                },
            },
        }),
        data=data,
    )

    prepared = preparer.prepared_data_
    split = prepared.split

    assert split is not None
    assert split.stage == "source"
    assert split.part_names == ("training", "validation", "test")
    assert split.indices_for("test").size == 0

    engine = prepared.lag_engine
    assert engine.fitted_config_ is not None

    training = split.indices_for("training")
    validation = split.indices_for("validation")
    lagged = engine.transform(data)

    dataset_owners = {}

    for segment in lagged.segments:
        segment_rows = np.arange(
            segment.prepared_row_slice.start,
            segment.prepared_row_slice.stop,
            dtype=np.intp,
        )
        assert segment_rows.size > 0

        in_training = np.isin(segment_rows, training)
        in_validation = np.isin(segment_rows, validation)

        assert np.all(in_training ^ in_validation)

        if np.all(in_training):
            owner = "training"
        elif np.all(in_validation):
            owner = "validation"
        else:
            pytest.fail(
                f"Dataset {segment.dataset_index} was split "
                "between training and validation."
            )

        previous_owner = dataset_owners.setdefault(
            segment.dataset_index,
            owner,
        )
        assert previous_owner == owner

    assert set(dataset_owners) == set(range(len(data)))
    assert list(dataset_owners.values()).count("training") == 2
    assert list(dataset_owners.values()).count("validation") == 1

    np.testing.assert_array_equal(
        np.sort(np.concatenate([training, validation])),
        np.arange(prepared.X_raw.shape[0], dtype=np.intp),
    )


def test_dataset_level_explicit_positions_and_validation_errors():
    preparer = fit_preparer(
        make_preparer(
            {"type": "dataset_level", "params": {"parts": {"training": [0, 2], "validation": [1]}}}
        )
    )
    assert preparer.prepared_data_.split.parts["training"].size > 0

    with pytest.raises(DataValidationError):
        fit_preparer(
            make_preparer(
                {"type": "dataset_level", "params": {"parts": {"training": [0, 1], "validation": [1]}}}
            )
        )

    with pytest.raises(DataValidationError):
        fit_preparer(
            make_preparer(
                {"type": "dataset_level", "params": {"parts": {"training": 5, "validation": 1}}}
            )
        )


def test_source_splitters_reject_duplicated_index_labels():
    data = make_datasets()
    data[0].index = pd.Index([0] * len(data[0]))

    with pytest.raises(DataValidationError):
        fit_preparer(
            make_preparer(
                {"type": "temporal_blocks", "params": {"parts": {"training": 0.6, "validation": 0.4}}}
            ),
            data=data,
        )


# ---------------------------------------------------------------------------
# Prepared-stage splitters
# ---------------------------------------------------------------------------


def test_random_rows_warns_about_information_leakage_and_is_deterministic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preparer = fit_preparer(
            make_preparer(
                {"type": "random_rows", "params": {"parts": {"training": 0.8, "validation": 0.2}, "random_state": 11}}
            )
        )

    assert any("information leakage" in str(item.message) for item in caught)

    split = preparer.prepared_data_.split
    assert split.stage == "prepared"

    n_rows = preparer.prepared_data_.X_raw.shape[0]
    assert split.parts["training"].size == int(round(0.8 * n_rows))
    assert split.parts["validation"].size == n_rows - split.parts["training"].size

    preparer_repeat = fit_preparer(
        make_preparer(
            {"type": "random_rows", "params": {"parts": {"training": 0.8, "validation": 0.2}, "random_state": 11}}
        )
    )
    np.testing.assert_array_equal(
        split.parts["training"],
        preparer_repeat.prepared_data_.split.parts["training"],
    )


def test_stratified_random_rows_preserves_proportions_per_series():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preparer = fit_preparer(
            make_preparer(
                {"type": "stratified_random_rows", "params": {"parts": {"training": 0.75, "validation": 0.25}, "random_state": 5}}
            )
        )

    assert any("information leakage" in str(item.message) for item in caught)

    split = preparer.prepared_data_.split
    engine = preparer.prepared_data_.lag_engine
    lagged = engine.transform(make_datasets())

    for segment in lagged.segments:
        segment_rows = np.arange(
            segment.prepared_row_slice.start,
            segment.prepared_row_slice.stop,
        )
        in_training = np.isin(segment_rows, split.parts["training"])
        in_validation = np.isin(segment_rows, split.parts["validation"])

        assert np.all(in_training | in_validation)
        assert abs(in_training.mean() - 0.75) < 0.34  # per-series proportion


def test_prepared_splitters_require_fractions():
    with pytest.raises(DataValidationError):
        StratifiedRandomRowSplitter(parts={"training": 100})


# ---------------------------------------------------------------------------
# Scalers and sampling under an active split
# ---------------------------------------------------------------------------


def test_scalers_are_fitted_on_training_rows_only():
    preparer = fit_preparer(
        make_preparer(
            {"type": "temporal_blocks", "params": {"parts": {"training": 0.6, "validation": 0.4}}}
        )
    )

    split = preparer.prepared_data_.split
    X_scaled = preparer.prepared_data_.X_scaled

    training_mean = X_scaled[split.training_indices].mean(axis=0)
    assert np.allclose(training_mean, 0.0, atol=1e-10)

    validation_mean = X_scaled[split.parts["validation"]].mean(axis=0)
    assert np.any(np.abs(validation_mean) > 1e-3)


def test_reduce_workload_rejects_row_removal_under_active_split():
    preparer = fit_preparer(
        make_preparer(
            {"type": "random_rows", "params": {"parts": {"training": 0.8, "validation": 0.2}}}
        )
    )

    with pytest.raises(DataValidationError):
        preparer.reduce_workload(backend_sample_fraction=0.5)

    # Keeping all rows remains valid and must not invalidate the split.
    preparer.reduce_workload()
    assert preparer.prepared_data_.split.training_indices.size > 0


# ---------------------------------------------------------------------------
# LagEngine fit/transform contract
# ---------------------------------------------------------------------------


def test_lag_engine_fit_and_transform_separate_selection_from_matrices():
    datasets = make_datasets()
    training = [frame.iloc[:40] for frame in datasets]

    engine = LagEngine(config=LagEngineConfig(max_lag=2), selector=None)
    fitted = engine.fit(training, outputs=["y1", "y2"], predictors=["a", "b", "c"])

    assert isinstance(fitted, LagFittedConfig)
    assert engine.fitted_config_ is fitted
    assert fitted.selection_results == ()

    lagged = engine.transform(datasets)
    assert isinstance(lagged, LaggedData)
    assert lagged.X.shape == (3 * 58, 6)
    assert len(lagged.segments) == 3

    for segment in lagged.segments:
        assert segment.retained_index[0] == segment.retained_index[0]  # labels preserved


def test_lag_engine_transform_requires_fit():
    engine = LagEngine(config=LagEngineConfig(max_lag=2), selector=None)

    with pytest.raises(RuntimeError):
        engine.transform(make_datasets())


def test_lag_engine_prepare_returns_legacy_result():
    engine = LagEngine(config=LagEngineConfig(max_lag=2), selector=None)
    result = engine.prepare(
        make_datasets(), outputs=["y1", "y2"], predictors=["a", "b", "c"]
    )

    assert isinstance(result, LagPreparationResult)
    assert result.X.shape[1] == 6
    assert len(result.segments) == 3


def test_lag_engine_transform_validates_schema():
    engine = LagEngine(config=LagEngineConfig(max_lag=2), selector=None)
    engine.fit(make_datasets(), outputs=["y1", "y2"], predictors=["a", "b", "c"])

    datasets = make_datasets()
    datasets[0] = datasets[0].drop(columns=["c"])

    with pytest.raises(ValueError):
        engine.transform(datasets)


# ---------------------------------------------------------------------------
# DataSplit artifact
# ---------------------------------------------------------------------------


def test_data_split_validation_rejects_bad_parts():
    """Reject malformed assignments and normalize optional parts."""
    bad_parts = [
        {},
        {"training": np.array([0, 0], dtype=np.intp)},
        {"training": np.array([2, 1], dtype=np.intp)},
        {"training": np.array([-1, 0], dtype=np.intp)},
        {"training": np.array([0, 5], dtype=np.intp)},
        {"training": np.array([0.0, 1.0], dtype=float)},
        {"training": np.array([[0, 1]], dtype=np.intp)},
        {
            "training": np.array([0, 1], dtype=np.intp),
            "validation": np.array([1, 2], dtype=np.intp),
        },
    ]

    for parts in bad_parts:
        with pytest.raises(DataValidationError):
            DataSplit(
                stage="prepared",
                splitter_name="x",
                parts=parts,
            ).validate(5)

    # An omitted optional part is normalized to an empty integer array.
    good = DataSplit(
        stage="prepared",
        splitter_name="x",
        parts={
            "training": np.array([0, 1, 2], dtype=np.intp),
            "validation": np.array([3, 4], dtype=np.intp),
        },
    ).validate(5)

    assert good.part_names == ("training", "validation", "test")

    np.testing.assert_array_equal(
        good.training_indices,
        np.array([0, 1, 2], dtype=np.intp),
    )
    np.testing.assert_array_equal(
        good.indices_for("validation"),
        np.array([3, 4], dtype=np.intp),
    )

    assert good.indices_for("test").shape == (0,)
    assert good.indices_for("test").dtype.kind in {"i", "u"}

    # Explicitly empty validation and test parts are also valid.
    training_only = DataSplit(
        stage="prepared",
        splitter_name=None,
        parts={
            "training": np.arange(5, dtype=np.intp),
            "validation": np.empty(0, dtype=np.intp),
            "test": np.empty(0, dtype=np.intp),
        },
    ).validate(5)

    assert training_only.part_names == (
        "training",
        "validation",
        "test",
    )
    np.testing.assert_array_equal(
        training_only.training_indices,
        np.arange(5, dtype=np.intp),
    )
    assert training_only.indices_for("validation").size == 0
    assert training_only.indices_for("test").size == 0
