from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from enhanced_granger_analysis.core.exceptions import (
    StationarityNotFittedError,
    StationarityTestError,
)
from enhanced_granger_analysis.core.registry import Registry
from enhanced_granger_analysis.preprocessing.pipeline import (
    DataPreparationConfig,
    DataPreparer,
)
from enhanced_granger_analysis.preprocessing.stationarity import (
    ADFStationarityEnforcer,
    BaseStationarityEnforcer,
    KPSSStationarityEnforcer,
    register_stationarity_enforcer,
    resolve_stationarity_enforcer,
)


enforcers_module = importlib.import_module(
    "enhanced_granger_analysis.preprocessing.stationarity.enforcers"
)
registry_module = importlib.import_module(
    "enhanced_granger_analysis.preprocessing.stationarity.registry"
)
pipeline_module = importlib.import_module(
    "enhanced_granger_analysis.preprocessing.pipeline"
)


@pytest.fixture
def frames() -> list[pd.DataFrame]:
    return [
        pd.DataFrame(
            {
                "x": [1.0, 3.0, 6.0, 10.0, 15.0, 21.0],
                "y": [2.0, 5.0, 9.0, 14.0, 20.0, 27.0],
            },
            index=pd.Index(
                [10, 11, 12, 13, 14, 15],
                name="sample",
            ),
        ),
        pd.DataFrame(
            {
                "x": [100.0, 103.0, 107.0, 112.0, 118.0, 125.0],
                "y": [50.0, 54.0, 59.0, 65.0, 72.0, 80.0],
            },
            index=pd.Index(
                [100, 101, 102, 103, 104, 105],
                name="sample",
            ),
        ),
    ]


@pytest.fixture
def isolated_registry(monkeypatch):
    """Avoid changing the application's global registry during tests."""
    registry = Registry(label="stationarity enforcer")

    monkeypatch.setattr(
        registry_module,
        "STATIONARITY_REGISTRY",
        registry,
    )

    registry_module.register_stationarity_enforcer("adf")(
        ADFStationarityEnforcer
    )
    registry_module.register_stationarity_enforcer("kpss")(
        KPSSStationarityEnforcer
    )

    return registry


@pytest.fixture
def adf_order(monkeypatch):
    """Replace numerical estimation with a controllable deterministic helper."""
    calls = []
    orders = {"x": 1, "y": 2}

    def estimate(
        *,
        series,
        max_differencing_order,
        maxlag,
        alpha,
    ):
        calls.append(
            {
                "series": series.copy(),
                "max_differencing_order": max_differencing_order,
                "maxlag": maxlag,
                "alpha": alpha,
            }
        )
        return orders[series.name]

    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        estimate,
    )

    return orders, calls


def difference_frame(
    frame: pd.DataFrame,
    orders: dict[str, int],
    *,
    dropna: bool,
) -> pd.DataFrame:
    expected = frame.astype(float).copy()

    for name, order in orders.items():
        for _ in range(order):
            expected[name] = expected[name].diff()

    return expected.dropna() if dropna else expected


class RecordingEnforcer(BaseStationarityEnforcer):
    """Small custom component for registry and pipeline contract tests."""

    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, data_list):
        self.fit_calls += 1
        return self

    def transform(self, data_list):
        self.transform_calls += 1
        return [frame.copy() + self.offset for frame in data_list]


class FailingEnforcer(BaseStationarityEnforcer):
    def fit(self, data_list):
        raise RuntimeError("Deliberate fit failure.")

    def transform(self, data_list):
        return [frame.copy() for frame in data_list]


class AbstractEnforcer(BaseStationarityEnforcer):
    pass


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_differencing_order": -1},
        {"max_differencing_order": 1.5},
        {"max_differencing_order": True},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"alpha": -0.1},
        {"alpha": np.nan},
        {"alpha": np.inf},
        {"alpha": True},
        {"maxlag": -1},
        {"maxlag": 1.5},
        {"maxlag": True},
        {"dropna": "yes"},
        {"dropna": 1},
        {"exclude": "x"},
        {"exclude": ["x", "x"]},
        {"exclude": [1]},
    ],
)
@pytest.mark.parametrize(
    "enforcer_class",
    [ADFStationarityEnforcer, KPSSStationarityEnforcer],
)
def test_constructor_rejects_invalid_configuration(
    enforcer_class,
    kwargs,
):
    with pytest.raises(StationarityTestError):
        enforcer_class(**kwargs)


@pytest.mark.parametrize(
    "enforcer_class",
    [ADFStationarityEnforcer, KPSSStationarityEnforcer],
)
def test_constructor_accepts_zero_order_and_zero_maxlag(enforcer_class):
    enforcer = enforcer_class(
        max_differencing_order=0,
        maxlag=0,
        alpha=0.1,
        exclude=["x"],
        dropna=False,
    )

    assert enforcer.max_differencing_order == 0
    assert enforcer.maxlag == 0
    assert enforcer.alpha == 0.1
    assert enforcer.exclude == ("x",)
    assert enforcer.dropna is False


# ---------------------------------------------------------------------------
# Algorithm delegation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("enforcer_class", "helper_name", "other_helper_name", "method"),
    [
        (
            ADFStationarityEnforcer,
            "static_adfuller_order",
            "static_kpss_order",
            "adf",
        ),
        (
            KPSSStationarityEnforcer,
            "static_kpss_order",
            "static_adfuller_order",
            "kpss",
        ),
    ],
)
def test_enforcer_delegates_to_its_own_algorithm(
    monkeypatch,
    frames,
    enforcer_class,
    helper_name,
    other_helper_name,
    method,
):
    calls = []

    def selected_helper(**kwargs):
        calls.append(kwargs)
        return 1

    def unexpected_helper(**kwargs):
        pytest.fail("The other algorithm must not be called.")

    monkeypatch.setattr(
        enforcers_module,
        helper_name,
        selected_helper,
    )
    monkeypatch.setattr(
        enforcers_module,
        other_helper_name,
        unexpected_helper,
    )

    enforcer = enforcer_class(
        max_differencing_order=3,
        maxlag=2,
        alpha=0.1,
    )

    returned = enforcer.fit(frames)

    assert returned is enforcer
    assert len(calls) == 4
    assert enforcer.fit_result_.method == method
    assert enforcer.fit_result_.max_differencing_order == 3
    assert enforcer.fit_result_.alpha == 0.1

    for call in calls:
        assert call["max_differencing_order"] == 3
        assert call["maxlag"] == 2
        assert call["alpha"] == 0.1


# ---------------------------------------------------------------------------
# Fitting and transformation
# ---------------------------------------------------------------------------


def test_transform_requires_fit(frames):
    with pytest.raises(StationarityNotFittedError):
        ADFStationarityEnforcer().transform(frames)


def test_fit_aggregates_maximum_order_across_datasets(
    monkeypatch,
    frames,
):
    def estimate(*, series, **kwargs):
        if series.name == "x":
            return 1 if series.index[0] == 10 else 2
        return 0 if series.index[0] == 10 else 1

    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        estimate,
    )

    enforcer = ADFStationarityEnforcer(
        max_differencing_order=2,
    ).fit(frames)

    assert enforcer.feature_names_in_ == ("x", "y")
    assert enforcer.differencing_orders_ == {"x": 2, "y": 1}
    assert enforcer.fit_result_.differencing_orders == {
        "x": 2,
        "y": 1,
    }


@pytest.mark.parametrize("dropna", [True, False])
def test_transform_applies_orders_without_crossing_dataset_boundaries(
    frames,
    adf_order,
    dropna,
):
    originals = [frame.copy(deep=True) for frame in frames]
    enforcer = ADFStationarityEnforcer(
        dropna=dropna,
    ).fit(frames)

    transformed = enforcer.transform(frames)

    assert len(transformed) == len(frames)

    for original, source, output in zip(
        originals,
        frames,
        transformed,
    ):
        expected = difference_frame(
            original,
            {"x": 1, "y": 2},
            dropna=dropna,
        )

        assert_frame_equal(output, expected)
        assert_frame_equal(source, original)
        assert output is not source


def test_transform_does_not_estimate_orders_again(
    monkeypatch,
    frames,
    adf_order,
):
    enforcer = ADFStationarityEnforcer().fit(frames)

    def unexpected_estimation(**kwargs):
        pytest.fail("transform() must not estimate differencing orders.")

    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        unexpected_estimation,
    )

    transformed = enforcer.transform(frames)

    assert len(transformed) == 2


def test_fit_transform_matches_separate_fit_and_transform(
    frames,
    adf_order,
):
    direct = ADFStationarityEnforcer().fit_transform(frames)

    separate_enforcer = ADFStationarityEnforcer().fit(frames)
    separate = separate_enforcer.transform(frames)

    for actual, expected in zip(direct, separate):
        assert_frame_equal(actual, expected)


def test_excluded_variables_are_not_estimated_or_differenced(
    frames,
    adf_order,
):
    _, calls = adf_order

    enforcer = ADFStationarityEnforcer(
        exclude=["x"],
        dropna=False,
    )

    transformed = enforcer.fit_transform(frames)

    assert enforcer.differencing_orders_ == {"x": 0, "y": 2}
    assert all(call["series"].name == "y" for call in calls)

    for source, output in zip(frames, transformed):
        pd.testing.assert_series_equal(output["x"], source["x"])


def test_all_variables_can_be_excluded(frames, monkeypatch):
    def unexpected_estimation(**kwargs):
        pytest.fail("Excluded variables must not be estimated.")

    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        unexpected_estimation,
    )

    enforcer = ADFStationarityEnforcer(exclude=["x", "y"])
    transformed = enforcer.fit_transform(frames)

    assert enforcer.fit_result_ is not None
    assert enforcer.differencing_orders_ == {"x": 0, "y": 0}

    for actual, expected in zip(transformed, frames):
        assert_frame_equal(actual, expected)


def test_unknown_excluded_variable_is_rejected(frames):
    enforcer = ADFStationarityEnforcer(exclude=["missing"])

    with pytest.raises(StationarityTestError, match="Unknown excluded"):
        enforcer.fit(frames)


def test_fit_result_contains_an_independent_orders_mapping(
    frames,
    adf_order,
):
    enforcer = ADFStationarityEnforcer().fit(frames)

    enforcer.differencing_orders_["x"] = 0

    assert enforcer.fit_result_.differencing_orders["x"] == 1


def test_refit_replaces_previous_fitted_schema(frames, adf_order):
    enforcer = ADFStationarityEnforcer().fit(frames)

    changed = [frame.rename(columns={"y": "z"}) for frame in frames]

    # The fixture's helper uses column names, so exclude the renamed column.
    enforcer.exclude = ("z",)
    enforcer.fit(changed)

    assert enforcer.feature_names_in_ == ("x", "z")
    assert enforcer.differencing_orders_ == {"x": 1, "z": 0}
    assert "y" not in enforcer.fit_result_.differencing_orders


def test_failed_refit_clears_previous_fitted_state(frames, adf_order):
    enforcer = ADFStationarityEnforcer().fit(frames)

    with pytest.raises(StationarityTestError):
        enforcer.fit([])

    assert enforcer.feature_names_in_ == ()
    assert enforcer.differencing_orders_ == {}
    assert enforcer.fit_result_ is None

    with pytest.raises(StationarityNotFittedError):
        enforcer.transform(frames)


@pytest.mark.parametrize("order", [-1, 6, 1.5, True])
def test_invalid_estimator_output_is_rejected(
    monkeypatch,
    frames,
    order,
):
    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        lambda **kwargs: order,
    )

    with pytest.raises(
        StationarityTestError,
        match="Invalid differencing order",
    ):
        ADFStationarityEnforcer(
            max_differencing_order=5,
        ).fit(frames)


def test_transform_rejects_complete_observation_loss(
    monkeypatch,
):
    frame = pd.DataFrame({"x": [1.0, 2.0]})

    monkeypatch.setattr(
        enforcers_module,
        "static_adfuller_order",
        lambda **kwargs: 2,
    )

    enforcer = ADFStationarityEnforcer(
        max_differencing_order=2,
    ).fit([frame])

    with pytest.raises(
        StationarityTestError,
        match="No observations remain",
    ):
        enforcer.transform([frame])


# ---------------------------------------------------------------------------
# Input and schema validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_data",
    [
        [],
        pd.DataFrame({"x": [1.0, 2.0]}),
        [np.array([1.0, 2.0])],
        [pd.DataFrame()],
        [pd.DataFrame({"x": []})],
        [pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])],
        [pd.DataFrame({1: [1.0, 2.0]})],
        [pd.DataFrame({"x": ["not numeric", "data"]})],
        [pd.DataFrame({"x": [1.0, np.nan]})],
        [pd.DataFrame({"x": [1.0, np.inf]})],
        [pd.DataFrame({"x": [1.0, -np.inf]})],
    ],
)
def test_fit_rejects_invalid_input(invalid_data):
    with pytest.raises(StationarityTestError):
        ADFStationarityEnforcer().fit(invalid_data)


def test_fit_rejects_different_dataset_column_order(frames):
    invalid = [frames[0], frames[1][["y", "x"]]]

    with pytest.raises(
        StationarityTestError,
        match="identical columns",
    ):
        ADFStationarityEnforcer().fit(invalid)


def test_datasets_may_have_different_lengths(frames, adf_order):
    unequal = [frames[0], frames[1].iloc[:4]]

    enforcer = ADFStationarityEnforcer().fit(unequal)

    assert enforcer.fit_result_ is not None


@pytest.mark.parametrize(
    "columns",
    [
        ["y", "x"],
        ["x"],
        ["x", "y", "z"],
    ],
)
def test_transform_requires_fitted_columns_and_order(
    frames,
    adf_order,
    columns,
):
    enforcer = ADFStationarityEnforcer().fit(frames)

    changed = []
    for frame in frames:
        extended = frame.assign(z=0.0)
        changed.append(extended.loc[:, columns])

    with pytest.raises(
        StationarityTestError,
        match="fitted columns",
    ):
        enforcer.transform(changed)


# ---------------------------------------------------------------------------
# Registry and specification resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("adf", ADFStationarityEnforcer),
        ("kpss", KPSSStationarityEnforcer),
        (" ADF ", ADFStationarityEnforcer),
    ],
)
def test_resolve_registered_names(
    isolated_registry,
    name,
    expected_class,
):
    resolved = resolve_stationarity_enforcer(name)

    assert isinstance(resolved, expected_class)


def test_none_disables_resolution(isolated_registry):
    assert resolve_stationarity_enforcer(None) is None


def test_none_rejects_constructor_parameters(isolated_registry):
    with pytest.raises(ValueError, match="disabled"):
        resolve_stationarity_enforcer(None, alpha=0.1)


def test_mapping_specification_constructs_component(isolated_registry):
    spec = {
        "type": "adf",
        "params": {
            "max_differencing_order": 2,
            "alpha": 0.1,
        },
    }

    resolved = resolve_stationarity_enforcer(spec)

    assert isinstance(resolved, ADFStationarityEnforcer)
    assert resolved.max_differencing_order == 2
    assert resolved.alpha == 0.1
    assert spec == {
        "type": "adf",
        "params": {
            "max_differencing_order": 2,
            "alpha": 0.1,
        },
    }


def test_mapping_parameters_can_be_combined_with_distinct_kwargs(
    isolated_registry,
):
    resolved = resolve_stationarity_enforcer(
        {
            "type": "adf",
            "params": {"alpha": 0.1},
        },
        max_differencing_order=2,
    )

    assert resolved.alpha == 0.1
    assert resolved.max_differencing_order == 2


def test_duplicate_parameter_sources_are_rejected(isolated_registry):
    with pytest.raises(ValueError, match="supplied twice"):
        resolve_stationarity_enforcer(
            {
                "type": "adf",
                "params": {"alpha": 0.1},
            },
            alpha=0.05,
        )


@pytest.mark.parametrize(
    ("spec", "error_type"),
    [
        ({}, ValueError),
        ({"params": {}}, ValueError),
        ({"type": "adf", "extra": 1}, ValueError),
        ({"type": "adf", "params": None}, TypeError),
        ({"type": "adf", "params": []}, TypeError),
        ({"type": "adf", "params": {1: 2}}, TypeError),
        ({"type": None}, TypeError),
        ({"type": ADFStationarityEnforcer}, TypeError),
    ],
)
def test_invalid_mapping_specifications_are_rejected(
    isolated_registry,
    spec,
    error_type,
):
    with pytest.raises(error_type):
        resolve_stationarity_enforcer(spec)


def test_unknown_name_uses_registry_error(isolated_registry):
    # Use the registry's own exception type without duplicating its API here.
    with pytest.raises(Exception) as registry_error:
        isolated_registry.resolve("not_registered")

    with pytest.raises(type(registry_error.value)):
        resolve_stationarity_enforcer("not_registered")


def test_direct_class_does_not_require_registration(isolated_registry):
    resolved = resolve_stationarity_enforcer(
        RecordingEnforcer,
        offset=3.0,
    )

    assert isinstance(resolved, RecordingEnforcer)
    assert resolved.offset == 3.0


def test_abstract_class_cannot_be_resolved(isolated_registry):
    with pytest.raises(TypeError, match="abstract"):
        resolve_stationarity_enforcer(AbstractEnforcer)


@pytest.mark.parametrize("spec", [123, object(), dict])
def test_invalid_selection_types_are_rejected(
    isolated_registry,
    spec,
):
    with pytest.raises(TypeError):
        resolve_stationarity_enforcer(spec)


def test_each_class_resolution_creates_a_fresh_instance(
    isolated_registry,
):
    first = resolve_stationarity_enforcer("adf")
    second = resolve_stationarity_enforcer("adf")

    assert first is not second


def test_instance_resolution_copies_before_fitting(
    isolated_registry,
    frames,
):
    original = RecordingEnforcer(offset=2.0)

    resolved = resolve_stationarity_enforcer(original)
    resolved.fit_transform(frames)

    assert resolved is not original
    assert resolved.fit_calls == 1
    assert resolved.transform_calls == 1
    assert original.fit_calls == 0
    assert original.transform_calls == 0


def test_instance_rejects_constructor_parameters(isolated_registry):
    with pytest.raises(ValueError, match="existing enforcer instance"):
        resolve_stationarity_enforcer(
            RecordingEnforcer(),
            offset=2.0,
        )


def test_custom_registration_and_alias(isolated_registry):
    returned = register_stationarity_enforcer(
        "custom",
        "custom_alias",
    )(RecordingEnforcer)

    assert returned is RecordingEnforcer
    assert isolated_registry.resolve("custom") is RecordingEnforcer
    assert isolated_registry.resolve("custom_alias") is RecordingEnforcer

    resolved = resolve_stationarity_enforcer(
        "custom_alias",
        offset=4.0,
    )

    assert isinstance(resolved, RecordingEnforcer)
    assert resolved.offset == 4.0

@pytest.mark.parametrize(
    "value",
    [
        RecordingEnforcer(),
        AbstractEnforcer,
        object,
        lambda: RecordingEnforcer(),
    ],
)
def test_registration_requires_a_concrete_enforcer_class(
    isolated_registry,
    value,
):
    with pytest.raises(TypeError):
        register_stationarity_enforcer("invalid")(value)

    assert "invalid" not in isolated_registry


# ---------------------------------------------------------------------------
# Pipeline stage handling
# ---------------------------------------------------------------------------


def test_pipeline_requires_data_before_stationarity():
    preparer = DataPreparer()

    with pytest.raises(RuntimeError, match="set_data"):
        preparer.force_stationarity("adf")


def test_pipeline_rejects_stationarity_after_lagging(frames):
    preparer = DataPreparer().set_data(frames)
    preparer.prepared_data_.lag_engine = object()

    with pytest.raises(RuntimeError, match="before lag"):
        preparer.force_stationarity(None)


def test_pipeline_inherits_configured_specification(
    isolated_registry,
    frames,
):
    preparer = DataPreparer(
        config=DataPreparationConfig(
            stationarity=RecordingEnforcer(offset=2.0),
        ),
    ).set_data(frames)

    returned = preparer.force_stationarity()

    assert returned is preparer

    fitted = preparer.prepared_data_.stationarity_enforcer
    assert isinstance(fitted, RecordingEnforcer)
    assert fitted.fit_calls == 1
    assert fitted.transform_calls == 1

    for actual, original in zip(preparer.data_list, frames):
        assert_frame_equal(actual, original + 2.0)


def test_pipeline_explicit_none_overrides_configured_component(
    isolated_registry,
    frames,
):
    configured = RecordingEnforcer(offset=2.0)

    preparer = DataPreparer(
        config=DataPreparationConfig(stationarity=configured),
    ).set_data(frames)

    preparer.force_stationarity(None)

    assert preparer.prepared_data_.stationarity_enforcer is None
    assert configured.fit_calls == 0

    for actual, original in zip(preparer.data_list, frames):
        assert_frame_equal(actual, original)
        assert actual is not original


def test_pipeline_explicit_specification_overrides_config(
    isolated_registry,
    frames,
):
    preparer = DataPreparer(
        config=DataPreparationConfig(
            stationarity=RecordingEnforcer(offset=10.0),
        ),
    ).set_data(frames)

    preparer.force_stationarity(
        RecordingEnforcer(offset=1.0),
    )

    fitted = preparer.prepared_data_.stationarity_enforcer
    assert fitted.offset == 1.0

    for actual, original in zip(preparer.data_list, frames):
        assert_frame_equal(actual, original + 1.0)


def test_pipeline_resolves_mapping_during_execution(
    isolated_registry,
    frames,
):
    register_stationarity_enforcer("recording")(RecordingEnforcer)

    spec = {
        "type": "recording",
        "params": {"offset": 3.0},
    }

    preparer = DataPreparer(
        config=DataPreparationConfig(stationarity=spec),
    ).set_data(frames)

    assert preparer.prepared_data_.stationarity_enforcer is None

    preparer.force_stationarity()

    fitted = preparer.prepared_data_.stationarity_enforcer
    assert isinstance(fitted, RecordingEnforcer)
    assert fitted.offset == 3.0

    assert spec == {
        "type": "recording",
        "params": {"offset": 3.0},
    }


def test_pipeline_calls_resolver_with_uninterpreted_specification(
    monkeypatch,
    frames,
):
    spec = {
        "type": "custom",
        "params": {"arbitrary": "value"},
    }
    received = []
    component = RecordingEnforcer()

    def resolve(value):
        received.append(value)
        return component

    monkeypatch.setattr(
        pipeline_module,
        "resolve_stationarity_enforcer",
        resolve,
    )

    preparer = DataPreparer(
        config=DataPreparationConfig(stationarity=spec),
    ).set_data(frames)

    preparer.force_stationarity()

    assert len(received) == 1
    assert received[0] is spec
    assert component.fit_calls == 1
    assert preparer.prepared_data_.stationarity_enforcer is component


def test_pipeline_does_not_mutate_configured_instance(
    isolated_registry,
    frames,
):
    configured = RecordingEnforcer(offset=2.0)

    preparer = DataPreparer(
        config=DataPreparationConfig(stationarity=configured),
    ).set_data(frames)

    preparer.force_stationarity()

    assert configured.fit_calls == 0
    assert configured.transform_calls == 0
    assert (
        preparer.prepared_data_.stationarity_enforcer
        is not configured
    )


def test_pipeline_rebuild_uses_independent_fitted_instances(
    isolated_registry,
    frames,
):
    preparer = DataPreparer(
        config=DataPreparationConfig(
            stationarity=RecordingEnforcer(),
        ),
    )

    preparer.set_data(frames).force_stationarity()
    first = preparer.prepared_data_.stationarity_enforcer

    preparer.set_data(frames).force_stationarity()
    second = preparer.prepared_data_.stationarity_enforcer

    assert first is not second
    assert first.fit_calls == 1
    assert second.fit_calls == 1


def test_pipeline_failure_preserves_previous_data_and_component(
    isolated_registry,
    frames,
):
    preparer = DataPreparer().set_data(frames)
    preparer.force_stationarity(RecordingEnforcer(offset=1.0))

    previous_data = [
        frame.copy(deep=True)
        for frame in preparer.data_list
    ]
    previous_component = (
        preparer.prepared_data_.stationarity_enforcer
    )

    with pytest.raises(RuntimeError, match="Deliberate fit failure"):
        preparer.force_stationarity(FailingEnforcer())

    assert (
        preparer.prepared_data_.stationarity_enforcer
        is previous_component
    )

    for actual, expected in zip(
        preparer.data_list,
        previous_data,
    ):
        assert_frame_equal(actual, expected)


def test_disabled_stage_does_not_reverse_previous_transformation(
    isolated_registry,
    frames,
):
    preparer = DataPreparer().set_data(frames)
    preparer.force_stationarity(RecordingEnforcer(offset=2.0))

    preparer.force_stationarity(None)

    assert preparer.prepared_data_.stationarity_enforcer is None

    for actual, original in zip(preparer.data_list, frames):
        assert_frame_equal(actual, original + 2.0)


def test_set_data_resets_stationarity_execution_state(
    isolated_registry,
    frames,
):
    preparer = DataPreparer().set_data(frames)
    preparer.force_stationarity(RecordingEnforcer())

    assert preparer.prepared_data_.stationarity_enforcer is not None

    preparer.set_data(frames)

    assert preparer.prepared_data_.stationarity_enforcer is None