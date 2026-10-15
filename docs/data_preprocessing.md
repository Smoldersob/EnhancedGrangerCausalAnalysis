# Data Preprocessing

This document explains the components in the `preprocessing` folder, their purpose,
and the data contract passed to the backend model.

## Module Scope

The `preprocessing` package contains:

- `preprocessing/prepare_data.py` and `preprocessing/pipeline.py`: the public
  preparation entry point and staged preparation pipeline.
- `preprocessing/stationarity`: stationarity testing and differencing.
- `preprocessing/lag`: lag selection and lagged feature matrix construction.
- `preprocessing/scaling`: deterministic scaling for inputs and targets.
- `preprocessing/missing` and `preprocessing/sampling`: optional imputation and
  sample-reduction strategies.

## Standard preparation order

`prepare_data(...)` creates a `DataPreparer` and executes:

1. Schema alignment.
2. Optional missing-value imputation.
3. Removal of constant and exactly duplicated variables.
4. Optional stationarity transformation.
5. Lagged input and target construction.
6. Input and target scaling.
7. Workload reduction through the configured sampling strategy.

Role assignment takes place before these stages.

The standard workflow fits scalers on the lagged arrays before workload
reduction. Consequently, reducing the workload does not reduce the data used
to fit those scalers.

The base model and all restricted runs use the same prepared observations and
effect ordering.

This workflow does not automatically create training, validation, and test
partitions. Applications requiring held-out evaluation must define the split
and transformation-fitting policy explicitly.

## Defaults differ between entry points

| Setting | `prepare_data()` | `DataPreparationConfig` | Analysis builder |
| --- | --- | --- | --- |
| Schema policy | `error` | `error` | `union` |
| Stationarity transformer | `None` | `None` | Default `StationarityTransformer` |
| Sampling | `NoSampling` | `NoSampling` | `None`, resolved to `NoSampling` |
| Input scaler | `standard` | `standard` | `standard` |
| Target scaler | `standard` | `standard` | `standard` |

Use explicit settings when reproducing an experiment across entry points.

For schema alignment:

- `error` requires matching column names and order across segments.
- `union` aligns segments to the union of columns. Absent columns become
  missing values; alignment alone does not impute them.

## Optional transformations

For `DataPreparer`, omitting a stage argument uses its configuration.
Passing `None` explicitly disables the optional transformation:

```python
preparer.imput(None)
preparer.force_stationarity(None)
```

For example:

```python
from enhanced_granger_analysis.preprocessing.pipeline import (
    DataPreparer,
    DataPreparationConfig,
)

preparer = DataPreparer(
    DataPreparationConfig(
        stationarity_transformer=None,
        missing_value_imputer=None,
    )
)
```

The analysis builder currently behaves differently:
`.stationarity(None)` creates a default transformer.

To disable stationarity through the builder configuration:

```python
builder.from_config({"stationarity_transformer": None})
```

For explicit no-op scaling, use `"identity"`:

```python
builder.scaling(x_scaler="identity", y_scaler="identity")
```

## Staged `DataPreparer` Example

Use `DataPreparer` when preprocessing should be inspected or modified before the
model is built. Its methods mutate one preparation object and return `self`, so
they can be chained or executed one at a time.

```python
import numpy as np
import pandas as pd

from enhanced_granger_analysis.core.lag_config import LagConfiguration
from enhanced_granger_analysis.preprocessing.lag import ICLagSelector
from enhanced_granger_analysis.preprocessing.missing import ConstantImputer
from enhanced_granger_analysis.preprocessing.pipeline import (
    DataPreparer,
    DataPreparationConfig,
)
from enhanced_granger_analysis.preprocessing.stationarity import (
    StationarityTransformer,
)

first = pd.DataFrame(
    {
        "cause": np.linspace(0.0, 1.0, 40),
        "context": np.sin(np.linspace(0.0, 4.0, 40)),
        "effect": np.linspace(1.0, 0.0, 40),
    }
)
second = first.copy()
second.loc[5, "context"] = np.nan

config = DataPreparationConfig(
    schema_policy="error",
    lag_config=LagConfiguration(max_lag=3, use_lag_zero=False),
    lag_selector=ICLagSelector(max_lag=3, use_bic=True),
    stationarity_transformer=StationarityTransformer(
        max_differencing_order=1,
        test_name="adf",
        alpha=0.05,
        dropna=True,
    ),
    missing_value_imputer=ConstantImputer(value=0.0),
    x_scaler="standard",
    y_scaler="standard",
)

preparer = DataPreparer(config=config)
preparer.set_data([first, second])
preparer.set_roles(
    causes=["cause", "context"],
    tested_causes=["cause"],
    effects=["effect"],
)

# These calls follow the same order as DataPreparer.fit().
preparer.align()
preparer.imput()
preparer.remove_redundant()
preparer.force_stationarity()
preparer.lag()
preparer.scale()
preparer.reduce_workload()

prepared = preparer.prepared_data_
print(preparer.get_report())
print(preparer.get_feature_map())
print(prepared.X_scaled.shape, prepared.y_scaled.shape)
print(prepared.cause_slices["cause"])
```

`prepared_data_` can be passed directly to the analysis builder. This avoids
running preprocessing a second time:

```python
from enhanced_granger_analysis.api import MultiTaskGrangerBuilder

builder = MultiTaskGrangerBuilder(backend="sklearn").prepared_data(prepared)
builder.prepare_data()
builder.prepare_components()
builder.build_model({"max_iter": 100})
builder.fit_base()
builder.fit_restricted_models()
results = builder.get_results()
```

The builder keeps the prepared arrays, feature slices, masks, scalers, and
role metadata in the `PreparedData` object used by every model run.

Supplying `PreparedData` bypasses preprocessing inside the builder.
The supplied arrays, role metadata, feature mappings, masks, and fitted
scalers are therefore authoritative for that execution.

Changing preprocessing settings on the builder does not transform an already
supplied `PreparedData` object. Rebuild prepared data explicitly when changing
the preprocessing workflow.

## Lag-and-Scaling Example

The example below shows the lag and scaling contracts explicitly. For the complete
standard order, including schema alignment, imputation, redundancy removal, and
workload reduction, use `prepare_data(...)` or
`MultiTaskGrangerBuilder.prepare_data()`.

```python
import pandas as pd

from enhanced_granger_analysis.core.lag_config import LagConfiguration
from enhanced_granger_analysis.preprocessing.stationarity import StationarityTransformer
from enhanced_granger_analysis.preprocessing.lag import LagEngine, ICLagSelector
from enhanced_granger_analysis.preprocessing.scaling import StandardScaler

# One or more datasets with the same schema
df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)
data_list = [df]

# Stage 1: stationarity
stationarity = StationarityTransformer(
    max_differencing_order=3,
    test_name="adf",
    alpha=0.05,
    dropna=True,
)
data_stationary = stationarity.fit_transform(data_list)

# Stage 2: lag preparation
selector = ICLagSelector(max_lag=12, use_bic=True)
engine = LagEngine(
    config=LagConfiguration(max_lag=12, use_lag_zero=False),
    selector=selector,
)
X_raw, y_raw, block_boundaries = engine.prepare(
    data_stationary,
    causes=["x1", "x2"],
    effects=["x1", "x2"],
)

# Stage 3: scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

print(X_scaled.shape, y_scaled.shape, block_boundaries)
```

`LagEngine.prepare()` returns block boundaries with `n_causes + 1` entries:
`block_boundaries[i]:block_boundaries[i + 1]` is the feature block for the
`i`-th cause. The higher-level `PreparedData` exposes the preferred named
mapping, `cause_slices`, so callers do not need to depend on cause order.

## Stationarity

Files:

- `preprocessing/stationarity/transformer.py`
- `preprocessing/stationarity/tests.py`

Main class: `StationarityTransformer`

Key capabilities:

- Per-feature differencing order selection using:
  - ADF (`test_name="adf"`)
  - KPSS (`test_name="kpss"`)
- Search range: `0..max_differencing_order`.
- Optional exclusion of selected variables from differencing via `non_stationary`.
- Consistent application of fitted differencing orders across a list of DataFrames.
- `dropna=True/False` control after differencing.

API:

- `fit_stationarity(data_list)`
- `transform(data_list)`
- `fit_transform(data_list)`
- `prepare_static(...)` (backward-compatible path)

Helper functions:

- `static_adfuller_order(...)`
- `static_kpss_order(...)`
- `apply_differencing(series, order)`

## Lag Processing

Files:

- `preprocessing/lag/lag_engine.py`
- `preprocessing/lag/lag_selectors.py`

Main class: `LagEngine`

Purpose:

- Determine lag structure,
- build lagged feature matrix,
- align targets with lagged predictors.

Key `LagEngine` capabilities:

- Operates on a list of DataFrames. Each segment is lagged independently before
  the resulting samples are combined; lags do not cross segment boundaries.
- Supports fixed-lag mode (no selector) and auto-lag mode (with selector).
- Supports overrides from config (`custom_lags`, `custom_pair_lags`).
- Parallel processing with joblib (`n_jobs`).
- Returns `block_boundaries` for positional predictor-block mapping.
- Stores selection artifacts (`mask_`, `selection_result_`).

Public lag selectors:

- `ICLagSelector` (AIC/BIC)
- `CVLagSelector` (cross-validation)
- `VARLagSelector` (VAR-based)

Shared interface:

- `BaseLagSelector.fit(X) -> LagSelectionResult`

`LagSelectionResult` includes, among others:

- `ar_lags`
- `pred_lag_matrix`
- `max_lags_per_pred`
- `col_offsets` in `LagSelectionResult` (the start offset of each predictor
  block used by the lag-selection mask)
- `mask`

## Scaling

File:

- `preprocessing/scaling/scalers.py`

All scalers implement:

- `fit_transform(data)`
- `transform(data)`
- `inverse_transform(data)`

Available implementations:

- `StandardScaler`: z-score (`(x-mean)/std`)
- `MinMaxScaler`: range scaling (default `[0,1]`)
- `RobustScaler`: median + IQR (outlier-robust)
- `MaxAbsScaler`: divides by max absolute value
- `IdentityScaler`: no-op transformation

Common behavior:

- Input validation to 2D arrays.
- Explicit fitted-state checks (`ScalerNotFittedError` when used before fit).
- `inverse_transform` to map outputs back to original scale.

## Standard and block-wise scaling

The standard `scale()` stage applies the configured scaler to the prepared
input array and independently scales the targets.

`block_scale()` is an alternative input-scaling stage. It constructs a
`BlockScaler` from `cause_slices`; target scaling remains separate.

```python
preparer.lag()
preparer.block_scale(x_scaler="standard", y_scaler="standard")
preparer.reduce_workload()
```

The standard `fit()` workflow calls `scale()`, not `block_scale()`.

## Workload reduction

`NoSampling` is the default strategy. To enable reduction, configure a
sampling strategy that selects a subset of observations.

For non-empty prepared data, the requested sample budget is calculated as:

```text
n_keep = max(1, floor(backend_sample_fraction * n_samples))
```

If `backend_max_samples` is set, it further limits that budget.
The configured strategy receives the requested budget and selects indices.

Selection must return valid, unique, strictly increasing integer indices.

Workload reduction is not a training/validation/test split.

The standard workflow samples after scaling. A manual workflow can sample
after lagging and before scaling, in which case the scalers are fitted on the
retained subset:

```python
preparer.lag()
preparer.reduce_workload()
preparer.scale()
```

Changing this order changes the data used to fit scaling parameters and should
be recorded as part of the experiment configuration.

## Integration with API

In the public preparation function and builder:

- `prepare_data(...)` returns a `PreparedData` object containing scaled and raw
  arrays, scalers, masks, role metadata, and feature slices;
- `MultiTaskGrangerBuilder.prepare_data()` stores that object and initializes
  the result context;
- `build_model()` passes the same scaled arrays to the backend's
  `initialize(X, targets=y)`;
- predictions are mapped back with
  `prepared_data.y_scaler.inverse_transform(...)`;
- `fit_base()` and `fit_restricted()` save snapshots directly to the result
  object.

This keeps preprocessing consistent across all supported backends (`pytorch`, `tensorflow`, `sklearn`).

## PreparedData contract

`PreparedData` connects preprocessing with model execution.

### Arrays and transforms

- `X_raw`: lagged input features before scaling.
- `y_raw`: aligned targets before scaling.
- `X_scaled`: model inputs after scaling.
- `y_scaled`: model targets after scaling.
- `x_scaler` and `y_scaler`: fitted scaling objects.
- `stationarity_transformer`: transformer used before lag construction,
  or `None`.
- `lag_engine`: lag-construction object and its selection artifacts.
- `base_mask`: configured lag-activity mask.

`raw` means unscaled at this preparation stage. It does not necessarily mean
the original input values: stationarity transformation may already have been
applied.

Inverse target scaling reverses scaling, not differencing.

### Variable roles

- `requested_roles`: input, output, and tested-cause roles requested before
  redundancy removal.
- `effective_roles`: roles remaining after cleanup.
- `causes`: effective model inputs.
- `effects`: effective model outputs.
- `tested_causes`: effective inputs tested through restricted runs.

`tested_causes` must be a subset of `causes`.

An effect does not have to belong to `causes`. In that case, its lagged history
is not automatically included in the model inputs, and role assignment emits
a warning.

Omitted or empty role lists are currently interpreted as defaults during
initial role assignment. They should not be used to request an intentionally
empty role.

If cleanup leaves no effective causes, effects, or tested causes, lag
preparation raises `DataValidationError`. This is an inability to execute the
requested analysis, not a statistical conclusion that causality is absent.

### Named input blocks

- `cause_slices`: maps each effective input name to its complete feature slice.
- `tested_cause_slices`: exposes mappings for effective tested causes.
- `predictor_slices`: compatibility alias; prefer `cause_slices` in new code.

Example:

```python
cause_slices = {
    "x1": slice(0, 2),
    "x2": slice(2, 4),
    "x3": slice(4, 6),
}
```

For three model inputs, this mapping contains three entries. Each slice
already stores both boundaries.

The equivalent positional boundary array is:

```python
block_boundaries =[0][2][4][6]
```

It contains four entries because adjacent boundaries define each block.

In general, a mapping for `n` inputs contains `n` slices, while a boundary
array contains `n + 1` endpoints.

Select an input block by name:

```python
block = prepared.cause_slices["x1"]
x1_features = prepared.X_scaled[:, block]
```

The order of `tested_causes` does not determine feature-block positions.

`PreparedData` does not expose `col_offsets` as its public mapping contract.
Lower-level lag-selection and component interfaces may still use positional
offsets internally.

### Removed variables

`removed_variables` maps names to `RemovedVariable` objects containing:

- `name`;
- `reason`: `"constant"` or `"duplicate"`;
- `removed_roles`;
- `retained_variable`, when the variable duplicates a retained signal.

Constant and duplicate diagnostics operate on the combined input segments.
Duplicate detection uses exact equality rather than a correlation threshold.

### How removal metadata is used by results

The preparation pipeline records removed variables but does not calculate
their causality results. The result layer uses this metadata when restoring
evaluated metric matrices to the requested roles.

Restoration is metric-dependent:

- newly added rows and columns use the metric's `removed_signal_value`;
- duplicate effect rows are copied from available fitted representatives;
- duplicate tested-cause columns are copied from available tested
  representatives.

A representative that was only a model input has no computed tested-cause
column to copy.

The sign metric uses the same restoration mechanism as other metrics, with
a fill value of `0`. Statistical metrics can define different fill values,
such as `1` for p-values or `0` for test statistics.

Restoration preserves the requested layout according to these rules.
It does not fit additional models or independently estimate signs for
removed variables.

### Observation tracking

`sample_indices` stores positions in the combined lagged arrays before
workload reduction. It is not a replacement for the original DataFrame index
and does not encode segment identifiers or timestamps.

Sampling updates raw arrays, already available scaled arrays, and
`sample_indices` using the same selected positions.

## When to Use Which Component

- `StationarityTransformer`: when data shows trend/non-stationarity and you need stable Granger inference.
- `LagEngine` + selector: when lag order should be data-adaptive.
- `LagEngine` without selector: when you need deterministic, fixed lag structure.
- `StandardScaler`: default choice for most workflows.
- `RobustScaler`: useful with strong outliers.
- `IdentityScaler`: when model/backend expects original scale.

## Related Documents

- [API Usage](api_usage.md)
- [Configuration File Usage](config_file_usage.md)
- [Script Usage](script_usage.md)
