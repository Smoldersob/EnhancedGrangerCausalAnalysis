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

## Standard Preparation Order

`prepare_data(...)` creates a `DataPreparer` and executes these stages:

1. **Schema alignment** (`schema_policy`)
2. **Missing-value imputation**
3. **Redundancy removal** (constant and duplicate variables)
4. **Stationarity transform** (`StationarityTransformer`)
5. **Lag preparation** (`LagEngine` plus an optional selector)
6. **Scaling** (`x_scaler` and `y_scaler`)
7. **Backend workload reduction** (`sampling_strategy`,
   `backend_sample_fraction`, and `backend_max_samples`)

The order is significant. Stationarity is fitted before lagging, scalers are
fitted on the lagged arrays, and workload reduction occurs after scaling. The
same prepared arrays and metadata are then used by the base and all restricted
model runs.

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

`PreparedData` is the contract between preprocessing and model execution:

- `requested_roles`: roles requested by the caller before cleanup;
- `effective_roles`: roles after constants and duplicates are removed;
- `causes`: effective model inputs;
- `effects`: effective model outputs; an effect does not have to be a cause;
- `tested_causes`: effective causes for which restricted runs are created;
- `cause_slices`: mapping from each effective cause name to its lagged feature
  slice in `X_scaled` and `X_raw`;
- `base_mask`: lag/relationship mask used when constructing constraints;
- `removed_variables`: typed metadata describing removed variables, including
  reason, removed roles, and a retained duplicate where applicable;
- `sample_indices`: indices retained after workload reduction.

`tested_causes` must be a subset of `causes`. The order of `tested_causes` only
controls the order of restricted runs and result columns; it does not determine
the feature block. Always use `prepared.cause_slices[cause_name]` when selecting
an input block.

For multiple input segments, use `schema_policy="error"` when all schemas must
match exactly. Use `schema_policy="union"` to align to the union of columns,
with missing columns represented as missing values for the subsequent imputation
stage.

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
