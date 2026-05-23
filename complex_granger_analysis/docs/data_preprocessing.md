# Data Preprocessing

This document explains the components in the `preprocessing` folder, their purpose, and how they are used in the Granger analysis pipeline.

## Module Scope

The `preprocessing` package is split into three submodules:

- `preprocessing/stationarity`: stationarity testing and differencing.
- `preprocessing/lag`: lag selection and lagged feature matrix construction.
- `preprocessing/scaling`: deterministic scaling for inputs and targets.

In the standard pipeline, preprocessing stages are executed in this order:

1. Stationarity transform (`StationarityTransformer`)
2. Lag preparation (`LagEngine` + optional selector)
3. Scaling (`StandardScaler`/`MinMaxScaler`/`RobustScaler`/`MaxAbsScaler`/`IdentityScaler`)

## Stage-by-Stage Example

The example below shows all preprocessing stages explicitly before model fitting.

```python
import pandas as pd

from complex_granger_analysis.core.lag_config import LagConfiguration
from complex_granger_analysis.preprocessing.stationarity import StationarityTransformer
from complex_granger_analysis.preprocessing.lag import LagEngine, ICLagSelector
from complex_granger_analysis.preprocessing.scaling import StandardScaler

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
X_train, y_train, col_offsets = engine.prepare(data_stationary, effects=["x1", "x2"])

# Stage 3: scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_train)
y_scaled = y_scaler.fit_transform(y_train)

print(X_scaled.shape, y_scaled.shape, col_offsets)
```

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

- Operates on a list of DataFrames (segments are processed and concatenated).
- Supports fixed-lag mode (no selector) and auto-lag mode (with selector).
- Supports overrides from config (`custom_lags`, `custom_pair_lags`).
- Parallel processing with joblib (`n_jobs`).
- Returns `col_offsets` for predictor block mapping.
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
- `col_offsets`
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

In the main orchestrator (`MultiTaskGrangerAPI`):

- stationarity is computed and applied before lag generation,
- `LagEngine` produces `X_train`, `y_train`, and `col_offsets`,
- X and y scalers are applied before backend training,
- predictions are mapped back with `y_scaler.inverse_transform(...)`.

This keeps preprocessing consistent across all supported backends (`pytorch`, `tensorflow`, `sklearn`).

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
- [Backend Usage](backend_usage.md)
- [Configuration File Usage](config_file_usage.md)
- [Script Usage](script_usage.md)
