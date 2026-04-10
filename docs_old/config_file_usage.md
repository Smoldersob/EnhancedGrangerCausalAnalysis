# Configuration Files for MultitaskGrangerBuilder

This document describes a simple JSON/YAML format for builder configuration and how to load it.

## Supported file types
- JSON (`.json`)
- YAML (`.yml`, `.yaml`) if `PyYAML` is installed

## Single-run config format
Top-level object maps directly to `MultiTaskGrangerAPI.fit(...)` argument names and builder keys.

Example (`single_run_config.json`):
```json
{
  "backend": "pytorch",
  "x_scaler": "standard",
  "y_scaler": "standard",
  "causes": ["x1", "x2"],
  "effects": ["y"],
  "tested_causes": ["x1", "x2"],
  "lag_config": {
    "max_lag": 8,
    "use_lag_zero": false,
    "custom_lags": {"x1": [3]},
    "custom_pair_lags": {"y,x2": [2, 6]}
  },
  "model_config": {
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001
  }
}
```

Notes:
- `lag_config` is automatically converted into `LagConfiguration` object.
- For YAML files, install `PyYAML` first.

## Loading with Builder
```python
from complex_granger_analysis.api import MultitaskGrangerBuilder

output = (
    MultitaskGrangerBuilder()
    .from_file("single_run_config.json")
    .data(df)
    .fit()
)
```

## Loading with API helper
```python
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder

cfg = BuilderConfigLoader.load_file("single_run_config.json")
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

## Error handling behavior
- Unsupported extension -> `DataValidationError`
- Missing file -> `DataValidationError`
- Invalid top-level format (not an object) -> `DataValidationError`
- YAML without `PyYAML` -> `DataValidationError`
