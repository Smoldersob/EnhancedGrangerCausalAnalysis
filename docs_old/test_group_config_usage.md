# Test Group Configuration Format

This document describes a compact format for running many tests with parameter sweeps.

## Chosen approach
We use:
- `base_config`: one base builder config
- `sweep.param_names`: list of dotted parameter names
- `sweep.cases`: list of value rows (each row = one concrete test config)

This approach is easy to validate and easy to map to concrete configurations.

## Example (`group_config.json`)
```json
{
  "base_config": {
    "backend": "pytorch",
    "x_scaler": "standard",
    "y_scaler": "standard",
    "lag_config": {
      "max_lag": 8,
      "use_lag_zero": false
    },
    "model_config": {
      "epochs": 50,
      "batch_size": 32,
      "optimizer": "adam",
      "learning_rate": 0.001
    }
  },
  "sweep": {
    "param_names": [
      "model_config.optimizer",
      "model_config.learning_rate",
      "lag_config.max_lag"
    ],
    "cases": [
      ["adam", 0.001, 8],
      ["adam", 0.0005, 8],
      ["sgd", 0.01, 10]
    ]
  }
}
```

## Rules
- Every row in `sweep.cases` must have the same length as `sweep.param_names`.
- Dotted keys create/update nested mappings.
- If `sweep` is omitted, one configuration is produced from `base_config`.

## Iteration API
Use `TestGroupConfigIterator` to generate configurations one-by-one.

```python
from complex_granger_analysis.api import TestGroupConfigIterator, MultitaskGrangerBuilder

it = TestGroupConfigIterator.from_file("group_config.json")

while it.has_next():
    cfg = it.next()
    output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

## Methods
- `has_next()` -> `bool`: whether another config is available
- `next()` -> `dict`: returns next concrete config, raises `StopIteration` at the end

## Normalization
Returned configs are normalized:
- `lag_config` mapping is converted to `LagConfiguration`
