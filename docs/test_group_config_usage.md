# Test Group Configuration Usage

This document explains how to use `TestGroupConfigIterator` for running a sweep of test configurations.

## Purpose

A test group file contains:
- one `base_config`,
- optional `sweep.param_names`,
- optional `sweep.cases`.

The iterator expands this into one concrete builder config per case.

## Example file

```json
{
  "reuse_data": true,
  "base_config": {
    "backend": "tensorflow",
    "lag_config": {
      "max_lag": 8,
      "use_lag_zero": false
    },
    "model_config": {
      "epochs": 50,
      "batch_size": 32
    }
  },
  "sweep": {
    "param_names": [
      "model_config.epochs",
      "lag_config.max_lag"
    ],
    "cases": [
      [5, 8],
      [10, 10],
      [20, 12]
    ]
  }
}
```

## How the iterator works

`TestGroupConfigIterator.from_file(...)` reads the JSON/YAML file, expands the sweep, and normalizes each generated config through `BuilderConfigLoader`.

That means each returned config is ready to pass into the builder:

```python
from complex_granger_analysis.api import TestGroupConfigIterator, MultitaskGrangerBuilder

it = TestGroupConfigIterator.from_file("group_config.json")

while it.has_next():
    cfg = it.next()
    output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

## `reuse_data`

The top-level `reuse_data` flag is preserved for every expanded test config.

Use it when you want the builder/orchestrator to keep prepared preprocessing artifacts instead of discarding them after the run.

Example:

```python
it = TestGroupConfigIterator.from_file("group_config.json")
cfg = it.next()

builder = MultitaskGrangerBuilder().from_config(cfg)
result = builder.data(df).fit()
```

If `reuse_data` is `true`, the orchestrator can return the prepared data in the output object when its reuse policy allows it.

## Dotted sweep keys

Dotted keys update nested configuration values:
- `model_config.epochs`
- `lag_config.max_lag`
- `model_config.optimizer`

This is the preferred format because it is simple to validate and easy to map back to nested builder config.

## Notes

- If `sweep` is omitted, the iterator returns one config built from `base_config`.
- Extra top-level keys are preserved, so shared flags such as `reuse_data` are not lost.
- Returned configs are normalized, so `lag_config` becomes a `LagConfiguration` instance.
