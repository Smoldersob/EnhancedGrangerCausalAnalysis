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

## What can and cannot be swept

`TestGroupConfigIterator` applies sweep values using dotted keys and then calls `BuilderConfigLoader.normalize_builder_config(...)`.

Important implementation detail:
- Dotted updates work only through mappings (dictionaries/objects).
- List indexing is not supported by dotted keys (for example, `callbacks.0.patience` will not work).

### Supported (directly or conditionally)

You can sweep these fields:

- `lag_selector` and its parameters
  - Supported: yes.
  - Examples: `lag_selector.type`, `lag_selector.use_bic`, `lag_selector.max_lag`.
  - Note: supported selector types are normalized to built-ins (`ic`, `cv`, `var`).

- `lag_config` parameters
  - Supported: yes.
  - Examples: `lag_config.max_lag`, `lag_config.use_lag_zero`.

- `relations`
  - Supported: yes, but prefer replacing the full `relations` object/list per case.
  - `relations` is normalized from mapping or list form.
  - Nested per-relation dotted edits are practical only when `relations` is represented as a mapping of mappings.

- `model_config` parameters
  - Supported: yes.
  - Examples: `model_config.optimizer`, `model_config.learning_rate`, `model_config.epochs`, `model_config.batch_size`.

- `backend`
  - Supported: yes.
  - Examples: `backend` (string alias), or `backend.type`/`backend.params.*` when backend is object form.

- scalers
  - Supported: yes.
  - Examples: `x_scaler`, `y_scaler` (for example `standard`, `minmax`, `robust`, `maxabs`, `none`).

- callbacks list
  - Supported: yes, by replacing the entire `callbacks` value per case.
  - Works with callback specs normalized by backend loaders.

- callback parameters
  - Supported conditionally.
  - Works via dotted keys only when callbacks are modeled as named mapping nodes (not as list items).
  - In common list form, use full-list replacement per case instead of trying list-index paths.

- regularizer type and presence
  - Supported: yes.
  - You can sweep `regularizer` / `regularizer_spec` and set them to `null` to disable regularization in a case.
  - Supported regularizer types: `l1`, `lag_dependent_l1`.

- regularizer parameters
  - Supported: yes, for object form (for example `regularizer.type`, `regularizer.l1`).

### Not directly supported (or limited)

- Dotted paths through list indices are not supported.
  - Not supported: `callbacks.0.patience`, `relations.0.min_abs_sum`.
  - Use whole-object/list replacement in `cases` values.

- `initializer` as plain string in generic iterator/loader flow is not normalized automatically.
  - The orchestrator expects callable/class initializer.
  - In `scripts/run_group_causality_tests.py`, there is an additional script-level mapping from known strings (`ols`, `zeros`, `random...`) to initializer classes.
  - Outside that script-level mapping, pass initializer object/class directly from Python code (not JSON string).

### Practical recommendation

For sweep dimensions with many list/object internals (`callbacks`, complex `relations`), treat each case value as a full replacement object.
For scalar and simple nested options (`model_config.*`, `lag_config.*`, `lag_selector.*`, `x_scaler`, `y_scaler`, `backend`), dotted keys are the most robust approach.

## Notes

- If `sweep` is omitted, the iterator returns one config built from `base_config`.
- Extra top-level keys are preserved, so shared flags such as `reuse_data` are not lost.
- Returned configs are normalized, so `lag_config` becomes a `LagConfiguration` instance.
