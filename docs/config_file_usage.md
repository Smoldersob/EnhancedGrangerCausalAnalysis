# Configuration File Usage

This document explains how to load a builder configuration file and pass it into the builder/orchestrator pipeline.

For the generic component spec format used by callbacks, regularizers, and constraints, see [components loading](components_loading.md).

## What the loader does

`BuilderConfigLoader` is responsible for normalization only. It does not run training and does not create the final analysis result.

It can:
- load JSON or YAML files,
- normalize `lag_config` into `LagConfiguration`,
- normalize relation constraints,
- validate and normalize backend aliases,
- preserve backend-specific callback specs for later resolution,
- normalize relation-based constraint declarations into the `relations` mapping,
- move backend defaults from a backend spec into `model_config` when present,
- normalize `initializer` strings to initializer classes (e.g., `"ols"`, `"zeros"`, `"random_normal"`),
- normalize `compute_device` strings to backend-specific device configuration.

The loader also accepts `relations` as a file path when you want to keep constraint rules in a separate JSON/YAML file.

## Supported backend forms

```json
{
  "backend": "pytorch"
}
```

or

```json
{
  "backend": {
    "type": "pytorch",
    "params": {
      "loading_verbose": true
    }
  }
}
```

The backend spec form is useful when you want to keep loader-level settings close to the config file while still letting the backend strategy build the concrete runtime objects.

## Variable Selection (causes, effects, tested_causes)

Three optional fields control which variables are used in the analysis:

- **`causes`** — The full set of predictor variables available to the model. If omitted or set to `null`, all columns in the data are used.
- **`effects`** — The output variables to predict. If omitted or set to `null`, all columns in the data are used.
- **`tested_causes`** — A subset of `causes` for which Granger causality is formally tested (via the reference loop). If omitted or set to `null`, defaults to the value of `causes`.

### Behavior when omitted or `null`

```
causes=null       → all columns
effects=null      → all columns
tested_causes=null → defaults to causes (which could be all columns if causes=null)
```

This distinction is useful when you want the model to see all available context (full `causes`) but test causality only for a specific subset (`tested_causes`), saving computation while preserving model accuracy.

### Example with variable selection

```json
{
  "causes": ["u", "f1", "f2", "x1", "x2"],
  "effects": ["x3", "x4"],
  "tested_causes": ["f1", "f2"],
  "model_config": { "epochs": 50 }
}
```

In this case:
- The model uses 5 input variables: `u`, `f1`, `f2`, `x1`, `x2`.
- It predicts 2 outputs: `x3`, `x4`.
- Causality is tested only for `f1→x3`, `f1→x4`, `f2→x3`, `f2→x4`, saving time on the reference loop for `u`, `x1`, `x2`.

## Example configuration

```json
{
  "backend": "tensorflow",
  "reuse_data": true,
  "lag_config": {
    "max_lag": 8,
    "use_lag_zero": false
  },
  "callbacks": [
    {
      "type": "early_stopping",
      "monitor": "loss",
      "patience": 5
    }
  ],
  "relations": {
    "y->x1": {
      "zero": true
    },
    "y->x2": {
      "min_abs_sum": 0.2
    }
  },
  "model_config": {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam"
  }
}
```

Or keep the relation rules in a separate file and point `relations` at it:

```json
{
  "relations": "./relations.json"
}
```

The referenced file may contain either the list-of-objects form or the mapping form.

## Group Test Configuration

Use a group-test file when you want to run multiple cases from one shared base config.

Example:

```json
{
  "reuse_data": true,
  "compute_device": "cpu",
  "base_config": {
    "backend": "tensorflow",
    "lag_config": {
      "max_lag": 8,
      "use_lag_zero": false
    },
    "model_config": {
      "epochs": 50,
      "batch_size": 32
    },
    "relations": "./relations.json"
  },
  "sweep": {
    "param_names": ["model_config.epochs", "lag_config.max_lag"],
    "cases": [[5, 8], [10, 10], [20, 12]]
  }
}
```

Important fields:

- `reuse_data` keeps prepared preprocessing artifacts available across compatible cases. It is a top-level flag copied into each expanded config, not a `sweep.param_names` target.
- `compute_device` may be `cpu`, `gpu`, or `auto`.
- `base_config.relations` may be an inline mapping/list or a path to a JSON/YAML file.
- `sweep.param_names` uses dotted keys for nested values.
- `sweep.cases` must match the order and length of `param_names`.

The group iterator expands the file into specific builder configs and then normalizes them.

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

- `causes`, `effects`, `tested_causes`
  - Supported: yes.
  - You can sweep these as lists of variable names.
  - Examples: `causes: ["u", "f1", "x1"]`, `effects: ["x3"]`, `tested_causes: ["f1"]`.
  - Defaults: if not specified or set to `null`, `causes` and `effects` default to all columns in data; `tested_causes` defaults to `causes`.
  - Useful for comparing the impact of different variable subsets on causality results without retraining on multiple datasets.

- `compute_device`
  - Supported: yes.
  - Examples: `compute_device: "cpu"`, `compute_device: "gpu"`, `compute_device: "auto"`.
  - For PyTorch this is forwarded to `model_config.device`; for TensorFlow it maps to the runtime CPU/GPU environment flags.
  - `auto` is supported: it leaves backend defaults in place, so PyTorch auto-selects CUDA when available and TensorFlow uses its normal runtime device selection.

- `initializer`
  - Supported: yes.
  - String specs are automatically normalized to initializer classes via `BuilderConfigLoader`.
  - Supported strings (case-insensitive): `"ols"`, `"zeros"`, `"random_normal"` and their aliases (`"olsinitializer"`, `"zero"`, `"zerosinitializer"`, `"randomnormal"`, `"random"`, `"randomnormalinitializer"`).
  - Examples: `initializer: "ols"`, `initializer: "zeros"`, `initializer: "random_normal"`.
  - You can also pass an initializer object/class directly from Python code (already resolved).
  - `null` is accepted and means no custom weight initialization; each backend uses its defaults (PyTorch: Kaiming, TensorFlow: Glorot, scikit-learn: zeros).

- `relations`
  - Supported: yes.
  - You can provide a full relation object/list inline or point `relations` to a JSON/YAML file path.
  - The file is resolved relative to the group-config file.
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

### Practical recommendation

For sweep dimensions with many list/object internals (`callbacks`, complex `relations`), treat each case value as a full replacement object.
For scalar and simple nested options (`model_config.*`, `lag_config.*`, `lag_selector.*`, `x_scaler`, `y_scaler`, `backend`), dotted keys are the most robust approach.

## How to load and use it

```python
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder

cfg = BuilderConfigLoader.load_file("config.json")
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

You can also load the raw mapping first and then adapt it yourself if needed:

```python
from complex_granger_analysis.api import BuilderConfigLoader

cfg = BuilderConfigLoader.load_file("config.json")
```

Then pass the normalized mapping to `MultitaskGrangerBuilder.from_config(...)`.

## Constraints

Constraints for Granger analysis are specified via `relations`. This is a mapping of `(effect, cause)` pairs to constraint rules, and the loader also accepts JSON-friendly string notation or a list of objects.

Examples:

```json
{
  "relations": {
    "y->x1": { "zero": true },
    "y->x2": { "min_abs_sum": 0.2 }
  }
}
```

and:

```json
{
  "relations": [
    { "effect": "y", "cause": "x1", "zero": true },
    { "effect": "y", "cause": "x2", "min_abs_sum": 0.2 }
  ]
}
```

This is consistent with the description in [components loading](components_loading.md): `relations` contains the rules, and the backend strategy translates them into backend-native constraint objects.

## `reuse_data`

If `reuse_data` is set to `true`, the orchestrator can keep the prepared data object on the output when the builder/orchestrator decides to preserve it. This is useful when you want to inspect or reuse preprocessing artifacts across repeated runs.

## Recommended workflow

- Keep the file focused on configuration data.
- Let the loader normalize the shape.
- Let the builder orchestrator create and train the model.
- Put backend-specific runtime behavior into the backend strategy, not the loader.
