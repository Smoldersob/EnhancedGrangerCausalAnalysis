# Configuration File Usage

This document explains how to load a builder configuration file and pass it into the builder/orchestrator pipeline.

For component-family-specific formats, see
[Components Loading](components_loading.md).
Constraints and regularizers use dedicated parsers and registries;
their accepted formats differ from callback and optimizer loader formats.

## What the loader does

`BuilderConfigLoader` is responsible for normalization only. It does not run training and does not create the final analysis result.

It can:
- load JSON or YAML files,
- normalize `lag_config` into `LagConfiguration`,
- normalize relation constraints,
- validate and normalize backend aliases,
- preserve backend-specific callback specs for later resolution,
- parse canonical relation declarations into `RelationConstraintsSpec`;
- resolve relation-file paths relative to the configuration file;
- preserve initializer specifications for resolution by the builder;
- leave backend-native constraint and regularizer construction to the
component preparation and backend construction stages.

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

## Model selection in config files

A configuration can also carry a `model` or `model_type` field. This is supported as a compatibility hook for the current backend model family.

Current behavior:

- The default implementation remains the active fallback for each backend.
- If `model` / `model_type` points to a subclass of the backend's current base model class, that subclass is used.
- There is no model registry or automatic plugin discovery yet.
- The model config remains the place for additional parameters such as hidden layers, dropout, or custom training settings.

Example:

```json
{
  "backend": "tensorflow",
  "model": "TensorFlowGrangerModel",
  "model_config": {
    "hidden_layers": 2,
    "epochs": 50,
    "batch_size": 32
  }
}
```

The value can also be passed as a Python class object when working directly from code rather than through JSON.

## Example configuration

```json
{
  "backend": "tensorflow",
  "reuse_data": true,
  "causes": ["x1", "x2", "y"],
  "effects": ["y"],
  "tested_causes": ["x1", "x2"],
  "lag_config": {
    "max_lag": 3,
    "use_lag_zero": false
  },
  "constraint_spec": {
    "type": "mask_and_min_abs_sum",
    "params": {
      "eps": 1e-8
    }
  },
  "relations": [
    {
      "cause": "x1",
      "effect": "y",
      "rule": {
        "type": "zero"
      }
    },
    {
      "cause": "x2",
      "effect": "y",
      "rule": {
        "type": "min_abs_sum",
        "params": {
          "value": 0.2
        }
      }
    }
  ],
  "regularizer_spec": {
    "type": "l1",
    "params": {
      "l1": 0.01
    }
  },
  "model_config": {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam"
  }
}
```

The top-level `regularizer` field is no longer supported.

Constraint and regularizer configuration must remain outside
`model_config`. An explicit `constraint_spec: null` is invalid;
`regularizer_spec: null` disables regularization.

To keep relation rules in a separate file:

```json
{
  "relations": "./relations.json"
}
```

The referenced file must contain the canonical declaration list itself,
without a `rules` or `relations` wrapper.

Relative paths are resolved against the configuration file location,
not the process working directory. Legacy relation mappings and
`{"path": "..."}` references are not supported by the new path.

## Optimizer and training parameter control
You can tune optimizer and training behavior directly in the builder config under `model_config` (or, for backward compatibility, by top-level keys such as `learning_rate`). Supported patterns are:

```json
{
  "model_config": {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 32
  }
}
```

```json
{
  "model_config": {
    "optimizer": {
      "type": "adam",
      "params": {
        "lr": 1e-4,
        "beta_1": 0.9,
        "beta_2": 0.999
      }
    }
  }
}
```

This works for both TensorFlow and PyTorch. The difference is how backend-specific optimizer settings are forwarded:

- TensorFlow/Keras: `optimizer.params` is the canonical location for optimizer arguments. If a backend-specific optimizer parameter is expected there, it should be placed under `optimizer.params` as part of the optimizer configuration.
- PyTorch: `gradient_accumulation_steps` is also accepted outside `optimizer.params` as a legacy compatibility alias, because the training loop consumes it at the model level rather than passing it to the optimizer constructor.

Canonical TensorFlow/Keras form:

```json
{
  "model_config": {
    "optimizer": {
      "type": "adam",
      "params": {
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 4
      }
    }
  }
}
```

Legacy PyTorch-compatible form:

```json
{
  "model_config": {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "gradient_accumulation_steps": 4
  }
}
```

The important distinction is that the backward-compatibility alias is PyTorch-only. In TensorFlow/Keras, the valid placement is under `optimizer.params`; the reverse is not the canonical or supported pattern.

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
- `base_config.relations` may be an inline canonical declaration list or a path to a JSON/YAML file containing that list.
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
  - Replace the complete canonical declaration list per case, or supply
    a relation-file path handled by the file-loading workflow.
  - Dotted updates through relation-list indices are not supported.
  - Legacy relation mappings are not a supported workaround.

- regularizer type and presence
  - Sweep `regularizer_spec`, not `regularizer`.
  - Set `regularizer_spec` to `null` to disable regularization.
  - Built-in names are `l1` and `lag_dependent_l1`.

- regularizer parameters
  - Use mapping-form specifications with nested `params`.
  - Example paths: `regularizer_spec.type`,
    `regularizer_spec.params.l1`,
    `regularizer_spec.params.relative_lag_indexing`.

- constraint implementation and parameters
  - Use `constraint_spec` or mapping-form paths such as
    `constraint_spec.type` and `constraint_spec.params.eps`.
  - Do not sweep prepared masks or indexed rules through
    `constraint_spec.params`.

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


### Not directly supported (or limited)

- Dotted paths through list indices are not supported.
  - Not supported: `callbacks.0.patience`, `relations.0.min_abs_sum`.
  - Use whole-object/list replacement in `cases` values.

### Practical recommendation

For sweep dimensions with many list/object internals (`callbacks`, `relations`), treat each case value as a full replacement object.
For scalar and simple nested options (`model_config.*`, `lag_config.*`, `lag_selector.*`, `x_scaler`, `y_scaler`, `backend`), dotted keys are the most robust approach.

## How to load and use it

```python
from enhanced_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder

cfg = BuilderConfigLoader.load_file("config.json")
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

You can also load the raw mapping first and then adapt it yourself if needed:

```python
from enhanced_granger_analysis.api import BuilderConfigLoader

cfg = BuilderConfigLoader.load_file("config.json")
```

Then pass the normalized mapping to `MultitaskGrangerBuilder.from_config(...)`.

## Constraints

`relations` describes named rules, while `constraint_spec` selects the
backend implementation enforcing the compiled data.

```json
{
  "constraint_spec": "mask_and_min_abs_sum",
  "relations": [
    {
      "cause": "x1",
      "effect": "y",
      "rule": {"type": "zero"}
    },
    {
      "cause": "x2",
      "effect": "y",
      "rule": {
        "type": "min_abs_sum",
        "params": {"value": 0.2}
      }
    }
  ]
}
```

Use an empty list for no explicit relation rules. Omit `relations` to use
the builder's empty default specification; do not set it to `null`.

A `min_abs_sum` value of zero adds no lower-bound restriction. It does not
mask the relation.

Named rules are compiled against effective variables after preprocessing.
Unknown effective causes or effects are rejected.

The preprocessing mask is preserved. A positive minimum-sum rule with no
remaining active coefficients raises an error in the builder path.

Relation-file loading is handled by `BuilderConfigLoader`, not by
`.relations(...)` or `.from_config(...)`.

## `reuse_data`

If `reuse_data` is set to `true`, the orchestrator can keep the prepared data object on the output when the builder/orchestrator decides to preserve it. This is useful when you want to inspect or reuse preprocessing artifacts across repeated runs.

## Recommended workflow

- Keep the file focused on configuration data.
- Let the loader normalize the shape.
- Let the builder orchestrator create and train the model.
- Put backend-specific runtime behavior into the backend strategy, not the loader.
