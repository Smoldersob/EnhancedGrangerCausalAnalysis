# Configuration File Usage

This document explains how to load a builder configuration file and pass it into the builder/orchestrator pipeline.

For component-family-specific formats, see
[Components Loading](components_loading.md).
Constraints and regularizers use dedicated parsers and registries;
their accepted formats differ from callback and optimizer loader formats.

# Table of content

- [Introduction](#introduction)
- [How to write different components](#how-to-write-different-components)
  - [Supported backend formats](#supported-backend-forms)
  - [Variable selection](#variable-selection-causes-effects-tested_causes)
  - [Model selection](#model-selection-and-construction)
  - [Optimizers and other components](#optimizers-losses-and-other-components)
  - [Constraints](#constraints)
- [Structural metadata and training observations](#structural-metadata-and-training-observations)
- [Hyperparameter optimization status](#hyperoptimization-status)
- [Group test configuration](#group-test-configuration)
  - [What can and cannot be swept)](#what-can-and-cannot-be-swept)
  - [Supported sweeping hyperparameters](#supported-directly-or-conditionally)
  - [Indirectly supported hyperparameters](#not-directly-supported-or-limited)
  - [How to load and use](#how-to-load-and-use-it)
  - [Reuse data optiom](#reuse_data)
- [Recommended workflow](#recommended-workflow)
# Introduction


`BuilderConfigLoader` is responsible for normalization only. It does not run training and does not create the final analysis result.

It can:
- load JSON or YAML files,
- preserve lag-engine and lag-selector specifications for resolution
  during preprocessing,
- normalize relation constraints,
- validate and normalize backend aliases,
- preserve backend-specific callback specs for later resolution,
- parse canonical relation declarations into `RelationConstraintsSpec`;
- resolve relation-file paths relative to the configuration file;
- preserve initializer specifications for resolution by the builder;
- leave backend-native constraint and regularizer construction to the
component preparation and backend construction stages.

The loader also accepts `relations` as a file path when you want to keep constraint rules in a separate JSON/YAML file.

Lag configuration is not instantiated by the loader or builder.
`DataPreparer.lag()` resolves the engine configuration and constructs the
selector through its registry immediately before lag preparation.

# How to write different components

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

## Lag preparation

Lag configuration uses four top-level fields:

| Field | Purpose |
| --- | --- |
| `lag_config` | Fixed baseline, aggregation policy, lag-0 permission, and overrides |
| `lag_selector` | Registered selector name or `type`/`params` specification |
| `lag_selector_params` | Optional constructor parameters supplied separately |
| `lag_n_jobs` | Parallel selector execution across datasets |

The loader and builder preserve these values. Resolution takes place in
the preprocessing pipeline.

### Fixed lags

Set `lag_selector` to `null` to disable automatic selection:

```yaml
lag_selector: null

lag_config:
  max_lag: 4
  use_lag_zero: false
```

This creates positive lags 1 through 4 for every eligible output/predictor
pair before manual overrides.

With zero lag enabled, cross-variable pairs also include lag 0. Own lag 0
remains forbidden.

### Automatic selection

Use a registered selector name:

```yaml
lag_selector: cv

lag_selector_params:
  max_lag: 12
  cv_folds: 5
  cv_metric: mae
  prune_lags: true

lag_config:
  use_lag_zero: false
  aggregation: union

lag_n_jobs: -1
```

Alternatively, place constructor parameters inside the selector
specification:

```yaml
lag_selector:
  type: cv
  params:
    max_lag: 12
    cv_folds: 5
    cv_metric: mae
    prune_lags: true
    addition_reference: initial
    pruning_reference: initial

lag_config:
  use_lag_zero: false
  aggregation: union

lag_n_jobs: -1
```

Do not define the same constructor parameter both inside `params` and in
`lag_selector_params`.

Only `type` and `params` are supported in mapping-form selector
specifications. Constructor parameters must not be flattened beside `type`.

### Search bounds versus engine settings

When a selector is enabled, its own `max_lag` controls the search bound.
`lag_config.max_lag` is used for the fixed baseline when no selector is
configured.

The pipeline does not inject selector settings from `lag_config`.

To evaluate contemporaneous predictors, explicitly enable lag 0 in both
the selector and the engine:

```yaml
lag_selector:
  type: ic
  params:
    max_lag: 8
    use_bic: true
    use_lag_zero: true

lag_config:
  use_lag_zero: true
```

The selector determines which permitted terms it returns. The engine
validates the final permission policy.

For VAR selection, enabling cross-variable lag 0 adds it by policy;
the VAR order criterion does not evaluate its inclusion.

### Selector families

| Registered name | Selection procedure |
| --- | --- |
| `ic` | Per-pair cumulative search scored by IC |
| `cv` | Per-pair cumulative search scored by time-series CV |
| `common_order_ic` | Shared order scored by aggregated IC |
| `common_order_cv` | Shared order scored by aggregated CV |
| `naive_var` / `var` | Historical VAR procedure |

These methods share the result format, not the same objective or search
algorithm.

See [Lag Selection](lag_selection.md) for interpretation.

### Per-pair IC/CV parameters

Shared search settings include:

- `max_lag`;
- `use_lag_zero`;
- `min_ar_order`;
- `prune_lags`;
- `addition_reference`;
- `pruning_reference`;
- `score_atol`, `score_rtol`;
- `rcond`.

IC additionally supports:

- `use_bic`;
- `delta_min_ic`;
- `delta_prune_ic`;
- `complexity_measure`: `columns` or `rank`;
- `variance_floor_rel`.

CV additionally supports:

- `cv_folds`;
- `cv_metric`: `mse` or `mae`;
- `cv_gap`;
- `cv_test_size`;
- `cv_max_train_size`;
- `delta_min_rel_cv`;
- `delta_prune_rel_cv`.

### Common-order IC

```yaml
lag_selector:
  type: common_order_ic
  params:
    max_lag: 6
    use_bic: true
    complexity_measure: columns
    variance_floor_rel: 1.0e-12
```

Common-order IC also accepts `use_lag_zero`, `output_weights`,
`score_atol`, `score_rtol`, and `rcond`.

It does not accept per-pair addition/pruning settings.

### Common-order CV

```yaml
lag_selector:
  type: common_order_cv
  params:
    max_lag: 6
    cv_folds: 5
    cv_metric: mse
    normalize_outputs: true
    scale_floor: 1.0e-12
```

Common-order CV also accepts the fold geometry settings,
`use_lag_zero`, `output_weights`, `score_atol`, `score_rtol`, and `rcond`.

Example output weights:

```yaml
output_weights:
  y1: 1.0
  y2: 2.0
```

The mapping must contain exactly the requested output names.
All weights must be finite and positive.

### Historical VAR

```yaml
lag_selector:
  type: var
  params:
    max_lag: 6
    min_lag: 1
    center: true
    maximum: true
    use_lag_zero: false
```

`type: naive_var` selects the same implementation.

`maximum: true` chooses the largest of the orders selected by AIC, BIC,
HQIC, and FPE. `false` chooses the smallest.

The historical selector does not accept the `criterion` or `trend`
parameters from the earlier single-criterion replacement proposal.
It retains four-criterion selection and constant-term fitting.

Provide `max_lag` explicitly when reproducing an experiment. The
configuration loader no longer copies it from `lag_config`.

### Aggregating independent datasets

```yaml
lag_config:
  aggregation: frequency
  selection_frequency: 0.5
```

Available methods:

- `union`: selected in at least one dataset;
- `intersection`: selected in every dataset;
- `frequency`: selected in at least the configured fraction of datasets.

Frequency uses equal dataset weights and an inclusive comparison.

For example, a threshold of `0.5` requires at least two of three datasets.
A threshold of `0.67` requires all three because `2 / 3` is less than
`0.67`.

`lag_n_jobs` controls dataset-level selection tasks. It does not configure
CV-fold parallelism or the model-training device.

### Manual lag specifications

Use explicit lag lists:

```yaml
lag_config:
  use_lag_zero: true
  aggregation: union

  predictor_overrides:
    x1:[1][3]
    x2: []

  pair_overrides:
    y2:
      x1:[0][2][4]
```

Interpretation:

1. Aggregate automatic selections, or create the fixed baseline.
2. Replace `x1` with lags 1 and 3 across outputs.
3. Exclude `x2` across outputs.
4. Replace the more specific `y2`/`x1` pair with lags 0, 2, and 4.

Lists are explicit sets in ascending order, not range endpoints.
`[1, 3]` does not include lag 2.

Empty lists exclude a predictor. Pair-specific overrides take precedence
over predictor-wide replacements.

A pair override requesting own lag 0 is invalid. Predictor-wide
zero-lag specifications omit lag 0 from the same-variable equation.

Manual overrides can include lags above the automatic search bound.
Datasets must still contain enough history for the final layout.

### Removed tested causes

A predictor with no final lags for any output is omitted from effective
inputs and recorded with reason `"no_selected_lags"`.

If it was a requested tested cause, it is also omitted from effective
restricted-model runs. Result restoration preserves the requested column
using the metric's neutral removal value.

This records exclusion by the configured selection/preparation procedure,
not a separately evaluated negative causality test.

### Sweeping lag settings

Use canonical nested parameter paths:

```json
{
  "base_config": {
    "lag_selector": {
      "type": "ic",
      "params": {
        "max_lag": 8,
        "use_bic": true
      }
    },
    "lag_config": {
      "aggregation": "union",
      "use_lag_zero": false
    }
  },
  "sweep": {
    "param_names": [
      "lag_selector.params.max_lag",
      "lag_selector.params.use_bic"
    ],
    "cases": [
      [8, false],
      [8, true],
      [12, true]
    ]
  }
}
```

If constructor parameters are supplied separately, use paths such as
`lag_selector_params.max_lag`.

Sweep `lag_config.max_lag` to change the fixed baseline, not the search
bound of an enabled selector.

Replace complete lag lists when sweeping overrides. Dotted paths do not
support list indexing.

### Migration from range-based configuration

Replace:

- `LagConfiguration` with `LagEngineConfig`;
- `custom_lags` with `predictor_overrides`;
- `custom_pair_lags` with nested `pair_overrides`;
- selector names encoding parameters, such as `bic`, with
  `type: ic` and `params.use_bic: true`.

Expand old range endpoints explicitly. An old range `(1, 3)` becomes
`[1, 2, 3]`, not `[1, 3]`.

Configuration mappings remain mappings after file loading. They become
runtime configuration objects only during preprocessing.

The historical `"var"` name remains supported and resolves to
`NaiveVARLagSelector`. `VARLagSelector` remains its Python alias.

Use `"common_order_ic"` or `"common_order_cv"` only when intentionally
switching to the new common-order heuristic.

For the historical VAR selector, use `maximum` to combine four criteria.
For common-order IC, use `use_bic` to choose its equation-wise criterion.
Do not treat these settings as equivalent.

## Model selection and construction

The builder constructs models through the selected backend strategy. Model
selection, ready backend components, and model constructor parameters are handled
separately.

Use `model_config.model` to select a registered model:

```json
{
  "model_config": {
    "model": "conditional_linear",
    "epochs": 50,
    "batch_size": 32
  }
}
```

`model_type` is accepted as an alternative to `model`. Specifying both keys is
an error.

When neither key is provided, the backend strategy selects the default
`conditional_linear` model.

Model selection is resolved by the selected backend's model registry. An unknown
registered name raises an error rather than silently falling back to the default
model.

### Typed model specifications

A model can also be selected using a typed specification:

```json
{
  "model_config": {
    "model": {
      "type": "conditional_linear",
      "params": {
        "use_bias": true
      }
    },
    "epochs": 50,
    "batch_size": 32
  }
}
```

Parameters inside the model specification and the remaining `model_config`
entries must not define the same constructor argument more than once.

For example, placing `epochs` both in `model.params` and directly in
`model_config` is an error.

JSON/YAML configuration uses registered string names. Python callers may also
supply compatible concrete model classes. Passing a ready model instance with
builder-managed constructor arguments and components is not supported by this
builder construction path.

## Optimizers, losses, and other components

Optimizer and loss specifications are configured through `model_config`:

```json
{
  "model_config": {
    "optimizer": {
      "type": "adam",
      "params": {
        "lr": 0.001
      }
    },
    "loss": "mse"
  }
}
```

The builder extracts these specifications, constructs backend-compatible
components, and passes the ready objects to the model separately from constructor
configuration.

Callbacks, regularization, and constraints use their dedicated configuration
fields:

| Configuration field | Responsibility |
| --- | --- |
| `callbacks` | Callback specifications |
| `regularizer_spec` | Regularizer selection and parameters |
| `relations` | Named cause/effect constraint declarations |
| `constraint_spec` | Constraint implementation and constructor parameters |
| `model_config.optimizer` | Backend optimizer specification |
| `model_config.loss` | Backend loss specification |

Do not place `callbacks`, `regularizer`, `regularizer_spec`, `constraint`, or
`constraint_spec` inside `model_config`.

The builder also manages structural arguments such as `input_spec`, `n_features`,
and `n_outputs`. These must not be supplied in `model_config`.

### PyTorch configuration

PyTorch optimizer parameters follow native PyTorch naming conventions. For Adam,
use `lr` inside optimizer `params`:

```json
{
  "backend": "pytorch",
  "model_config": {
    "model": "conditional_linear",
    "epochs": 50,
    "batch_size": 32,
    "device": "cpu",
    "optimizer": {
      "type": "adam",
      "params": {
        "lr": 0.001
      }
    },
    "loss": "mse"
  }
}
```

Do not use a model-level `learning_rate` as a replacement for the PyTorch
optimizer's `lr`.

### Keras configuration

Keras optimizer parameters follow native Keras naming conventions. For Adam,
use `learning_rate` inside optimizer `params`:

```json
{
  "backend": "tensorflow",
  "model_config": {
    "model": "conditional_linear",
    "epochs": 50,
    "batch_size": 32,
    "device": "cpu",
    "optimizer": {
      "type": "adam",
      "params": {
        "learning_rate": 0.001
      }
    },
    "loss": "mse"
  }
}
```

The `tensorflow`, `tf`, and `keras` backend selection names select the Keras
strategy.

Backend selection identifies the library strategy; it does not make optimizer
configuration interchangeable with the native PyTorch strategy.

### NumPy configuration

The NumPy adapter uses a built-in MSE objective and an internal solver loop.
Configure its solver directly:

```json
{
  "backend": "numpy",
  "model_config": {
    "model": "conditional_linear",
    "solver": "auto",
    "epochs": 1000,
    "learning_rate": 1.0,
    "tol": 1e-8,
    "use_bias": true
  }
}
```

Do not configure a custom optimizer or loss for the NumPy adapter. Its optimizer
and loss resolvers return `None`; supplying such specifications does not enable
custom optimizer or loss behavior.

NumPy's `learning_rate` is a relaxation/step-control parameter, not the same
quantity as a neural-network optimizer's learning rate.

Available solver selections are:

- `auto`: select a solver according to the fitting configuration and component
  capabilities.
- `restricted_lstsq`: request direct restricted least-squares fitting.
- `coefficient`: use the iterative coefficient-update path.

Automatic direct least-squares fitting requires a full batch, no regularizer
object, and either no constraint or a constraint declaring only coefficient
zeroing. Additional minimum-absolute-sum rules require the iterative path.

A regularizer object with zero penalty strength still prevents automatic direct
least-squares selection.

See [NumPy Backend Optimization](./numpy_backend_optimization.md) for solver
eligibility, objective normalization, proximal updates, stopping criteria, and
limitations.

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


## Structural metadata and training observations

The builder creates `ModelInputSpec` from the effective prepared-data layout,
including:

- The number of prepared input features and outputs.
- Ordered effective cause and effect names.
- Named cause slices.
- Actual feature lags.

Training observations are not supplied during model construction. The builder
passes the prepared input and target arrays explicitly to each base and
restricted fit.

Do not configure structural metadata manually to reproduce the raw input-column
layout: preprocessing may change the effective variables and expand each cause
into multiple lagged features.

# Hyperoptimization status

Hyperoptimization remains disabled in the standard builder pipeline, as in the
previous implementation.

The builder may retain hyperoptimization settings for compatibility, but these
settings do not trigger parameter search, alter model selection, or modify the
training configuration.

Enabling hyperoptimization is outside the scope of this update.

# Group Test Configuration

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

### What can and cannot be swept

`TestGroupConfigIterator` applies sweep values using dotted keys and then calls `BuilderConfigLoader.normalize_builder_config(...)`.

Important implementation detail:
- Dotted updates work only through mappings (dictionaries/objects).
- List indexing is not supported by dotted keys (for example, `callbacks.0.patience` will not work).

## Supported (directly or conditionally)

You can sweep these fields:

- `lag_selector` and its constructor parameters
  - Replace the entire specification, or use canonical mapping paths.
  - Examples: `lag_selector.type`,
    `lag_selector.params.max_lag`, and
    `lag_selector.params.use_bic`.
  - With separate constructor parameters, use
    `lag_selector_params.max_lag`.
  - Registry resolution and construction occur during preprocessing,
    after sweep expansion.

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


## Not directly supported (or limited)

- Dotted paths through list indices are not supported.
  - Not supported: `callbacks.0.patience`, `relations.0.min_abs_sum`.
  - Use whole-object/list replacement in `cases` values.

## Practical recommendation

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

## `reuse_data`

If `reuse_data` is set to `true`, the orchestrator can keep the prepared data object on the output when the builder/orchestrator decides to preserve it. This is useful when you want to inspect or reuse preprocessing artifacts across repeated runs.

# Recommended workflow

- Keep the file focused on configuration data.
- Let the loader normalize the shape.
- Let the builder orchestrator create and train the model.
- Put backend-specific runtime behavior into the backend strategy, not the loader.
