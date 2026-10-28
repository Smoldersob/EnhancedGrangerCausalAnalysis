# API Usage Guide

This document provides a comprehensive guide to the main API elements available in `enhanced_granger_analysis` for building, configuring, and running Granger causality tests. It covers available components, their applications, simple test patterns, and the key distinctions between the builder, orchestrator, and simple Granger implementations.

## Overview

The library provides three analysis entry points:

1. `MultiTaskGrangerBuilder` — a stateful builder that stores configuration
   and executes individual analysis stages.
2. `MultiTaskGrangerAPI` — a coordinator for the standard analysis workflow
   implemented by the builder.
3. `SimpleGrangerAPI` — an independent pair-wise statsmodels baseline.

`MultitaskGrangerBuilder` is a compatibility alias of
`MultiTaskGrangerBuilder`.

Related utilities:

- `BuilderConfigLoader`: loads and normalizes JSON/YAML configuration.
- `TestGroupConfigIterator`: expands grouped experiment configurations.
- `GrangerAnalysisResults`: stores model snapshots and evaluates metrics.

Configuration chaining and execution are separate operations. Configuration
methods update settings; execution methods immediately perform a stage and
store its artifacts.

## Detailed Component Overview

### 1. MultiTaskGrangerAPI (Orchestrator)

**Purpose:** The standard orchestration entry point for multitask Granger causality
analysis.

**Scope:** Creates a builder, invokes the standard preparation, component, model, base-fit, and restricted-fit stages, then returns the accumulated results. It does not maintain a second model implementation.

**Key Method:**

### Constraint and regularizer configuration

```python
output = api.fit(
    data=df,
    causes=["x1", "x2", "y"],
    effects=["y"],
    tested_causes=["x1"],
    constraint_spec="mask_and_min_abs_sum",
    relations=[
        {
            "cause": "x2",
            "effect": "y",
            "rule": {"type": "zero"},
        },
    ],
    regularizer_spec={
        "type": "l1",
        "params": {"l1": 0.01},
    },
    model_config={"epochs": 20},
)
```

`relations` accepts a canonical declaration list or a
`RelationConstraintsSpec`. Its default is an empty specification.

`constraint_spec` selects the enforcing implementation and defaults to
`"mask_and_min_abs_sum"`.

`regularizer_spec=None` disables regularization. The former `regularizer`
argument is no longer part of this interface.

This example illustrates the version 2.6 configuration contract.
End-to-end execution remains subject to the experimental integration
limitations described in the changelog.

**Variable Selection Parameters:**

- **`causes`** — Predictor variables available to the model. If `None`, all columns are used.
- **`effects`** — Target variables to predict. If `None`, all columns are used.
- **`tested_causes`** — A subset of `causes` for which Granger causality is formally tested. If `None`, defaults to all of `causes`.

The distinction is important: `causes` define what the model can *see*, while `tested_causes` define which relationships are *analyzed*. This allows you to include contextual variables without explicitly testing their causality, reducing computation while maintaining model fidelity.

**Output:**
```python
results: GrangerAnalysisResults
```

**Features:**
- One model object is reused for the base and restricted runs.
- Each restricted run starts from an independent copy of the base weights.
- Named relation declarations configured separately from constraint
  implementation selection.
- Backend abstraction via `BackendFactory` strategy pattern.
- Optional initializer support through the builder.

`hiperoptimalization_state` and `hiperoptimalization_conf` are accepted for
configuration compatibility, but hyperoptimization is not currently executed by the
standard pipeline.

**When to use:**
- Production analyses requiring fine-grained control.
- Complex causal structures with constraints or regularization.
- Multi-effect or multi-tested-cause scenarios.

---

### 2. MultiTaskGrangerBuilder

**Purpose:** Fluent builder for configuring and running MultiTaskGrangerAPI workflows.

**Scope:** Provides method chaining for configuration and executable pipeline stages. Each stage runs immediately and stores its artifacts, so an analysis can be inspected or paused before training.

**Configuration and execution**

Configuration methods return the builder and can be chained:

```python
builder = (
    MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(
        causes=["x1", "x2"],
        effects=["y"],
        tested_causes=["x2", "x1"],
    )
    .lag(LagConfiguration(max_lag=5, use_lag_zero=False))
    .model({"epochs": 20, "batch_size": 16})
    .scaling(x_scaler="standard", y_scaler="standard")
)
```

The order of `tested_causes` controls restricted-run order, not feature-block
positions. Blocks are selected by name from prepared data.

Run the complete pipeline:

```python
results = builder.fit()
```

`run()` is an alias of `fit()`. Both execute the complete pipeline from data
preparation; they do not resume a partially completed analysis.

For explicit execution:

```python
prepared = builder.prepare_data()
builder.prepare_components()
model = builder.build_model()
builder.initialize_weights()
builder.fit_base()
builder.fit_restricted_models()
results = builder.get_results()
```

Execution methods do not share one fluent return contract:

| Method | Return value |
| --- | --- |
| `prepare_data()` | `PreparedData` |
| `prepare_components()` | Builder |
| `build_model()` | Working model |
| `initialize_weights()` | Assigned initial weights, or `None` |
| `fit_base()` | Copied base-model weights |
| `fit_restricted(cause)` | `None` |
| `fit_restricted_models()` | Result container |
| `get_results()` | Current result container |
| `prepare_constraint_data()` | `PreparedConstraintData` |
| `prepare_regularizer_data()` | `PreparedRegularizerData` or `None` |
| `prepare_backend()` | Backend strategy |
| `prepare_constraint()` | Backend constraint |
| `prepare_regularizer()` | Backend regularizer or `None`; current integration caveat applies |
| `prepare_callbacks()` | Copied callback configuration list or `None` |

**In Practice:**

```python
from enhanced_granger_analysis.api import MultiTaskGrangerBuilder
from enhanced_granger_analysis.preprocessing.lag import LagEngineConfig
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(lag_config=LagEngineConfig(max_lag=5,use_lag_zero=False,),)
    .model(model_config={"epochs": 20, "batch_size": 16})
    .scaling(x_scaler="standard", y_scaler="standard")
    .fit()
)

causality = output.result(threshold=0.01, with_sign=True)
print(causality)  # DataFrame with signed causality indicators
```

### Configuring lag preparation

The builder stores lag configuration without constructing a selector or
executing lag selection. Resolution takes place in `DataPreparer.lag()`.

Engine configuration, selector configuration, and execution parallelism
are separate:

```python
builder.lag(
    lag_config={
        "use_lag_zero": False,
        "aggregation": "frequency",
        "selection_frequency": 0.5,
        "predictor_overrides": {
            "x1": (1, 3),
        },
        "pair_overrides": {
            "y": {
                "x2": (),
            },
        },
    },
    lag_selector={
        "type": "cv",
        "params": {
            "max_lag": 12,
            "cv_folds": 5,
            "cv_metric": "mae",
            "prune_lags": True,
            "addition_reference": "initial",
            "pruning_reference": "initial",
        },
    },
    n_jobs=-1,
)
```

`lag_config` accepts a `LagEngineConfig`, a mapping, or `None` for the
default engine configuration.

`lag_selector` accepts a registered name, a `type`/`params` specification,
a compatible class, a compatible instance, a factory, or `None`.
The pipeline resolves the specification using the selector registry.

Constructor parameters may alternatively be supplied separately:

```python
builder.lag(
    lag_config={"aggregation": "union"},
    lag_selector="ic",
    lag_selector_params={
        "max_lag": 12,
        "use_bic": True,
    },
    n_jobs=-1,
)
```

Supplying the same constructor parameter both inside `params` and through
`lag_selector_params` is an error. Existing selector instances must not be
combined with constructor parameters.

Calling `builder.lag(...)` replaces the stored lag setup and invalidates
dependent preparation artifacts.

#### Fixed versus selected lags

Without an automatic selector, `lag_config.max_lag` defines the fixed
positive-lag range:

```python
builder.lag(
    lag_config={"max_lag": 4},
    lag_selector=None,
)
```

With a selector, its own `max_lag` defines the automatic search bound.
The pipeline does not copy that value from the engine configuration.

#### Explicit lag sets

Override values contain actual lag values, not interval endpoints:

- `(1, 3)` includes exactly lags 1 and 3.
- `(0,)` includes only the contemporaneous predictor.
- `()` excludes the predictor for the applicable output equations.

Predictor-wide overrides replace the selected lag set across outputs.
Pair-specific overrides are applied afterwards and take precedence.

Lag 0 is permitted only when `lag_config.use_lag_zero=True`.
A variable's own contemporaneous value is never a valid predictor of
itself.

#### Dataset-level parallelism

The builder's `n_jobs` argument is passed to the preparation pipeline as
`lag_n_jobs`. It controls independent selector execution across input
DataFrames.

Each dataset is processed with an independent selector instance. The
resulting specifications are aggregated before user overrides and final
feature construction.

For one dataset, dataset-level parallelism provides no additional
selection tasks.

### Selecting a lag-search family

Per-pair selection permits different ranges for different relationships:

```python
builder.lag(
    lag_selector="ic",
    lag_selector_params={
        "max_lag": 6,
        "use_bic": True,
        "complexity_measure": "columns",
        "prune_lags": True,
    },
)
```

Common-order selection evaluates complete ranges `1..p` and returns one
order for all pairs:

```python
builder.lag(
    lag_selector="common_order_ic",
    lag_selector_params={
        "max_lag": 6,
        "use_bic": True,
        "complexity_measure": "columns",
    },
)
```

To use common-order time-series validation:

```python
builder.lag(
    lag_selector="common_order_cv",
    lag_selector_params={
        "max_lag": 6,
        "cv_folds": 5,
        "cv_metric": "mse",
        "normalize_outputs": True,
    },
)
```

To retain the historical VAR procedure:

```python
builder.lag(
    lag_selector="var",
    lag_selector_params={
        "max_lag": 6,
        "min_lag": 1,
        "center": True,
        "maximum": True,
    },
)
```

`"var"` and `"naive_var"` resolve to the same historical implementation.

The common-order methods do not accept per-pair pruning or addition
settings.

All selectors are preprocessing heuristics. Choosing a different family
changes the auxiliary selection procedure, even if the returned lag
format is unchanged.

See [Lag Selection](lag_selection.md) for methodological details.

### Component preparation stages

After preparing data, backend-independent component data can be inspected
separately:

```python
prepared = builder.prepare_data()

constraint_data = builder.prepare_constraint_data()
regularizer_data = builder.prepare_regularizer_data()
```

`prepare_constraint_data()` returns `PreparedConstraintData`.

`prepare_regularizer_data()` returns `PreparedRegularizerData`, or `None`
when regularization is disabled. Refreshing regularizer data clears the
existing regularizer and invalidates the working model and its snapshots.

Backend-native construction requires a selected strategy:

```python
strategy = builder.prepare_backend()
constraint = builder.prepare_constraint()
regularizer = builder.prepare_regularizer()
callbacks = builder.prepare_callbacks()
```

`prepare_constraint()` recompiles the named rules and updates the result
container's mask when that container is available.

`prepare_callbacks()` stores a copy of the configured callback sequence;
backend-native callback resolution occurs in the backend construction
path.

`prepare_components()` groups backend preparation, component-data
preparation, component construction, and callback preparation.

The explicit calls above illustrate the intended stage contract. In the
current experimental commit, a remaining legacy definition overrides the
new `prepare_regularizer()` implementation and must be removed before
relying on the new regularizer construction path.

### Configuring constraints and regularization

```python
builder = (
    MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(
        causes=["x1", "x2", "y"],
        effects=["y"],
        tested_causes=["x1"],
    )
    .lag(LagConfiguration(max_lag=3, use_lag_zero=False))
    .relations([
        {
            "cause": "x2",
            "effect": "y",
            "rule": {"type": "zero"},
        },
    ])
    .constraint({
        "type": "mask_and_min_abs_sum",
        "params": {"eps": 1e-8},
    })
    .regularization({
        "type": "l1",
        "params": {"l1": 0.01},
    })
    .model({"epochs": 20})
)
```

These methods store configuration; they do not construct backend component
instances or train the model.

Use `.relations([])` for no explicit named rules and
`.regularization(None)` to disable regularization.

`.constraint(None)` is not supported: the preprocessing mask must remain
part of the constraint preparation path.

Do not put `regularizer`, `regularizer_spec`, `constraint`, or
`constraint_spec` inside `model_config`.

`.relations(...)` does not read files. Use `.from_file(...)` or
`BuilderConfigLoader.load_relations_file(...)` for external rule files.


### Configuration changes and invalidation

Configuration changes invalidate dependent runtime artifacts.

- `.backend(...)` clears the selected strategy, compiled component data,
  component instances, and working model. Prepared data is retained.
  An existing result container is preserved, but its model snapshots
  are cleared.
- `.regularization(...)` clears compiled component data and component
  instances and invalidates the working model. An existing preparation
  result context is preserved, but model snapshots are cleared.
- `.relations(...)` and `.constraint(...)` invalidate component and model
  artifacts and currently discard the result container.
- `prepare_regularizer_data()` clears the existing regularizer and
  invalidates the working model while preserving an existing preparation
  result context.
- `prepare_data()` invalidates old components and creates a new result
  container for the supplied or newly prepared data.

After changing configuration, rerun the affected stages. If a setter has
discarded the result container, rerun data preparation before fitting;
reconstructing components alone does not recreate that container.

`fit()` and `run()` execute the complete pipeline rather than resume a
partially completed analysis.

The builder configures a default `StationarityTransformer`.
Calling `.stationarity()` stores a default transformer;
`.stationarity(None)` disables the transformation.

Hyperoptimization settings can be stored, but the standard pipeline does
not currently execute an optimization stage.

Version 2.6 remains experimental. Component import, registration, and
construction inconsistencies must be resolved before the new workflow
can be considered validated end to end.

**When to use:**
- Simpler, more readable configuration workflows.
- When building from YAML/JSON config files.
- When chaining configuration steps is clearer than inline dicts.
- When building you on granger tests pipeline 

---

### 3. SimpleGrangerAPI

**Purpose:** Quick pair-wise Granger causality analysis using statsmodels.

**Scope:** Tests causality for each (cause, effect) pair independently without multi-output regression or backend abstraction.

**Key Method:**

```python
from enhanced_granger_analysis.api import SimpleGrangerAPI

api = SimpleGrangerAPI()
output = api.fit(
    data: pd.DataFrame,
    causes: Optional[Sequence[str]] = None,
    effects: Optional[Sequence[str]] = None,
    test: str = "ssr_chi2test",
    lag: Optional[int] = None,  # inferred via AIC if None
    lag_max: int = 20,
    threshold: float = 0.01,
) -> SimpleGrangerOutput
```

**Output:**
```python
SimpleGrangerOutput(
    causality_matrix: CausalityMatrix,  # signed causality (1, 0, -1)
    p_value: pd.DataFrame,              # p-values for each pair
    sign: pd.DataFrame,                 # sign of strongest lag coefficient
)
```

**In Practice:**

```python
from enhanced_granger_analysis.api import SimpleGrangerAPI
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = SimpleGrangerAPI().fit(
    data=df,
    causes=["x1", "x2"],
    effects=["y"],
    lag=2,
    threshold=0.01,
)

print(output.causality_matrix.data)
print(output.p_value)
```

**When to use:**
- Exploratory analysis to quickly test causality hypotheses.
- Benchmarking or validation against statsmodels baseline.
- Simple pair-wise models without regularization or multi-output constraints.
- When stationarity and constant lag order suffice.

---

### 4. BuilderConfigLoader

**Purpose:** Loads YAML/JSON configuration files and normalizes them for the builder.

**What it does:**
- reads JSON/YAML configuration;
- resolves backend aliases and existing backend configuration defaults;
- resolves relation-file references relative to the configuration file;
- preserves lag-engine and lag-selector specifications without
  constructing those components;
- preserves callback specifications for later backend resolution;
- applies the existing normalization rules for other configuration
  families.

Lag specifications remain declarative when passed to the builder.

`DataPreparer.lag()` calls `resolve_lag_engine_config()` and
`resolve_lag_selector()`, then constructs the lag engine. Neither the
configuration loader nor the analysis builder constructs lag selectors.

For lag configuration formats, see
[Configuration File Usage](config_file_usage.md#lag-preparation).

**Given Example Config:**
```json
{
  "backend": "pytorch",
  "lag_config": {
    "max_lag": 8,
    "use_lag_zero": false
  },
  "callbacks": [
    {"type": "early_stopping", "params":{"patience": 5}}
  ],
  "relations": [
    {
        "cause": "x1",
        "effect": "y",
        "rule": {"type": "zero"}
    }
  ],
  "model_config": {
    "epochs": 50,
    "learning_rate": 0.001
  }
}
```

**In Practice:**

```python
from enhanced_granger_analysis.api import BuilderConfigLoader, MultiTaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

cfg = BuilderConfigLoader.load_file("config.json")
output = MultiTaskGrangerBuilder().from_config(cfg).data(df).fit()
```

For details on configuration structure, see [Configuration File Usage](config_file_usage.md).

---

### 5. TestGroupConfigIterator

**Purpose:** Expands test sweeps into individual builder configurations and iterates over them.

**Given Sweep File:**
```json
{
  "reuse_data": true,
  "base_config": {
    "backend": "tensorflow",
    "lag_config": { "max_lag": 8 },
    "model_config": { "epochs": 20 }
  },
  "sweep": {
    "param_names": ["model_config.epochs", "lag_config.max_lag"],
    "cases": [[5, 8], [10, 10], [20, 12]]
  }
}
```

**In Practice:**

```python
from enhanced_granger_analysis.api import TestGroupConfigIterator, MultiTaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
it = TestGroupConfigIterator.from_file("group_config.json")

results = []
for cfg in it:
    print(
        f"Running: epochs={cfg['model_config']['epochs']}, "
        f"max_lag={cfg['lag_config']['max_lag']}"
    )
    
    output = MultiTaskGrangerBuilder().from_config(cfg).data(df).fit()
    results.append(output)
```

Lag configuration remains a mapping after file loading and sweep expansion.
Use dictionary access when inspecting an unresolved configuration.

When automatic selection is enabled, sweep its search bound through
`lag_selector.params.max_lag` or `lag_selector_params.max_lag`, depending
on the chosen specification form.

**Features:**
- Expands cartesian product of parameter values.
- Preserves top-level metadata (e.g., `reuse_data`) across sweeps.
- Each config is normalized via `BuilderConfigLoader`.
- Dotted keys (e.g., `"model_config.epochs"`) update nested config.

For details, see [Configuration File Usage](config_file_usage.md).

---

### 6. GrangerAnalysisResults

**Purpose:** Aggregates causality test results into matrices of p-values, test statistics, and signed indicators.

**Scope:** Compares base model predictions/weights against restricted snapshots (one per
tested cause) to derive causality signatures.

**Key Information:**
```python
results = builder.fit()

# Populated by fit_base() and fit_restricted_models().
# Metrics are exposed as DataFrames:
results.p_value                  # p-values
results.f_test                    # F-test statistics
results.sign                      # sign of the strongest contribution
```

**Accessible After Fit:**
```python
results = builder.fit()
causality = results.result(threshold=0.01, with_sign=True)
print(causality)
# Output example:
#      x1   x2
# y1  1.0 -1.0
# y2  0.0  1.0
```

---

### Result access

```python
results = builder.fit()

# Convenience properties return DataFrames:
p_values = results.p_value
f_values = results.f_test
signs = results.sign

# evaluate() returns a CausalityMatrix:
f_matrix = results.evaluate("f_statistic")
sign_matrix = results.evaluate("sign")

print(f_matrix.data)
print(sign_matrix.data)

# result() returns a thresholded DataFrame:
binary = results.result(threshold=0.05)
signed = results.result(threshold=0.05, with_sign=True)
```

Do not append `.data` to `results.result(...)`: it already returns a
`pandas.DataFrame`.

Use `.data` for the `CausalityMatrix` returned by `evaluate(...)`.

Metric evaluation restores removed variables by default when a preparation
context is available. This also applies to the `sign` metric.

To inspect only effective fitted variables:

```python
fitted_signs = results.evaluate(
    "sign",
    restore_removed=False,
).data
```

`results.sign` is equivalent to:

```python
results.evaluate("sign", restore_removed=True).data
```

`result()` applies the comparison `metric_value < threshold`. Use a p-value
metric for significance thresholding; passing a raw statistic does not
automatically select an appropriate upper-tail critical value.

### Signed results

A signed result is calculated in three steps:

1. Evaluate and restore the selected metric to the requested effect and
   tested-cause axes.
2. Evaluate and restore the sign metric to the same axes.
3. Multiply thresholded significance indicators by the restored signs.

```python
signed = results.result(
    threshold=0.05,
    metric="f_test_p_value",
    with_sign=True,
)
```

The implementation checks that both matrices have identical row and column
labels before multiplication. An axis mismatch raises `ValueError` rather
than relying on implicit pandas alignment.

The sign is derived from the sum of base-model coefficients in the feature
block belonging to each cause. It is not the direction of the tested causal
arrow.

For effective fitted variables, sums with absolute value below `1e-12`
retain the existing convention of sign `+1`.

During removed-variable restoration:

- duplicate effects copy the sign row of an available fitted representative;
- duplicate tested causes copy the sign column of an available tested
  representative;
- entries without an available representative use the sign metric's
  fill value, `0`.

A restored sign of `0` is a reconstruction convention, not an independently
estimated coefficient sign.

If a removed duplicate cause's representative was an input but was not
formally tested, no representative result column is available to copy.

#### Causes excluded during lag preparation

An input with no final selected lags for any effective output is removed
from the effective predictor roles. If it was a tested cause, it is also
removed from the effective tested-cause set and receives no restricted
model fit.

Requested roles remain available for result restoration. Restored
lag-excluded columns use the metric's configured neutral value:

- p-values: `1`;
- F, Wald, and likelihood-ratio statistics: `0`;
- coefficient-derived signs: `0`;
- metrics without an explicit restoration value: `NaN`.

These are structural restoration conventions for the selected model, not
additional statistical tests demonstrating absence of causality.

A variable excluded as an input can remain an effective output.

A lag-excluded tested cause is restored using a neutral reporting value.
This records exclusion by the preprocessing procedure, not an independent
statistical conclusion about the underlying relationship.

Rank deficiency alone is not recorded as `"no_selected_lags"`.

## Key Distinctions: Builder vs. Orchestrator vs. Simple Granger

| Aspect | **Builder** | **Orchestrator** | **Simple Granger** |
|--------|-----------|-----------------|-------------------|
| **Purpose** | Configuration and executable stages | Standard pipeline coordinator | Pair-wise statsmodels |
| **API Complexity** | Fluent or explicit stages | One fit() call | Simple fit() call |
| **Multi-output** | ✓ Yes | ✓ Yes (primary) | ✗ No (per pair) |
| **Constraints** | ✓ Yes (via relations) | ✓ Yes | ✗ No |
| **Regularization** | ✓ Yes (L1, lag-dependent) | ✓ Yes | ✗ No |
| **Callbacks** | ✓ Yes | ✓ Yes | ✗ No |
| **Config files** | ✓ Yes (from_file, from_config) | ✗ Manual dicts | ✗ Manual dicts |
| **Backend choice** | "pytorch" \ "tensorflow" \ "sklearn" | "pytorch" \ "tensorflow" \ "sklearn" | Fixed (statsmodels) |
| **Hyperopt** | Configuration accepted; execution deferred | Configuration accepted; execution deferred | ✗ No |
| **Use case** | Config-driven workflows | Fine-grained control | Quick exploration |

**In Summary:**
- **Use Builder** when you have YAML/JSON configs or prefer method chaining.
- **Use Orchestrator** when you want the standard pipeline through one `fit()` call.
- **Use explicit builder stages** when you need to inspect or pause between preparation,
  model construction, base fitting, and restricted fitting.
- **Use Simple Granger** for rapid pair-wise tests, validation, or when statsmodels is preferred.

---

## Simple Test Examples

### Minimal example with a lagged dependency

```python
import numpy as np
import pandas as pd

from enhanced_granger_analysis.api import MultiTaskGrangerBuilder
from enhanced_granger_analysis.preprocessing.lag import LagEngineConfig

rng = np.random.default_rng(42)
n_samples = 400
x = rng.normal(size=n_samples)
y = np.zeros(n_samples)

for t in range(1, n_samples):
    y[t] = (
        0.4 * y[t - 1]
        + 0.8 * x[t - 1]
        + rng.normal(scale=0.5)
    )

df = pd.DataFrame({"x": x, "y": y})

results = (
    MultiTaskGrangerBuilder(backend="sklearn")
    .data(df)
    .variables(
        causes=["x", "y"],
        effects=["y"],
        tested_causes=["x"],
    )
    .from_config({"stationarity_transformer": None})
    .lag(lag_config=LagEngineConfig(max_lag=2,use_lag_zero=False,))
    .fit()
)

print(results.result(threshold=0.01))
print(results.p_value)
```

Here `y` is included in `causes` so that its own lagged history is available
to the model. Only `x` is tested through a restricted run.

This example illustrates the API and data construction; it is not an
automated validation test.

### Test with Config File

```python
from enhanced_granger_analysis.api import BuilderConfigLoader, MultiTaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

# Load from config file
cfg = BuilderConfigLoader.load_file("config.json")

# Run analysis
output = MultiTaskGrangerBuilder().from_config(cfg).data(df).fit()

# Examine results
print("Causality Matrix:")
print(output.result(threshold=0.01, with_sign=True))
print("\nP-values:")
print(output.p_value)
```

### Test with Sweep (Multiple Configurations)

```python
from enhanced_granger_analysis.api import TestGroupConfigIterator, MultiTaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
it = TestGroupConfigIterator.from_file("group_config.json")

for i, cfg in enumerate(it):
    output = MultiTaskGrangerBuilder().from_config(cfg).data(df).fit()
    print(f"Test {i}:\n{output.result(threshold=0.01, with_sign=True)}")
```

### Test with Relations (Constraints)

```python
from enhanced_granger_analysis.api import MultiTaskGrangerBuilder
from enhanced_granger_analysis.preprocessing.lag import LagEngineConfig
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(
        lag_config=LagEngineConfig(
            max_lag=5,
            use_lag_zero=False,
        ),
    )
    .model({"epochs": 30})
    .relations([
        {
            "cause": "x1",
            "effect": "y",
            "rule": {"type": "zero"},
        },
        {
            "cause": "x2",
            "effect": "y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0.5},
            },
        },
    ])
    .constraint("mask_and_min_abs_sum")
    .fit()
)

print(output.result(threshold=0.01, with_sign=True))
```

### Quick Exploration with SimpleGrangerAPI

```python
from enhanced_granger_analysis.api import SimpleGrangerAPI
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = SimpleGrangerAPI().fit(
    data=df,
    causes=["x1", "x2"],
    effects=["y1", "y2"],
    lag=2,
    threshold=0.05,
)

print("Causality Matrix (signed):")
print(output.causality_matrix.data)
print("\nP-values:")
print(output.p_value)
```

---

## Typical Analysis Workflow

1. **Prepare Data**
   ```python
   import pandas as pd
   df = pd.read_csv("data.csv", index_col=0)
   ```

2. **Choose Configuration Method**
   - Option A: Load from file
     ```python
     cfg = BuilderConfigLoader.load_file("config.json")
     ```
   - Option B: Build in code
     ```python
     cfg = {
         "backend": "pytorch",
         "lag_config": {"max_lag": 5},
         "model_config": {"epochs": 20},
     }
     ```

3. **Create Builder**
   ```python
   builder = MultitaskGrangerBuilder().from_config(cfg).data(df)
   ```

4. **Run Analysis**
   ```python
   output = builder.fit()
   ```

5. **Extract Results**
   ```python
   causality = output.result(threshold=0.01, with_sign=True)
   p_values = output.p_value
   print(causality)
   ```

---

## When to Use Each Component

| Task | Recommended | Why |
|------|-------------|-----|
| Load config from JSON | `BuilderConfigLoader` | Clean separation of concerns |
| Standard analysis through one call | `MultiTaskGrangerAPI` | Coordinates the standard builder workflow |
| Inspect or customize execution stages | `MultiTaskGrangerBuilder` | Exposes preparation, construction, and fitting stages |
| Quick pair-wise test | `SimpleGrangerAPI` | Simple, no backends needed |
| Run multiple configs | `TestGroupConfigIterator` | Automatic sweep expansion |
| Extract causality results | `GrangerAnalysisResults` | Aggregates p-values, statistics |

---

## References

- [Configuration File Usage](config_file_usage.md) — Detailed config structure reference.
- [Configuration File Usage](config_file_usage.md) — Sweep expansion and iteration.
- [Components Loading](components_loading.md) — Backend-specific component formats.
