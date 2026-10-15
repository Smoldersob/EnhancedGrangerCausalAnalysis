# API Usage Guide

This document provides a comprehensive guide to the main API elements available in `enhanced_granger_analysis` for building, configuring, and running Granger causality tests. It covers available components, their applications, simple test patterns, and the key distinctions between the builder, orchestrator, and simple Granger implementations.

## Overview

The library provides three analysis entry points:

1. `MultiTaskGrangerBuilder` — a stateful builder that stores configuration
   and executes individual analysis stages.
2. `MultiTaskGrangerAPI` — a coordinator for the standard analysis workflow
   implemented by the builder.
3. `SimpleGrangerAPI` — an independent pair-wise statsmodels baseline.

`MultiTaskGrangerBuilder` is a compatibility alias of
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

```python
output = api.fit(
    data: DataFrame | Sequence[DataFrame],
    causes: Optional[Sequence[str]] = None,
    effects: Optional[Sequence[str]] = None,
    tested_causes: Optional[Sequence[str]] = None,
    relations: Optional[Mapping[Tuple[str, str], Any]] = None,
    lag_config: Optional[LagConfiguration] = None,
    regularizer: Optional[Any] = None,
    callbacks: Optional[Sequence[Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    ...
) -> GrangerAnalysisResults
```

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
- Relation-based constraint specification (e.g., `{"y->x1": {"zero": true}}`).
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

Prepare components individually when needed:

```python
builder.prepare_backend()
builder.prepare_constraint()
builder.prepare_regularizer()
builder.prepare_callbacks()
```

`fit_restricted(cause)` requires a completed base fit. In normal use, `cause`
should be one of the effective `prepared.tested_causes`.

Every restricted run restores the base weights, selects the cause's input
block, resets callbacks for that run, fits the working model, and saves a
reference snapshot.

`get_results()` does not fit a model or evaluate all metrics. A result container
is created during data preparation, so its availability does not imply that
base and restricted fitting have completed.

**In Practice:**

```python
from enhanced_granger_analysis.api import MultiTaskGrangerBuilder
from enhanced_granger_analysis.core.lag_config import LagConfiguration
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(lag_config=LagConfiguration(max_lag=5, use_lag_zero=False))
    .model(model_config={"epochs": 20, "batch_size": 16})
    .scaling(x_scaler="standard", y_scaler="standard")
    .fit()
)

causality = output.result(threshold=0.01, with_sign=True)
print(causality)  # DataFrame with signed causality indicators
```


**Configuration changes and current caveats**

Data, variable, lag, scaling, relation, regularization, callback, initializer,
and model changes invalidate different dependent runtime artifacts.

After changing settings in a partially executed analysis, rerun the affected
stages. In the current implementation, invalidation may also remove the result
container; rerunning the full pipeline is the simplest way to rebuild a
consistent result context.

Create a new builder when changing backends after execution has started.
The current `.backend(...)` setter does not invalidate existing runtime
artifacts.

The builder configures a `StationarityTransformer` by default.
Calling `.stationarity()` currently disables stationarity.

Hyperoptimization can be configured through `.hyperoptimization(...)`, but
`fit()` does not currently execute an optimization stage.

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
- Reads JSON/YAML files.
- Normalizes `lag_config` dict into `LagConfiguration` objects.
- Resolves backend aliases (e.g., `"torch"` → `"pytorch"`).
- Preserves callback specs for backend-native resolution.
- Validates component types (lag selectors, regularizers).
- Moves backend-specific defaults into `model_config`.

**Given Example Config:**
```json
{
  "backend": "pytorch",
  "lag_config": {
    "max_lag": 8,
    "use_lag_zero": false
  },
  "callbacks": [
    {"type": "early_stopping", "patience": 5}
  ],
  "relations": {
    "y->x1": {"zero": true}
  },
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
    print(f"Running: epochs={cfg['model_config']['epochs']}, max_lag={cfg['lag_config'].max_lag}")
    
    output = MultiTaskGrangerBuilder().from_config(cfg).data(df).fit()
    results.append(output)
```

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
from enhanced_granger_analysis.core.lag_config import LagConfiguration

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
    .lag(LagConfiguration(max_lag=2, use_lag_zero=False))
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
from enhanced_granger_analysis.core.lag_config import LagConfiguration
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(LagConfiguration(max_lag=5))
    .model({"epochs": 30})
    .relations({
    ("y", "x1"): {"zero": True},
    ("y", "x2"): {"min_abs_sum": 0.5},
    })
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
