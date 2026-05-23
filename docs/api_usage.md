# API Usage Guide

This document provides a comprehensive guide to the main API elements available in `complex_granger_analysis` for building, configuring, and running Granger causality tests. It covers available components, their applications, simple test patterns, and the key distinctions between the builder, orchestrator, and simple Granger implementations.

## Overview

The library provides three main analysis APIs:

1. **MultiTaskGrangerAPI** — Advanced multi-variable Granger causality analysis with configurable backends, regularization, constraints, and callbacks. Best for production workflows and complex causal studies.
2. **MultitaskGrangerBuilder** — Fluent builder interface for MultiTaskGrangerAPI. Best for building workflows step-by-step or working with configuration files.
3. **SimpleGrangerAPI** — Pair-wise Granger causality analysis using statsmodels. Best for quick exploratory analysis and benchmarking.

Related utilities:
- **BuilderConfigLoader** — Loads and normalizes YAML/JSON configuration files for builders.
- **TestGroupConfigIterator** — Expands test sweeps for running multiple configurations.
- **GrangerAnalysisResults** — Aggregates causality signatures (p-values, test statistics) across base and reference models.

## Detailed Component Overview

### 1. MultiTaskGrangerAPI (Orchestrator)

**Purpose:** The main orchestration engine for multitask Granger causality analysis.

**Scope:** Handles data preparation, stationarity transformation, lag engineering, scaling, backend model creation, hyperoptimization, base/reference model training, and result aggregation.

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
) -> MultitaskGrangerOutput
```

**Variable Selection Parameters:**

- **`causes`** — Predictor variables available to the model. If `None`, all columns are used.
- **`effects`** — Target variables to predict. If `None`, all columns are used.
- **`tested_causes`** — A subset of `causes` for which Granger causality is formally tested. If `None`, defaults to all of `causes`.

The distinction is important: `causes` define what the model can *see*, while `tested_causes` define which relationships are *analyzed*. This allows you to include contextual variables without explicitly testing their causality, reducing computation while maintaining model fidelity.

**Output:**
```python
MultitaskGrangerOutput(
    results: GrangerAnalysisResults,  # causality matrix, p-values, statistics
    base_model: Any,                  # trained reference model
    reference_models: Dict[str, ...], # predictions/weights for each cause
    stationarity_transformer: Any,
    lag_engine: LagEngine,
    X_scaler: Any,
    y_scaler: Any,
    prepared_data: Optional[_PreparedData],  # reused if reuse_data=True
)
```

**Features:**
- Conditional model initialization based on backend capabilities (`needs_reinit` flag).
- Hot-start from base model weights for reference model training.
- Relation-based constraint specification (e.g., `{"y->x1": {"zero": true}}`).
- Hyperoptimization over model config or regularization parameters.
- Backend abstraction via `BackendFactory` strategy pattern.

**When to use:**
- Production analyses requiring fine-grained control.
- Complex causal structures with constraints or regularization.
- Multi-effect or multi-tested-cause scenarios.

---

### 2. MultitaskGrangerBuilder

**Purpose:** Fluent builder for configuring and running MultiTaskGrangerAPI workflows.

**Scope:** Provides method chaining for builder initialization, data assignment, variable specification, configuration loading, and final execution.

**Key Methods:**

```python
builder = MultitaskGrangerBuilder(backend="pytorch", reuse_data=True)
builder = builder.data(df)
builder = builder.variables(causes=["x1"], effects=["y"], tested_causes=["x1"])
builder = builder.lag(lag_config=LagConfiguration(max_lag=10))
builder = builder.model(model_config={"epochs": 50})
builder = builder.scaling(x_scaler="standard", y_scaler="standard")
builder = builder.regularization(regularizer_spec={"type": "l1", "l1": 0.01})
builder = builder.callbacks([callback1, callback2])
builder = builder.hyperoptimization(state="model", config={"n_trials": 20})
builder = builder.from_config(cfg)  # load from dict
builder = builder.from_file("config.json")  # load from file

output = builder.fit()  # or .run()
```

**In Practice:**

```python
from complex_granger_analysis.api import MultitaskGrangerBuilder
from complex_granger_analysis.core.lag_config import LagConfiguration
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultitaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(lag_config=LagConfiguration(max_lag=5, use_lag_zero=False))
    .model(model_config={"epochs": 20, "batch_size": 16})
    .scaling(x_scaler="standard", y_scaler="standard")
    .fit()
)

causality_matrix = output.results.causality_matrix
print(causality_matrix.data)  # DataFrame with signed causality indicators
```

**When to use:**
- Simpler, more readable configuration workflows.
- When building from YAML/JSON config files.
- When chaining configuration steps is clearer than inline dicts.

---

### 3. SimpleGrangerAPI

**Purpose:** Quick pair-wise Granger causality analysis using statsmodels.

**Scope:** Tests causality for each (cause, effect) pair independently without multi-output regression or backend abstraction.

**Key Method:**

```python
from complex_granger_analysis.api import SimpleGrangerAPI

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
from complex_granger_analysis.api import SimpleGrangerAPI
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
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

cfg = BuilderConfigLoader.load_file("config.json")
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
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
from complex_granger_analysis.api import TestGroupConfigIterator, MultitaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
it = TestGroupConfigIterator.from_file("group_config.json")

results = []
for cfg in it:
    print(f"Running: epochs={cfg['model_config']['epochs']}, max_lag={cfg['lag_config'].max_lag}")
    
    output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
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

**Scope:** Compares base model predictions/weights against reference models (one per tested cause) to derive causality signatures.

**Key Information:**
```python
results = GrangerAnalysisResults(effects=["y1", "y2"], causes=["x1", "x2"])

# Populated by orchestrator.fit() via update_cause()
# Contains:
results.causality_matrix          # DataFrame: signed causality (1, 0, -1) per pair
results.p_value_matrix            # DataFrame: corrected p-values
results.test_statistic_matrix     # DataFrame: F/chi2 test statistics
```

**Accessible After Fit:**
```python
output = builder.fit()
causality_matrix = output.results.causality_matrix  # DataFrame
print(causality_matrix.data)
# Output example:
#       x1   x2
# y1  1.0 -1.0
# y2  0.0  1.0
```

---

## Key Distinctions: Builder vs. Orchestrator vs. Simple Granger

| Aspect | **Builder** | **Orchestrator** | **Simple Granger** |
|--------|-----------|-----------------|-------------------|
| **Purpose** | Fluent config interface | Direct multitask engine | Pair-wise statsmodels |
| **API Complexity** | High-level method chains | Lower-level fit() params | Simple fit() call |
| **Multi-output** | ✓ Yes | ✓ Yes (primary) | ✗ No (per pair) |
| **Constraints** | ✓ Yes (via relations) | ✓ Yes | ✗ No |
| **Regularization** | ✓ Yes (L1, lag-dependent) | ✓ Yes | ✗ No |
| **Callbacks** | ✓ Yes | ✓ Yes | ✗ No |
| **Config files** | ✓ Yes (from_file, from_config) | ✗ Manual dicts | ✗ Manual dicts |
| **Backend choice** | "pytorch" \ "tensorflow" \ "sklearn" | "pytorch" \ "tensorflow" \ "sklearn" | Fixed (statsmodels) |
| **Hyperopt** | ✓ Yes | ✓ Yes | ✗ No |
| **Use case** | Config-driven workflows | Fine-grained control | Quick exploration |

**In Summary:**
- **Use Builder** when you have YAML/JSON configs or prefer method chaining.
- **Use Orchestrator** when you need direct Python control or are building custom workflows.
- **Use Simple Granger** for rapid pair-wise tests, validation, or when statsmodels is preferred.

---

## Simple Test Examples

### Minimal Test: MultiTaskGrangerBuilder with Synthetic Data

```python
import pandas as pd
import numpy as np
from complex_granger_analysis.api import MultitaskGrangerBuilder

# Create synthetic dataset
np.random.seed(42)
n_samples = 100
x1 = np.random.randn(n_samples).cumsum()  # Random walk
x2 = x1[:-1].mean() * np.ones(n_samples) + 0.5 * np.random.randn(n_samples)  # Dependent on x1+noise
df = pd.DataFrame({"x1": x1, "x2": x2})

# Run Granger analysis
output = (MultitaskGrangerBuilder(backend="sklearn")
    .data(df)
    .variables(causes=["x1"], effects=["x2"], tested_causes=["x1"])
    .lag(LagConfiguration(max_lag=3))
    .model({"max_iter": 100})
    .fit()
)

print(output.results.causality_matrix.data)
```

### Test with Config File

```python
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

# Load from config file
cfg = BuilderConfigLoader.load_file("config.json")

# Run analysis
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()

# Examine results
print("Causality Matrix:")
print(output.results.causality_matrix.data)
print("\nP-values:")
print(output.results.p_value_matrix)
```

### Test with Sweep (Multiple Configurations)

```python
from complex_granger_analysis.api import TestGroupConfigIterator, MultitaskGrangerBuilder
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
it = TestGroupConfigIterator.from_file("group_config.json")

for i, cfg in enumerate(it):
    output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
    print(f"Test {i}: {output.results.causality_matrix.data}")
```

### Test with Relations (Constraints)

```python
from complex_granger_analysis.api import MultitaskGrangerBuilder
from complex_granger_analysis.core.lag_config import LagConfiguration
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)

output = (MultitaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(LagConfiguration(max_lag=5))
    .model({"epochs": 30})
    .fit()
)

# Define constraints (relations): force x1→y=0, bound x2→y coefficients
relations = {
    ("y", "x1"): {"zero": True},
    ("y", "x2"): {"min_abs_sum": 0.5},
}

# This would be done through orchestrator directly or via config:
# See orchestrator.fit(relations=relations, ...)
```

### Quick Exploration with SimpleGrangerAPI

```python
from complex_granger_analysis.api import SimpleGrangerAPI
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
   causality = output.results.causality_matrix
   p_values = output.results.p_value_matrix
   print(causality.data)
   ```

---

## When to Use Each Component

| Task | Recommended | Why |
|------|-------------|-----|
| Load config from JSON | `BuilderConfigLoader` | Clean separation of concerns |
| Chain config steps | `MultitaskGrangerBuilder` | Readable, method chaining |
| Direct Python workflow | `MultiTaskGrangerAPI` | Full control, no builder overhead |
| Quick pair-wise test | `SimpleGrangerAPI` | Simple, no backends needed |
| Run multiple configs | `TestGroupConfigIterator` | Automatic sweep expansion |
| Extract causality results | `GrangerAnalysisResults` | Aggregates p-values, statistics |

---

## References

- [Configuration File Usage](config_file_usage.md) — Detailed config structure reference.
- [Configuration File Usage](config_file_usage.md) — Sweep expansion and iteration.
- [Backend Usage](backend_usage.md) — Creating components (callbacks, regularizers, constraints).
- [Components Loading](components_loading.md) — Backend-specific component formats.

