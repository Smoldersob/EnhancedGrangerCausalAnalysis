# API Usage

The standard multi-output workflow has three entry points: a fluent builder, a convenience orchestrator, and explicit `AnalysisSpec`/`AnalysisRunner` execution. They share execution services; they are not separate model implementations.

## 1. Fluent builder

```python
from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder

builder = (
    MultiTaskGrangerBuilder(backend="numpy")
    .data(frame)
    .variables(causes=["x", "y"], effects=["y"], tested_causes=["x"])
    .stationarity(None)
    .lag(lag_config={"max_lag": 2, "use_lag_zero": False})
    .model({"model": "conditional_linear", "epochs": 100})
)
results = builder.fit()
```

Examples assume an existing numeric, time-ordered DataFrame named `frame`. Own output history is available only if that variable is also included among `causes`.

- `causes`: available model inputs.
- `effects`: jointly predicted outputs.
- `tested_causes`: inputs tested by restricted fits; must be a subset of causes.

Omitted roles use preprocessing defaults. Effective roles may differ after redundancy cleanup and lag selection.

`fit()` and its alias `run()` execute a complete analysis; they do not resume intermediate work. `fit(fit_restricted=False)` fits only the full model. Metrics requiring reference snapshots remain unavailable until the relevant restricted fits exist.

### Configuration updates and replacement

```python
builder.from_config({"model_config": {"epochs": 200}})
builder.replace_config({"backend": "numpy", "model_config": {"epochs": 100}})
```

`from_config()` updates supplied top-level fields. It is not a recursive merge: supplying `model_config` replaces that whole field. Unspecified settings and source data remain as configured.

`replace_config()` resets configuration and execution artifacts using a fresh builder. Source data is not retained unless explicitly supplied in the mapping; call `.data(frame)` again when needed. It is designed for the concrete builder, not as a generic reconstruction protocol for subclasses.

### Create a specification

```python
spec = builder.to_spec()
```

A specification excludes source data, prepared data, live models, weights, and analysis results. Creating it does not execute the analysis. Frozen dataclasses are not a promise of deep immutability of component instances or nested mappings.

### Explicit stages

```python
prepared = builder.prepare_data()
builder.prepare_components()
model = builder.build_model()
builder.initialize_weights()
base_weights = builder.fit_base()
builder.fit_restricted_models()
results = builder.get_results()
```

| Method | Return |
| --- | --- |
| `prepare_data()` | `PreparedData` |
| `prepare_components()` | Builder |
| `prepare_backend()` | Backend strategy |
| `prepare_constraint_data()` | `PreparedConstraintData` |
| `prepare_constraint()` | Backend constraint |
| `prepare_regularizer_data()` | `PreparedRegularizerData` or `None` |
| `prepare_regularizer()` | Backend regularizer or `None` |
| `prepare_callbacks()` | Resolved callback list |
| `build_model()` | Working model |
| `initialize_weights()` | Copied assigned weights or `None` |
| `fit_base()` | Copied fitted full-model weights |
| `fit_restricted(cause, ...)` | `None` |
| `fit_restricted_models()` | `GrangerAnalysisResults` |
| `get_results()` | Existing result container; no additional fit |

`fit_restricted()` defaults to restoring full-model weights. `start_with_initial=True` takes precedence when initial weights exist; otherwise `start_with_base=True` restores full-model weights. If neither restoration applies, current weights remain in use. A completed base fit is required regardless of the chosen start policy.

Configuration changes invalidate dependent builder artifacts. Prepare data again when a setter has discarded the result context. Do not assume an existing result container means a model has already been fitted.

Complete `fit()` delegates directly to the runner. Overriding stage methods in a builder subclass is not a supported way to customize that complete workflow.

## 2. Convenience orchestrator

```python
from enhanced_granger_analysis.api.orchestrator import MultiTaskGrangerAPI

results = MultiTaskGrangerAPI(backend="numpy").fit(
    data=frame,
    causes=["x", "y"],
    effects=["y"],
    tested_causes=["x"],
    stationarity=None,
    lag_config={"max_lag": 2, "use_lag_zero": False},
    model_config={"model": "conditional_linear", "epochs": 100},
    fit_restricted=True,
)
```

The orchestrator creates an `AnalysisSpec` and calls `AnalysisRunner.run_analysis()`. It does not create a builder or expose intermediate models. Its existing public `fit()` signature is retained. Do not assume optional convenience methods discussed during design are implemented.

## 3. Specification and runner

```python
from enhanced_granger_analysis.configuration import AnalysisSpec
from enhanced_granger_analysis.execution import AnalysisRunner

spec = AnalysisSpec.from_mapping({
    "backend": "numpy",
    "causes": ["x", "y"],
    "effects": ["y"],
    "tested_causes": ["x"],
    "stationarity": None,
    "lag_config": {"max_lag": 2, "use_lag_zero": False},
    "model_config": {"model": "conditional_linear", "epochs": 100},
})
results = AnalysisRunner().run_analysis(spec=spec, data=frame)
```

Use `run_analysis()` for a complete analysis. The lower-level `run()` accepts an already prepared model and internal progress object; it is not an alias of `run_analysis()`.

A standalone runner does not update a builder from which `spec` was obtained. The runner does not retain the last model or cache preparation between calls.

## 4. Prepared data reuse

```python
prepared = builder.prepare_data()
results = AnalysisRunner().run_analysis(spec=builder.to_spec(), prepared_data=prepared)
```

A supplied `prepared_data` artifact takes precedence over source data. The caller must ensure it is appropriate for the intended configuration. The runner does not re-run lag selection or scaling to reconcile it with changed preparation settings.

The `reuse_data` field does not create an internal runner/orchestrator cache. Explicit reuse and the experiment script's external cache are different mechanisms.

## 5. Constraints, regularization, and initialization

```python
builder.relations([
    {"cause": "x", "effect": "y", "rule": {"type": "zero"}},
])
builder.constraint("mask_and_min_abs_sum")
builder.regularization({"type": "l1", "params": {"l1": 0.01}})
builder.initializer("zeros")
```

This illustrates configuration placement, not a scientifically useful combination of rules for every experiment. `relations` describes rules; `constraint_spec` selects their implementation. Empty relations preserve the preparation mask. `constraint_spec=None` is rejected. `regularizer_spec=None` disables regularization.

Initializers use the existing registry: names, supported mappings, classes, or `None`; not arbitrary instances/factories. `initialize_weights(None)` uses the stored specification; it is not a per-call disable override.

See [Components](components_loading.md) for family-specific contracts.

## 6. Results

```python
statistics = results.evaluate("f_statistic").data
p_values = results.evaluate("f_test_p_value").data
binary = results.result(threshold=0.01, metric="f_test_p_value")
signed = results.result(threshold=0.01, metric="f_test_p_value", with_sign=True)
```

`evaluate()` returns a `CausalityMatrix`; `result()` returns a DataFrame. Sign summarizes full-model coefficient sums within a cause's lagged block, not the direction of the causal arrow. Removed-variable restoration is a reporting operation, not an extra hypothesis test.

`FitOutcome` is not the public return of builder/orchestrator fitting. It contains one fit's model-scale predictions, complete copied weights, diagnostics, and timings. The runner converts it to inverse-scaled analysis snapshots. See [Execution architecture](execution_architecture.md).

## 7. Errors and builder state

Full builder execution synchronizes completed artifacts in a `finally` block. A failed restricted fit does not erase the completed full-model snapshot or its saved weights. Exceptions are propagated; partial results do not imply a successful complete analysis.

`AnalysisRunProgress` is internal bookkeeping, not a resumable session API. Exact intermediate private state is not guaranteed to match every historical builder failure path.

## 8. Separate baseline and future work

`SimpleGrangerAPI` remains a separate pair-wise statsmodels baseline. It does not use the standard multi-output runner described here.

Automatic HPO remains inactive. Existing HPO fields do not trigger search. Future HPO is intended to optimize model quality, not causal-analysis scores. Sessions and interchangeable high-level procedures remain future development, not current APIs.

Related: [Configuration](config_file_usage.md), [Migration](api_refactor_migration.md), [Results](results_and_metrics.md), [Lag selection](lag_selection.md).
