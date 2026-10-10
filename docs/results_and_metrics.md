# Results and Metrics

This document describes how fitted Granger models are represented, how
causality metrics are calculated, and how to inspect or threshold the results.
The main result-handling classes are:

- `GrangerAnalysisResults`, which stores fitted-model outputs and evaluates
  metrics.
- `ModelSnapshot`, which stores predictions, weights, and optional metadata for
  one fitted model.
- `PreparationContext`, which stores preprocessing information required to
  restore variables removed before model fitting.
- `CausalityMatrix`, which wraps a metric result in a labelled
  effects-by-causes `pandas.DataFrame`.

The implementation is in:

- `enhanced_granger_analysis/results/granger_results.py`
- `enhanced_granger_analysis/results/metrics.py`
- `enhanced_granger_analysis/results/causality_matrix.py`

## Result lifecycle

A multitask analysis produces an unrestricted **base model** and one restricted
**reference model** for each tested cause. The restricted model omits the cause
being tested while retaining the rest of the model configuration.

```text
prepared data
    |
    +--> base model --------------------+
    |                                    |
    +--> reference model without x1 ----+--> metric matrix
    +--> reference model without x2 ----+
```

`GrangerAnalysisResults` stores these model outputs in snapshots:

```python
results.set_base_snapshot(
    base_predictions=base_predictions,
    base_weights=base_weights,
    metadata={"backend": "pytorch"},
)

results.set_reference_snapshot(
    cause="x1",
    reference_predictions=x1_reference_predictions,
    reference_weights=x1_reference_weights,
)
```

The result object also stores:

- `effects`: output/effect variable names.
- `causes`: tested cause variable names.
- `mask`: coefficient activity mask.
- `col_offsets`: boundaries of the coefficient block belonging to each cause.
- `y_true`: observed target values.
- `X_true`: design matrix used for covariance-based metrics.
- `preparation_context`: optional preprocessing and removed-variable metadata.

## Base and reference snapshots

`ModelSnapshot` contains:

```python
ModelSnapshot(
    predictions=np.ndarray,
    weights=np.ndarray | None,
    metadata=dict,
)
```

The unrestricted model is available through:

```python
results.base_snapshot
results.base_predictions
results.base_weights
```

Restricted models are keyed by cause name:

```python
results.reference_snapshots["x1"]
results.ref_predictions["x1"]
results.ref_weights["x1"]
```

The predictions from the base and reference models must refer to the same
observations and use the same effect ordering. Their residuals are compared
when computing RSS-based metrics.

## Degrees of freedom

The result object supports two restriction-counting modes:

```python
results.set_dof_method("structural")
results.set_dof_method("active_support")
```

### `structural`

Counts the coefficient positions selected by `mask` for each cause and effect.
This is the default mode and reflects the configured model structure.

### `active_support`

Counts only structurally selected coefficients whose corresponding base-model
weight is non-zero. This mode is useful when sparsity or regularization makes
the fitted active support different from the configured structure.

`n_restrictions_for(cause)` returns one restriction count per effect. The
restriction count is used by F, Wald, and likelihood-ratio p-value
calculations.

`n_unrestricted_parameters()` returns the unrestricted model width. It uses the
base-model weights when available and falls back to the width of `X_true`.

## Available metrics

Metrics are registered by name and can also have aliases. Names are normalized
to lowercase by `resolve_metric`.

| Metric | Aliases | Description |
| --- | --- | --- |
| `f_statistic` | `f` | RSS-based Granger F statistic for each effect and cause. |
| `f_statistic_p_value` | `f_test_p_value`, `f_p_value`, `p_value_f` | Upper-tail p-values derived from the F statistic. |
| `wald_test` | `wald`, `wald_statistic` | Wald statistic based on base weights and OLS weight covariance. |
| `wald_test_p_value` | `wald_statistic_p_value`, `wald_p_value`, `p_value_wald` | P-values derived from the Wald statistic. |
| `likelihood_ratio` | `likelihood_ratio_test`, `lr`, `lr_test` | RSS-based likelihood-ratio statistic. |
| `likelihood_ratio_p_value` | `likelihood_ratio_test_p_value`, `lr_p_value`, `p_value_lr`, `p_value_likelihood_ratio` | P-values derived from the likelihood-ratio statistic. |
| `p_value` | — | Uses the p-value companion of a cached statistic; defaults to F-test p-values. |
| `sign` | — | Sign of summed base-model coefficients for each effect and cause. |
| `base_mse` | — | Base-model RSS divided by the number of samples, repeated for each cause. |
| `base_rss` | — | Base-model RSS, repeated for each cause. |

## Evaluating a metric

Use `evaluate` when a `CausalityMatrix` is required:

```python
f_statistics = results.evaluate("f_statistic")
f_p_values = results.evaluate("f_test_p_value")
wald = results.evaluate("wald_test")
likelihood_ratio = results.evaluate("likelihood_ratio")
```

The returned object contains:

```python
f_statistics.name       # "f_statistic"
f_statistics.data       # effects x causes DataFrame
f_statistics.metadata   # metric metadata
```

Metric callables can be supplied directly:

```python
def custom_metric(results) -> CausalityMatrix:
    values = ...
    return results.make_matrix(values, name="custom_metric")

custom = results.evaluate(custom_metric)
```

### Caching

Metric evaluation is cached by default:

```python
first = results.evaluate("f_statistic")
second = results.evaluate("f_statistic")
```

The second call reuses the cached matrix. To bypass the cache:

```python
fresh = results.evaluate("f_statistic", force=True)
```

To evaluate without storing the result:

```python
uncached = results.evaluate("f_statistic", cache=False)
```

Setting a new base or reference snapshot clears the metric cache and the
corresponding RSS cache because previously calculated values are no longer
valid.

## Restoring removed variables

Preprocessing can remove variables before fitting. If a
`PreparationContext` is attached, `evaluate` restores those variables by
default:

```python
results.set_preparation_context(context)

restored = results.evaluate("f_test_p_value")
```

To inspect only the fitted-variable result matrix:

```python
fitted_only = results.evaluate(
    "f_test_p_value",
    restore_removed=False,
)
```

Restoration is applied to the returned matrix; the cached metric itself
remains the result calculated from the fitted variables.

## Convenience properties

The most common metrics are exposed as properties. These properties evaluate
the metric and return the underlying `DataFrame`:

```python
results.p_value
results.f_test
results.wald_test
results.likelihood_ratio
results.sign
```

For example:

```python
p_values = results.p_value
signs = results.sign
```

The properties restore removed variables when a preparation context is
available.

## Thresholded causality results

`result` converts a metric matrix into a binary significance table. The default
metric is `f_test_p_value` and the default threshold is `0.01`:

```python
causality = results.result()
```

Equivalent explicit usage:

```python
causality = results.result(
    threshold=0.05,
    metric="f_test_p_value",
)
```

Cells with metric values below the threshold become `1`; other cells become
`0`. The returned object is a `pandas.DataFrame` indexed by effects and
labelled by causes.

To include the estimated causal direction:

```python
signed_causality = results.result(
    threshold=0.05,
    with_sign=True,
)
```

When `with_sign=True`, the binary significance table is multiplied by the
sign matrix. Significant relationships therefore become `1` or `-1`, while
non-significant relationships remain `0`.

## Working with `CausalityMatrix`

`CausalityMatrix` is a small wrapper around a `pandas.DataFrame`:

```python
matrix = results.evaluate("f_statistic")

matrix.data       # labelled DataFrame
matrix.name       # metric name
matrix.metadata   # metric-specific metadata
```

It can threshold its own data:

```python
binary = matrix.threshold(threshold=0.05)
```

This is useful when a metric matrix has already been evaluated and no result
restoration or sign multiplication is needed.

## Interpreting the metrics

### F statistic

The F statistic compares the restricted model's residual error with the
unrestricted model's residual error while accounting for:

- the number of restrictions for the cause;
- the number of unrestricted parameters;
- the number of samples.

Use the F statistic together with `f_test_p_value`; the statistic alone does
not provide a significance decision.

### Wald test

The Wald metric uses the base-model coefficient weights and an estimated
covariance matrix. It requires:

- a base snapshot with weights;
- non-empty `X_true`;
- compatible target, prediction, mask, and coefficient shapes.

If the required base data is unavailable, the implementation returns a matrix
of `NaN` values.

### Likelihood-ratio test

The likelihood-ratio metric compares the RSS of each restricted model with the
RSS of the unrestricted model. It requires a base snapshot and a reference
snapshot for every tested cause.

### Base RSS and MSE

`base_rss` and `base_mse` are diagnostic metrics. Because they describe the
unrestricted model rather than a cause-specific comparison, the same
per-effect value is repeated across all cause columns.

## Example workflow

```python
from enhanced_granger_analysis.api import MultitaskGrangerBuilder

results = (
    MultitaskGrangerBuilder(backend="pytorch")
    .data(data)
    .variables(
        causes=["x1", "x2"],
        effects=["y"],
        tested_causes=["x1", "x2"],
    )
    .fit()
)

# Continuous metrics.
f_statistics = results.evaluate("f_statistic")
p_values = results.evaluate("f_test_p_value")

# Binary and signed causality decisions.
causality = results.result(threshold=0.05)
signed_causality = results.result(
    threshold=0.05,
    with_sign=True,
)
```

## Troubleshooting

### Unknown metric

An unknown metric name raises `ResultsError` and includes the registered metric
names in the error message:

```python
results.evaluate("not_a_metric")
```

Use one of the registered names in the metric table or pass a callable that
returns `CausalityMatrix`.

### Missing base or reference model

F, likelihood-ratio, and RSS metrics require the relevant snapshots. A missing
base snapshot or reference snapshot causes the RSS helper methods to raise a
`ValueError`. This normally indicates that model fitting did not finish or
that the result object was constructed manually without all model outputs.

### Unexpected `NaN` Wald values

Check that the result contains:

1. `base_snapshot.weights`;
2. a non-empty `X_true`;
3. predictions and targets with matching rows;
4. a coefficient mask and column offsets consistent with the weight matrix.

