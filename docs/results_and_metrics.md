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
- `cause_slices`: named feature slices used to select coefficient blocks
  for tested causes.
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

## Restriction counts and parameter counts

The result object supports two restriction-counting modes:

```python
results.set_dof_method("structural")
results.set_dof_method("active_support")
```

### Structural restrictions

`structural` counts active coefficient positions in the configured mask
for each tested cause and effect.

The cause's coefficient block is selected by name from `cause_slices`.
This is the default mode.

### Active-support restrictions

`active_support` counts structurally active coefficient positions whose
corresponding base-model weights are non-zero.

The current implementation uses an exact non-zero comparison. It does not
apply a numerical tolerance to identify small coefficients.

When base weights are unavailable, the current implementation falls back
to structural counting.

### Restriction-count contract

`n_restrictions_for(cause)` returns one restriction count per effective effect.

A tested cause must have a valid named feature slice. Missing or invalid
mappings raise an error rather than being interpreted as zero influence.

A zero restriction count means that no coefficient positions are counted
under the selected method. It is not, by itself, a statistical conclusion
about the tested relationship.

### Unrestricted parameter count

`n_unrestricted_parameters()` currently uses the feature width inferred
from base-model weights, or the width of `X_true` when weights are unavailable.

This is a model-width proxy used by the implemented statistics. It should
not be interpreted as a general effective-degrees-of-freedom estimator for
arbitrary regularized, constrained, or nonlinear models.

Availability of a statistic does not guarantee that its reference
distribution is appropriate for every model configuration.

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

## Coefficient-derived signs

The `sign` metric summarizes the sum of base-model coefficients in each
cause's named feature block.

It does not report the sign of the strongest individual lag coefficient
and does not determine the direction of the causal arrow.

A cause may have positive and negative lag coefficients whose contributions
cancel in the sum. The resulting sign is therefore a summary of the block,
not a complete description of lag-dependent effects.

For effective fitted variables, coefficient sums with absolute value below
`1e-12` retain the current convention of sign `+1`.

This near-zero convention differs from removed-variable restoration,
which uses sign `0` when no representative value is available.

The `results.sign` property evaluates the registered sign metric with
restoration enabled. Use `evaluate("sign", restore_removed=False)` to inspect
only fitted-variable signs.


# User interface properties


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

## Caching

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

Preprocessing can remove constant or exactly duplicated variables before
model fitting. A `PreparationContext` preserves the requested roles and
removal metadata.

Metric evaluation restores the requested effect and tested-cause axes by
default:

```python
restored = results.evaluate("f_test_p_value")
```

To inspect only effective fitted variables:

```python
fitted_only = results.evaluate(
    "f_test_p_value",
    restore_removed=False,
)
```

Restoration follows two rules:

1. Newly introduced entries use the metric's `removed_signal_value`.
2. Duplicate rows and columns are copied from available computed
   representatives.

Duplicate effects can copy representative rows. Duplicate tested causes
can copy representative columns only when the representative has an
evaluated tested-cause column.

A representative that was only a model input has no tested-cause result
column to copy.

Typical fill values are:

| Metric category | Fill value for newly restored entries |
| --- | --- |
| F, Wald, and likelihood-ratio statistics | `0` |
| Their p-value metrics | `1` |
| Sign | `0` |
| Metrics without an explicit fill value | `NaN` |

These values are restoration conventions, not results of additional fitted
models or independently executed statistical tests.

Restoration is applied to the returned matrix. The cached metric remains
the matrix calculated for effective fitted variables.

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

The statistical convenience properties and `sign` return DataFrames with
removed-variable restoration enabled when a preparation context is available.

The `sign` property delegates to the registered metric:

```python
results.evaluate("sign", restore_removed=True).data
```

To inspect signs for effective fitted variables only, use:
```python
results.evaluate("sign", restore_removed=False).data
```

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

To include coefficient-derived signs:

```python
signed_causality = results.result(
    threshold=0.05,
    with_sign=True,
)
```

When `with_sign=True`, the thresholded metric matrix is multiplied by the
sign metric after both have been restored to the requested axes.

The matrices must have identical effect and tested-cause labels.
An axis mismatch raises `ValueError`.

For effective fitted variables, signs are calculated from the sum of
base-model coefficients in each named cause block. Near-zero sums retain
the existing convention of `+1`.

Restored entries without an available representative use sign `0`.
Consequently, a zero in the signed result can reflect either a
non-significant relationship or a neutral sign introduced during restoration.
Use the unsigned result and evaluated metric matrix when this distinction
matters.

# Working with `CausalityMatrix`

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

# Interpreting the metrics

## F statistic

The F statistic compares the restricted model's residual error with the
unrestricted model's residual error while accounting for:

- the number of restrictions for the cause;
- the number of unrestricted parameters;
- the number of samples.

Use the F statistic together with `f_test_p_value`; the statistic alone does
not provide a significance decision.

## Wald test

The Wald metric uses the base-model coefficient weights and an estimated
covariance matrix. It requires:

- a base snapshot with weights;
- non-empty `X_true`;
- compatible target, prediction, mask, and coefficient shapes.

If the required base data is unavailable, the implementation returns a matrix
of `NaN` values.

## Likelihood-ratio test

The likelihood-ratio metric compares the RSS of each restricted model with the
RSS of the unrestricted model. It requires a base snapshot and a reference
snapshot for every tested cause.

## Base RSS and MSE

`base_rss` and `base_mse` are diagnostic metrics. Because they describe the
unrestricted model rather than a cause-specific comparison, the same
per-effect value is repeated across all cause columns.

# Example workflow

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

# Troubleshooting

## Unknown metric

An unknown metric name raises `ResultsError` and includes the registered metric
names in the error message:

```python
results.evaluate("not_a_metric")
```

Use one of the registered names in the metric table or pass a callable that
returns `CausalityMatrix`.

## Missing base or reference model

F, likelihood-ratio, and RSS metrics require the relevant snapshots. A missing
base snapshot or reference snapshot causes the RSS helper methods to raise a
`ValueError`. This normally indicates that model fitting did not finish or
that the result object was constructed manually without all model outputs.

## Unexpected `NaN` Wald values

Check that the result contains:

1. `base_snapshot.weights`;
2. a non-empty `X_true`;
3. predictions and targets with matching rows;
4. a coefficient mask and named cause slices consistent with the weight matrix.

# Evaluating an analysis

Evaluate statistical evidence, model behavior, and preprocessing decisions
separately.

## Base versus restricted models

The base model is unrestricted relative to the tested cause.
The corresponding reference model disables that cause's feature block.

The comparison asks whether the restriction increases residual error.
It does not ask whether the restricted model improves on the base model.

Base and reference predictions must correspond to identical observations
and effect ordering.

## Statistical evidence

Use a statistic together with its matching p-value metric.
An F statistic, Wald statistic, or likelihood-ratio statistic alone is not
a significance decision.

Interpret p-values under the assumptions of the selected test and fitted
model. Regularization, constraints, preprocessing, and model selection
should be reported as part of the analysis.

## Predictive diagnostics

Inspect base-model errors and changes in residual error after restriction.
These diagnostics help explain model behavior but are not interchangeable
with a calibrated statistical test.

## Removed variables

Inspect preparation metadata when a result contains restored variables.
A restored entry may follow a fill or duplicate-copy rule rather than
represent an independently fitted comparison.

## Signed results

Inspect unsigned significance and coefficient-derived signs separately
when their distinction matters.

A zero in a signed matrix may reflect a non-significant relationship or
a neutral sign introduced during restoration.

## Benchmarking

When a ground-truth relationship matrix is available, report classification
metrics together with the threshold, tested axes, sign convention, and
handling of removed variables.

The entries being compared must describe the same cause-effect pairs.

## Training and benchmark diagnostics

`TrainingResult` contains adapter-level fitting diagnostics. It does not replace
`GrangerAnalysisResults`, model snapshots, statistical metrics, or causality
matrices.

Evaluation against externally supplied ground truth is handled separately by
`MetricCalculator`. Version `2.7` changes its alignment and missing-value policy;
see [Ground-Truth Evaluation](ground_truth_evaluation.md).