# Results and Metrics

This page describes the refactored results interface. Integration and regression
validation are pending. Apply the corresponding code changes before using the
new interface; this document does not establish that they are already present
in every execution path.

## Responsibilities

- `GrangerAnalysisResults` stores snapshots, preparation context, and cached
  metric values. It provides shared mask, coordinate, and design-geometry access.
- `ModelSnapshot` stores predictions, optional slope weights, the structural
  slope mask, intercept configuration, and provenance for one model state.
- `PreparationContext` connects results to preprocessing coordinates and
  removed-variable metadata.
- `metrics.py` registers metrics and orchestrates their evaluation.
- `statistics.py` implements numerical statistics and distribution conversions.
- `CausalityMatrix` wraps an effects-by-causes `pandas.DataFrame` and metadata.

Models remain joint multi-output predictors. Per-effect inference does not
introduce separately trained models for individual outputs.

## Snapshot lifecycle

Each tested cause identifies an unrestricted/restricted comparison:

```python
unrestricted = results.get_snapshot("x1", "unrestricted")
restricted = results.get_snapshot("x1", "restricted")
```

The canonical mappings are:

```python
results.unrestricted_snapshots
results.restricted_snapshots
```

There is no separate base/reference snapshot store. A shared model is fitted
once and the same snapshot object can be registered for multiple causes:

```python
results.register_snapshots(
    causes=results.causes,
    role="unrestricted",
    snapshot=shared_snapshot,
)
```

This stores references, not copies of prediction and coefficient arrays.
Use `register_snapshot(cause, role, snapshot)` for a single comparison.
Both methods invalidate RSS, metric, and rank-geometry caches.

Construct snapshots explicitly when assembling results manually:

```python
from enhanced_granger_analysis.results.granger_results import ModelSnapshot

snapshot = ModelSnapshot(
    predictions=predictions_raw.copy(),
    weights=weights_model.copy(),
    active_slope_mask=structural_mask.copy(),
    bias_enabled=True,
    metadata={
        "coefficient_space": "model",
        "model_predictions": predictions_model.copy(),
        "fit_performed": True,
    },
)
results.register_snapshot("x1", "unrestricted", snapshot)
```

Attach the matching preparation context for model-space snapshots.
Slope weights and masks use `(n_effects, n_features)` order. Predictions use
`(n_samples, n_effects)` order. Compared masks must be nested: restricted
positions must be a subset of unrestricted positions. Intercept settings must
match. Predictions and targets must use identical observations and output order.

`clear_snapshots()` removes both mappings and clears derived caches without
changing DOF settings or preparation context. Treat registered arrays and
coordinate metadata as immutable; replace snapshots through registration
methods rather than changing them in place.

### Shared training is not a separate result representation

`fit_base()`, `model.base_model_role`, and native `base_weights` remain execution
concepts. They select shared U or R fitting and support initialization of later
runs. They do not require a `results.base_snapshot` field.
Native initialization weights may include a bias and backend-specific arrays;
they are not interchangeable with `ModelSnapshot.weights`.

## Coordinates and masking

`predictions` are stored in target reporting coordinates. For model-space
coefficients, metadata records `coefficient_space="model"` and
`model_predictions`; features and targets come from the attached scaled
preparation arrays. For raw coefficients, declare `coefficient_space="raw"`
and provide matching raw arrays. Ambiguous legacy coordinates are rejected.

A structural zero excludes a slope from the model. It is different from a
free coefficient estimated as zero. Covariance estimation selects enabled
feature columns first; it does not invert the full unmasked design and then
zero covariance entries afterward.

## Degrees-of-freedom policies

```python
results.set_dof_method("structural")
results.set_dof_method("active_support")
results.set_dof_method("rank", rank_rcond=1e-10)
```

Use the setter to change the method or rank tolerance. It invalidates metric
and rank-geometry caches. Direct assignment to configuration fields does not
provide the same invalidation contract.

### Structural

For structural slope masks \(M_U\) and \(M_R\):

\[
T=M_U\land\neg M_R,\qquad
q_j=\sum_k T_{jk},\qquad
k_{U,j}=\sum_k M_{U,jk}+I_{\mathrm{bias}}.
\]

Counts use the actual snapshot pair. `cause_slices` preserves feature-block
metadata, but restriction counts are not silently clipped to a cause slice.

### Active support

This is an intentional heuristic for sparse fits, including L1 regularization.
With tolerance \(\epsilon\):

\[
S_U=M_U\land(|\hat B_U|>\epsilon),\qquad
M_U^*=M_U\land S_U,\qquad M_R^*=M_R\land S_U.
\]

Both effective masks use unrestricted weights. Restricted weights are not
thresholded independently. Count restrictions from
\(M_U^*\land\neg M_R^*\), and count unrestricted slopes from \(M_U^*\), plus
the enabled intercept. Bias is counted by configuration, not its fitted value.
Missing unrestricted weights raise an error; there is no structural fallback.

Wald uses the effective unrestricted mask for covariance estimation as well
as the effective tested mask for its hypothesis. It retains original fitted
weights and predictions; it does not refit the support-selected model.
Thresholding small nonzero values can therefore make the effective design
and fitted predictions imperfectly aligned. Resulting p-values are heuristic,
not selection-adjusted or guaranteed to have classical calibration.

The current configuration has no dedicated support-tolerance setter. When
changing it, invalidate derived values using the existing method setter:

```python
results.active_support_tolerance = 1e-10
results.set_dof_method("active_support")
```

### Numerical rank

Rank uses structural masks, not coefficient-value thresholding. For each
output, form designs \(D_U,D_R\) from enabled feature columns and the optional
intercept:

\[
k_U=\operatorname{rank}(D_U),\qquad
q=\operatorname{rank}(D_U)-\operatorname{rank}(D_R),\qquad
df_{\mathrm{resid}}=n-k_U.
\]

SVD uses a shared absolute cutoff
\(\tau=r_{\mathrm{cond}}s_{\max}(D_U)\). With `rank_rcond=None`, the relative
cutoff is machine precision times `max(D_U.shape)`.

Rank counts can remain available even when independently truncated spaces
are not nested. In that case an incremental basis is unavailable and
rank-Wald returns `NaN` with a warning for the affected output.
Use `rank_diagnostics()` to inspect ranks, counts, cutoffs, nesting, and
incremental-basis validity.

A zero restriction count means the selected policy has no testable restriction;
it is not evidence that the underlying process has no causal relationship.

## Available metrics

| Metric | Aliases | Meaning |
| --- | --- | --- |
| `f_statistic` | `f` | RSS-based F statistic. |
| `f_statistic_p_value` | `f_test_p_value`, `f_p_value`, `p_value_f` | F upper-tail p-value. |
| `wald_test` | `wald`, `wald_statistic` | Policy-specific Wald statistic. |
| `wald_test_p_value` | `wald_statistic_p_value`, `wald_p_value`, `p_value_wald` | Wald chi-square upper-tail p-value. |
| `likelihood_ratio` | `likelihood_ratio_test`, `lr`, `lr_test` | RSS-based LR statistic. |
| `likelihood_ratio_p_value` | `likelihood_ratio_test_p_value`, `lr_p_value`, `p_value_lr`, `p_value_likelihood_ratio` | LR chi-square upper-tail p-value. |
| `unrestricted_rss` | — | U residual sum of squares per effect and cause. |
| `restricted_rss` | — | R residual sum of squares per effect and cause. |
| `unrestricted_mse` | — | U RSS divided by sample count. |
| `restricted_mse` | — | R RSS divided by sample count. |
| `sign` | — | Summary sign of unrestricted coefficients in the tested cause block. |

There is no generic `p_value` metric. Choose the test explicitly; cache history
does not determine the test. `base_rss` and `base_mse` are removed rather than
aliased to U diagnostics, since a shared training base can also have role R.
Shared snapshots naturally produce repeated diagnostic columns.

## Statistical tests

### F and likelihood ratio

F and LR compare the registered U/R predictions for each cause. F uses the
selected policy's restriction count and unrestricted parameter count. LR
retains its RSS statistic and uses the selected restriction count for its
chi-square p-value.

The RSS-based implementations clamp negative improvement to zero when R has
lower RSS than U. This accommodates imperfect optimization; it does not make
an incompatible estimator classically calibrated.

### Wald: structural and active support

Wald uses unrestricted slopes and an OLS-form covariance on the enabled
design. All enabled U regressors, including nuisance regressors, participate
in covariance estimation. The tested block is selected afterward:

\[
W=\hat\beta_T^\top
\left[\widehat{\operatorname{Cov}}(\hat\beta)_T\right]^{-1}
\hat\beta_T.
\]

The active design must have full column rank and positive residual degrees
of freedom. Failures raise an error; neither policy silently switches to
rank mode. Covariance is computed only for outputs with tested coefficients.
Missing structural-mode weights yield unavailable values; active-support
preparation requires weights and raises if they are absent.

The covariance is OLS-form. For regularized, constrained, or incompletely
optimized fits its classical interpretation is not automatically valid.
`active_support` explicitly accepts a heuristic interpretation.

### Wald: rank

Rank-Wald tests additional retained prediction directions in U relative to R.
It uses existing U predictions and residual variance, not arbitrary coefficient
representatives. It does not require stored weights or restricted predictions,
but requires both structural masks and matching snapshot coordinates.

The numerical implementation is `statistics.rank_wald_test_value()`;
`metrics.wald_test()` orchestrates it. No Wald execution method remains on
`GrangerAnalysisResults`.

For each output it checks:

1. A valid incremental basis exists and `q > 0`.
2. Residual degrees of freedom are positive.
3. Existing predictions lie in the retained U space.
4. Residuals are numerically orthogonal to that space (OLS equivalence).

Failed compatibility checks produce `NaN` with a warning when `q > 0`.
Invalid input dimensions or nonfinite coordinate arrays raise an error.
The implementation does not refit, project predictions to replace them, or
silently reinterpret an incompatible fit.

### Wald metadata and p-values

The statistic records `dof_method`, tolerances, `hypothesis`,
`n_restrictions`, per-cell `status`, and `heuristic`.
Wald p-values use the restriction counts recorded with the statistic.
Metadata arrays follow effective fitted-variable axes, not restored axes.

`q=0` produces `NaN` for both statistic and p-value. A zero residual variance
with positive numerator gives an infinite statistic; the corresponding
positive-DF upper-tail p-value is zero. A zero numerator and zero variance
leave the statistic unavailable.

## Evaluation and caching

```python
wald = results.evaluate("wald_test", restore_removed=False)
p_wald = results.evaluate("wald_test_p_value", restore_removed=False)
p_f = results.evaluate("f_statistic_p_value")
p_lr = results.evaluate("likelihood_ratio_p_value")

wald.data
wald.metadata["status"]
```

Repeated evaluation reuses cached matrices. `force=True` recomputes the
requested metric; `cache=False` does not store that evaluation. Neither is
a general recursive invalidation mechanism for dependencies.
Registering or clearing snapshots invalidates all snapshot-derived caches.
Change DOF settings through the supported setter. Do not mutate predictions,
weights, masks, or preparation arrays in place after caching.

A metric callable can also be supplied:

```python
def custom_metric(results):
    values = ...
    return results.make_matrix(values, name="custom_metric")

custom = results.evaluate(custom_metric)
```

### Convenience properties

After applying the matching property migration:

```python
results.f_p_value
results.wald_p_value
results.likelihood_ratio_p_value
results.f_test
results.wald_test
results.likelihood_ratio
results.sign
```

These return DataFrames with removed-variable restoration enabled. The old
`results.p_value` property is removed. Use `evaluate()` for metadata or explicit
control of restoration.

## Restoration and thresholding

Preparation can remove constant variables, duplicates, or inputs with no
selected lags. Restoration uses requested axes and role-specific metadata:

- Statistics use neutral fill `0`; test-specific p-values use `1`.
- Sign uses neutral fill `0`; diagnostics without a fill use `NaN`.
- Duplicate axes copy available representative values.
- An input removed only in its predictor role can retain an effective output row.
- A representative not evaluated as a tested cause has no column to copy.

A restored p-value of `1` is a reporting convention, not an independently
fitted test. Cached values and array-valued statistical metadata remain on
effective fitted-variable axes. Compare metadata with
`restore_removed=False`.

```python
causality = results.result(
    threshold=0.05,
    metric="wald_test_p_value",
)
signed = results.result(
    threshold=0.05,
    metric="wald_test_p_value",
    with_sign=True,
)
```

`result()` returns a DataFrame. Values below the threshold are significant.
Its default metric remains `f_test_p_value`, with default threshold `0.01`.
Use explicit metric names when comparing tests. A signed result multiplies
significance by the restored sign matrix and requires matching axes.

`CausalityMatrix.threshold(threshold=0.05)` thresholds an already evaluated
matrix without additional restoration.

### Coefficient-derived signs

Sign summarizes unrestricted coefficients in the named cause block, using
positions present in U and removed in R. It is not the strongest lag's sign,
the direction of a causal arrow, or an active-support/rank hypothesis.
Opposing lag effects can cancel. Effective near-zero sums retain the existing
`+1` convention below absolute tolerance `1e-12`; restored unavailable signs
use `0`. A zero signed result can therefore represent nonsignificance or a
neutral restored sign.

## Migration summary

| Removed interface | Replacement |
| --- | --- |
| `set_base_snapshot(...)` | Construct once and use `register_snapshots(...)`. |
| `set_reference_snapshot(...)` | `register_snapshot(cause, "restricted", snapshot)`. |
| `set_unrestricted_snapshot(...)` | `register_snapshot(cause, "unrestricted", snapshot)`. |
| `set_restricted_snapshot(...)` | `register_snapshot(cause, "restricted", snapshot)`. |
| `base_snapshot`, `base_predictions`, `base_weights` on results | Select a cause and role with `get_snapshot(...)`. |
| `reference_snapshots`, `ref_predictions`, `ref_weights` | `restricted_snapshots` and its snapshot fields. |
| `reference_rss_per_output(cause)` | `restricted_rss_per_output(cause)`. |
| `base_rss_per_output()` | `rss_per_output(cause, role)`. |
| `base_rss`, `base_mse` metrics | Explicit restricted/unrestricted RSS or MSE metrics. |
| `p_value` metric / property | Explicit test-specific p-value metric / property. |
| `rank_wald_per_output()` | Evaluation through the registered `wald_test` metric. |

## Troubleshooting and interpretation

Missing U/R snapshots raise lookup errors. Check completion of the appropriate
training path and manual registration. With `train_only_unrestricted=True`,
restricted snapshots describe captured masked states, not independently
optimized restricted models; inspect `fit_performed` provenance before
interpreting RSS comparisons.

For unavailable Wald cells, inspect `status`, DOF settings, coordinate
metadata, and rank diagnostics. Restored cells must be distinguished from
computed cells. A singular structural/support design is not handled by
silently switching inference policies.

Report regularization, constraints, selection, preprocessing, optimizer
convergence, rank tolerance, support tolerance, and restoration choices.
A computable statistic is not a guarantee of a valid reference distribution.
Rank counting does not establish inference validity for arbitrary nonlinear
or penalized estimators.

Prediction diagnostics, statistical evidence, and evaluation against known
ground truth are separate tasks. See [Ground-Truth Evaluation](ground_truth_evaluation.md)
for alignment and missing-ground-truth rules and [Lag Selection](lag_selection.md)
for feature-selection semantics. `TrainingResult` records fitting diagnostics;
it does not replace the analysis result container.
