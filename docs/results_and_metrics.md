# Results and Metrics

This page describes the results interface after the 2.12.2 cleanup. The
numerical formulas live in `statistics.py`, per-cause orchestration lives in
`metrics.py`, and degrees of freedom live in `dof.py` as a registered,
extensible strategy registry.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `granger_results.py` | Snapshot storage, preparation context, DOF selection, metric evaluation, caching, thresholding. No statistic formulas. |
| `dof.py` | `DofContext`, `DofResult`, the DOF strategy registry, and built-in strategies. |
| `statistics.py` | Numerical statistic values, distribution conversions, and geometry helpers. Pure array in/out; no container access. |
| `metrics.py` | Registered metrics that orchestrate evaluation and record metadata. Thin over `statistics.py`. |
| `causality_matrix.py` | Effects-by-causes DataFrame wrapper with metadata. |

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

A shared model is fitted once and the same snapshot object can be registered
for multiple causes:

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

## Coordinates and masking

`predictions` are stored in target reporting coordinates. For model-space
coefficients, metadata records `coefficient_space="model"` and
`model_predictions`; features and targets come from the attached scaled
preparation arrays. For raw coefficients, declare `coefficient_space="raw"`
and provide matching raw arrays. Ambiguous legacy coordinates are rejected.

A structural zero excludes a slope from the model. It is different from a
free coefficient estimated as zero.

## Degrees of freedom

DOF strategies are registered callables. Each receives a `DofContext` for one
cause and returns a `DofResult` with per-output `model_df_u`, `model_df_r`,
`restriction_df`, `residual_df`, `status`, `heuristic`, and optional effective
masks. Values are floats; fractional dimensions are allowed.

```python
from enhanced_granger_analysis.results.dof import DofResult, register_dof

@register_dof("external_dimensions", "external_df")
def external_dimensions(context):
    u = context.unrestricted_metadata["dof"]
    r = context.restricted_metadata["dof"]
    return DofResult(
        model_df_u=np.asarray(u["model_df"], dtype=float),
        model_df_r=np.asarray(r["model_df"], dtype=float),
        restriction_df=np.asarray(u["restriction_df"], dtype=float),
        residual_df=np.asarray(u["residual_df"], dtype=float),
        status=("user_supplied",) * len(context.effects),
        heuristic=True,
    )
```

### Selecting a strategy and passing parameters

```python
results.set_dof_method("structural")
results.set_dof_method("active_support", params={"active_support_tolerance": 1e-10})
results.set_dof_method("rank", params={"rank_rcond": 1e-10})
results.set_dof_method(external_dimensions)          # ready callable, no registration needed
```

Strategy-specific numerical options travel inside the `params` mapping and are
forwarded verbatim to `DofContext.params`. There are no dedicated container
attributes for individual strategies, so a custom strategy can define its own
parameter names without container changes. The setter invalidates metric and
rank-geometry caches; changing `results.dof_params` directly does not, so
prefer the setter.

Built-in strategies and their parameters:

| Strategy | Parameters | Definition |
| --- | --- | --- |
| `structural` | — | Counts structurally enabled slopes plus the intercept. |
| `active_support` | `active_support_tolerance` (default `1e-12`) | Counts slopes retained by the unrestricted support; heuristic. |
| `rank` | `rank_rcond` (default machine precision) | Counts retained design directions via nested SVD geometry. |

A strategy never changes which Wald algorithm runs; it only supplies
dimensions. `wald_projection` additionally requires that the selected DOF
match its geometry (see below).

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

An intentional heuristic for sparse fits, including L1 regularization. With
tolerance \(\epsilon\):

\[
S_U=M_U\land(|\hat B_U|>\epsilon),\qquad
M_U^*=M_U\land S_U,\qquad M_R^*=M_R\land S_U.
\]

Both effective masks use unrestricted weights. Restricted weights are not
thresholded independently. Missing unrestricted weights raise an error; there
is no structural fallback. Resulting p-values are heuristic, not
selection-adjusted or guaranteed to have classical calibration.

### Numerical rank

Rank uses structural masks, not coefficient-value thresholding. For each
output, form designs \(D_U,D_R\) from enabled feature columns and the optional
intercept:

\[
k_U=\operatorname{rank}(D_U),\qquad
q=\operatorname{rank}(D_U)-\operatorname{rank}(D_R),\qquad
df_{\mathrm{resid}}=n-k_U.
\]

SVD uses a shared absolute cutoff \(\tau=r_{\mathrm{cond}}s_{\max}(D_U)\).
With `rank_rcond=None`, the relative cutoff is machine precision times
`max(D_U.shape)`.

Rank counts can remain available even when independently truncated spaces
are not nested; in that case an incremental basis is unavailable and
`wald_projection` returns `NaN` for the affected output. Use
`results.rank_diagnostics()` to inspect ranks, counts, cutoffs, nesting, and
incremental-basis validity.

A zero restriction count means the selected policy has no testable
restriction; it is not evidence that the underlying process has no causal
relationship.

## Available metrics

| Metric | Aliases | Meaning |
| --- | --- | --- |
| `f_statistic` | `f` | RSS-based F value. |
| `f_statistic_p_value` | `f_test_p_value`, `f_p_value`, `p_value_f` | F upper-tail p-value. |
| `wald` | `wald_test`, `wald_statistic` | Coefficient Wald statistic (U weights, OLS-form covariance). |
| `wald_p_value` | `wald_test_p_value`, `wald_statistic_p_value`, `p_value_wald` | Coefficient Wald chi-square p-value. |
| `wald_projection` | `wald_svd`, `wald_projection_test` | Projection Wald statistic in retained linear spaces. |
| `wald_projection_p_value` | `wald_svd_p_value`, `wald_projection_test_p_value`, `p_value_wald_projection` | Projection Wald chi-square p-value. |
| `likelihood_ratio` | `likelihood_ratio_test`, `lr`, `lr_test` | RSS log-ratio value. |
| `likelihood_ratio_p_value` | `likelihood_ratio_test_p_value`, `lr_p_value`, `p_value_lr`, `p_value_likelihood_ratio` | LR chi-square p-value. |
| `unrestricted_rss`, `restricted_rss` | — | Residual sums of squares per effect and cause. |
| `unrestricted_mse`, `restricted_mse` | — | RSS divided by sample count. |
| `sign` | — | Summary sign of unrestricted coefficients in the tested cause block. |

There is no generic `p_value` metric or property. Choose the test explicitly.
`wald` and `wald_projection` are separate algorithms: the former tests slope
coefficients and needs U weights; the latter tests additional retained
prediction directions and needs neither weights nor R predictions.

## Statistical tests

### F and likelihood ratio

F and LR compare the registered U/R predictions for each cause. F uses the
selected strategy's `restriction_df` and `residual_df`; LR uses `n` and the
restriction count for its p-value. Both accept fractional DOF.

Unavailable inputs produce `NaN` with a `status` reason (`invalid_rss`,
`invalid_restriction_df`, `invalid_residual_df`, `zero_rss_both_models`).
When R attains a lower RSS than U, the value is clipped to zero and reported
through the status `restricted_rss_below_unrestricted`; this accommodates
imperfect optimization without hiding it.

### Wald (coefficient)

`wald` tests zero slope coefficients in the tested block. It uses U weights
and an OLS-form covariance computed from the enabled design via SVD, so no
Gram matrix is explicitly inverted. The design must have full retained column
rank and positive residual degrees of freedom; otherwise the cell is `NaN`
with status `coefficient_design_rank_deficient` or `invalid_residual_df`.

The covariance is OLS-form. For regularized, constrained, or incompletely
optimized fits its classical interpretation is not automatically valid; the
metadata marks such cells as `heuristic` when the selected DOF or diagnostics
indicate it.

### Wald (projection)

`wald_projection` tests additional retained prediction directions in U
relative to R. It uses existing U predictions and residual variance, not
coefficient representatives. It does not require stored weights or restricted
predictions, but requires both structural masks and matching snapshot
coordinates.

The selected DOF must agree with the projection geometry: `restriction_df`
must equal the geometry's `q` and `residual_df` must equal `n - rank_u`.
A custom strategy may be used, but incompatible dimensions produce `NaN`
with status `dof_incompatible_with_projection`.

For each output the implementation checks that a valid incremental basis
exists, that `q > 0`, that residual degrees of freedom are positive, that
predictions lie in the retained U space, and that residuals are numerically
orthogonal to that space (OLS equivalence). Failed checks produce `NaN` with
the reason recorded in `status`.

### Statuses, diagnostics, and p-values

Every test metric records:

- `n_restrictions`, `residual_df` — the exact DOF used;
- `dof_method`, `dof_params` — the selected strategy and its parameters;
- `status` — per-cell availability of the value or p-value;
- `statistic_status` — on p-value metrics, the underlying statistic status;
- `dof_status` — the DOF strategy status;
- `diagnostics` — per-cell lists of assumption warnings;
- `heuristic` — whether the cell relies on heuristic assumptions;
- `calibration` — always `not_verified`: a computable p-value is not a
  guarantee of a valid reference distribution.

P-value metrics convert the recorded statistics using the DOF stored with
them; they never recompute DOF and never coerce DOF to integers. `q=0`
produces `NaN` for both statistic and p-value. An infinite statistic yields
a zero p-value.

## Evaluation and caching

```python
f_matrix = results.evaluate("f_statistic", restore_removed=False)
p_wald = results.evaluate("wald_test_p_value", restore_removed=False)
p_projection = results.evaluate("wald_projection_p_value")
wald.metadata["status"]
```

Repeated evaluation reuses cached matrices, including cache hits with
restoration. `force=True` recomputes the requested metric; `cache=False` does
not store that evaluation. Neither is a general recursive invalidation
mechanism for dependencies. Registering or clearing snapshots invalidates all
snapshot-derived caches. Change DOF settings through `set_dof_method`; the
rank-geometry cache key includes the DOF parameters.

A metric callable can also be supplied:

```python
def custom_metric(results):
    values = ...
    return results.make_matrix(values, name="custom_metric")

custom = results.evaluate(custom_metric)
```

### Convenience properties

```python
results.f_p_value
results.wald_p_value
results.wald_projection_p_value
results.likelihood_ratio_p_value
results.wald_test
results.wald_projection
results.likelihood_ratio
results.sign
```

These return DataFrames with removed-variable restoration enabled.

## Restoration and thresholding

Preparation can remove constant variables, duplicates, or inputs with no
selected lags. Restoration uses requested axes and role-specific metadata:

- Statistics use neutral fill `0`; test-specific p-values use `1`.
- Sign uses neutral fill `0`; diagnostics without a fill use `NaN`.
- Duplicate axes copy available representative values.

A restored p-value of `1` is a reporting convention, not an independently
fitted test. Cached values and array-valued statistical metadata remain on
effective fitted-variable axes; metadata arrays record their axes in
`effective_effects` / `effective_causes` with `diagnostic_scope="effective"`.
Compare metadata with `restore_removed=False`.

### Thresholding and unavailable tests

```python
causality = results.result(
    threshold=0.05,
    metric="wald_projection_p_value",
)
```

`result()` thresholds p-values and applies an explicit policy for tests that
are unavailable (`NaN`):

| Policy | Behavior |
| --- | --- |
| `zero` (default) | Missing decisions become `0` — no detected edge. Compatible with ground-truth comparison. |
| `preserve` | Missing decisions stay `pd.NA`. |
| `raise` | Fails when any test is unavailable. |

The policy can be set per call (`unavailable_policy=...`) or container-wide
(`results.unavailable_policy`). Raw statistics and p-values always keep
`NaN`; the policy applies only to the binary decision layer. A zero decision
means "not rejected", not "proven absent".

A signed result multiplies significance by the restored sign matrix and
requires matching axes.

## Migration summary

| Removed interface | Replacement |
| --- | --- |
| `results.p_value` property | `results.f_p_value` or an explicit p-value metric. |
| `results.n_restrictions_for(cause)` | `results.dof_for(cause).restriction_df`. |
| `results.n_unrestricted_parameters(cause)` | `results.dof_for(cause).model_df_u`. |
| `results.residual_df_for(cause)` | `results.dof_for(cause).residual_df`. |
| `results.unrestricted_rss_per_output(cause)` | `results.rss_per_output(cause, "unrestricted")`. |
| `results.restricted_rss_per_output(cause)` | `results.rss_per_output(cause, "restricted")`. |
| `set_dof_method(..., rank_rcond=..., active_support_tolerance=...)` | `set_dof_method(..., params={"rank_rcond": ..., "active_support_tolerance": ...})`. |
| `results.rank_rcond` / `results.active_support_tolerance` attributes | `results.dof_params` entries. |
| `statistics.wald_test_value` / `estimate_masked_weight_covariance` | `statistics.coefficient_wald_value` (covariance computed internally via SVD). |
| `statistics.f_test_value` / `likelihood_ratio_test_value` | `statistics.f_statistic_values` / `likelihood_ratio_values` (value + status, float DOF). |
| `statistics.p_value_from_*` (integer-DOF converters) | `statistics.f_test_p_values` / `chi_square_p_values` (float DOF, status-aware). |
| `statistics.error_values` | `residual_sum_of_squares` divided by the sample count at the call site. |
| `_tested_mask_for`, `_effective_masks_for`, `_parameter_counts_for`, `_sign_matrix`, `_snapshot_for` | Structural masks via `_structural_masks_for`; DOF data via `dof_for`; signs via the `sign` metric; snapshots via `get_snapshot`. |

## Troubleshooting and interpretation

Missing U/R snapshots raise lookup errors. Check completion of the appropriate
training path and manual registration. With `train_only_unrestricted=True`,
restricted snapshots describe captured masked states, not independently
optimized restricted models; inspect `fit_performed` provenance before
interpreting RSS comparisons.

For unavailable cells, inspect `status`, `statistic_status`, `dof_status`,
DOF settings, coordinate metadata, and rank diagnostics. A singular
structural design is not handled by silently switching inference policies.

Report regularization, constraints, selection, preprocessing, optimizer
convergence, DOF strategy and parameters, and restoration choices. A
computable statistic is not a guarantee of a valid reference distribution;
rank counting does not establish inference validity for arbitrary nonlinear
or penalized estimators.

## References

Prediction diagnostics, statistical evidence, and evaluation against known
ground truth are separate tasks. See [Ground-Truth Evaluation](ground_truth_evaluation.md)
for alignment and missing-ground-truth rules and [Lag Selection](lag_selection.md)
for feature-selection semantics. `TrainingResult` records fitting diagnostics;
it does not replace the analysis result container.
