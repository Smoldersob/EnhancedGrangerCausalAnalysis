# Lag Selection: Methods and Interpretation

## Purpose

Lag selection is an auxiliary preprocessing step. It proposes which
past values may be supplied to the final model.

The selectors are heuristics. They aim to reduce the size of the later
optimization problem while retaining useful predictive history. They do
not guarantee an optimal lag set for the final model.

The auxiliary model used for selection is not the final Granger model.
The final model may apply additional constraints, regularization, or a
different architecture.

Selecting a lag does not establish causality. Excluding a lag does not
prove that it is unnecessary for every possible final model.

## Available selectors

| Registered name | Class | Result |
| --- | --- | --- |
| `ic` | `ICLagSelector` | Separate cumulative lag ranges per output/predictor pair |
| `cv` | `CVLagSelector` | Separate cumulative lag ranges per output/predictor pair |
| `common_order_ic` | `CommonOrderICLagSelector` | One common order selected by aggregated IC scores |
| `common_order_cv` | `CommonOrderCVLagSelector` | One common order selected by aggregated CV losses |
| `naive_var` | `NaiveVARLagSelector` | Historical VAR-based common-order procedure |
| `var` | Alias of `NaiveVARLagSelector` | Historical configuration compatibility |

`VARLagSelector` remains a Python alias of `NaiveVARLagSelector`.

The historical VAR method and the new common-order methods are
alternatives. They do not use interchangeable scoring rules.

## Input and output contract

Selectors implement:

```python
fit(
    data: pd.DataFrame,
    *,
    outputs: Sequence[str] | None = None,
) -> LagSpec
```

Rows must be chronologically ordered observations. Columns must have
unique string names.

All supplied columns are candidate predictors. `outputs` selects which
response variables are requested; `None` requests all columns.

A result contains explicit lag tuples:

```python
{
    "y": {
        "y": (1, 2),
        "x1": (1, 2, 3),
        "x2": (),
    }
}
```

Rules:

- Lag `d` means the predictor value at time `t - d`.
- Tuples are sorted, unique non-negative integers.
- An empty tuple excludes that predictor for the output.
- Own-variable lag 0 is forbidden.
- Tuples contain actual values, not range endpoints.
- Every requested output includes every supplied predictor key.

The selector does not construct the final dataset or coefficient mask.

## Per-pair IC and CV selection

### Initial search

For each output:

1. Start with the smallest permitted AR model.
2. Evaluate cumulative own-history ranges.
3. Screen each other predictor conditional on the selected AR component.
4. Combine the selected ranges.
5. Optionally apply backward pruning.

Own-history candidates use:

```text
(1,), (1, 2), (1, 2, 3), ...
```

`min_ar_order=0` permits an empty initial AR component.

External-predictor candidates use:

```text
Without lag 0:
(), (1,), (1, 2), ...

With lag 0:
(), (0,), (0, 1), (0, 1, 2), ...
```

All configured orders are considered. Failure to accept an intermediate
candidate does not stop the search.

External predictors are initially screened independently against the
selected AR component. A predictor useful only jointly with another
predictor can therefore be missed.

### Addition references

`addition_reference` controls the improvement reference:

- `initial`: the smallest permitted AR model for AR expansion, or the
  selected AR-only model for external-predictor expansion;
- `current`: the best candidate accepted so far in that expansion search.

A candidate must also improve on the best accepted score. Passing the
initial threshold alone does not allow replacing a better model with a
worse one.

Positive improvement thresholds can retain a model other than the
absolute score minimum. `None` disables the minimum improvement margin,
but numerical comparison tolerances still apply.

If the initial baseline cannot be scored, the first scorable candidate
becomes the comparison baseline. This recovery is recorded through
candidate-failure diagnostics.

### Tail pruning

Each pruning iteration evaluates removal of the largest active lag from
each eligible predictor.

The best acceptable reduction is selected, and the process repeats.

- External predictors may become empty.
- Own-history terms retain at least `min_ar_order` lags.
- Interior lags are not removed independently.
- The comparison response window does not change during pruning.

`pruning_reference="initial"` compares reductions against the combined
model at the start of pruning.

`pruning_reference="current"` compares each reduction against the most
recent accepted model. Per-step deterioration can accumulate under this
policy.

If the combined model cannot be scored, the implementation attempts
recovery through a scorable one-step reduction. It does not guarantee
recovery when several removals would be necessary before any valid
candidate can be evaluated.

## IC scoring

IC selection fits auxiliary OLS regressions with an intercept.

The implemented score omits common likelihood constants:

```text
AIC-based score = n * log(residual_variance) + 2 * k
BIC-based score = n * log(residual_variance) + log(n) * k
```

`complexity_measure` determines `k`:

- `columns`: number of regression columns including the intercept;
- `rank`: numerical rank of the augmented design.

For rank-deficient designs, `columns` is an explicit penalty on
representation size, rather than a count of independent estimable
regression parameters.

A relative residual-variance floor prevents degenerate logarithms and
limits the numerical preference for almost exact fits. Its use is
reported in diagnostics.

Models with no residual degrees of freedom are not used for IC scoring.

## CV scoring

CV uses the package's NumPy-only forward-chaining splitter.

Configuration includes:

- `cv_folds`;
- `cv_test_size`;
- `cv_gap`;
- `cv_max_train_size`.

Training observations precede validation observations. With no training
window limit, the training window expands.

Candidate estimation uses minimum-norm OLS. `cv_metric="mae"` changes
evaluation, not the fitting objective.

Rank-deficient and underdetermined training fits are allowed if their
predictions and validation loss remain computable.

Intercept-only candidates are trained and evaluated within the same
folds as other candidates.

A candidate that fails in one fold is rejected as a whole. It is not
compared using only the remaining folds.

CV does not fall back to IC.

The validation procedure uses observed lagged histories. It is not a
recursive multi-step forecast simulation.

## Common-order selection

Common-order selectors search:

```text
p = 1, 2, ..., max_lag
```

Candidate `p` contains every positive lag from 1 through `p` for every
predictor. One order is selected jointly for all requested outputs.

For example, selecting order 3 produces:

```python
{
    "y1": {
        "y1": (1, 2, 3),
        "y2": (1, 2, 3),
        "x": (1, 2, 3),
    },
    "y2": {
        "y1": (1, 2, 3),
        "y2": (1, 2, 3),
        "x": (1, 2, 3),
    },
}
```

The auxiliary structure is VAR-like when inputs and outputs coincide,
and VARX-like when additional input variables are present.

The X designation does not establish statistical exogeneity of a signal.

### Common-order IC

`CommonOrderICLagSelector` evaluates the same auxiliary IC score used by
`ICLagSelector` for each requested output and aggregates the scores.

This is not the classical joint VAR criterion: it does not require the
log determinant of the full residual covariance matrix.

### Common-order CV

`CommonOrderCVLagSelector` aggregates validation losses across outputs.

With `normalize_outputs=True`, each training fold determines its response
scale. Validation observations do not determine normalization.

The scale is bounded below by `scale_floor`. A nearly constant training
response can therefore lead to large normalized validation errors if
the validation response subsequently changes.

Set `normalize_outputs=False` to aggregate losses in their original
units. Outputs with larger scales may then dominate the result.

### Output aggregation

Output weights are equal by default. Explicit `output_weights` must
contain exactly the requested output names and have positive finite
values. Weights are normalized internally.

Redundant output variables can influence the aggregate repeatedly.
Weighting is explicit; outputs are not automatically removed.

An order is valid only if every requested output can be scored.

Common-order selectors do not use per-pair addition or pruning thresholds.

## Historical VAR selection

`NaiveVARLagSelector` preserves the earlier procedure:

1. Optionally center all supplied variables.
2. Fit a separate VAR for every order from `min_lag` to `max_lag`.
3. Evaluate AIC, BIC, HQIC, and FPE.
4. Find the optimal order for each criterion.
5. Return the maximum or minimum of those orders according to `maximum`.

All supplied columns participate as endogenous variables. The `outputs`
argument restricts the returned specification, not the fitted VAR system.

Unlike the new common-order selectors, the historical procedure retains
candidate-dependent sample trimming through separate `VAR.fit(p)` calls.

`min_lag` restricts candidate orders. A selected order `p` still returns
the complete positive range `1..p`.

Expected candidate failures are recorded. Orders without four finite
criteria are excluded. If no valid order remains, selection raises an
error rather than returning the first candidate.

This corrected failure handling can expose earlier experiments in which
the old code returned an order despite unsuccessful criterion evaluation.

The method remains subject to statsmodels VAR requirements, including
the numerical behavior of residual covariance calculations.

## Zero-lag behavior

Own-variable lag 0 is always forbidden.

For per-pair IC/CV, enabling lag 0 allows cumulative external-predictor
searches to start at 0.

For common-order IC/CV, cross-variable lag 0 is included in every
candidate when enabled; there is no separate inclusion/exclusion search.

For historical VAR, cross-variable lag 0 is added to the returned
specification by configuration policy. It is not assessed by the fitted
positive-order VAR models.

The engine must also permit lag 0 in its final configuration.

## Numerical failures and redundancy

Rank deficiency is diagnostic information, not an automatic reason to
reject an auxiliary OLS model.

However, computable coefficients do not guarantee:

- a finite scoring criterion;
- stable validation predictions;
- identifiable individual coefficients;
- a suitable representation for every final model.

Expected numerical failures reject a candidate. Invalid configuration
and programming errors propagate rather than being hidden as poor scores.

If no valid candidate remains, selection fails explicitly.

No selector silently changes its criterion or removes original signal
names to construct an independent basis.

## Relationship to LagEngine

Each selector processes one independent dataset.

The engine then:

1. validates each returned specification;
2. aggregates selections across datasets;
3. applies manual overrides;
4. validates the final lag specification;
5. builds features, masks, and metadata.

A common-order selector returns a common order within one dataset.
Subsequent engine aggregation and pair overrides can produce a final
specification that is no longer uniform across pairs.

Dataset-level parallelism is managed by the engine. Selector searches
remain sequential.

## Interpretation and limitations

A selected lag means that the heuristic retained it under its auxiliary
model, criterion, candidate space, and data.

An excluded lag is not guaranteed to be unnecessary for:

- a nonlinear model;
- a constrained model;
- a regularized model;
- a model using different interactions or conditioning variables.

A smaller lag specification reduces the candidate representation but
does not guarantee better final optimization or predictive performance.

Manual overrides allow domain knowledge to modify the selected proposal.

When reporting an experiment, record the selector name, constructor
settings, aggregation policy, overrides, and any numerical failures or
stabilization policies that affected selection.

Lag exclusion is not an independently executed negative Granger test.

## Related documentation

- [Data Preprocessing](data_preprocessing.md)
- [Configuration File Usage](config_file_usage.md)
- [API Usage](api_usage.md)
- [Results and Metrics](results_and_metrics.md)