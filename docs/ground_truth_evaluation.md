# Ground-Truth Evaluation

`MetricCalculator` compares exported causality matrices against ground truth.
This benchmark evaluation is separate from statistical metrics and p-values
calculated inside the Granger-analysis result subsystem.

## Alignment

Rows represent effects and columns represent causes. Labels are normalized to
strings.

Row labels are aligned over their sorted union. Column labels are aligned
independently over their sorted union. Rectangular matrices are not forced into
square matrices.

Nonzero numeric values are interpreted as edges. This is binarization, not
probability or p-value thresholding.

## Evaluation domain

| Option | Default | Meaning |
| --- | --- | --- |
| `evaluate_only_known_gt` | `True` | Restrict evaluation using ground-truth coverage |
| `gt_nan_means_unknown` | `True` | Treat missing ground-truth values as unknown when restricting evaluation |
| `drop_diagonal` | `False` | Remove diagonal entries only when aligned row/column label sets are equal |
| `double_edges` | `False` | Select reciprocal-pair discrepancy counting |

With the defaults, only non-missing ground-truth cells are evaluated.
Predictions outside that domain do not contribute false positives.

Missing predictions become zero. A missing prediction at a known positive
ground-truth cell therefore contributes a false negative.

With `evaluate_only_known_gt=True` and `gt_nan_means_unknown=False`, evaluation
covers the original ground-truth row-by-column rectangle; missing values inside
that rectangle become zero.

With `evaluate_only_known_gt=False`, the complete aligned domain is evaluated
and missing ground truth becomes zero.

For unequal row/column label sets, `drop_diagonal=True` currently does not remove
matching-label cells. Do not interpret it as unconditional self-edge removal
for every rectangular matrix.

## Reported metrics

`evaluate()` returns:

- `tp`, `fp`, `tn`, and `fn`.
- `fdr`, `tpr`, and `fpr`.
- `accuracy`, `precision`, `recall`, and `f1`.
- `shd`.
- `n_evaluated`.

`n_evaluated` counts evaluated cells after masking, not variables or positive
edges. Record it alongside reported scores.

Rate metrics return `0.0` when their denominator is zero. Such a value may
represent an empty-denominator case rather than an informative measured rate.

## SHD interpretation

For unordered pairs whose two directional cells are both evaluated:

- `double_edges=True` counts directed-cell mismatches separately.
- `double_edges=False` assigns one discrepancy whenever the two-direction pair
  differs.

The second rule applies to any reciprocal-pair mismatch, not only a single
reversed edge.

Remaining evaluated cells are counted individually. When only one direction is
known, its discrepancy is handled by the remaining-cell Hamming calculation.

State the counting setting explicitly when reporting SHD.

## Compatibility

The class exposes raw, aligned, and binary tables through `gt_raw`, `pred_raw`,
`gt_aligned`, `pred_aligned`, `gt_bin`, and `pred_bin`, together with `eval_mask`.

Consumers of the previous `ground_truth`, `predictions`, `adjusted_gt`, and
`adjusted_pred` attributes require migration.

The new defaults are not equivalent to the previous intersection alignment and
square-completion behavior. Historical benchmark scores should not be compared
without accounting for the changed evaluation domain.