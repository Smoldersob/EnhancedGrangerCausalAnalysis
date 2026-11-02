# Model Contract

The model layer separates structural metadata, predictor parameters, input
omission, and training components.

Each built-in predictor jointly predicts all configured effects. Evaluating
individual Granger relations does not imply training a separate predictor for
each output.

## Structural specification

`ModelInputSpec` contains:

| Field | Meaning |
| --- | --- |
| `n_features` | Number of prepared, lag-expanded input columns |
| `n_outputs` | Number of model outputs |
| `causes` | Ordered effective input-variable names |
| `effects` | Ordered effective output-variable names |
| `cause_slices` | Named contiguous feature blocks |
| `feature_lags` | One actual lag per prepared feature |

Cause slices must cover all feature columns exactly once without overlap.
Dimensions, names, bounds, and lag values are validated.

The specification does not contain observations, fitted preprocessing objects,
tested-cause selection, or training diagnostics.

## Adapter and predictor

| Backend | Adapter | Native coefficient layout |
| --- | --- | --- |
| NumPy | `NumpyConditionalAdapter` | `(n_outputs, n_features)` |
| PyTorch | `PyTorchConditionalAdapter` | `(n_outputs, n_features)` |
| Keras | `KerasConditionalAdapter` | `(n_features, n_outputs)` |

Predictors own the input-control layer and coefficient layer. Adapters manage
construction, fitting, component resets, diagnostics, and parameter access.

Built-in constructors create an unfitted predictor without retaining training
observations.

## Training inputs

`fit(X, y)` receives prepared arrays:

- `X`: `(n_samples, n_features)`.
- `y`: `(n_samples, n_outputs)`.

Arrays must be real, finite, two-dimensional, and sample-aligned.

NumPy uses float64 training arrays. The shared Keras/PyTorch validation path
copies training arrays to float32 and rejects values that cannot remain finite
at that precision.

The adapter does not prepare lags, fit scalers, transform stationarity,
inverse-transform predictions, or calculate Granger significance tests.

## Lifecycle

| Operation | Predictor parameters | Input omission | Training components |
| --- | --- | --- | --- |
| `rebuild()` | Reinitialized | Baseline restored | Optimizer baseline recreated; callback-run preparation remains separate |
| `fit(X, y)` | Updated from current values | Preserved | Reset by default |
| `get_weights()` | Independent copies returned | Not exported | Not exported |
| `set_weights()` | Restored | Preserved | Preserved |
| `omit_variables()` | Preserved | Replaced | Preserved |
| `reset_components()` | Preserved | Preserved | Reset selectively |

`set_weights()` invalidates fitted status. Component reset and input omission
preserve fitted status.

`fit()` never implicitly rebuilds the predictor. Repeated fitting can warm-start
from existing parameters while resetting optimizer and callback state.

## Parameter export

Built-in linear adapters export a list containing:

- A kernel of shape `(n_features, n_outputs)`.
- An optional bias of shape `(n_outputs,)`.

The generic protocol leaves parameter ordering and shapes to each model
implementation. The built-in linear layout must not be assumed for every future
registered model.

Input masks, optimizer slots, callback state, and diagnostics are excluded from
weight export.

## Feature omission

`omit_variables()` accepts prepared feature-column indices, not unlagged
variable positions or cause names.

Use the named cause slice to omit all prepared features belonging to one cause.
Each call replaces the previous omission set.

Input omission and structural coefficient constraints are different mechanisms.
Omission suppresses input columns across outputs; structural constraints may
differ by output-feature pair.

Regularization acts on coefficient parameters and is not automatically disabled
for coefficients whose input columns are omitted.

## Training diagnostics

With `collect_history=True`, fitting returns `TrainingResult` containing:

- `history`.
- `metrics`.
- `n_iterations`.
- `stop_reason`.

Diagnostic content and aggregation are backend-specific.

With `collect_history=False`, fitting returns an empty diagnostic container.
An absent iteration count does not imply that no training occurred.

`TrainingResult` does not replace analysis snapshots, statistical tests,
p-values, or causality matrices.

## Extension boundary

Custom models must satisfy `GrangerModelProtocol` and backend-specific instance
validation; inheritance from the removed common model base is not required.

`CallbackManagedModelProtocol` additionally requires `reset_callbacks()`.
A custom model satisfying only the basic protocol is not required to implement
that additional operation.

Registry validation checks required members, not complete signature correctness,
numerical equivalence, or lifecycle semantics.
