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

| Backend | Conditional adapter | Classical adapter | Native coefficient layout |
| --- | --- | --- | --- |
| NumPy | `NumpyConditionalAdapter` | `NumpyClassicalAdapter` | `(n_outputs, n_features)` |
| PyTorch | `PyTorchConditionalAdapter` | `PyTorchClassicalAdapter` | `(n_outputs, n_features)` |
| Keras | `KerasConditionalAdapter` | `KerasClassicalAdapter` | `(n_features, n_outputs)` |

Predictor architectures are defined in backend-local
`models/network_parts.py`. Conditional and classical adapters are defined
in `models/conditional.py` and `models/classic.py`, respectively.

The adapters manage construction, fitting, component resets, diagnostics,
and parameter access. Predictors own coefficient parameters and input
control.

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
| `omit_variables()` | Preserved | Replaced according to model-family semantics | Preserved |
| `add_variables()` | Preserved | Replaced according to model-family semantics | Preserved |
| `reset_components()` | Preserved | Preserved | Reset selectively |

`set_weights()` invalidates fitted status. Component reset and input omission
preserve fitted status.

`fit()` never implicitly rebuilds the predictor. Repeated fitting can warm-start
from existing parameters while resetting optimizer and callback state.

The baseline restored by `rebuild()` is model-dependent:

- Conditional adapters restore the unrestricted input configuration.
- Classical adapters restore the restricted autoregressive configuration.

Input-control operations preserve fitted status but do not imply that the
new configuration has been independently fitted.

## Parameter export

Built-in linear adapters export a list containing:

- A kernel of shape `(n_features, n_outputs)`.
- An optional bias of shape `(n_outputs,)`.

The generic protocol leaves parameter ordering and shapes to each model
implementation. The built-in linear layout must not be assumed for every future
registered model.

Input masks, optimizer slots, callback state, and diagnostics are excluded from
weight export.

## Input Control and Baseline Role

`omit_variables()` and `add_variables()` accept prepared feature-column
indices, not raw variable positions or cause names.

Use `ModelInputSpec.cause_slices` or `PreparedData.cause_slices` to select
all prepared columns belonging to a cause.

| Family | `base_model_role` | `omit_variables(indices)` | `add_variables(indices)` |
| --- | --- | --- | --- |
| Conditional | `"unrestricted"` | Disable exactly the selected input columns across outputs | Restore all inputs |
| Classical | `"restricted"` | Restore the own-history-only baseline | Enable exactly the selected additional columns relative to that baseline |

Selections replace previous input-control state; they do not accumulate.

Classical own-history terms bypass the additional-input control vector.
Each output requires at least one positive-lag own-history feature in X.

Input control is distinct from structural coefficient constraints.
Constraints may differ by output-feature pair, while dynamic additional
input selection remains vector-valued.

Regularization operates on coefficient parameters and is not automatically
disabled for connections inactive in prediction.

Changing input control does not restore baseline weights. Weight
restoration remains a separate operation.

Built-in `get_active_mask()` reports architectural/input-control activity
intersected with the structural allowance mask, in shape
`(n_outputs, n_features)`. It does not report fitted nonzero support.

See [Classical Granger models](classical_granger_models.md).

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
