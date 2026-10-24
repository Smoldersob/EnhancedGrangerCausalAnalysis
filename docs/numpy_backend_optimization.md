# NumPy Backend: Model Fitting and Solver Selection

The NumPy backend implements a multi-output linear predictor with optional
coefficient constraints and regularization. It selects a fitting method
according to the structure of the configured optimization problem.

This document describes the adapter with automatic solver selection,
`restricted_lstsq`, the iterative coefficient-update path, and the
`zero_only_mask()` constraint interface.

## Model

The predictor has the form

\[
\widehat{Y} = X_{\mathrm{masked}} W^\top + \mathbf{1}b^\top,
\]

where:

- \(X_{\mathrm{masked}}\) contains the prepared input features after feature
  omission;
- \(W\) has shape `(n_outputs, n_features)`;
- \(b\) has shape `(n_outputs,)` when an intercept is enabled;
- predictions have shape `(n_samples, n_outputs)`.

Input dimensions and feature/output metadata are provided through
`ModelInputSpec`. The adapter constructs its predictor immediately.
Construction does not fit the model.

Coefficients are initialized to zero. Construction validates the optional seed
but does not use it to randomize coefficients.

## Objective Function

The fitting objective is

\[
F(W,b)
=
\frac{1}{nm}
\left\|
X_{\mathrm{masked}}W^\top+\mathbf{1}b^\top-Y
\right\|_F^2
+
R(W),
\]

subject to the configured coefficient constraint, where:

- \(n\) is the number of samples;
- \(m\) is the number of outputs;
- \(R(W)\) is the regularization penalty, or zero when no regularizer is
  configured.

The data term is MSE averaged over both samples and outputs. The regularizer
returns its penalty separately; the adapter adds that value to the MSE.

The intercept is not passed to the coefficient regularizer or constraint.

## Feature Omission and Coefficient Constraints

The backend distinguishes two mechanisms:

1. Feature omission applies an input mask shared by all outputs.
2. Coefficient constraints operate on the coefficient matrix and may differ
   between outputs.

Omitted input columns contribute nothing to predictions. A coefficient
constraint may additionally forbid individual output–feature connections.

Feature omission must not change during a `fit()` call. Change it between
fitting runs instead.

## Solver Configuration

The `solver` parameter accepts:

| Value | Behavior |
|---|---|
| `"auto"` | Selects a solver from the configured problem structure. |
| `"restricted_lstsq"` | Requires a problem supported by the direct least-squares path. |
| `"coefficient"` | Uses iterative gradient/proximal coefficient updates. |

The selected method is available after solver preparation through:

```python
model.solver_used_
```

Solver selection occurs once per `fit()` call. It is not repeated between
epochs.

### Automatic Selection

With `solver="auto"`, the backend selects `restricted_lstsq` only when all of
the following hold:

- fitting uses a full batch;
- `network.regularizer is None`;
- no coefficient constraint is configured, or the constraint returns a mask
  from `zero_only_mask()`.

A full batch means that `batch_size` is `None` or is at least the number of
training samples.

All other configurations use `coefficient`.

An existing regularizer object blocks automatic direct fitting even if its
configured penalty strength is zero. The adapter does not inspect arbitrary
regularizer parameters to determine whether a penalty is inactive.

Explicitly requesting `restricted_lstsq` for an unsupported configuration
raises an error instead of silently dropping a penalty or constraint.

## Pure Zero-Mask Constraint Interface

A constraint may expose its complete feasible set through:

```python
def zero_only_mask(self) -> np.ndarray | None:
    ...
```

The return value has the following meaning:

- A boolean array with shape `(n_outputs, n_features)` describes a constraint
  consisting exclusively of coefficient zeroing.
- `False` requires the corresponding coefficient to equal zero.
- `True` leaves the corresponding coefficient unrestricted.
- `None` means that the constraint does not declare this representation.

The returned mask must describe the entire constraint, not merely its
zeroing component.

For example, a mask-and-minimum-absolute-sum constraint may return its mask
when no additional sum rules are active. It must return `None` when active
rules impose further restrictions.

The adapter validates the mask shape and dtype. It does not select the direct
solver from a constraint's class name.

## Direct Restricted Least Squares

### Problem Preparation

Feature omission is applied before fitting.

When an intercept is enabled, the backend centers the masked inputs and
targets using full-dataset means:

\[
X_c = X_{\mathrm{masked}}-\mathbf{1}\bar{x}^\top,
\qquad
Y_c = Y-\mathbf{1}\bar{y}^\top.
\]

Without an intercept, the centering offsets are zero.

### Coefficient Solution

For each output \(j\), the backend selects the input columns allowed by both
the feature-omission mask and the coefficient zero mask.

It solves

\[
W^\star_{j,A_j}
=
\arg\min_v
\left\|
X_{c,A_j}v-Y_{c,j}
\right\|_2^2
\]

using:

```python
np.linalg.lstsq(
    X_centered[:, active],
    y_centered[:, output_index],
    rcond=None,
)
```

Forbidden and omitted coefficients are assigned zero. If an output has no
allowed input columns, all its coefficients remain zero.

Each output is solved separately because the supported objective and
zero-mask constraints do not couple outputs.

The procedure refits the allowed coefficients. It does not fit an
unrestricted model and then simply zero selected entries.

The target coefficient matrix is computed once per fitting run. No Hessian
is explicitly constructed for this path.

### Relaxation

The target is applied during the epoch loop:

\[
W_{k+1}
=
(1-\alpha_k)W_k+\alpha_kW^\star,
\]

followed by explicit zeroing of forbidden and omitted coefficients.

Here, \(\alpha_k\) is the current callback-controlled learning-rate value.

- At \(\alpha_k=1\), the first executed update applies the least-squares
  solution.
- For \(0<\alpha_k<1\), updates move progressively toward that solution.
- At \(\alpha_k=0\), the coefficient update is skipped.

Constraint satisfaction is checked before assigning the candidate.

The epoch loop remains active for callbacks, loss history, and stopping
criteria. Consequently, direct fitting does not necessarily terminate after
one reported epoch.

## Iterative Coefficient Updates

The `coefficient` path supports regularizers, additional constraints, and
mini-batch fitting while retaining the existing component interfaces.

### Data Gradient

For a batch containing \(n_b\) samples, the data gradient is

\[
G_b
=
\frac{2}{n_bm}
\left(X_bW^\top-Y_b\right)^\top X_b.
\]

With an intercept enabled, batches are taken from arrays centered using
full-dataset means. They are not centered independently per batch.

### Curvature-Based Step Size

For each batch, the backend computes

\[
L_b
=
\frac{2\|X_b\|_2^2}{n_bm},
\]

where \(\|X_b\|_2\) is the matrix spectral norm.

The coefficient step size is

\[
\eta_b=\frac{\alpha_k}{L_b}.
\]

A zero computed curvature is replaced by `1.0` to avoid division by zero.

This is a scalar curvature-scaled update, not a full Newton update.
The backend does not solve a Hessian system on this path.

### Regularizers with a Proximal Operator

If the runtime regularizer provides a callable `prox()` method, the update is:

```python
candidate = regularizer.prox(
    weights - step_size * data_gradient,
    step_size,
)
```

For built-in L1 penalties, the proximal operation performs
soft-thresholding. Weighted L1 penalties use coefficient-dependent
thresholds according to their configuration.

### Gradient/Subgradient Fallback

If no callable proximal operator is available, the backend obtains the
penalty gradient from the network and uses:

\[
W_{\mathrm{candidate}}
=
W-\eta_b\left(G_b+\nabla R(W)\right).
\]

The batch curvature accounts for the data term only. It does not include
additional curvature introduced by an arbitrary custom regularizer.
Such regularizers may require a smaller learning rate or a dedicated solver.

### Constraint Enforcement

The candidate is passed through:

```python
network.constrain(candidate)
```

before assignment.

This preserves the configured enforcement behavior, but arbitrary enforcement
is not necessarily a mathematical projection. In particular, composing a
regularizer's proximal operation with generic constraint enforcement is not
automatically an exact proximal operation for the combined problem.

The adapter therefore does not claim global convergence for arbitrary
regularizer–constraint combinations.

## Intercept Update

After coefficient updates, the intercept is recomputed analytically:

\[
b=\bar{y}-W\bar{x}.
\]

For the direct solver, this occurs after the epoch's coefficient update.
For the iterative solver, it occurs after each batch.

Without an intercept, no bias parameter is updated.

## Learning Rate and Callbacks

The backend does not construct a native optimizer object. A compatibility
object exposes:

```python
state["optimizer"].param_groups["lr"]
```

to existing learning-rate callbacks.

Its meaning depends on the selected solver:

- `restricted_lstsq`: relaxation toward the precomputed solution;
- `coefficient`: a multiplier of the inverse-curvature step size.

The configured learning rate must lie in `(0, 1]`. Callback-provided values
are validated in `[0, 1]`.

The adapter reads the callback-controlled value after
`on_epoch_beginning`. A zero value skips coefficient updates; intercept
recomputation and callback processing may still occur.

The lifecycle hooks are:

1. `on_train_beginning`
2. `on_epoch_beginning`
3. `on_epoch_end`
4. `on_train_end`

All callbacks receive their hook even when an earlier callback requests
stopping.

## Stopping and Training Results

Fitting may stop because of:

- a callback request;
- an absolute loss change no greater than `tol`;
- reaching the configured epoch limit.

The loss-change criterion is evaluated only when the current relaxation is
positive.

This criterion detects objective stagnation. It is not a general certificate
of optimality or constraint-qualified convergence.

The recorded loss is the full-dataset MSE plus regularization penalty,
evaluated after an epoch.

When `collect_history=True`, `TrainingResult` contains:

- the recorded loss history;
- final loss, MSE, and regularization penalty;
- the number of completed epochs;
- the stopping reason.

The final objective is evaluated after `on_train_end`, preserving weights
and bias restored by callbacks. It may therefore differ from the last
recorded epoch loss.

When `collect_history=False`, fitting returns an empty `TrainingResult`.
The adapter is marked fitted when at least one epoch has completed.

## Component Lifecycle

Regularizers and constraints supplied to the adapter are configuration
templates.

During `rebuild()`, runtime instances are reconstructed from their
configuration and attached to a new predictor. Rebuilding also resets
coefficients, fitting state, loss history, and loop-control state.

Resetting components before `fit()` does not itself rebuild or zero the
predictor. Existing coefficients can therefore act as a warm start.

Callback reset creates callbacks for a new run. Predictor rebuilding and
callback resetting are separate operations.

## Limitations and Interpretation

- Constraints containing additional rules, such as a positive lower bound
  on a sum of coefficient magnitudes, are not handled by direct restricted
  least squares.
- The iterative fallback preserves existing enforcement behavior; it is not
  a universal constrained optimizer.
- Solver selection assumes that regularizers and constraint definitions
  remain unchanged during a fitting run.
- Numerical fitting and statistical inference are separate concerns.
  Selecting a solver does not establish the validity of a Granger test for
  a regularized or otherwise constrained estimator.
- The direct path uses NumPy linear algebra. Neither fitting path calls a
  scikit-learn estimator.