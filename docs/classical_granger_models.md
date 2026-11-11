# Classical Granger Model Adapters

## Scope

The classical linear adapters implement a shared restricted autoregressive
baseline and cause-specific unrestricted extensions.

They complement, rather than replace, the conditional adapters.

| Model family | Shared baseline | Cause-specific modification |
| --- | --- | --- |
| Conditional | Unrestricted model using all structurally permitted inputs | Omit the tested cause's prepared feature block |
| Classical | Restricted model using each output's own history | Add the tested cause's prepared feature block |

These names describe the available predictor information sets. They do not
by themselves establish the validity of a particular significance test for
a regularized or constrained estimator.

Each predictor jointly returns all configured outputs. Cause-specific
Granger comparisons do not require a separate adapter or predictor for
each output.

## Backend Implementations

| Backend | Adapter | Native coefficient layout |
| --- | --- | --- |
| PyTorch | `PyTorchClassicalAdapter` | `(n_outputs, n_features)` |
| Keras | `KerasClassicalAdapter` | `(n_features, n_outputs)` |
| NumPy | `NumpyClassicalAdapter` | `(n_outputs, n_features)` |

For each backend:

- `models/classic.py` defines the adapter.
- `models/network_parts.py` defines `ClassicalLinearGrangerNetwork`.
- The existing model registry provides `classical` and `classical_linear`.

Adapters reuse their conditional counterparts' constructor and training
interfaces. They specialize predictor construction, input control, active
coefficient reporting, and, where necessary, fitting calculations.

## Prepared Input Contract

The model receives already prepared arrays:

- X: `(n_samples, n_features)`.
- y: `(n_samples, n_outputs)`.

Lag construction, sample alignment, stationarity transformations, and
scaler fitting remain preprocessing responsibilities.

The autoregressive selector is derived from:

- `ModelInputSpec.effects`;
- `ModelInputSpec.cause_slices`;
- `ModelInputSpec.feature_lags`.

For each output, its own-history block must exist among the prepared input
features and contain at least one positive lag. Construction raises an
error when this requirement is not satisfied.

Output histories are not automatically inserted into X.

The implementation does not assume equal lag counts across variables or
that outputs and input blocks have the same ordering.

A lag-zero feature is not included in the autoregressive baseline. It may
be selected as an additional input if its column is explicitly activated.
Applications requiring strictly lagged predictors must enforce that
requirement in their input configuration or validation.

## Predictor Architecture

For the equations below, W uses the conceptual orientation
`(n_outputs, n_features)`.

Let:

- A select each output's positive-lag own-history coefficients;
- E = 1 - A select the remaining coefficients;
- m be a vector of length `n_features` controlling additional inputs.

The predictor is

$
\widehat Y
=
X(W\odot A)^\top
+
(X\odot m)(W\odot E)^\top
+
\mathbf{1}b^\top.
$

Equivalently,

$
M=A\lor(E\land m[None,:]),
\qquad
\widehat Y=X(W\odot M)^\top+\mathbf{1}b^\top.
$

A and E are fixed architectural selectors. Only m changes between
cause-specific configurations.

The predictor retains one coefficient matrix and an optional single bias
vector. The two computation paths do not introduce duplicate coefficient
parameters.

### Why Own History Bypasses Additional-Input Control

Disabling a variable as an additional predictor must not remove its own
autoregressive terms from its corresponding output equation.

For example, when predicting a and b:

- the history of a remains available for predicting a;
- activating a additionally exposes its history for predicting b;
- the history of a is not counted twice in the equation for a.

The additional-input vector therefore controls only the additional path,
not the complete input of the predictor.

## Input-Control Methods

All indices below refer to columns of prepared X.

| Operation | Result |
| --- | --- |
| Construction or `rebuild()` | Autoregressive baseline |
| `add_variables(indices)` | Baseline plus exactly the selected additional columns |
| `add_variables([])` | Autoregressive baseline |
| `omit_variables(indices)` | Autoregressive baseline |
| `omit_variables([])` | Autoregressive baseline |

Each call to `add_variables()` constructs a new selection relative to the
baseline. Selections do not accumulate.

`omit_variables()` validates its argument for interface compatibility but
disables the complete additional-input path, regardless of which valid
indices were supplied.

Neither operation changes predictor parameters, resets training
components, or invalidates fitted status.

### Selecting a Cause Block

Use the existing named feature mapping:

```python
block = prepared.cause_slices[cause]
indices = list(range(block.start, block.stop))

model.add_variables(indices)
```

Do not infer block positions from a variable index multiplied by a presumed
common lag count.

Activating several cause blocks at once produces a joint extension. It is
not a sequence of independent single-cause tests.

## Shared Restricted Baseline

Classical adapters expose:

```python
model.base_model_role == "restricted"
```

The standard fitted-baseline workflow is:

1. Configure the autoregressive baseline with `add_variables([])`.
2. Fit the shared restricted predictor.
3. Capture its predictions, active coefficient mask, and copied weights.
4. For each tested cause:
   - restore copied baseline weights;
   - activate that cause's prepared feature block;
   - reset training components;
   - fit the corresponding unrestricted extension;
   - capture an independent snapshot.

The restricted baseline is shared across causes, but contains a distinct
autoregressive equation for each output.

Baseline weights and baseline input configuration are separate state:

- `omit_variables(...)` or `add_variables([])` restores the configuration;
- `set_weights(base_weights)` restores parameter values.

Neither operation substitutes for the other.

## Constraints, Regularization, and Activity

The implementation separates:

1. Fixed AR/additional-input architecture.
2. Dynamic additional-input selection.
3. Structural coefficient constraints.
4. Coefficient regularization.

Changing additional inputs does not replace or reconfigure the constraint.

Regularization and constraint enforcement continue to operate on raw
coefficient parameters. They are not automatically limited to coefficients
currently contributing to prediction.

Consequently, constraints or penalties coupling coefficients may allow
inactive parameters to affect the optimization of active parameters.
The resulting estimator should not automatically be interpreted as an
independent unregularized pairwise regression.

For a structural allowance mask S, reported coefficient activity is

$
M_{\mathrm{active}}=S\land M.
$

This reports permitted active connections, not the numerical nonzero support
of fitted coefficients.

## Parameter Export

Adapter `get_weights()` returns independent copies of:

- kernel: `(n_features, n_outputs)`;
- optional bias: `(n_outputs,)`.

It excludes:

- the additional-input control vector;
- fixed architectural selectors;
- optimizer state;
- callback state.

`set_weights()` preserves input control and invalidates fitted status,
following the existing adapter contract.

For Keras, adapter-level `get_weights()` must not be confused with native
`model.get_weights()`. The native whole-model operation also includes
tracked non-trainable weights, including the feature-control vector.

## NumPy Optimization

NumPy preserves its existing solver choices:

- `auto`;
- `restricted_lstsq`;
- `coefficient`.

The classical adapter changes the design/gradient calculations rather than
introducing a neural-network optimizer.

The direct path solves each output on its active columns. The iterative
path masks the data gradient, preserves regularizer and constraint
interfaces, and recomputes bias using effective coefficients.

See [NumPy optimization](numpy_backend_optimization.md) for solver
eligibility, relaxation, curvature scaling, and limitations.

## Interpretation and Validation

- Adding a variable already used as an output does not add duplicate terms
  to that variable's own equation.
- A structural constraint can forbid every added coefficient for a
  particular relation, leaving no effective extension for that relation.
- A coefficient numerically equal to zero is not automatically a
  structural restriction.
- Capturing a restricted predictor by masking a fitted unrestricted model
  does not refit its remaining coefficients.
- Such a captured predictor is not the same artifact as the fitted shared
  autoregressive baseline.
- The model family does not change preprocessing order or authorize
  fitting transformations on held-out observations.
- Compared snapshots must retain identical observation alignment, output
  order, and prediction scale.
- Numerical fitting does not establish the reference distribution of F,
  likelihood-ratio, Wald, or other statistics for arbitrary constrained
  or regularized estimators.

Examples describe the interface and are not reports of executed
cross-backend validation.