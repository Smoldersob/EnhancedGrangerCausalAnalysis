# Single-model Granger Fitting Lifecycle

The standard workflow reuses one working model-adapter instance.
Prepared X and y retain their dimensions across cause-specific runs.

The baseline role determines which fitted state can be shared across
tested causes:

| Model family | Shared fitted baseline | Cause-specific fit |
| --- | --- | --- |
| Conditional | Unrestricted model | Restricted model with the tested cause omitted |
| Classical | Restricted autoregressive model | Unrestricted extension with the tested cause added |

Each predictor jointly predicts all configured outputs.

Implementation is split between `execution/granger_trainer.py`, which
performs individual fits and captures outcomes, and
`execution/runner.py`, which chooses fit order and stores snapshots.

## Shared Base Fit

1. Restore optional configured initial weights.
2. Select the adapter's baseline with `add_variables([])`.
3. Reset training components with run name `base_model`.
4. Fit on prepared `X_scaled` and `y_scaled`, with
   `reset_components=False`.
5. Capture independent predictions, parameter arrays, active coefficient
   masks, diagnostics, and timings.
6. Store the snapshot under the adapter's `base_model_role`.
7. Save copied baseline weights for subsequent fits.

For conditional adapters, the baseline is unrestricted.
For classical adapters, it contains only each output's own history.

## Conditional Cause-specific Fits

For each tested cause:

1. Restore copied unrestricted baseline weights.
2. Obtain the cause's prepared columns from `cause_slices`.
3. Call `omit_variables(indices)`.
4. Reset training components.
5. Fit the restricted configuration.
6. Capture and store the corresponding restricted snapshot.

Omission replaces the previous omitted block.

## Classical Cause-specific Fits

For each tested cause:

1. Restore copied restricted autoregressive baseline weights.
2. Obtain the cause's prepared columns from `cause_slices`.
3. Call `add_variables(indices)`.
4. Reset training components.
5. Fit the unrestricted extension.
6. Capture and store the corresponding unrestricted snapshot.

Additional-input selection starts from the autoregressive baseline and
does not accumulate across causes.

Calling `omit_variables(...)` restores only the baseline input
configuration. It does not restore baseline coefficient values.

## Restricted Capture Without Refitting

A workflow may capture a restricted snapshot by applying input control to
a fitted unrestricted model and predicting without another fit.

Such a snapshot retains coefficients learned in the unrestricted fit.
It is not equivalent to independently refitting the restricted model.

For classical models, this also differs from reusing a fitted shared
autoregressive baseline. Preserve fit provenance and do not describe a
masked capture as a fitted baseline.

## Separation of Concerns

- The predictor implements prediction and owns coefficient parameters.
- The adapter implements the fitting and input-control contract.
- `FitExecutor` performs one protocol fit.
- `GrangerTrainer` controls weight restoration, input selection,
  component reset, and outcome capture.
- `AnalysisRunner` determines comparison order and stores snapshots.
- `GrangerAnalysisResults` evaluates saved snapshots rather than
  training models.

Construction uses `ModelInputSpec`, not training observations.
Initialization remains a separate operation.

## Isolation Requirements

- Weight restoration must preserve input-control state.
- Input selection replaces rather than accumulates previous selections.
- Independent runs explicitly reset training components.
- Captured arrays must remain independent of later model mutation.
- Compared predictions must share observation alignment, output order,
  and scale.
- Active coefficient masks must describe the actual captured
  configuration.
- Restoring input configuration and restoring fitted weights are
  separate operations.

## Results and Failures

The working model remains in its last run's state. No automatic
restoration of baseline parameters follows the cause loop.

Completed snapshots may remain available when a later fit fails; they do
not constitute a completed analysis.

`get_results()` does not trigger training. A complete fit is not a resume
operation, and internal progress bookkeeping is not a public session API.

See [Execution architecture](execution_architectur.md),
[Model contract](model_contract.md),
[Classical Granger models](classical_granger_models.md), and
[Results](results_and_metrics.md).