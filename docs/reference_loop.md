# Single-model Restricted-run Lifecycle

The standard workflow reuses one model for the full fit and subsequent restricted fits. Array dimensions remain unchanged; omission controls prepared feature columns.

Implementation is split between `execution/granger_trainer.py` (one fit) and `execution/runner.py` (fit order and snapshot storage), not embedded in the builder.

## Base fit

1. Restore optional configured initial weights.
2. Clear omitted columns with `omit_variables([])`.
3. Reset optimizer and callbacks with run name `base_model`.
4. Call the executor with prepared `X_scaled`, `y_scaled`, and `reset_components=False`.
5. Capture model-scale predictions, full copied weights, training diagnostics, and timings in `FitOutcome`.
6. The runner inverse-scales predictions and saves the base snapshot.
7. Save copied full-model weights for restricted starts.

## Restricted fit

A completed base fit is required. For each effective tested cause:

1. Restore initial weights when explicitly requested and available; otherwise restore base weights when requested (the standard default).
2. Select `prepared.cause_slices[cause]` and replace omitted columns with that block.
3. Reset optimizer and callbacks using `reference_cause_{index}`, where index is the cause's position among effective causes, not among tested causes.
4. Fit on the same prepared observations with `reset_components=False`.
5. Capture `FitOutcome` and immediately store the reference snapshot.

If neither weight restoration applies, current model weights are retained. The model is not rebuilt for each cause and is not restored to the full-model state after the loop.

## Separation of concerns

`FitExecutor` performs one protocol fit. `GrangerTrainer` controls weights, omission, component reset, and artifact capture. `AnalysisRunner` stores outcomes as analysis snapshots. `GrangerAnalysisResults` evaluates saved snapshots rather than fitting models.

Construction uses structural `ModelInputSpec`, not training observations. Optional initialization is a separate operation using `InitializationContext`.

## Isolation requirements

- Weight restoration must preserve the intended input-control contract.
- Omission replaces, rather than accumulates, omitted blocks.
- Optimizer and callbacks reset explicitly for independent runs.
- Predictions and weights captured for completed runs must remain independent of later training.
- Compared predictions must share observation alignment, output order, and scale.

These are implementation contracts, not automatic consequences of sharing a Python model object.

## Results and failures

Result snapshots store inverse-scaled predictions and the current first-weight-array coefficient representation. Completed snapshots remain stored if a later fit fails. Exceptions propagate; partially populated results are not a completed analysis.

`get_results()` does not trigger training. `fit()` is a complete run, not resume. Internal progress bookkeeping is not a public resume API.

See [Architecture](execution_architecture.md), [API](api_usage.md), [Model contract](model_contract.md), and [Results](results_and_metrics.md).
