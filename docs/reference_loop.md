# Single-model restricted-run lifecycle

The multitask builder reuses one working model for the unrestricted base run
and subsequent restricted runs.

Each restricted run changes the active input block, not the dimensions of
the prepared input and output arrays.

The builder stores completed runs as snapshots rather than retaining one
live model object per tested cause.

## Standard lifecycle

```text
prepare data and components
        |
build and initialize the working model
        |
apply optional initial weights
        |
fit the unrestricted base model
        |
copy base weights and save the base snapshot
        |
for each effective tested cause:
    restore copied base weights
    select the cause's feature block by name
    apply the omission
    reset callbacks for the run
    fit the working model
    save a reference snapshot
```

The standard implementation is in
`enhanced_granger_analysis/api/builder.py`.

## Initialization versus weight restoration

`build_model()` constructs the working model and initializes it with the
prepared inputs and targets.

Restricted runs do not rebuild the model through the builder.
They restore the fitted base weights through `set_weights()`.

Weight restoration is not equivalent to a complete reset of runtime state.
The backend model must define how fitting handles optimizer state, omitted
variables, buffers, and other training state.

The builder explicitly resets callbacks for each run. Other reset semantics
belong to the backend model implementation and should be verified when
adding or changing a backend.

## Required isolation properties

A compatible backend should ensure that:

- setting weights does not unintentionally replace the configured mask;
- omitting a cause selects the intended feature block;
- omissions from one run do not accumulate in later runs;
- callback state and run-specific logging do not leak between runs;
- saved weights and predictions remain independent of subsequent training;
- base and restricted predictions use the same observations and effect order.

These are lifecycle requirements, not consequences of reusing one Python
object.

## Named feature blocks

The builder selects a block through:

```python
block = prepared.cause_slices[cause]
```

It does not infer the block from the position of the cause in `tested_causes`.

`cause_slices` contains one complete slice per effective input.
It is not an array of block boundaries.

## Snapshots and result retrieval

The base run and each restricted run save predictions and coefficient weights
to `GrangerAnalysisResults`.

`get_results()` returns the current result container without additional fitting.

Metric evaluation is separate from model execution. Statistical and sign
metrics use the saved snapshots and preparation context.

## Why reuse one model?

A separate live model object is unnecessary when the architecture is unchanged
and completed runs are preserved as independent snapshots.

Reusing one model simplifies the execution lifecycle, but correctness depends
on explicit state-management contracts rather than object reuse alone.

## Related documentation

- [API Usage](.api_usage.md)
- [Data Preprocessing](.data_preprocessing.md)
- [Results and Metrics](.results_and_metrics.md)
- [Backend Usage](.backend_usage.md)