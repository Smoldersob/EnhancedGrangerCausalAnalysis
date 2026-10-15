# Evaluating Granger Analysis Results

Result evaluation and interpretation are documented in
[Results and Metrics](results_and_metrics.md).

That guide covers:

- unrestricted base models and restricted reference models;
- statistical metrics and their p-value companions;
- residual-error diagnostics;
- coefficient-derived signs;
- restoration of variables removed during preprocessing;
- thresholded and signed causality matrices;
- comparison with ground-truth relationship matrices.

The base model contains the tested cause's active predictors.
The restricted reference model disables that cause's feature block.

The comparison evaluates whether the restriction increases residual error
relative to the base model.

For API usage and execution stages, see [API Usage](api_usage.md).