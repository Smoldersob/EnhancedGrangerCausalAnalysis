# Hyperparameter Optimization: Current Status and Intended Boundary

## Current status

Automatic hyperparameter optimization is not executed by the standard builder, orchestrator, or `AnalysisRunner.run_analysis()` workflow.

The retained fields `hiperoptimalization_state` and `hiperoptimalization_conf` are inactive. Their presence does not select parameters, adjust regularization, or change final fitting. Existing code under `hiperopt/` must not be presented as an integrated, validated search API.

No `ModelSearchSpec`, `ModelQualityEvaluator`, `ModelSearchRunner`, or `search` configuration section is introduced by the execution refactor documented here. Those names describe intended future work only.

## Intended purpose

Future optional HPO should select configuration according to model quality. Its scope may include model/training parameters, regularization, and preprocessing such as lag selection or scaling.

Candidate evaluation must not fit unrestricted/restricted pairs to score causal-analysis results. Do not optimize p-values, Granger statistics, recovered-edge F1, or ground-truth causal accuracy. An apparently favorable causal matrix does not establish adequate model quality.

The intended high-level flow is:

```text
optional model-quality search
    -> selected configuration
    -> final model fitting
    -> standard Granger analysis
```

Search should reuse existing preparation, model assembly, and single-fit services rather than implement another component resolver or model-construction path. Validation-specific prediction must remain under the evaluator's control; `FitExecutor` should not force prediction on training observations.

## Separate experiment sweeps

`TestGroupConfigIterator` expands explicitly specified experiments. Sweeping L1 values runs separate analyses for those values; it does not automatically choose L1 by validation quality.

Ground-truth metrics computed by the experiment script are reports on completed analyses, not feedback into HPO. Future optional HPO may run inside each explicit experiment without taking over the iterator's role.

## Validation requirements for future implementation

A future implementation must specify the objective, temporal validation boundaries, preprocessing fitting scope, candidate isolation, final refit policy, failure handling, and recorded selected configuration. It must not infer statistical calibration of causal tests from predictive tuning alone.

See [Architecture](execution_architecture.md) and [Configuration](config_file_usage.md).
