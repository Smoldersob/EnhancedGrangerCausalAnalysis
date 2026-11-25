# Project Context

## Purpose

EnhancedGrangerCausalAnalysis analyzes predictive dependencies among variables measured over time. Its Granger effects describe whether a variable's past adds predictive information about another variable, conditional on other included variables. They are not proof of intervention-level causality.

The standard workflow is multivariate and multi-output: it tests one source variable at a time across the configured output variables. For each source-to-output effect, the selected lags are tested jointly. `SimpleGrangerAPI` is a separate pair-wise statsmodels reference, not a substitute for the conditional multivariate workflow.

## Measurement data

A **measurement segment** is a regular, time-ordered recording window of the same object. Multiple DataFrames may represent non-contiguous windows, with unobserved time between them. Treat each frame as a separate sequence: construct lags within its own rows and never carry lag history across frame boundaries.

Validation can hold out whole measurement segments or split every segment into time-ordered training and validation portions. The selected method determines which approach applies. Do not shuffle prepared rows for validation.

## Current implementation and intended direction

Keep current behavior distinct from project direction:

- The standard workflow does not currently run automatic hyperparameter optimization.
- The built-in preparation/model path is currently numeric and continuous-target oriented; binary and count target families are future work. Categorical and ordinal targets, and native categorical-predictor encoding, are deferred.
- Current source-level splitters provide whole-segment holdout and a temporal block split; prepared-row random splitters also still exist. The intended HPO direction is to keep whole-segment holdout, add expanding walk-forward folds within every segment, and remove prepared-row random splitting.
- Test metrics can produce p-values, but their calibration is not verified for every fit. The core result path uses raw p-values and does not apply false-discovery-rate correction.
- The intended next modeling direction is to develop linear methods first, with explicit target types and target-specific linear equations. See [AGENTS.md](../../AGENTS.md) for the agreed development requirements.

## Where to look next

- [Project structure](../project_structure.md) for package responsibilities and shared contracts.
- [Model contract](../model_contract.md) for predictor and fit boundaries.
- [Data preprocessing](../data_preprocessing.md) and [data splitting](../data_splitting.md) for preparation and validation details.
- [Results and metrics](../results_and_metrics.md) for statistic availability, p-values, and interpretation limits.
- [Classical Granger models](../classical_granger_models.md) for the reference linear model family.
