# Execution Architecture

This document describes the implemented separation of responsibilities in the standard analysis workflow. It is not a specification of future sessions or interchangeable procedures.

## Dependency flow

```text
Builder / Orchestrator / direct specification
                  |
             AnalysisSpec
                  |
       AnalysisRunner.run_analysis
          |       |        |
   DataPreparer   |   ModelInitializer
             ModelAssembler
                  |
           BackendStrategy
                  |
       existing resolvers/registries
                  |
           GrangerTrainer
                  |
              FitExecutor
                  |
              model.fit
```

The diagram summarizes dependencies rather than exact chronological placement of every operation. Actual order is preparation, component/model construction, initialization, base fit, and optional restricted fits.

## Configuration is not execution state

`AnalysisSpec` groups `DataPreparationConfig`, requested `VariableRoles`, `ModelingSpec`, and retained top-level settings. The external document remains flat. Source observations and prepared artifacts are passed separately.

`AnalysisConfigLoader` only reads JSON/YAML and relation references. `BackendFactory` owns backend aliases and declaration handling. Component-specific interpretation remains in existing parsers and resolvers.

Frozen specifications do not deeply freeze or clone arbitrary objects. Passing an instance does not establish safe reuse across independent fits.

## Construction services

`ModelAssembler` prepares constraint and regularizer data against the effective layout, delegates backend-native construction, creates `ModelInputSpec`, and assembles a model. It does not fit, initialize weights, or evaluate causal results.

`ModelInitializer` uses the existing initializer resolver and `InitializationContext` to compute weights. The caller assigns them and captures the model-accepted representation. It does not implement another registry or weight-shape validator.

`FitExecutor` calls the existing model `fit(X, y, ...)` protocol and returns its `TrainingResult`. It does not rebuild, omit features, restore weights, or predict automatically.

## Three artifact contracts

| Type | Contents | Not responsible for |
| --- | --- | --- |
| `TrainingResult` | Optional history, metrics, iteration count, stop reason | Predictions, model state, causal evaluation |
| `FitOutcome` | Predictions in model output scale, all copied weight arrays, `TrainingResult`, timings | Inverse scaling, causal metrics, full/restricted relationships |
| `GrangerAnalysisResults` | Base/reference snapshots, preparation context, caches and metric access | Training models or choosing hyperparameters |

`FitOutcome` is produced by `GrangerTrainer`, after it uses `FitExecutor` and captures predictions and weights. It does not retain a live model or `PreparedData`.

The runner inverse-transforms target predictions and preserves the current snapshot convention: only `outcome.weights[0]` is stored as coefficient weights. This first-array assumption belongs to current analysis snapshot integration, not a universal rule for all future models.

Training diagnostics are not automatically inserted into final snapshot metadata. Current metadata preserves timings and requested backend information. Calling standard fit without history collection produces a valid empty `TrainingResult`, not evidence of zero training iterations.

## Standard trainer

`GrangerTrainer` operates on the supplied working model. It returns one `FitOutcome` per successful fit and has no dependency on `GrangerAnalysisResults`.

Base fitting restores optional initial weights, clears omission, explicitly resets optimizer and callbacks, and calls fit with `reset_components=False`.

Restricted fitting requires saved base weights, chooses the start state, replaces the omitted cause block, resets optimizer/callbacks, and fits on the same prepared arrays. Default start is copied base weights.

The model is left in its last fit's state. No automatic restoration of the full model follows the restricted loop.

## Runner

`run_analysis(spec=..., data=..., prepared_data=..., fit_restricted=...)` performs the full standard workflow. Supplied prepared data takes precedence.

The lower-level `run(model=..., prepared=..., progress=...)` coordinates fitting for already constructed inputs. It is not a resume API.

The runner creates the result context and stores outcomes immediately after each fit, preserving completed snapshots if a later fit fails. It does not create a cross-call preparation cache.

## Builder synchronization

`AnalysisRunProgress` publishes artifacts of completed stages: preparation, strategy, constraint/regularizer data and instances, model construction, initial weights, base weights, and results.

Full builder execution synchronizes these artifacts in `finally`, including when a later operation raises. A failure before new preparation completes leaves previous builder artifacts unchanged. After successful preparation, dependent old components are invalidated.

Progress is mutable internal bookkeeping. It does not provide configuration editing, evaluation, resume, scheduling, or a public session lifecycle. Callers must supply a fresh object to a full run.

Exact historical intermediate private fields after every possible failure are not guaranteed. In particular, model construction publishes callbacks only after successful assembly.

## Compatibility boundaries

- Explicit builder stages remain available.
- Complete fit uses runner services directly rather than dispatching through overridden builder stage methods.
- Ready models remain incompatible with conflicting constructor overrides in the standard assembly path.
- Backend-specific device managers are reused. NumPy ignores the device constructor parameter at its strategy boundary, including resolved model params.
- Constraint/regularizer data depend on final prepared layout, not requested raw-column positions.

## HPO and experiment boundaries

Automatic HPO is not active. Future HPO should use preparation, assembly, and individual fitting to evaluate model quality. It must not optimize Granger statistics, p-values, ground-truth causal F1, or restricted/full comparisons.

`TestGroupConfigIterator` defines explicit experiments. Its role is not transferred to HPO. A future experiment may optionally select model hyperparameters internally before running one final analysis.

## Validation status

The architecture documentation assumes the refactor has been applied. It does not claim the examples or cross-backend workflows were executed. Numerical equivalence, resource isolation, and statistical calibration require separate validation.
