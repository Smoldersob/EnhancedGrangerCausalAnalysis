# Hyperparameter Optimization

## Scope

This document describes the standalone HPO implementation prepared for
EnhancedGrangerCausalAnalysis: contracts, parsing, objectives, strategies,
candidate evaluation, budgets, and the search loop.

It assumes the code fragments and subsequent corrections from the HPO work
have been applied. It is not an independent audit of the local repository and
does not claim successful execution across all backends.

The standard analysis runner does not automatically invoke this subsystem.
Automatic public API integration and file-based analysis integration are not
part of this document's implemented workflow.
s
## Purpose

HPO selects a modeling configuration using prediction quality on validation
observations. Each candidate is one jointly trained multi-output model.

HPO does not:

- optimize Granger p-values, test statistics, or graph recovery scores;
- train restricted models for individual Granger relations;
- replace an outer experiment/configuration iterator;
- perform mandatory final test-set evaluation;
- automatically refit the selected model for Granger analysis.

The output is selected modeling settings and search history, not a winning
trained model to reuse in the analysis.

## Package layout

```text
enhanced_granger_analysis/
    configuration/
        model_search_spec.py
    hiperopt/
        space/
            base.py
            parser.py
            application.py
            __init__.py
        strategies/
            base.py
            registry.py
            implementations.py
            __init__.py
        objectives/
            base.py
            registry.py
            implementations.py
            __init__.py
        results.py
    modeling/
        search_configuration.py
        search_evaluator.py
        initialization.py
    execution/
        model_search_runner.py
```

Use `modeling/search_evaluator.py` for `ModelCandidateEvaluator`. A second
`candidate_evaluator.py` implementation is not needed.

## Execution flow

```text
prepared data + baseline ModelingSpec + ModelSearchSpec
    -> parse/validate the search space
    -> resolve objective and strategy
    -> validate shared data and budget conflicts
    -> generate candidate updates
    -> construct a fresh model and components
    -> initialize from training rows
    -> fit on training rows
    -> predict validation rows
    -> evaluate the objective
    -> record TrialResult
    -> select the best completed trial
    -> return ModelSearchResult
```

The evaluator shares the existing domain preparation functions and backend
factories through `ModelAssembler`. It does not implement a separate model
registry or name-to-class construction path.

## Data preparation and splitting

### Required input

`ModelSearchRunner.run()` accepts an existing `PreparedData` object.
The caller performs HPO preprocessing once before calling the runner.
No candidate invokes preprocessing again.

The artifact must provide aligned `X_raw`, `y_raw`, `X_scaled`, `y_scaled`,
output names, feature metadata, and a split referring to prepared row positions.

The search requires:

- a non-empty `training` part;
- a non-empty part selected by `validation_part`;
- disjoint, integer, increasing, in-range indices;
- finite selected inputs and targets.

`validation_part` cannot be `training` or `test`.
The optional test part is not used by this search implementation.

### Explicit split representation

The updated splitting contract uses the standard parts:

```python
parts = {
    "training": training_indices,
    "validation": validation_indices,
    "test": test_indices,
}
```

Unused validation/test parts are empty integer arrays. Without a splitter,
all prepared rows belong to training, while validation and test are empty.
That is valid ordinary preparation, but insufficient for enabled HPO.

Indices refer to prepared matrices, not directly to original source rows.
Retain existing source-correspondence metadata when relating them to input
series. Do not reinterpret HPO indices as final-analysis indices.

### Fitting transformations

Scalers must be fitted on HPO training rows and applied to the other rows.
Data-dependent initialization also uses training rows only.

Keep the splitter stage's existing meaning:

- `source`: applicable preprocessing decisions are fitted on source training
  blocks; full series are transformed with row correspondence preserved.
- `prepared`: complete lagged samples are split before scaling. Earlier
  globally fitted decisions are not thereby made independent of validation.

The evaluator checks arrays and assignments. It cannot infer a supplied
scaler's fitting provenance from prepared matrices alone.

### Separate final Granger preparation

HPO preparation and final Granger preparation are separate contexts:

```text
HPO preprocessing once, with the HPO split
    -> search and select modeling settings
    -> final Granger preprocessing separately
    -> construct fresh analysis models and components
```

The HPO split must not automatically constrain the final Granger analysis.
The current final procedure may use all analysis observations; any future
analysis split must be configured separately.

Only the selected modeling configuration crosses this boundary. HPO scalers,
lag-engine state, initial weights, models, optimizers, and callbacks do not.
The same preparation declarations can be used in both contexts without
sharing their fitted instances.

Automatic orchestration of these two contexts is not implemented by the
standalone search module. The caller must prepare the HPO artifact explicitly.
Do not assume that `ModelSearchSpec` already has a working `preparation`
override field or a file-loading integration.

## Search space

### Declaration

A space maps dotted configuration paths to parameter domains:

```python
space = {
    "model_config.learning_rate": {"choices": [0.1, 0.5, 1.0]},
    "model_config.batch_size": {"choices": [None, 32]},
}
```

Use names supported by the selected model/backend. For an optimizer component,
a path can instead address its declaration, for example
`model_config.optimizer.params.<parameter_name>`.

The implemented roots are `model_config` and `regularizer_spec`.
Assembler-owned dimensions, component injection, constraints, regularizers,
and callbacks cannot be placed inside `model_config` as ordinary parameters.

### Domains

| Domain | Declaration | Grid | Random |
|---|---|---|---|
| Choices | `{"choices": [a, b]}` | Enumerate values | Sample uniformly with replacement |
| Integer range | `{"type": "int", "low": 1, "high": 5}` | Enumerate 1 through 5 | Sample integers, both bounds included |
| Float range | `{"type": "float", "low": 0.01, "high": 1.0}` | Rejected | Sample on the linear scale |
| Log float range | `{"type": "float", "low": 1e-5, "high": 1e-2, "log": True}` | Rejected | Sample on the logarithmic scale |

Use explicit choices to discretize a floating-point domain for grid search.
Log domains require positive lower bounds. A singleton domain is valid.

Lists can themselves be candidate values:

```python
{"choices": [[16, 8], [32, 16]]}
```

An empty space means one baseline candidate. A domain with no choices is an
error. Conditional spaces, list-index paths, and preprocessing search are not
implemented.

### Parser output

`SearchSpaceParser.parse()` returns `SearchSpace`, containing `SearchParameter`
objects with tuple paths and `Choices`, `IntegerRange`, or `FloatRange` domains.
The parser does not generate candidates or construct components.

It rejects malformed paths, unsupported roots, reserved fields, invalid
domains, and parent/child path conflicts. Backend parameter validity remains
the responsibility of existing component parsers and factories.

### Applying candidates

`apply_candidate()` applies `SearchCandidate.updates` to copied declaration
containers. Intermediate sections must exist and be mappings. The final
parameter may be absent, allowing an implicit constructor default to be tuned.

For example, this baseline supports a nested optimizer update:

```python
{"model_config": {"optimizer": {"type": "adam", "params": {}}}}
```

A string optimizer selector does not provide a traversable `params` mapping.
The application function does not silently expand it or mutate a ready object.

`apply_modeling_candidate()` normalizes the regularizer declaration and returns
a new `ModelingSpec`. Containers and numeric NumPy arrays are copied; arbitrary
backend objects are not cloned. Avoid mutable opaque state in constructor
parameters. Object-dtype arrays are rejected by configuration application.

## Search strategies

Strategies implement `BaseSearchStrategy.candidates(space)` and generate
`SearchCandidate` objects. They never train or score models.

### Grid

```python
strategy = {"type": "grid"}
```

Aliases: `grid`, `grid_search`.
Grid lazily enumerates combinations of discrete values. Continuous float
domains are rejected. Enumeration stops at exhaustion or the runner's budget.
A truncated grid explores only its initial combinations; it is not randomized.

### Random

```python
strategy = {"type": "random", "params": {"seed": 123}}
```

Aliases: `random`, `random_search`.
Random samples dimensions independently with replacement, so candidates may
repeat. For a non-empty space its generator is unbounded; consume it through
the budgeted runner, not `list(strategy.candidates(space))`.

The random constructor no longer takes `n_trials`. The runner owns that limit.

### Registry and resolver

`register_search_strategy(*names)` uses the shared `Registry` decorator.
The first name is canonical; the rest are aliases. Names are stripped and
lowercased. Internal spaces, underscores, and hyphens remain distinct.

`resolve_search_strategy()` accepts a registered name, `{type, params}` mapping,
concrete strategy class, or ready strategy instance. An instance cannot be
combined with constructor parameters. Ready strategies are returned unchanged;
custom stateful strategies must define their own run-state behavior.

Duplicate registration, unknown names, and conflicting embedded/separate
parameters are errors. Importing the strategies package registers built-ins.

The current generator contract supports grid/random. Adaptive strategies
requiring previous scores need a future feedback-contract extension.

## Objectives

### Base contract

```python
class BaseObjective:
    @property
    def direction(self): ...

    def evaluate(self, y_true, y_pred, *, output_names): ...
```

`direction` is `minimize` or `maximize`. `evaluate()` returns `ObjectiveResult`
with a finite scalar `score` and optional finite `scores_by_output`.
Arrays have shape `(n_samples, n_outputs)`.

Objectives do not choose data parts, scale arrays, build models, or fit them.
The evaluator supplies arrays in the selected scoring scale.

### Built-ins

| Name | Aliases | Aggregate |
|---|---|---|
| `mse` | `mean_squared_error` | Weighted mean of per-output MSE |
| `mae` | `mean_absolute_error` | Weighted mean of per-output MAE |
| `rmse` | `root_mean_squared_error` | Weighted mean of per-output RMSE |
| `callable` | None | User-supplied scalar or `ObjectiveResult` |

MSE/MAE/RMSE minimize their score. With no output weights, all outputs have equal
weight. If provided, `output_weights` must name exactly the model outputs, use
finite non-negative values, and include at least one positive weight.

```python
objective = {
    "type": "mse",
    "params": {"output_weights": {"y1": 1.0, "y2": 2.0}},
}
```

Per-output diagnostics are unweighted. RMSE takes the square root per output
before aggregating, not after aggregating MSE.

`_PredictionErrorObjective` shares array validation, weight handling, and
aggregation. It is an internal helper, not a registered objective.

### Callable

```python
objective = CallableObjective(
    function=lambda truth, prediction: float(
        np.mean((truth - prediction) ** 2)
    ),
    direction="minimize",
)
```

The function receives copies of validated arrays. It returns a real scalar or
`ObjectiveResult`. A raw function is not automatically resolved as an objective;
wrap it explicitly. User functions and custom objectives must not accumulate
unintended state across trials.

### Registry

Use `register_objective(*names)` for custom `BaseObjective` subclasses.
`resolve_objective()` accepts a name, declaration mapping, class, or ready
objective. A ready objective is preserved and cannot receive constructor
overrides. Objective instances are scoring objects, not training components.

### Scoring scale

- `scaled`: compare predictions with validation `y_scaled`.
- `prepared_raw`: inverse-transform predictions through the fitted target
  scaler and compare with validation `y_raw`.

`prepared_raw` requires a target scaler exposing `inverse_transform()`, including
an identity scaler when no numerical scaling is desired. It does not undo
stationarity transformations or reconstruct levels before differencing.
Training loss is not the selection objective and may include regularization.

## Budgets and reproducibility

```python
budget = SearchBudget(max_trials=30, epochs_per_trial=20)
```

- `max_trials` is positive and counts completed, invalid, and failed trials.
- `epochs_per_trial` is positive or `None`. `None` keeps model settings.
- An explicit epoch limit replaces the trial's baseline `epochs`/`max_iter`
  settings without changing the selected final-training configuration.
- The evaluator checks that the model exposes the requested `epochs` limit.
- Conflicting epoch search dimensions are rejected when a fixed limit is set.
  Parent updates that could replace protected model-parameter sections are
  also rejected by the implemented conservative conflict check.
- Callbacks or convergence may stop a fit before its limit.
- No wall-clock, memory, or GPU-resource budget is implemented.

`training_seed` controls model rebuilding and is the same across candidates.
The random strategy's `seed` controls candidate generation separately.
These controls do not establish deterministic behavior for every backend or
custom component. With no training seed, the model's normal construction and
seed behavior applies.

`collect_history=True` requests available training diagnostics, including
actual iterations where a backend reports them. The epoch limit is not a
measurement of actual work or a promise of equal wall-clock cost.

## Candidate evaluation and component lifetime

`ModelCandidateEvaluator.evaluate()`:

1. Validates shared arrays, split parts, options, and scoring requirements.
2. Applies candidate updates to baseline declarations.
3. Applies the temporary trial epoch limit.
4. Prepares constraints/regularizer layouts for the supplied feature layout.
5. Builds fresh backend components and a fresh model through `ModelAssembler`.
6. Optionally rebuilds with the training seed and clears input omissions.
7. Computes initializer weights from training rows using the compiled mask.
8. Fits through `FitExecutor` on training rows only.
9. Predicts validation rows, validates predictions, and computes the objective.
10. Returns `TrialResult`.

No winning model is stored in `TrialResult`. Initialization uses the extended
`ModelInitializer.compute_weights(row_indices=..., initialization_mask=...)`.
Calls omitting those arguments retain the previous initialization behavior.

The initial evaluator requires constructible training-component declarations
rather than ready model/component instances. Each component still follows its
own existing parser's accepted formats. A constructor parameter containing
shared mutable opaque state is outside the supported isolation guarantee.

The existing model protocol can reset run components during `fit()`. HPO does
not use reset as a way to recycle objects between candidates: fresh construction
provides trial independence.

## Results and errors

### TrialResult

Records trial ID, candidate, status, objective result, optional `TrainingResult`,
training seed, duration, requested epoch limit, and optional `TrialError`.

Statuses:

- `completed`: a valid objective result is available;
- `invalid`: candidate application or construction failed before the evaluator
  entered the initialized/training phase;
- `failed`: a later rebuild, initialization, fit, prediction, or scoring failed.

The status reflects execution phase, not a guarantee that every invalid
configuration is identified before training.

### Error policy

- `raise`: propagate the first candidate exception and stop.
- `record`: retain the candidate error and continue within the budget.

Shared input errors and strategy/parser errors are not silently converted into
successful searches. If no trial completes successfully, the runner raises
`ModelSearchFailedError`, whose `trials` attribute preserves recorded history.
No automatic baseline fallback is performed.

### ModelSearchResult

Provides:

- `best_modeling`, `best_candidate`, `best_trial`, and `best_score`;
- all trial records and `n_trials`/`n_completed`;
- objective direction, scale, and validation-part name;
- requested budget and total duration;
- a snapshot of split parts as prepared-row index tuples;
- `budget_reached` or `space_exhausted` stop reason.

Only completed trials compete. Exact ties keep the first completed candidate.
Reaching exactly `max_trials` reports `budget_reached`, even if a finite grid
also ends there; the runner does not fetch an extra candidate to distinguish it.

The result is an in-memory object, not a built-in JSON serializer. Callable
objectives, class selectors, and training diagnostics may require explicit
serialization decisions. Split snapshots alone are not complete preparation
provenance or a fully reproducible experiment manifest.

## Standalone example

This example uses a NumPy model and explicitly prepares HPO data. It is
illustrative code, not a claim of runtime verification.

```python
import numpy as np
import pandas as pd

from enhanced_granger_analysis.configuration.modeling_spec import ModelingSpec
from enhanced_granger_analysis.configuration.model_search_spec import (
    ModelSearchSpec,
    SearchBudget,
)
from enhanced_granger_analysis.execution.model_search_runner import ModelSearchRunner
from enhanced_granger_analysis.preprocessing.prepare_data import prepare_data

rng = np.random.default_rng(7)
a = rng.normal(size=240)
b = rng.normal(size=240)
y = np.zeros(240)
y[2:] = 0.6 * a[1:-1] - 0.2 * b[:-2] + rng.normal(scale=0.2, size=238)
data = pd.DataFrame({"a": a, "b": b, "y": y})

hpo_prepared = prepare_data(
    data,
    causes=["a", "b"],
    tested_causes=["a", "b"],
    effects=["y"],
    stationarity=None,
    lag_config={"max_lag": 2},
    splitter={
        "type": "temporal_blocks",
        "params": {"parts": {"training": 0.75, "validation": 0.25}},
    },
)

baseline = ModelingSpec(
    backend="numpy",
    model_config={
        "model": "conditional_linear",
        "solver": "coefficient",
        "epochs": 200,
        "learning_rate": 1.0,
    },
)

search = ModelSearchSpec(
    space={
        "model_config.learning_rate": {"choices": [0.1, 0.5, 1.0]},
    },
    strategy={"type": "grid"},
    objective={"type": "mse"},
    budget=SearchBudget(max_trials=10, epochs_per_trial=20),
    training_seed=42,
    collect_history=True,
    error_policy="record",
)

result = ModelSearchRunner().run(
    spec=search,
    modeling=baseline,
    prepared=hpo_prepared,
)

selected_modeling = result.best_modeling
selected_score = result.best_score
history = result.trials
# selected_modeling retains the final-training epoch setting of 200.
# No final model or Granger analysis has been fitted by these statements.
```

For random search, replace the strategy declaration with:

```python
{"type": "random", "params": {"seed": 123}}
```

The maximum number of evaluations still comes from `SearchBudget`.

## Current boundaries

- No automatic connection to the standard analysis execution path.
- No automatic final refit, test-part scoring, or restricted Granger training.
- No preprocessing hyperparameter search, conditional search space, K-fold,
  adaptive score feedback, pruning, or parallel candidate execution.
- No general cloning contract for ready training-component instances.
- No general file-loader support or complete search-result serializer supplied
  by the standalone code described here.
- Separate HPO and final Granger preprocessing is the intended caller workflow;
  orchestration of both contexts remains a separate integration task.
- Selected hyperparameters may behave differently if final preparation changes
  scales, selected lags, or dimensionality. Rebuild layout-dependent components
  against the final data; do not copy fitted HPO objects.
- HPO validation does not by itself establish the calibration of later Granger
  p-values after selection, constraints, or regularization.
