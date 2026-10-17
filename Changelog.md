# History of Changes

## Version Changes 1.0 

This version is still available on branch version_1.0, but is considered to be out dated. It was the first version which was consisted of three classes of tests which corresponed with diffrent backends. It had regularizer, constraints and callback. The data preparation was automatic and had one precedure.

## Version Changes 2.0 

The previous README described the project using the older `granger_tests/...` entry points. The current codebase has moved to a new, modular API and added several practical capabilities.

Implemented additions and upgrades:

- **New public API layer**: `api/` now provides `MultitaskGrangerBuilder`, `MultiTaskGrangerAPI`, `SimpleGrangerAPI`, and config loaders, replacing legacy direct class usage from `granger_tests`.
- **Config-first workflows**: JSON/YAML loading (`BuilderConfigLoader`) and sweep expansion (`TestGroupConfigIterator`) are implemented for reproducible experiments.
- **Scripted benchmark runs**: `scripts/run_group_causality_tests.py` executes grouped experiments and writes per-case and summary outputs against ground truth.
- **Extended lag control**: in addition to automatic lag selection, the library now supports `custom_lags` and per-pair `custom_pair_lags` overrides through `LagConfiguration`.
- **Explicit preprocessing modules**: stationarity (`preprocessing/stationarity`) and scaling (`preprocessing/scaling`) are now separate reusable components.
- **Multiple deterministic scalers**: `standard`, `minmax`, `robust`, `maxabs`, and `identity` scalers are available in the core pipeline.
- **Structured output containers**: output dataclasses and `GrangerAnalysisResults` provide standardized access to p-values, F-tests, errors, signs, predictions, and weights.
- **Backend orchestration improvements**: backend load throttling (`backend_sample_fraction`, `backend_max_samples`) and optional hyperparameter optimization stages are available.
- **Run-scoped callback handling**: callbacks can be cloned and specialized per run (base/reference/hyperopt), improving logging and training loop isolation.
- **Expanded tests**: dedicated tests exist for builder API, config loader, custom lag behavior, backend factory, callbacks, regularizers, and result objects.

Items from the Vision that are still planned (not fully implemented yet):

- Native missing-value handling/imputation inside the pipeline.
- Distributed execution beyond current local parallelization.
- Full documentation generation and end-user CLI productization.

## Version Changes 2.1

The 2.1.0 line continues the 2.0.0 refactor with practical usability and workflow improvements, aligned with the current README and docs set.

Implemented and stabilized between 2.0.0 and 2.1.0:

- **Cleaner top-level project summary and navigation**: README now acts as a concise capability summary and points users directly to focused docs pages instead of duplicating long explanations.
- **Dedicated preprocessing documentation**: added explicit documentation for stationarity, lag preparation, and scaling in one place (`docs/data_preprocessing.md`), including stage order and example usage.
- **Stronger script-driven evaluation flow**: grouped experiment execution is documented and structured around list-of-DataFrame input, per-case result matrices, and a consolidated `summary.csv` with execution metrics.
- **More explicit API positioning**: distinction between orchestrator (`MultiTaskGrangerAPI`), fluent builder (`MultitaskGrangerBuilder`), and pairwise baseline (`SimpleGrangerAPI`) is now clearer in user-facing docs.
- **Improved config-first reproducibility story**: docs now better connect config loading, sweep expansion, and script execution into one reproducible pipeline.
- **Better discoverability of backend capabilities**: backend aliases and backend-specific component resolution are surfaced more clearly via direct links to backend/component documentation.

These improvements do not replace the long-term roadmap, but they significantly reduce onboarding friction and make the 2.x architecture easier to adopt in real testing workflows.


## Version Changes 2.2.0 (pip support and further consistency upgrades)

The 2.2.0 once again changes the folder structure of repository. Additional improvements in documentation and script usage. Changes allow:
- **Choosing computational unit in grouped experiments**: user can choose to calculate on **GPU** or **CPU**
- **Easier control of grouped experiment**: configuration of grouped experiment has been further development to allow better and more universal control
- **Development of saved parameters**: grouped experiments can be dane in two modes:
    - minimum: binary causality matrix + summary.csv
    - matrices: binary, p-values, F-test, sign + summary.csv
- **Pip support**: pyproject.toml has been added and module have been moved in to subfolder to allow pip install
- **Documentation update**: further development of docs to allow easier introduction to new users and development of library
 
## Version Changes 2.3.0

This section describes the functionality currently present on the
`version_2-3` development branch. It does not imply that every execution
path or backend/runtime combination has been validated for a stable release.

### Added

#### Executable analysis pipeline

- `MultiTaskGrangerBuilder` now exposes explicit execution stages:
  `prepare_data()`, `prepare_components()`, `build_model()`,
  `initialize_weights()`, `fit_base()`, `fit_restricted(cause)`,
  `fit_restricted_models()`, and `get_results()`.
- Preparation of runtime components can also be executed separately through
  `prepare_backend()`, `prepare_constraint()`, `prepare_regularizer()`,
  and `prepare_callbacks()`.
- Users can inspect intermediate artifacts and pause execution between
  stages instead of running the complete analysis through a single call.
- Externally prepared `PreparedData` can be supplied to the builder,
  avoiding repeated preprocessing.
- Optional weight initializers can be applied before fitting the base model.

#### Staged data preparation

- Added the stateful `DataPreparer` and `DataPreparationConfig`.
- Exposed separate stages for schema alignment, missing-value imputation,
  redundancy removal, stationarity transformation, lag construction,
  scaling, and workload reduction.
- Added schema policies for exact schema matching (`error`) and alignment
  to the union of input columns (`union`).
- Added optional missing-value imputation to the preparation pipeline.
- Added diagnostics and removal of constant and exactly duplicated variables.
- Added typed preprocessing metadata through `VariableRoles` and
  `RemovedVariable`.
- Added named feature-block mappings through `PreparedData.cause_slices`.
- Added `tested_cause_slices` for accessing only the tested input blocks.
- Added `sample_indices` to track observations retained from the prepared
  lagged arrays.
- Added block-wise input scaling through `DataPreparer.block_scale()` as
  an alternative to the standard `scale()` stage.

#### Results and metrics

- Added `ModelSnapshot` for storing predictions, coefficient weights,
  and metadata from individual model runs.
- Added `PreparationContext` to connect fitted results with preprocessing
  metadata and removed-variable restoration.
- Added a metric registry and evaluation by registered name or custom callable.
- Added cached metric evaluation, explicit cache bypass, and cache
  invalidation when snapshots are updated.
- Added F, Wald, likelihood-ratio, sign, base RSS, and base MSE metrics.
- Added structural and active-support restriction-counting modes.
- Added restoration of removed-variable entries when evaluating metric
  matrices with a preparation context.

### Changed

#### Builder and orchestrator responsibilities

- The builder now owns the executable pipeline and its runtime artifacts.
- `MultiTaskGrangerAPI` coordinates the standard workflow through the builder.
- Configuration methods and execution methods are distinct:
  configuration methods update settings, while execution methods perform
  the corresponding stage immediately.
- Selected configuration changes invalidate dependent execution artifacts.
  Backend switching after execution requires particular care; see the
  limitations below.

#### Model lifecycle

- Base and restricted runs reuse one working model object.
- Each restricted run inititaly starts by restoring copied base-model weights, but it can be change to initialized weights.
- Restricted feature blocks are selected by cause name using `cause_slices`,
  rather than inferred from the position in `tested_causes`.
- Callbacks are reset through the model interface for the base run and
  each restricted run.
- Predictions are inverse-scaled through the prepared target scaler before
  being stored in result snapshots.
- `get_results()` returns the current result container without triggering
  additional fitting.
- `fit()` and its alias `run()` execute the complete pipeline; they are not
  resume operations for a partially executed pipeline.

#### Data and variable contracts

- `causes` define model inputs, `effects` define model outputs, and
  `tested_causes` define the input variables tested through restricted runs.
- `tested_causes` must be a subset of `causes`.
- Effects outside `causes` are allowed. Their lagged values are not
  automatically added to the model inputs.
- Requested roles are preserved separately from effective roles after
  redundancy removal.
- Redundancy cleanup selects the variables needed by the configured
  input and output roles before lag construction.
- Empty effective input, output, or tested-cause roles cause preparation
  to fail rather than being interpreted as a negative causality result.
- Standard scaling is fitted after lag construction.
- The standard pipeline applies workload reduction after scaling.
- Workload reduction keeps raw arrays, scaled arrays, and sample indices
  aligned.
- `NoSampling` is the default sampling strategy.

#### Keras modeling implementation

- The former TensorFlow-oriented modeling implementation has been revised
  to use Keras-based model components.
- The documented design supports selecting a Keras runtime before importing
  the library.
- Library strategy aliases and the runtime used internally by Keras should
  be treated as separate configuration choices.
- Runtime-specific compatibility must be verified for the components used
  by an experiment.

#### Experiment output and optimization subsystem

- Result evaluation is organized around metrics rather than a fixed set
  of output matrices.
- Script output supports thresholded results and metric/p-value exports.
- The `with_sign` option controls whether thresholded relationships include
  coefficient-derived signs.
- Hyperoptimization code has been separated from the main orchestration
  workflow and its configuration supports model, optimizer, and regularizer
  parameter groups.

### Compatibility and API notes

- `MultitaskGrangerBuilder` remains an alias of `MultiTaskGrangerBuilder`.
- `PreparedData.predictor_slices` remains a compatibility alias;
  new code should prefer `cause_slices`.
- `PreparedData` no longer exposes positional `col_offsets` as its feature
  mapping contract. Some lower-level lag or component interfaces may still
  use offsets internally.
- `GrangerAnalysisResults.result(...)` returns a `pandas.DataFrame`.
- `GrangerAnalysisResults.evaluate(...)` returns a `CausalityMatrix`;
  access its DataFrame through `.data`.
- `results.sign` returns the restored DataFrame produced by evaluating
  the registered `sign` metric.
- Use `results.evaluate("sign", restore_removed=False)` to inspect signs
  for effective fitted variables only.
- `cause_slices` contains one named slice per effective model input.
  An array of block boundaries contains one additional endpoint; these
  representations must not be used interchangeably.
- The sign metric summarizes coefficient signs and is not the direction
  of the tested causal arrow.
- For fitted variables, coefficient sums with absolute value below
  `1e-12` retain the existing sign convention of `+1`.
- Newly restored sign entries without an available representative use `0`.

### Current limitations and behavior to note

- Hyperoptimization settings are accepted, but the standard builder pipeline does not currently execute hyperoptimization.
- The default builder and orchiestrator configures `StationarityTransformer`  to used. 
Calling `.stationarity()`, which is the default for the stage, will disable the stage.
- The low-level `prepare_data()` defaults to `schema_policy=""`,
  while the builder and `DataPreparationConfig` default to `"union"`.
- Changing the backend through `.backend(...)` does not currently invalidate
  existing runtime artifacts. Create a new builder when switching backends
  after pipeline execution.
- Computing a statistic does not by itself establish that its reference
  distribution is valid for every regularized or constrained model.

### Fixed

#### Named feature-block handling

- Removed a positional boundary check from restriction counting that
  incorrectly treated `cause_slices` as an array of block boundaries.
- Restriction counting now selects blocks by cause name and validates
  the corresponding slice.
- Corrected restriction counting for a single input and for the final
  input block.
- Migrated sign and Wald metric calculations from legacy `col_offsets`
  access to named `cause_slices`.

#### Signed result restoration

- The `results.sign` property now evaluates the registered sign metric
  with removed-variable restoration enabled.
- Removed duplicated sign-calculation logic from the result container.
- Sign restoration now follows the same mechanism as other metrics:
  duplicate rows and columns are copied from available representatives,
  while other newly restored entries use the metric's configured fill value.
- Added an explicit axis-consistency check before multiplying significance
  indicators by signs.
- Updating the base snapshot invalidates cached sign evaluations through
  the existing metric-cache invalidation mechanism.

### Version Changes 2.3.1

Old test files have been move to new directory, and new test naming startegy has been introduced were test have digist of version and 
subversion of packege - to know for which verision they were made. 

Additonaly registry mechanizm have been generalized and slightly updated. New core class of `Registry` is now the base class for all registries, and it provides a more flexible and robust way to manage components. Already existing registries have been updated to use the new registry mechanism.