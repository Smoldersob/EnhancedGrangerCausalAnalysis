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

### 2.3.1

Old test files have been move to new directory, and new test naming startegy has been introduced were test have digist of version and 
subversion of packege - to know for which verision they were made. 

Additonaly registry mechanizm have been generalized and slightly updated. New core class of `Registry` is now the base class for all registries, and it provides a more flexible and robust way to manage components. Already existing registries have been updated to use the new registry mechanism.

## Version Changes 2.4

### Added
- Added the optional `fit_restricted` parameter to the builder and
  orchestrator `fit()` methods, allowing the analysis to run without
  fitting restricted models.

### Changed
- Updated `run_group_causality_tests.py` to automatically skip restricted-model
  training when `--test wald` is selected, reducing computation time by
  fitting only the base model.
- Preserved the full fitting workflow for F and LR tests and for default
  `fit()` calls (`fit_restricted=True`).

### Notes
- This optimization does not change the calculation of the Wald statistic,
  its p-values, or the thresholding procedure.
- The change applies to the current Wald test implementation. Variants
  intended for regularized and multilayer models are outside the scope
  of this release and may be introduced as separate metrics in the future.

## Version Changes 2.5
### Added
- Added an `Initializer` protocol and a class registry with decorator-based
  registration and alias support.
- Added YAML/JSON initializer configuration using `type` and `params`.
- Added `InitializationContext` to provide data, the optional mask, and
  current model weights to initializers.

### Changed
- Moved initializer resolution from the configuration loader to the builder.
- Replaced the fixed `(A, B)` return contract with a list of weight arrays,
  allowing custom initializers to support multilayer models.
- Removed `LinearARXInitializer`; constructors now receive strategy parameters,
  while execution context is passed to `__call__()`.
- Delegated weight validation and assignment to models, removing
  initializer-specific transposition and shape checks from the builder.
- Updated zeros and random-normal initializers to use model weight templates.
- Updated OLS to infer dimensions from data, fit only mask-enabled
  coefficients, and return `[kernel, bias]`. Added a warning when the model
  exposes more than two weight arrays.
- Restricted initializer specifications to names, configuration mappings,
  classes, or `None`; instances and factory functions are not supported.

### Fixed
- Unified initializer name handling across file-based and programmatic configuration.
- Removed the `TypeError` fallback that could mask initializer execution errors.

## Version Changes 2.6

This version introduces a staged configuration and construction workflow
for constraints and regularizers.

User specifications, backend-independent preparation data, class
resolution, and backend-native component instances are handled separately.
The builder integrates these stages into the analysis pipeline and exposes
them individually for inspection and custom execution.

This release also updates backend package organization, component import
paths, and staged builder tests to match the new construction contract.

### Added

#### Backend-independent component preparation

- Added `core/components/` with constraint and regularizer specifications,
  parsing, preparation, class registries, and serialization utilities.
- Added `RelationConstraintsSpec`, `RelationConstraint`,
  `ZeroRelationRule`, and `MinAbsSumRelationRule` for named relation
  declarations.
- Added `PreparedConstraintData` for compiled masks and indexed
  minimum-absolute-sum rules.
- Added `RegularizerSpec`, `RegularizationLayout`, and
  `PreparedRegularizerData` for regularizer selection, feature metadata,
  and constructor parameters.
- Added backend-specific class registries with decorator-based
  registration and alias support. Registry resolution selects a class
  without constructing its instance.
- Added serialization of regularizer specifications and prepared data.
  Prepared regularizer data includes an explicit schema version.

#### Builder stages and configuration

- Added `.constraint(...)` for selecting the constraint implementation
  independently of the named rules supplied through `.relations(...)`.
- Added `prepare_constraint_data()` and `prepare_regularizer_data()` for
  inspecting backend-independent component inputs before construction.
- Added a default `mask_and_min_abs_sum` constraint selection, including
  when no explicit relation rules are supplied.
- Added validation preventing constraint and regularizer configuration
  from being supplied through `model_config`.

#### Explicit lag metadata

- Added `PreparedData.feature_lags`, containing the actual lag associated
  with each input-feature column.
- Extended `LagEngine.prepare()` to return
  `(X, y, block_boundaries, feature_lags)`.
- Propagated feature-lag metadata through the preparation pipeline.
- Added lag-dependent L1 parameter preparation using actual feature lags,
  with absolute or relative lag indexing.

#### Tests

- Added tests for canonical relation parsing, invalid declarations,
  feature-layout validation, mask preservation, infeasible rules,
  JSON/YAML references, and builder integration.

### Changed

#### Component organization and construction

- Moved backend-specific constraints and regularizers into
  `backends/keras/`, `backends/pytorch/`, and `backends/scikit/`.
- Introduced backend-specific constraint and regularizer base classes.
- Replaced strategy-level `build_constraint_from_relations(...)` with
  explicit preparation followed by `build_constraint(spec, data=...)`.
- Changed regularizer construction to
  `build_regularizer(selector, data=...)`.
- Removed constraint and regularizer resolution from the backend object
  loaders. Those components now use their dedicated class registries.
- Introduced an explicit NumPy regularizer penalty-and-gradient contract.
- Added diagnostic messages for backend import failures.

#### Relation declarations

- Replaced legacy relation mappings and shorthand values with a strict
  list of declarations containing `cause`, `effect`, and `rule`.
- Restricted rule types to `zero` and `min_abs_sum`.
- Added rejection of duplicate cause/effect declarations.
- Changed `min_abs_sum=0` to mean no additional lower-bound restriction;
  it does not force the relation to zero.
- Changed infeasible positive minimum-sum rules to raise an error by
  default when no active coefficients remain. The preparation utility
  also supports an explicit warning policy.

#### Builder invalidation

- Changing the backend through `.backend(...)` now clears the selected
  strategy and invalidates dependent component and model artifacts.
- Backend and regularization setters preserve an existing preparation
  result context while clearing model snapshots.
- Preparing data invalidates previously constructed components before
  creating the new result context.

### Breaking changes and migration

- Replace the top-level `regularizer` field with `regularizer_spec`.
- Replace `.regularization(regularizer=...)` with
  `.regularization(regularizer_spec=...)`.
- Put regularizer parameters inside `params`; flattened parameter
  mappings and the `name`/`kind` alternatives are not supported by the
  new regularizer parser.
- Replace legacy relation mappings, tuple keys, and shorthand rule values
  with canonical relation declarations.
- Use `[]` or `RelationConstraintsSpec()` for no explicit relation rules.
  `relations=None` is not accepted by the new parser.
- Resolve relation-file references through `BuilderConfigLoader` or
  `.from_file(...)`; `.relations(...)` does not perform file I/O.
- Do not pass `mask`, indexed `rules`, or layout metadata through
  `constraint_spec.params`; these belong to prepared constraint data.
- Constraint selection does not accept `None`, instances, or factory
  functions. Regularizer selection accepts `None` to disable
  regularization, but does not accept instances or factory functions.
- Update imports from the removed shared `backends/constraints/` and
  `backends/regularizers/` packages to the backend-specific packages.
- Update callers unpacking the former three-value return from
  `LagEngine.prepare()`.

### Current integration limitations

- The builder still contains a legacy `prepare_regularizer()` definition
  that overrides the new implementation.
- Backend import, component registration, and model-construction paths
  still contain inconsistencies introduced during the refactor.
- The new interfaces and added tests do not establish that the complete
  analysis pipeline works across all backends.
- End-to-end validation is required before treating this version as a
  stable release.

  ### Fixed

- Removed the legacy `prepare_regularizer()` definition that overrode the
  new staged implementation in `MultiTaskGrangerBuilder`.
- Corrected PyTorch dependency detection and the import/export name of
  `BasePyTorchRegularizer`.
- Replaced the repository-dependent absolute import in the orchestrator
  with a package-relative import.
- Removed obsolete component-resolution calls from PyTorch model
  construction. Prepared constraint and regularizer instances are now
  passed directly to the model.
- Removed an undefined legacy regularizer variable from scikit model
  construction.
- Aligned the scikit regularizer abstract interface with the built-in
  `value_and_grad()` implementations.
- Removed the import of the deleted `constraints_config` module from
  core package initialization.
- Updated staged builder tests to use the new component construction
  signatures and nested regularizer parameters.

## Version Changes 2.7

This section describes the backend refactor and its integration into the standard
builder pipeline. The analysis script and unit tests were successfully executed
after the integration changes. Validation is limited to the exercised environment,
configurations, and test cases; numerical equivalence and statistical calibration
across backends are not implied.

### Added

- Added `ModelInputSpec` for immutable structural input/output metadata,
  including named feature blocks and per-feature lags.
- Added `GrangerModelProtocol`, separating the model lifecycle contract
  from backend-specific implementations.
- Added `TrainingResult` for optional training history and diagnostics.
- Added backend-local model registries supporting registered names,
  specification mappings, concrete classes, and ready model instances.
- Added backend-local conditional linear adapters and separate
  `LinearGrangerNetwork` implementations for NumPy, PyTorch, and Keras.
- Added explicit strategy builders for optimizers, losses, and callbacks.
- Added a PyTorch optimizer adapter for model-owned optimizer construction
  and independent optimizer-state reset.
- Added backend-local callback registries for NumPy and PyTorch and shared
  callback lifecycle utilities.
- Added NumPy solver selection through `auto`, `restricted_lstsq`, and
  `coefficient`, with the selected solver exposed as `solver_used_`.
- Added proximal operators for the built-in NumPy L1 and lag-dependent L1
  regularizers.
- Added shared prepared-constraint validation and configuration conversion.
- Added ground-truth evaluation options:
  `evaluate_only_known_gt`, `gt_nan_means_unknown`, and `drop_diagonal`.
- Added `n_evaluated` to ground-truth evaluation output.
- Added `docs/numpy_backend_optimization.md`.
- Added backend construction and component test modules for NumPy,
  PyTorch, and Keras.

### Changed

- Changed model construction to:
  `build_model(spec=None, *, components=None, config=None)`.
- Separated ready component objects from ordinary constructor settings.
- Changed model construction to require `ModelInputSpec` instead of
  receiving training observations through model initialization.
- Changed fitting to explicit `fit(X, y, ...)` calls returning
  `TrainingResult`.
- Added explicit lifecycle separation between predictor rebuilding,
  weight restoration, input omission, and training-component reset.
- Replaced `backends/scikit/` with `backends/numpy/`.
  The selection aliases `sklearn`, `scikit`, and `scikit-learn` remain
  accepted and resolve to the NumPy strategy.
- Moved model and component construction responsibilities from shared
  object loaders into backend strategies and dedicated resolvers.
- Strengthened validation of model specifications, constructor arguments,
  training arrays, coefficients, constraints, and regularizers.
- Changed ground-truth alignment from axis intersection to separate
  row/column unions with an explicit evaluation mask.
- Revised SHD calculation for reciprocal edge pairs and rectangular
  effect-by-cause matrices.
- Replaced staged-builder test definitions with tests targeting the new
  construction and explicit-data fitting interfaces.
- Renamed `test_scalers_2_3.py` to `test_scalers_2_7.py` without changing
  its contents (tested to be stil valid).

### Removed

- Removed the shared `backends/models/` implementations and
  `BaseGrangerModel`.
- Removed `backends/object_loaders/`.
- Removed the former scikit backend implementation package.
- Removed the implementation from the former shared Keras callback-reset
  module. An empty file remains; new adapters use backend-local helpers.

### Breaking Changes

- Replace imports from removed model, object-loader, and scikit packages
  with the corresponding backend-local or `core/models/` modules.
- Replace dimension-keyword model construction with the new
  `spec` / `components` / `config` interface.
- Replace model-level `initialize(X, targets=y)` with structural
  construction through `ModelInputSpec`.
- Pass prepared observations explicitly to each `fit(X, y, ...)` call.
- Do not rely on silent fallback from an unknown model name to a default
  model.
- For PyTorch, place native optimizer options such as `lr` in optimizer
  `params`; keep `gradient_accumulation_steps` in model configuration.
- Custom NumPy regularizers should implement `_value_and_grad()` through
  the validated `BaseNumpyRegularizer` interface.
- Update direct access to `MetricCalculator` working attributes to the
  new raw, aligned, binary, and evaluation-mask attributes.
- Recalculate benchmark baselines when adopting the new evaluation-domain
  and SHD conventions.

### Builder integration

- Migrated `MultiTaskGrangerBuilder` to the new backend construction contract:
  `build_model(spec=None, *, components=None, config=None)`.
- Added construction of `ModelInputSpec` from effective prepared-data metadata,
  including ordered causes/effects, named feature slices, and actual feature lags.
- Routed callback specifications through backend-specific callback construction.
- Added explicit optimizer and loss preparation before model construction.
- Separated ready backend components from non-component model configuration.
- Replaced observation-based model initialization and parameterless fitting
  with explicit `fit(X, y)` calls.
- Added independent optimizer and callback resets for base and restricted runs
  without rebuilding predictor parameters.
- Preserved explicit weight restoration and named feature-block omission for
  restricted fits.
- Avoided duplicate component resets by resetting components before fitting and
  calling `fit(..., reset_components=False)`.

### Validation

- Successfully ran the analysis script and unit tests after migrating the
  builder to the new backend/model interface.
- These checks validate the exercised execution paths and test cases; they do
  not establish exhaustive coverage of every backend, device, component, or
  configuration combination.
- No new performance, cross-backend numerical equivalence, or statistical
  calibration claim is made.

### Known limitations

- Hyperoptimization remains disabled in the standard builder pipeline, as in
  the previous implementation. This update neither enables automatic parameter
  search nor introduces a hyperoptimization regression.
- Backend-specific optimizer parameters and training diagnostics are not
  interchangeable merely because they share a common construction interface.

## Version Changes 2.8

This update redesigns lag selection and preparation around explicit
output/predictor lag specifications.

Lag selectors are auxiliary preprocessing heuristics. Their purpose is
to propose a manageable set of lagged inputs for subsequent modeling,
not to establish causality or reproduce the assumptions and architecture
of the final model.

The final model may apply additional constraints, regularization, or a
different predictive architecture.

### Added

#### Selector interfaces and resolution

- Added registry-based lag-selector registration, discovery, and resolution.
- Added selector specifications using registered names and canonical
  `type`/`params` mappings.
- Added separate forwarding of selector constructor parameters through
  `lag_selector_params`.
- Added `resolve_lag_engine_config()` for resolving declarative engine
  configuration during preprocessing.

#### Selection methods

- Added a shared cumulative-expansion and backward-pruning implementation
  for `ICLagSelector` and `CVLagSelector`.
- Added independent `initial` and `current` reference policies for
  addition and pruning.
- Added `CommonOrderICLagSelector`, which selects one common order by
  aggregating auxiliary IC scores across requested outputs.
- Added `CommonOrderCVLagSelector`, which selects one common order by
  aggregating time-series validation losses across requested outputs.
- Added optional output weights for common-order selection.
- Added training-fold response normalization for common-order CV.
- Added a NumPy-only forward-chaining splitter with configurable validation
  size, gap, and maximum training-window length.

#### Preparation results and provenance

- Added union, intersection, and selection-frequency aggregation across
  independently processed datasets.
- Added explicit predictor-wide and output/predictor lag overrides.
- Added `LagPreparationResult` and `LagSegmentInfo` in `core/lag_config.py`.
- Added direct result fields for feature identities, actual lags,
  predictor-block boundaries, and the output-by-feature mask.
- Added `PreparedData.lag_result`.
- Added active and removed predictor reporting.
- Added the removal reason `"no_selected_lags"` and metric-dependent
  restoration of excluded tested causes.

#### Diagnostics

- Added reporting of numerically unsuccessful candidates.
- Added diagnostics for rank-deficient auxiliary fits and use of the
  residual-variance floor.
- Added per-order scores for common-order selectors.
- Added criterion tables and criterion-specific optimal orders for the
  historical VAR selector.

### Changed

#### Shared lag contracts

- Changed `BaseLagSelector` into an abstract interface without imposing
  constructor parameters, a specific search procedure, or mask construction.
- Changed selector outputs to nested dictionaries of explicit lag tuples.
- Moved shared lag configuration and result contracts to
  `core/lag_config.py`.
- Flattened feature-layout fields into `LagPreparationResult` instead of
  maintaining a separate `LagFeatureLayout` object.

#### IC/CV selection

- Changed initial selection to evaluate cumulative lag ranges rather than
  isolated delays.
- Changed pruning to remove the largest active lag while retaining the
  remaining selected terms.
- Changed candidate comparisons to use identical response timestamps.
- Allowed rank-deficient auxiliary regressions to be fitted using
  least-squares solutions rather than rejecting them unconditionally.
- Separated candidate admissibility for IC and CV.
- Added a choice between column-count and numerical-rank complexity
  penalties for IC.
- Added an explicit residual-variance floor for IC scoring.
- Changed numerically equivalent score comparisons to favor smaller
  candidate representations.
- Changed expected candidate-specific numerical failures to reject the
  candidate rather than immediately abort the complete search.
- Kept configuration and programming errors distinct from candidate
  failures.

#### CV scoring

- Changed intercept-only validation to estimate its prediction from
  training responses within each fold.
- Removed the fallback from CV loss to information-criterion scoring.
- Kept validation observations consistent across candidate models.
- Allowed rank-deficient and underdetermined training regressions when
  their validation predictions remain computable.
- Removed the scikit-learn splitter dependency from the CV implementation.

#### Engine and pipeline integration

- Changed parallel selection to operate across independent datasets.
- Changed feature construction to derive data columns, masks, block
  boundaries, and lag metadata from the same ordered feature identities.
- Changed `LagEngine.prepare()` to return a structured result.
- Moved selector construction and engine-configuration resolution into
  `DataPreparer.lag()`.
- Removed selector construction and lag-specific default injection from
  the configuration loader.
- Updated effective predictor and tested-cause roles after lag exclusion
  while preserving requested roles.
- Preserved output roles for variables excluded only as predictors.

### Historical VAR compatibility

- Retained the historical VAR selection procedure as
  `NaiveVARLagSelector`.
- Preserved `VARLagSelector` as a Python alias.
- Preserved `"var"` as a registered selection name and added `"naive_var"`.
- Preserved optional centering, separate `VAR.fit(p)` calls, the four
  AIC/BIC/HQIC/FPE criteria, and the maximum/minimum combination of their
  selected orders.
- Changed the return value to the current `LagSpec` contract.
- Added explicit reporting of failed candidate orders.
- Rejected non-finite criterion results and the case where no candidate
  produces four finite criteria.
- Removed the accidental selection of the first order when all criterion
  values are invalid.

The historical selector retains its original candidate-dependent sample
trimming. The new common-order selectors compare orders on a shared
response window.

### Fixed

- Corrected the mismatch between isolated-delay scoring and cumulative
  lag interpretation.
- Prevented unsuccessful predictor screening from automatically including
  the initialized minimum lag.
- Corrected pruning so reducing an order removes a feature instead of
  replacing one isolated delay with another.
- Prevented rank deficiency alone from aborting IC/CV selection.
- Prevented unsuccessful candidate evaluations from being mistaken for
  valid optimal orders.
- Made self-lag-0 exclusion explicit.
- Preserved earlier removal metadata when adding lag-exclusion records.
- Supported neutral restoration of duplicate tested causes whose
  representatives were subsequently excluded by lag preparation.

### Breaking changes and migration

- Replace `LagConfiguration` with `LagEngineConfig`.
- Replace `custom_lags` with `predictor_overrides`.
- Replace tuple-keyed `custom_pair_lags` with nested `pair_overrides`.
- Treat lag lists and tuples as explicit lag values, not interval endpoints.
- Replace the old selector result with
  `fit(data, outputs=...) -> LagSpec`.
- Replace positional unpacking of `LagEngine.prepare()` with named access
  to `LagPreparationResult`.
- Replace `result.layout.*` access with direct result fields.
- Configure automatic search bounds on the selector. Engine `max_lag`
  defines the fixed baseline when no selector is enabled.
- Use `lag_selector.params.*` or `lag_selector_params.*` for selector
  parameter sweeps.
- Do not replace the historical `"var"` selection with a common-order
  selector when reproducing earlier experiments without recording the
  methodological change.

### Interpretation and limitations

- All selectors are preprocessing heuristics; none guarantees the best
  lag specification for the final model.
- Linear auxiliary scoring can overlook terms useful to a different
  final architecture or in interactions with other predictors.
- Initial per-pair screening is conditional on the selected AR component,
  not on all other external predictors.
- Tail pruning does not remove arbitrary interior lags.
- Rank-deficient fitting does not make individual coefficients identifiable.
- Column-count IC penalization is an explicit representation-size
  heuristic when the design is rank deficient.
- Common-order IC is an aggregate of equation-wise scores, not the
  classical joint VAR information criterion.
- Historical VAR selection may remain unavailable when its residual
  covariance requirements are not satisfied.
- Restored neutral results for lag-excluded causes are reporting
  conventions, not independently calculated hypothesis-test results.
- Zero-feature designs require compatible scaling and backend handling.
- Integration tests and numerical regression checks are required before
  treating these changes as a validated release.

## Version Changes 2.8.1 - Stationarity preprocessing refactor

### Changed

- Replaced `StationarityTransformer` with independent registry-selected
  `ADFStationarityEnforcer` and `KPSSStationarityEnforcer` implementations.
- Added a shared `BaseStationarityEnforcer` interface and reusable
  differencing workflow.
- Unified configuration under `stationarity`; the loader and builder
  preserve selections, and the preparation stage resolves them.
- Made stationarity processing disabled by default, with explicit
  `None` support across preparation and public APIs.
- Renamed fitted-state access to `stationarity_enforcer` in `PreparedData`
  and `PreparationContext`.
- Added validation of input schema and finite values, fitted-state
  isolation, and explicit `exclude` configuration.

### Removed

- Removed the legacy transformer API, `test_name` selection,
  `fit_stationarity()`, and `prepare_static()`.

### Tests and documentation

- Added unit-test code for enforcers, registry resolution, validation,
  state isolation, and pipeline handling.
- Updated preprocessing, API, configuration, component-loading, and
  project-structure documentation.
- Tests were not executed as part of this change preparation.
  Numerical helper search and fallback behavior remain unchanged.

## Version Changes 2.8.2 — Shared analysis execution and configuration refactor

This entry describes the adopted architecture refactor. No new package version
is assigned here. Implementation examples and this documentation update are not
a report of executed regression tests or cross-backend validation.

### Added

- Added `AnalysisSpec` and `ModelingSpec`, reusing the existing preparation
  configuration and variable-role contracts rather than duplicating them.
- Added `ModelAssembler`, `ModelInitializer`, and `FitExecutor` as focused services.
- Added `FitOutcome` for one fit's model-scale predictions, copied full weights,
  existing `TrainingResult`, and timing information.
- Added `AnalysisRunner` for full execution and standard fit coordination.
- Added internal `AnalysisRunProgress` for publishing artifacts and synchronizing
  builder state after successful stages, including partial failures.
- Added document-only `AnalysisConfigLoader` and an `experiments` package for
  explicit configuration-case expansion.
- Added architecture and migration documentation.

### Changed

- Both the builder and orchestrator execute the shared runner. The orchestrator
  no longer constructs a builder or mutates its private configuration mapping.
- Extracted standard full/restricted fitting into `GrangerTrainer`; it returns
  fit artifacts instead of modifying `GrangerAnalysisResults`.
- Moved outcome-to-snapshot conversion to the runner, preserving target inverse
  scaling, current first-weight-array coefficient extraction, and snapshot metadata.
- Retained `TrainingResult` as the training diagnostics contract; diagnostics are
  not automatically added to final snapshot metadata or script summary columns.
- Centralized backend aliases and mapping-form declaration handling in the
  existing `BackendFactory`, without adding another backend resolver.
- Kept the external analysis mapping flat; specifications group configuration
  internally and exclude source/prepared data and live model state.
- Clarified `from_config()` as a top-level update and `replace_config()` as a
  reset/replacement; nested maps are not recursively merged.
- Device values are interpreted by existing Keras/PyTorch managers. NumPy ignores
  `device` at strategy model construction, including resolved model params, to
  support shared experiment configuration and automatic backend selection.
- Group iteration returns mappings, expands explicit case rows, and resolves
  document references without component normalization or Cartesian expansion.

### Removed

- Removed transitional configuration normalization through `configuration/legacy.py`.
- Removed normalizing loader helpers `normalize_builder_config()` and
  `normalize_relations()`; existing domain parsers remain available directly.
- Removed duplicate backend-alias/device-value handling from document loading.

### Compatibility and migration

- `BuilderConfigLoader` remains an alias of `AnalysisConfigLoader`; it no longer
  has the former normalizing behavior. Relation-file loading returns raw data.
- `MultitaskGrangerBuilder` remains an alias of `MultiTaskGrangerBuilder`.
- Explicit builder stages and public builder fit return values are retained.
  Complete fit does not dispatch through overridden builder stage methods.
- `run_analysis()` is the full runner entry point; lower-level `run()` has a
  different contract and is not a public resume operation.
- Prepared data is an explicit runtime input. `reuse_data` alone does not create
  a runner cache; the script's external cache retains its own limitations.
- Backend mapping params retain their role as model defaults, not arbitrary
  strategy-constructor options. Explicit model settings take precedence.

### Not included

- Automatic HPO remains inactive despite retained compatibility fields. Future
  HPO targets model quality, not Granger statistics or ground-truth causal scores.
- No interchangeable high-level procedure registry, resumable session API,
  universal component resolver, or new training-device manager is introduced.
