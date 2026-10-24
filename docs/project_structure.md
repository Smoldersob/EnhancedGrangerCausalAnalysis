# Project Structure

This document describes the repository layout and the role of each major package and module. The installable Python package lives in `enhanced_granger_analysis/` inside the repository root.

## Repository Layout

```text
enhanced_granger_analysis/
├── enhanced_granger_analysis/   # Installable package source
│   ├── api/                    # Stateful builder, standard orchestration, config loading
│   ├── backends/               # Strategies, models, and backend-specific components
│   ├── core/                   # Shared configurations, protocols, and exceptions
│   ├── hiperopt/               # Hyperoptimization subsystem
│   ├── initializers/           # Optional model-weight initialization
│   ├── preprocessing/          # Staged preparation and PreparedData contract
│   ├── results/                # Snapshots, metrics, restoration, and causality matrices
│   └── utilities/              # Shared helper functions (validation, metrics, etc.)
├── docs/                       # User and developer documentation
├── example/                    # Example data and notebooks
├── CONTRIBUTING.md             # Contribution workflow and development rules
├── README.md                   # Project overview and quick start
├── VISION.md                   # Roadmap and long-term direction
├── Changelog.md                # Release notes and change history
├── pyproject.toml              # Build metadata and package configuration
├── requirements.txt            # Base dependencies
├── requirements-torch.txt      # PyTorch backend extras
├── requirements-tensorflow.txt # TensorFlow backend extras
└── requirements-full.txt       # Combined backend dependency set
```

## Module Responsibilities

### api

Public analysis entry points and configuration loading:

- `builder.py`: `MultiTaskGrangerBuilder`, which stores configuration and
  runtime artifacts and executes individual pipeline stages.
- `orchestrator.py`: `MultiTaskGrangerAPI`, which coordinates the standard
  workflow through the builder.
- `simple_granger.py`: independent pair-wise statsmodels analysis.
- `config_loader.py`: JSON/YAML configuration normalization and grouped
  experiment configuration expansion.

`MultitaskGrangerBuilder` is a compatibility alias of
`MultiTaskGrangerBuilder`.

Configuration methods and execution stages are distinct. The builder is not
only a facade over the orchestrator.

### backends

### backends

Backend abstraction, model implementations, and backend-native components:

- `backend_factory.py`: backend selection and strategy dispatch.
- `base_backend.py`: shared strategy contract.
- `keras/keras_backend.py`: Keras strategy.
- `pytorch/pytorch_backend.py`: native PyTorch strategy.
- `scikit/scikit_backend.py`: scikit strategy.
- `keras/constraints/`, `pytorch/constraints/`, `scikit/constraints/`:
  constraint bases, class registries, and implementations.
- `keras/regularizers/`, `pytorch/regularizers/`, `scikit/regularizers/`:
  regularizer bases, class registries, and implementations.
- `models/`: model implementations and lifecycle contracts.
- `callbacks/`: shared callback infrastructure and implementations.
- `object_loaders/`: backend object loading, including callback and
  optimizer resolution.

Constraint and regularizer specifications are no longer resolved by
the object loaders. Their dedicated registries select classes, and
strategies construct instances from prepared component data.

### backends

Backend selection, registered model construction, and backend-native components:

- `backend_factory.py`: strategy selection and backend aliases.
- `base_backend.py`: shared strategy construction interface.
- `keras/`, `pytorch/`, `numpy/`: backend-specific strategies and components.
- `models/conditional.py` within each backend: training/lifecycle adapter.
- `models/network_parts.py` within each backend: predictor architecture.
- Backend-local `callbacks/`, `optimizers/`, and `losses/`: component resolution.
- Backend-local `constraints/` and `regularizers/`: implementations and registries.

The former shared `backends/models/`, `backends/object_loaders/`, and
`backends/scikit/` packages are removed.

### core models

Shared contracts, configuration objects, registries, and exceptions.

`core/models/` defines structural input metadata, model protocols,
model-class registry infrastructure, training diagnostics, and shared array
validation.

`core/components/` additionally contains shared callback registry and lifecycle
utilities alongside constraint and regularizer infrastructure.

- `constraint_spec.py`: named relation rules and prepared constraint data;
- `constraint_parsing.py`: canonical relation parsing;
- `constraint_preparation.py`: compilation against effective feature layout;
- `constraint_registry.py`: reusable constraint class-registry logic;
- `constraint_serialization.py`: prepared constraint serialization;
- `regularizer_spec.py`: normalized regularizer selection and layout data;
- `regularizer_parsing.py`: regularizer specification parsing;
- `regularizer_preparation.py`: layout and constructor-data preparation;
- `regularizer_parameters.py`: shared built-in parameter validation and
  lag-weight mapping;
- `regularizer_registry.py`: reusable regularizer class-registry logic;
- `regularizer_serialization.py`: specification and prepared-data serialization.

Backend-independent preparation does not construct backend-native
component instances.

### initializers

Utilities for building model state before training:

- reusable initialization helpers,
- backend-independent weight initialization logic.

### preprocessing

Staged data preparation and the contract passed to model execution:

- `prepare_data.py`: the convenience entry point and preparation data models,
  including `PreparedData`, `VariableRoles`, and `RemovedVariable`.
- `pipeline.py`: `DataPreparer` and `DataPreparationConfig`.
- `missing/`: optional missing-value imputation.
- `stationarity/`: stationarity tests and transformations.
- `lag/`: lag selection, feature construction, and target alignment.
- `scaling/`: input and target scaling, including block-wise scaling.
- `sampling.py`: workload-reduction strategies.

Prepared data separates requested roles from effective roles and maps inputs
to feature blocks through named `cause_slices`.

For stage order and defaults, see [Data Preprocessing](data_preprocessing.md).

### results

Model-run storage, metric evaluation, and labelled result matrices:

- `granger_results.py`: result container, model snapshots, preparation context,
  metric evaluation, and convenience accessors.
- `metrics.py`: registered statistical, diagnostic, and sign metrics.
- `restoration.py`: restoration of requested result axes using preprocessing
  metadata.
- `causality_matrix.py`: labelled metric-matrix wrapper.
- `statistics.py`: statistical helper calculations.

Metrics are calculated for effective fitted variables.
Restoration is applied when returning evaluated matrices.

Signed results combine restored metric and sign matrices.
Coefficient blocks are selected through named cause slices.

For the public result contract, see [Results and Metrics](results_and_metrics.md).

### hiperopt

Hyperoptimization subsystem for evaluating candidate configurations.

Its configuration can describe model, optimizer, and regularizer parameter
groups.

The standard builder pipeline currently accepts hyperoptimization settings
but does not execute an optimization stage.

Keep subsystem capabilities separate from their integration into the standard
analysis workflow.

(still under development)

### scripts

Automation for reproducible experiment runs:

- group/sweep execution,
- per-case matrix export,
- summary metrics vs. ground-truth.

For details, see [script usage](script_usage.md).

### tests

Unit and integration tests for library components and execution workflows.

The test directory is located inside the source package, while package
configuration excludes it from installed distribution packages.

New lifecycle and result-contract changes should include regression tests
for:

- named feature-block selection;
- single-input and final-input restriction counting;
- tested-cause ordering independent of input ordering;
- restored sign and metric axes;
- callback and omission isolation;
- snapshot independence;
- metric-cache invalidation.

### utilities

Shared helper functions used across the package:

- validation helpers,
- metric calculations,
- small reusable analysis utilities.

## Related Documentation

- [API usage](api_usage.md)
- [Backend usage](backend_usage.md)
- [Components loading](components_loading.md)
- [Data preprocessing](data_preprocessing.md)
- [Configuration files](config_file_usage.md)
- [Script usage](script_usage.md)
