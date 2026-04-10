# Project Structure

This document describes the file and folder structure of the library and the role of each major module.

## Top-Level Layout

```text
complex_granger_analysis/
├── api/                  # Public API: builder, orchestrator, simple API, config loading
├── backends/             # Backend strategies and backend-specific implementations
├── core/                 # Core configs, protocols, exceptions, output dataclasses
├── docs/                 # User and developer documentation
├── example/              # Example data and notebooks
├── initializers/         # Weight initialization utilities for models
├── preprocessing/        # Stationarity, lag preparation, scaling
├── results/              # Result containers and causality/statistics utilities
├── scripts/              # Scripted experiment runners (config-driven workflows)
├── tests/                # Unit/integration tests
├── utilities/            # Shared helper functions (validation, metrics, etc.)
├── CONTRIBUTING.md       # Contribution workflow and development rules
├── README.md             # Project overview and quick start
├── VISION.md             # Roadmap and long-term direction
├── requirements.txt      # Base dependencies
├── requirements-torch.txt
└── requirements-tensorflow.txt
```

## Module Responsibilities

### api

Main entry points for users:

- `builder.py`: `MultitaskGrangerBuilder` (fluent API).
- `orchestrator.py`: `MultiTaskGrangerAPI` (direct orchestration control).
- `simple_granger.py`: lightweight pair-wise Granger API.
- `config_loader.py`: JSON/YAML config normalization and test-group iterator.

### backends

Backend abstraction layer and implementations:

- `backend_factory.py`: backend selection and strategy dispatch.
- `base_backend.py`: shared backend contract.
- `pytorch_backend.py`, `tensorflow_backend.py`, `scikit_backend.py`: backend-specific logic.
- Subfolders (`callbacks`, `constraints`, `models`, `object_loaders`, `regularizers`): backend-native components.

### core

Low-level shared contracts and data models:

- configuration objects (e.g. lag/training constraints),
- protocol-style interfaces,
- exception classes,
- output dataclasses.

### preprocessing

Data preprocessing pipeline used before model fitting:

- `stationarity/`: differencing and stationarity tests,
- `lag/`: lag order selection and lagged matrix construction,
- `scaling/`: deterministic scalers (`standard`, `minmax`, `robust`, `maxabs`, `identity`).

For details, see [data preprocessing](data_preprocessing.md).

### results

Result containers and matrix/statistics logic:

- causality matrices,
- p-values and test statistics,
- aggregation of base/reference model outputs.

### scripts

Automation for reproducible experiment runs:

- group/sweep execution,
- per-case matrix export,
- summary metrics vs. ground-truth.

For details, see [script usage](script_usage.md).

### tests

Covers key functionality:

- API behavior,
- backend factory and backend integrations,
- config loading and sweeps,
- lag/constraints/regularization,
- result-object correctness.

## Related Documentation

- [API usage](api_usage.md)
- [Configuration files](config_file_usage.md)
- [Backend usage](backend_usage.md)
- [Components loading](componets_loading.md)
- [Data preprocessing](data_preprocessing.md)
- [Test group configuration](test_group_config_usage.md)
- [Script usage](script_usage.md)
