# Project Structure

This document describes the repository layout and the role of each major package and module. The installable Python package lives in `complex_granger_analysis/` inside the repository root.

## Repository Layout

```text
complex_granger_analysis/
├── complex_granger_analysis/   # Installable package source
│   ├── api/                    # Public API: builder, orchestrator, simple API, config loading
│   ├── backends/               # Backend strategies and backend-specific implementations
│   ├── core/                   # Core configs, protocols, exceptions, output dataclasses
│   ├── initializers/           # Weight initialization helpers
│   ├── preprocessing/          # Stationarity, lag preparation, scaling
│   ├── results/                # Result containers and causality/statistics utilities
│   ├── scripts/                # Scripted experiment runners and packaged data files
│   ├── tests/                  # Unit/integration tests packaged with the library
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

- configuration objects such as lag and training constraints,
- protocol-style interfaces,
- exception classes,
- output dataclasses.

### initializers

Utilities for building model state before training:

- reusable initialization helpers,
- backend-independent weight initialization logic.

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
- aggregation of base/reference model outputs,
- helpers for converting fitted model outputs into final causality reports.

### scripts

Automation for reproducible experiment runs:

- group/sweep execution,
- per-case matrix export,
- summary metrics vs. ground-truth.

For details, see [script usage](script_usage.md).

### tests

Covers the main execution paths:

- backend factory and backend integrations,
- builder and orchestrator behavior,
- config loading and group sweeps,
- lag selection, constraints, and regularization,
- result-object correctness and smoke tests.

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
