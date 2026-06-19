# EnhancedGrangerCausalAnalysis

EnhancedGrangerCausalAnalysis is a compact framework for advanced Granger-causality workflows on multivariate time series.

It extends classic pair-wise Granger tests with configurable multitask modeling, backend abstraction, constraint-aware training, and config-driven experiment sweeps.

Current package version: `2.2.0`

## Architecture at a glance

The library is organized into a small set of focused layers:

- `api/` for public entry points (`MultiTaskGrangerAPI`, `MultitaskGrangerBuilder`, `SimpleGrangerAPI`),
- `preprocessing/` for stationarity, lag construction, and scaling,
- `backends/` for TensorFlow/PyTorch/scikit-learn strategy implementations,
- `results/` for causality/statistical output containers,
- `scripts/` for reproducible config-driven experiment runs.

For a full module-by-module map, see [docs/project_structure.md](docs/project_structure.md).

## What Is Implemented

Compared to a standard Granger test, the library adds:

- multitask analysis (many causes/effects in one run),
- lag engineering with selectors and optional lag masks,
- stationarity preprocessing and X/y scaling,
- relation-based constraints (e.g. force or limit selected cause-effect paths),
- regularization (`l1`, lag-dependent `l1` variants),
- callbacks and optional hyperparameter sweeps,
- unified outputs (`p_value`, `F_test`, sign, error matrices),
- config-driven group runs with `summary.csv`, per-case result files, `reuse_data`, and `compute_device` selection.

For details, see:
- [docs/api_usage.md](docs/api_usage.md)
- [docs/config_file_usage.md](docs/config_file_usage.md) — Builder and group config format, including sweepable fields
- [docs/data_preprocessing.md](docs/data_preprocessing.md)
- [docs/project_structure.md](docs/project_structure.md)
- [docs/script_usage.md](docs/script_usage.md) — Script runner and group execution workflow

## APIs In One Line

- `MultiTaskGrangerAPI`: low-level orchestrator with full control.
- `MultitaskGrangerBuilder`: fluent, config-friendly wrapper over the orchestrator.
- `SimpleGrangerAPI`: lightweight pair-wise baseline (statsmodels).

API comparison and usage examples:
- [docs/api_usage.md](docs/api_usage.md)

## Backends

Supported backends:

- `pytorch` (`torch`)
- `tensorflow` (`tf`, `keras`)
- `sklearn` (`scikit`, `scikit-learn`)

When not specified, backend selection prefers: PyTorch -> TensorFlow -> scikit-learn.

Backend-specific component resolution (callbacks, regularizers, constraints) is documented in:
- [docs/backend_usage.md](docs/backend_usage.md)
- [docs/components_loading.md](docs/components_loading.md)

## Quick Example (Orchestrator)

```python
import pandas as pd

from enhanced_granger_analysis.api import MultiTaskGrangerAPI
from enhanced_granger_analysis.core.lag_config import LagConfiguration

df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)

api = MultiTaskGrangerAPI(backend="pytorch", reuse_data=True)

output = api.fit(
    data=df,
    causes=["u", "f1", "f2"],
    effects=["x1", "x2", "u1", "e1"],
    tested_causes=["u", "f1", "f2"],
    lag_config=LagConfiguration(max_lag=12, use_lag_zero=False),
    relations={
        ("x1", "f2"): {"zero": True},
    },
    model_config={"epochs": 40, "batch_size": 32, "learning_rate": 1e-3},
)

causality = output.results.result(threshold=0.01, with_sign=True)
print(causality)
```

If you prefer fluent chaining or config files, use `MultitaskGrangerBuilder`.

## Config-Driven Test Runs

Run grouped experiments from JSON/YAML with per-case matrices and summary metrics:

```bash
python scripts/run_group_causality_tests.py --config scripts/run_group_causality_tests.config.json
```

This flow supports loading a list of input DataFrames, running sweep cases, measuring execution time, comparing against ground truth, and saving `summary.csv`.

Details:
- [docs/config_file_usage.md](docs/config_file_usage.md) — Group config structure and sweepable fields
- [docs/script_usage.md](docs/script_usage.md) — Script runner options and output layout

## Installation

### Prerequisites

- Python 3.9+ is recommended.
- Git must be installed and available on your `PATH`.
- Using a virtual environment such as `venv` or `conda` is recommended.

### Install from the public Git repository

If the repository is public on GitHub, it can be installed directly with `pip` over HTTPS:

```bash
pip install "enhanced_granger_analysis @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"
```

This installs the package directly from the repository without publishing it to PyPI. Currently this repo has no PyPI support.

### Clone and install locally

If local development or source inspection is preferred:

```bash
git clone https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git
cd EnhancedGrangerCausalAnalysis
pip install .
```

For an editable install:

```bash
pip install -e .
```

### Optional backend dependencies

The project exposes three optional dependency sets:

- `torch`
- `tensorflow`
- `full`

These extras can be installed from a local clone:

```bash
# core only
pip install .

# PyTorch backend
pip install ".[torch]"

# TensorFlow backend
pip install ".[tensorflow]"

# all optional dependencies
pip install ".[full]"
```

They can also be installed directly from the public GitHub repository. Pip requires the package name to be provided when requesting extras from a VCS URL; extras cannot be requested with a bare Git URL alone.

```bash
# core + PyTorch backend
pip install "enhanced_granger_analysis[torch] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# core + TensorFlow backend
pip install "enhanced_granger_analysis[tensorflow] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# full environment
pip install "enhanced_granger_analysis[full] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"
```

### Requirements files

If the repository keeps backend-specific requirements files, they can still be used after cloning:

```bash
pip install -r requirements-torch.txt
pip install -r requirements-tensorflow.txt
pip install -r requirements-full.txt
```

Use this option when a fully explicit environment is preferred over extras.

## Development

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Project roadmap: [VISION.md](VISION.md)

Run tests:

```bash
pytest tests -v
```
