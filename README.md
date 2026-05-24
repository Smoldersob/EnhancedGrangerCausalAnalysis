# Complex Granger Analysis

Complex Granger Analysis is a compact framework for advanced Granger-causality workflows on multivariate time series.

It extends classic pair-wise Granger tests with configurable multitask modeling, backend abstraction, constraint-aware training, and config-driven experiment sweeps.

Current package version: `2.1.0`

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
- [docs/componets_loading.md](docs/componets_loading.md)

## Quick Example (Orchestrator)

```python
import pandas as pd

from complex_granger_analysis.api import MultiTaskGrangerAPI
from complex_granger_analysis.core.lag_config import LagConfiguration

df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)

api = MultiTaskGrangerAPI(backend="pytorch", reuse_data=True)

output = api.fit(
    data=df,
    causes=["u", "f1", "f2"],
    effects=["x1", "x2", "x3", "x4"],
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

### Using pip (for private repository)

Since this repository is currently private, you need to install it via SSH. Ensure you have SSH keys configured for GitHub, then:

```bash
pip install git+ssh://git@github.com/Smoldersob/complex_granger_analysis.git
```

Or clone first and install from your local copy:

```bash
git clone git@github.com:Smoldersob/complex_granger_analysis.git
cd complex_granger_analysis
pip install .
```

### Installing with specific backends

To install with only core dependencies:

```bash
pip install .
```

To install with PyTorch backend:

```bash
pip install -e ".[torch]"
# or install requirements manually:
pip install -r requirements-torch.txt
```

To install with TensorFlow backend:

```bash
pip install -e ".[tensorflow]"
# or install requirements manually:
pip install -r requirements-tensorflow.txt
```

To install repo with backend dependencies via pip you need to use:

```bash
pip install "complex_granger_analysis[full] @ git+ssh://git@github.com/Smoldersob/complex_granger_analysis.git"
```
There are three optional dependencies version torch,tensorflow and full.

### Installing all dependencies (full environment)

To install all dependencies for both PyTorch and TensorFlow backends in one go:

```bash
pip install -e ".[full]"
# or install requirements manually:
pip install -r requirements-full.txt
```

The `requirements-full.txt` file contains all packages from core, PyTorch, and TensorFlow dependencies, suitable for development or testing multiple backends.

## Development

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Project roadmap: [VISION.md](VISION.md)

Run tests:

```bash
pytest tests -v
```
