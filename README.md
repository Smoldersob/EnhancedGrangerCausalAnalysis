# Complex Granger Analysis

Complex Granger Analysis is a modular library for Granger-causality workflows on
multivariate time series. The current implementation focuses on a unified API,
backend flexibility, reproducible preprocessing, and result containers that are
easy to post-process in research and engineering pipelines.

Current package version: `2.0.0`

## What The Library Provides Today

- Multitask Granger analysis through `MultitaskGrangerBuilder` and `MultiTaskGrangerAPI`.
- Pair-wise statsmodels analysis through `SimpleGrangerAPI`.
- Backend abstraction for `pytorch`, `tensorflow`, and `scikit-learn`.
- Automatic lag selection with information criteria and configurable lag masks.
- Fine-grained lag overrides per predictor and per `(target, predictor)` pair.
- Stationarity preprocessing (ADF/KPSS differencing) on one or many datasets.
- Scalers for X/y data (`standard`, `minmax`, `robust`, `maxabs`, `identity`).
- Constraint creation from relation dictionaries and backend-specific enforcement.
- Regularization support including `l1` and lag-dependent L1 variants.
- Callback support with run-level cloning for base/reference/hyperopt loops.
- Structured outputs with p-values, F-test statistics, errors, and sign matrices.
- Config-driven experiment sweeps via JSON/YAML and script automation.

## Motivation

Granger causality is a statistical hypothesis test used to determine whether one time series helps predict another. This library was built to support scalable, reproducible Granger causality workflows across multiple backends:

- **Multi-framework flexibility**: TensorFlow, PyTorch, and scikit-learn backends allow users to choose the right tool for their data and constraints.
- **Automated preprocessing**: Stationarity testing, lag selection, and data scaling are integrated into a configurable pipeline.
- **Fine-grained control**: Override lag orders per pair, apply backend-specific constraints and regularization, and track reference models for each tested cause.
- **Research-friendly outputs**: Structured result containers with p-values, F-test statistics, and weight snapshots support post-processing and visualization.

For implementation details on orchestration strategies and model reuse patterns,
see [orchestrator reference loop design](memories/repo/orchestrator_reference_loop.md).

## Current Architecture

```
complex_granger_analysis/
├── api/                 # builder, orchestrator, simple API, config loader
├── backends/            # backend factory + TF/Torch/Sklearn strategies
├── components/          # models, regularizers, constraints, initializers
├── core/                # configs, protocols, exceptions, output dataclasses
├── preprocessing/       # lag engine/selectors, stationarity, scaling
├── results/             # causality matrices and statistics
├── scripts/             # config-driven group test runner
└── tests/               # unit tests for major modules
```

**For deep dives into architectural patterns and design decisions, see architecture notes in [memories/repo/](memories/repo/):**
- [Reference loop orchestration pattern](memories/repo/orchestrator_reference_loop.md) – how models are reused across cause iterations
- [Lag layout in regularizers](memories/repo/regularizers_lag_layout.md) – lag-dependent L1 regularizer structure and weight mapping

## Installation

Install dependencies from one of the requirement files.

```bash
git clone https://github.com/Smoldersob/complex_granger_analysis.git
cd complex_granger_analysis
pip install -r requirements.txt
```

Backend-specific options:

```bash
pip install -r requirements-torch.txt
# or
pip install -r requirements-tensorflow.txt
```

Notes:
- PyTorch and TensorFlow are optional but recommended for neural backends.
- `SimpleGrangerAPI` requires `statsmodels`.

## Quick Start (Builder API)

```python
import pandas as pd

from complex_granger_analysis.api import MultitaskGrangerBuilder
from complex_granger_analysis.core.lag_config import LagConfiguration

df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)

output = (
    MultitaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(
        causes=["u", "f1", "f2"],
        effects=["x1", "x2", "x3", "x4"],
        tested_causes=["u", "f1", "f2"],
    )
    .lag(lag_config=LagConfiguration(max_lag=12, use_lag_zero=False))
    .scaling(x_scaler="standard", y_scaler="standard")
    .model(model_config={"epochs": 100, "batch_size": 32, "learning_rate": 1e-3})
    .fit()
)

causality = output.results.result(threshold=0.01, with_sign=False)
print(causality)
```

## Quick Start (Simple Pair-wise API)

```python
import pandas as pd

from complex_granger_analysis.api import SimpleGrangerAPI

df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)

simple_output = SimpleGrangerAPI().fit(
    data=df,
    causes=["u", "f1", "f2"],
    effects=["x1", "x2", "x3", "x4"],
    lag_max=20,
    threshold=0.01,
)

print(simple_output.causality_matrix.data)
print(simple_output.p_value)
print(simple_output.sign)
```

## Config-Driven Runs

Load builder configuration from JSON/YAML:

```python
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder

config = BuilderConfigLoader.load_file("path/to/config.json")
output = MultitaskGrangerBuilder().from_config(config).data(df).fit()
```

Run sweep experiments from script config:

```bash
python scripts/run_group_causality_tests.py \
  --config scripts/run_group_causality_tests.config.json
```

This script reads:
- dataset paths
- group config with sweep cases
- ground truth matrix

and writes per-case outputs plus a summary table with metrics.

**Note on model orchestration:** The library reuses a single reference model instance across cause iterations with re-initialization between runs. This pattern is documented in [orchestrator reference loop design](memories/repo/orchestrator_reference_loop.md).

## Supported Backends

- `pytorch` / `torch`
- `tensorflow` / `tf` / `keras`
- `sklearn` / `scikit` / `scikit-learn`

When backend is not specified, the preferred order is:
1. PyTorch
2. TensorFlow
3. scikit-learn

## TensorFlow GPU/CPU Management

The library automatically manages TensorFlow device placement with WSL-aware defaults:

**Default behavior:**
- On **WSL**: Uses CPU-only mode for stability (CUDA/cuDNN can be unstable in WSL)
- On **native Linux/Mac with GPU**: Auto-detects and enables GPU with dynamic memory growth
- On **systems without GPU**: Automatically falls back to CPU

**Explicit control via environment variables:**

```bash
# Force CPU-only mode (most stable option)
export CGA_TF_FORCE_CPU=1
python your_script.py

# Enable GPU on WSL (if your CUDA setup is stable)
export CGA_TF_USE_GPU=1
python your_script.py
```

**Troubleshooting GPU issues:**

If you encounter `CUDNN_STATUS_NOT_INITIALIZED` errors, run with CPU mode:
```bash
CGA_TF_FORCE_CPU=1 python your_script.py
```

For detailed GPU policy, troubleshooting matrix, device mode states, and testing guidance,
see [GPU/CPU management notes in memories](memories/repo/tensorflow_gpu_policy.md).

## Regularization and Lag-Dependent Weights

The library supports lag-dependent L1 regularization through `LagDependentL1` regularizers (available for TensorFlow, PyTorch, and NumPy backends). These regularizers apply different penalty weights to different lag positions within each predictor.

**For details on lag layout, weight mapping, and the `set_lag_layout()` API:**
see [lag layout in regularizers documentation](memories/repo/regularizers_lag_layout.md).

## Result Objects

`MultitaskGrangerOutput` exposes:
- `results`: `GrangerAnalysisResults`
- `base_model`
- `reference_models`
- `stationarity_transformer`
- `lag_engine`
- `X_scaler`, `y_scaler`

`GrangerAnalysisResults` exposes:
- `result(threshold=0.01, with_sign=False)`
- `p_value`, `F_test`, `base_error`, `ref_error`, `sign`
- snapshots of base/reference predictions and weights

## Testing

Run the test suite from repository root:

```bash
pytest tests -v
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development and pull request guidelines.

## Project Direction

See [VISION.md](VISION.md) for roadmap details and planned enhancements.
