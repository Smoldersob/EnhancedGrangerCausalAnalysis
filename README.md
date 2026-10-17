# EnhancedGrangerCausalAnalysis

EnhancedGrangerCausalAnalysis is a compact framework for advanced Granger-causality workflows on multivariate time series.

It extends classic pair-wise Granger tests with configurable multitask modeling, backend abstraction, constraint-aware training, and config-driven experiment sweeps.

Current package version: `2.3.1`

## Architecture at a glance

The library is organized into focused layers:

- `api/`: public APIs, a stateful analysis builder, orchestration, and
  configuration loading.
- `preprocessing/`: schema alignment, optional imputation, redundancy
  removal, stationarity transformation, lag construction, scaling,
  and sampling.
- `backends/`: backend strategies and model-specific components.
- `results/`: model snapshots, metric evaluation, and causality matrices.
- `scripts/`: configuration-driven experiment execution.

See [docs/project_structure.md](docs/project_structure.md) for the module map.

## Implemented capabilities

- Joint prediction of multiple configured effects.
- Separate selection of model inputs (`causes`), outputs (`effects`),
  and formally tested input variables (`tested_causes`).
- Explicitly executable analysis stages.
- Reuse of externally prepared data.
- Configurable lag construction and selection.
- Optional imputation and stationarity transformation.
- Removal and tracking of constant and exactly duplicated variables.
- Input and target scaling.
- Relation-based constraints and regularization.
- Reuse of one working model for base and restricted runs.
- Model snapshots and extensible metric evaluation.
- Coefficient-derived signs evaluated through the same metric interface
  as statistical and diagnostic metrics.
- Restoration of removed-variable rows and columns before combining
  significance decisions with coefficient-derived signs.
- Configuration-driven grouped experiments.

Hyperoptimization settings are accepted, but automatic hyperoptimization
is not currently executed by the standard builder pipeline.

## Public APIs

- `MultiTaskGrangerBuilder`: a stateful pipeline with fluent configuration
  and explicitly executable stages.
- `MultiTaskGrangerAPI`: a coordinator for the standard builder workflow.
- `SimpleGrangerAPI`: a pair-wise statsmodels baseline.

`MultitaskGrangerBuilder` remains an alias of `MultiTaskGrangerBuilder`.

Use the builder's explicit stages to inspect intermediate artifacts or pause
between preparation, model construction, base fitting, and restricted fitting.

Related documentation:

- [API usage](./docs/api_usage.md)
- [Data preprocessing](./docs/data_preprocessing.md)
- [Results and metrics](./docs/results_and_metrics.md)
- [Configuration files](./docs/config_file_usage.md)
- [Experiment scripts](./docs/script_usage.md)

## Backends

Supported backends:

- `pytorch` (`torch`)
- `tensorflow` (`tf`, `keras`)
- `sklearn` (`scikit`, `scikit-learn`)

When not specified, backend selection prefers: PyTorch -> TensorFlow -> scikit-learn.

Backend-specific component resolution (callbacks, regularizers, constraints) is documented in:
- [docs/backend_usage.md](docs/backend_usage.md)
- [docs/components_loading.md](docs/components_loading.md)

## Quick example

```python
import pandas as pd

from enhanced_granger_analysis.api import MultiTaskGrangerBuilder
from enhanced_granger_analysis.core.lag_config import LagConfiguration

df = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col=0)

results = (
    MultiTaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(
        causes=["u", "f1", "f2"],
        effects=["x1", "x2", "u1", "e1"],
        tested_causes=["u", "f1", "f2"],
    )
    .lag(LagConfiguration(max_lag=12, use_lag_zero=False))
    .model({
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 1e-3,
    })
    .fit()
)

# result() returns a DataFrame directly.
causality = results.result(threshold=0.01, with_sign=True)
print(causality)

# evaluate() returns a CausalityMatrix.
f_statistics = results.evaluate("f_statistic")
print(f_statistics.data)
```

`result(...)` returns a DataFrame directly, while `evaluate(...)` returns a
`CausalityMatrix`.

When `with_sign=True`, both the selected metric and the sign metric are
restored to the requested effect and tested-cause axes before multiplication.

```python
signs = results.evaluate("sign").data
signed_causality = results.result(threshold=0.01, with_sign=True)
```

The sign summarizes the sum of base-model coefficients in each cause's
lagged feature block. It does not describe the direction of the causal arrow.

Effects outside `causes` are allowed. Their lagged values are not automatically
included in the input matrix, and preprocessing reports this configuration
with a warning.

For execution stage by stage, see [API usage](./docs/api_usage.md).

## Config-Driven Test Runs

Run grouped experiments from JSON/YAML with per-case matrices and summary metrics:

```bash
python enhanced_granger_analysis/scripts/run_group_causality_tests.py --config enhanced_granger_analysis/scripts/run_group_causality_tests.config.json
```

This flow supports loading a list of input DataFrames, running sweep cases, measuring execution time, comparing against ground truth, and saving `summary.csv`.

Details:
- [docs/config_file_usage.md](docs/config_file_usage.md) — Group config structure and sweepable fields
- [docs/script_usage.md](docs/script_usage.md) — Script runner options and output layout

## Installation

### Prerequisites

- Package metadata requires Python 3.10 or newer. Check the Python and
  platform requirements of the pinned dependencies and selected backend
  before creating the environment.
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

The project exposes three optional dependency sets:

- `torch`
- `tensorflow`
- `full`

This set of dependencies can be downloaded using bellow commands:

```bash
# core + PyTorch backend
pip install "enhanced_granger_analysis[torch] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# core + TensorFlow backend
pip install "enhanced_granger_analysis[tensorflow] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# full environment
pip install "enhanced_granger_analysis[full] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"
```

These commands install the current branch state. For reproducible experiments,
replace the branch reference with a recorded commit SHA.

### Clone and install locally

```bash
git clone https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git
cd EnhancedGrangerCausalAnalysis
pip install -e .
```

Optional local installations:

```bash
pip install -e ".[torch]"
pip install -e ".[tensorflow]"
pip install -e ".[full]"
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
