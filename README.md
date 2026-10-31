# EnhancedGrangerCausalAnalysis

EnhancedGrangerCausalAnalysis provides configurable, multi-output Granger-causality analysis for multivariate time series. It separates data preparation, model construction, fitting, and evaluation of completed model snapshots.

Current package version: `2.8.2`.

## Main capabilities

- Joint prediction of configured effects, with independent input and tested-cause roles.
- Configurable lag construction and selector-based preprocessing.
- Optional imputation, stationarity transformation, scaling, and sampling.
- Tracking of variables removed during preprocessing.
- Layout-aware constraints, optional regularization, and weight initialization.
- Standard unrestricted/restricted fitting with explicit weight, omission, optimizer, and callback lifecycle management.
- Config-driven experiment cases and result export.
- Extensible evaluation of saved model snapshots.

Automatic hyperparameter optimization is not executed by the standard workflow. Existing hyperoptimization fields are retained but inactive.

## Entry points

| Interface | Purpose |
| --- | --- |
| `MultiTaskGrangerBuilder` | Fluent configuration, complete execution, and explicit execution stages |
| `MultiTaskGrangerAPI` | One-call interface for the standard workflow |
| `AnalysisSpec` + `AnalysisRunner` | Execute a configuration without constructing a builder |
| `SimpleGrangerAPI` | Separate pair-wise statsmodels baseline |

The builder and orchestrator both use `AnalysisRunner`; the orchestrator no longer creates a builder. `MultitaskGrangerBuilder` remains a compatibility alias.

## Quick start

```python
import pandas as pd

from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder

frame = pd.read_csv("data.csv", index_col=0)

results = (
    MultiTaskGrangerBuilder(backend="numpy")
    .data(frame)
    .variables(causes=["x", "y"], effects=["y"], tested_causes=["x"])
    .stationarity(None)
    .lag(lag_config={"max_lag": 2, "use_lag_zero": False})
    .model({"model": "conditional_linear", "epochs": 100})
    .fit()
)

print(results.result(threshold=0.01))
print(results.evaluate("f_statistic").data)
```

The example assumes numeric time-ordered columns `x` and `y`. Disabling stationarity transformation is an explicit modeling choice, not a general recommendation.

`result()` returns a DataFrame. `evaluate()` returns a `CausalityMatrix`, whose DataFrame is available through `.data`. Statistical interpretation depends on the assumptions of the selected metric and model; constructing a statistic does not establish its calibration for every constrained or regularized fit.

## Configuration-driven execution

```python
from enhanced_granger_analysis.configuration import AnalysisSpec
from enhanced_granger_analysis.configuration.loader import AnalysisConfigLoader
from enhanced_granger_analysis.execution import AnalysisRunner

config = AnalysisConfigLoader.load_file("analysis.yaml")
spec = AnalysisSpec.from_mapping(config)
results = AnalysisRunner().run_analysis(spec=spec, data=frame)
```

The loader reads documents and relation-file references. It does not resolve components. The external configuration remains a flat mapping; `AnalysisSpec` groups it internally.

## Architecture

- `api/`: builder, orchestrator, baseline, and compatibility imports.
- `configuration/`: analysis/modeling specifications and document loading.
- `preprocessing/`: existing staged preparation and layout metadata.
- `modeling/`: assembly, initialization, single-fit execution, and `FitOutcome`.
- `execution/`: standard fitting coordination and conversion to result snapshots.
- `backends/`: strategies, existing resolvers/registries, models, and device handling.
- `experiments/`: explicit configuration-case expansion.
- `results/`: snapshots, metric evaluation, restoration, and causality matrices.

`TrainingResult` contains training diagnostics. `FitOutcome` captures one fit in model output scale. `GrangerAnalysisResults` stores analysis snapshots after target inverse scaling and evaluates their relationships. These are different contracts.

## Backends and devices

| Canonical backend | Selection aliases |
| --- | --- |
| `pytorch` | `torch` |
| `keras` | `tensorflow`, `tf` |
| `numpy` | `sklearn`, `scikit`, `scikit-learn` |

Automatic selection prefers PyTorch, Keras, then NumPy, subject to availability. Explicit selection is preferable when recording an experiment configuration.

PyTorch and Keras interpret `device` using their existing device managers. NumPy ignores `device` during model construction so that experiment cases can share a device declaration. Legacy scikit names select the NumPy implementation; they do not restore the removed scikit package.

## Grouped experiments

```bash
python enhanced_granger_analysis/scripts/run_group_causality_tests.py --config enhanced_granger_analysis/scripts/run_group_causality_tests.config.json --save matrices
```

The iterator expands explicitly listed cases, not an automatic Cartesian product. Experiment sweeps are not HPO. The script can compare completed analysis matrices with ground truth; those scores are not a model-selection objective.

## Installation

Python 3.10 or newer is required by package metadata. Check the requirements of the selected runtime and dependencies separately.

```bash
git clone https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git
cd EnhancedGrangerCausalAnalysis
pip install -e .
```

Optional dependency sets:

```bash
pip install -e ".[torch]"
pip install -e ".[tensorflow]"
pip install -e ".[full]"
```

Direct Git installation:

```bash
pip install "enhanced_granger_analysis @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"
```

Select the intended branch or commit explicitly when reproducing a development configuration. Installing the default branch does not guarantee installation of the API described for another branch. No PyPI publication is implied.

Backend-specific requirements files may also be installed from the repository. Do not interpret the library's Keras selection alias as a command that changes the active Keras runtime; configure that runtime before importing Keras.

If you want to install it with chosen dependencies use one of bellow commands:

```bash
# core + PyTorch backend
pip install "enhanced_granger_analysis[torch] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# core + TensorFlow backend
pip install "enhanced_granger_analysis[tensorflow] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"

# full environment
pip install "enhanced_granger_analysis[full] @ git+https://github.com/Smoldersob/EnhancedGrangerCausalAnalysis.git"
```

## Documentation

- [API usage](docs/api_usage.md)
- [Configuration files](docs/config_file_usage.md)
- [Execution architecture](docs/execution_architecture.md)
- [Migration guide](docs/api_refactor_migration.md)
- [Project structure](docs/project_structure.md)
- [Backend usage](docs/backend_usage.md)
- [Component loading](docs/components_loading.md)
- [Data preprocessing](docs/data_preprocessing.md)
- [Lag selection](docs/lag_selection.md)
- [Results and metrics](docs/results_and_metrics.md)
- [Restricted-run lifecycle](docs/reference_loop.md)
- [Experiment scripts](docs/script_usage.md)
- [HPO status](docs/hipero_optimization_documentation.md)

## Development and validation

- [Contributing](CONTRIBUTING.md)
- [Roadmap](VISION.md)
- [Changelog](Changelog.md)

This documentation describes the adopted refactor. Its examples are not a report of executed tests or proof of cross-backend numerical equivalence.
