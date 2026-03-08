# Complex Granger Analysis Library

A data‑driven toolset for automatically generating fault‑symptom relations from
process signals using Granger causality. The project originated in a BSc/MA thesis
focused on industrial fault diagnosis and offers a flexible Python implementation
with multiple backends (scikit‑learn, TensorFlow, PyTorch, Statsmodels).

> **Motivation.** Fault diagnosis in complex industrial plants is often based on
> expert knowledge and manual signal inspection. This library automates the
> extraction of causal relations between faults and symptoms, enabling rapid
> construction of diagnostic matrices and the incorporation of expert constraints.
> The core algorithm selects time lags automatically, detects direct effects,
> suppresses spurious links and reuses intermediate results to cut analysis time.
> It also returns the **sign** of a dependency (signal increase/decrease due to a
> fault) and allows users to inject prior knowledge via causal constraints.

## Key Features

- **Automatic lag selection** using VAR information criteria and extendable with
  cross‑validation strategies.
- **Granger‑based causality inference** with support for multivariate, sparse,
  multitask and physics‑informed variants.
- **Sign information** (positive/negative effect) alongside p‑values and F‑tests.
- **Expert knowledge integration**: force or forbid specific causal links during
  model fitting.
- **Multiple framework backends** (sklearn, TF, PyTorch, Statsmodels) under a
  unified interface.
- **Stationarity preprocessing** (ADF/KPSS differencing) and user‑specified
  non‑stationary variables.
- **Sparsity via L1 regularization** to reduce false positives.
- **Parallel computation** with joblib for faster analysis on multicore systems.
- **Modular architecture** – models, regularizers, lag engine, callbacks and
  utilities are decoupled for easy extension.

## Installation

### From source

```bash
git clone https://github.com/Smoldersob/complex_granger_analysis.git
cd complex_granger_analysis
pip install -e .
```

> TensorFlow and PyTorch are optional; install them separately if you plan to
> use the corresponding backends:

```bash
pip install tensorflow    # or torch
```

### From PyPI (future)

This package is expected to land on PyPI; install with `pip install
complex_granger_analysis` once available.

## Quick Start

```python
import complex_granger_analysis as cga
import pandas as pd

# load sample process data
data = pd.read_csv('example/PID_no_fault.csv', sep=';', index_col=0)

# choose a model (TF backend shown here)
gc = cga.granger_tests.tensorflow_granger.TFNeuralSparseConstrainedMVGC()
# fit with explicit causes/effects or let the library infer them
gc.fit(data, causes=['u','f1','f2'], effects=['x1','x2','x3','x4'])

# retrieve binary causality matrix (threshold 0.01 by default)
print(gc.results.result())
```

## Library Layout

```
complex_granger_analysis/
├── __init__.py
├── lag_engine.py           # lag selection & lagged-data builders
├── utilits.py              # stationarity tests and transformations
├── granger_analysis_results.py
├── granger_tests/           # core models (complex, tensorflow, pytorch,...)
├── models/                  # custom layers and linear models
├── regularizers/            # framework‑specific regularizers
└── callbacks/               # training/event callbacks
```

## Usage Examples

See the `example/` directory for Jupyter notebooks and CSV files.  Typical
workflows include:

- computing causality matrices for fault‑symptom analysis,
- adding expert constraints to force/forbid relations,
- evaluating models on synthetic and real datasets (TEP, LiU‑ICE).

## Theoretical Background

The algorithm builds on Granger’s definition of causality: a time series X
`Granger-causes` Y if past values of X improve predictions of Y beyond what past
values of Y alone can achieve.  Extensions implemented in the code include:

1. **Multivariate and multitask tests** – single matrix regression handling all
   variables simultaneously.
2. **Sparse Granger** – ℓ₁ regularization to promote sparsity and reduce
   false positives.
3. **Physics‑informed constraints** – embed expert knowledge by forcing sums of
   coefficients to zero or above a threshold.
4. **Three‑value outputs** – infer effect direction (+1/0/−1) from coefficient
   signs.

For more details, consult the original MA thesis and the inline documentation.

## Dependencies

- Python ≥ 3.8
- numpy, pandas, scikit-learn, statsmodels
- Optional: tensorflow, torch

## Testing (to be added)

```bash
pytest complex_granger_analysis/granger_tests -v
```

## Contributing

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for
coding standards and pull‑request guidelines.  Please file issues for bugs or
feature requests.

## License

MIT License – see [LICENSE](LICENSE).

## References & Further Reading

The approach and implementations are described in Janek’s thesis *"Automatic
generation of faults–symptoms relations based on process data"*. Key topics
include Granger causality, fault diagnosis, and physics-informed sparse
inference.

For background and datasets, consult the Tennessee Eastman Process (TEP) and
LiU‑ICE benchmark publications.

---

See also [VISION.md](./VISION.md) for the project roadmap and future plans.
