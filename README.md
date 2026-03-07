# Complex Granger Analysis Library

A comprehensive library for Granger causality analysis, developed as part of BSc and MA theses. It provides various implementations of Granger analysis algorithms with custom modifications, supporting frameworks like TensorFlow, PyTorch, Scikit-learn, and Statsmodels. Designed for easy customization and extension, it enables researchers and practitioners to analyze causal relationships in multivariate time series data.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Features
- **Multiple Framework Support**: Implementations using TensorFlow, PyTorch, Scikit-learn, and Statsmodels for flexibility across environments.
- **Customizable Models**: Includes constrained linear regression, sparse models, and neural network-based approaches (e.g., MaskedDenseLayer, MultiTaskConstrainedLinearRegression).
- **Advanced Regularizers**: Built-in regularizers for Keras and PyTorch to enhance model performance.
- **Lag Engine**: Tools for handling time lags in causality detection.
- **Callbacks and Utilities**: Extensible callbacks and utility functions for preprocessing and analysis.
- **Easy Integration**: Automatic inclusion/exclusion of framework-specific components based on available libraries.

## Installation

### From PyPI (To be added)
```bash
```

### From Source
Clone the repository and install in editable mode:
```bash
git clone https://github.com/Smoldersob/complex_granger_analysis.git
cd complex_granger_analysis
pip install -e .
```

**Note**: TensorFlow and PyTorch are optional dependencies. Install them separately if needed:
```bash
pip install tensorflow  # or torch
```

## Quick Start
Import the library and perform a basic Granger causality analysis:

```python
import complex_granger_analysis as cga
import pandas as pd

# Load sample data
data = pd.read_csv("example/PID_no_fault.csv", sep=";", index_col='Unnamed: 0')

# Initialize a TensorFlow-based model
gc_model = cga.granger_tests.tensorflow_granger.TFNeuralSparseConstrainedMVGC()
gc_model.fit(data, causes=['u', 'f1', 'f2'], effects=['x1', 'x2', 'x3', 'x4'])

# Get results
causal_matrix = gc_model.results.result()
print(causal_matrix)
```

## Usage Examples
See the `example/` directory for sample data and notebooks, including `PID_reg.ipynb`.

### Example with PyTorch
```python
from complex_granger_analysis.granger_tests import pytorch_granger
import torch

# Assuming data is a PyTorch tensor
model = pytorch_granger.PytorchSparseLinearModel()
# ... fit and analyze
```

For more examples, visit the [documentation](https://github.com/Smoldersob/complex_granger_analysis/wiki) or check the `example/` folder.

## API Reference
- **granger_tests**: Core test implementations (`complex_granger.py`, `tensorflow_granger.py`, etc.).
- **models**: Custom model classes (e.g., `MaskedDenseLayer.py`, `PytorchSparseLinearModel.py`).
- **regularizers**: Regularization utilities for different frameworks.
- **lag_engine**: Tools for lag selection and processing.
- **callbacks**: Event handlers for training and analysis.

Full API docs: [Link to Docs](https://github.com/Smoldersob/complex_granger_analysis/wiki/API)

## Dependencies
- Python >= 3.8
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
- Optional: TensorFlow, PyTorch

See `requirements.txt` for full details.

## Contributing
We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/Smoldersob/complex_granger_analysis/blob/main/CONTRIBUTING.md) for guidelines. Report issues or suggest features via [GitHub Issues](https://github.com/Smoldersob/complex_granger_analysis/issues).

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Smoldersob/complex_granger_analysis/blob/main/LICENSE) file for details.

## Authors
- Developed by Janek as part of BSc and MA theses.
- Repository: [Smoldersob/complex_granger_analysis](https://github.com/Smoldersob/complex_granger_analysis)
