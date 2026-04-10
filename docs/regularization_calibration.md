# Regularization Calibration Across Backends

This note compares the **effect strength** of L1 regularization across:
- NumPy regularizers with Scikit model
- PyTorch regularizers with PyTorch model
- TensorFlow regularizers with TensorFlow model

The goal is practical calibration, not implementation details.

## Method

Common synthetic setup used for effect comparison:
- random seed: 7
- samples: 220
- features: 6
- target: linear combination + Gaussian noise
- metric: weight shrink ratio

Shrink ratio definition:

$$
\text{shrink ratio} = \frac{\|W_{\text{with L1}}\|_1}{\|W_{\text{no reg}}\|_1}
$$

Lower ratio means stronger regularization effect.

## Calibration Table (L1 = 0.02)

| Backend pair | No-reg \|W\|1 | L1 \|W\|1 | Shrink ratio | Effect strength |
|---|---:|---:|---:|---|
| NumPy -> Scikit | 3.5930 | 3.5867 | 0.9982 | very weak |
| PyTorch -> PyTorch | 3.5240 | 3.4664 | 0.9837 | weak/moderate |
| TensorFlow -> TensorFlow | 3.6265 | 2.9193 | 0.8050 | strong |

## Lag-dependent L1 (qualitative consistency)

Observed with lag-dependent L1:
- all backends penalize weighted lag blocks in the expected direction,
- but magnitude differs significantly by backend/training dynamics.

Example block mean abs values (first block vs second block):
- Scikit: 0.9975 vs 0.1960
- PyTorch: 0.9681 vs 0.1743
- TensorFlow: 0.6420 vs 0.2963

## Practical guidance

- Do not reuse the same numeric `l1` across backends expecting equal effect.
- Start from backend-specific defaults and tune by target shrink ratio.
- For rough equivalence to PyTorch `l1=0.02`:
  - Scikit usually needs a larger `l1` than PyTorch,
  - TensorFlow often needs a smaller `l1` than PyTorch.

## Notes

- Results depend on optimizer, epochs, batch size, and data scale.
- Treat the table as a calibration baseline for this repository/runtime, not a universal constant.
