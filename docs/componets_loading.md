# Components Info

This document lists supported component types and dictionary names used to build components in each backend.

## Generic Spec Format
You can pass either:

- A concrete component object.
- A string type name.
- A dictionary with fields:
  - `type` (preferred) or `name` or `kind`
  - `params` (optional object)
  - Extra keys outside `params` are also accepted and merged into params.

Example:

```python
{"type": "early_stopping", "patience": 5}
{"name": "l1", "params": {"l1": 0.01}}
```

## Scikit/Numpy Backend
Backend names: `sklearn` / `scikit` / `scikit-learn`

### Callbacks
Supported dictionary types:

- `early_stopping` / `earlystopping`
- `reduce_lr` / `reduce_learning_rate` / `reducelearningrate`
- `convergence_check` / `convergencecheck`

Simple example:

```python
callbacks = [
    {"type": "early_stopping", "patience": 10},
    {"type": "reduce_lr", "factor": 0.5},
]
```

### Regularizers
Supported dictionary types:

- `l1` / `numpy_l1`
- `lag_dependent_l1` / `lagdependentl1` / `numpy_lag_dependent_l1`

Simple example:

```python
reg = {"type": "l1", "l1": 0.01}
```

Lag-dependent example:

```python
reg = {
    "type": "lag_dependent_l1",
    "l1": 0.01,
    "lag_weights": [1.0, 2.0, 3.0],
    "max_lags_per_pred": [2, 2],
    "col_offsets": [0, 3],
}
```

### Constraints
Supported dictionary types:

- `mask` / `mask_constraint` / `numpy_mask`
- `mask_and_min_abs_sum` / `mask_min_abs_sum` / `numpy_mask_and_min_abs_sum`

Simple mask example:

```python
constraint = {
    "type": "mask",
    "mask": [[1, 1, 0], [1, 0, 0]],
}
```

Mask + rules example:

```python
constraint = {
    "type": "mask_and_min_abs_sum",
    "mask": [[1, 1, 1]],
    "rules": [
        {"output_index": 0, "feature_indices": [1, 2], "min_abs_sum": 1.0}
    ],
    "eps": 1e-8,
}
```

## PyTorch Backend
Backend names: `pytorch` / `torch`

### Callbacks
Supported dictionary types:

- `early_stopping` / `earlystopping`
- `reduce_lr` / `reduce_learning_rate` / `reducelearningrate`
- `convergence_check` / `convergencecheck`
- `torch_tensorboard` / `tensorboard` / `tensorboard_logger`

Simple example:

```python
callbacks = [{"type": "torch_tensorboard", "log_dir": "runs/demo"}]
```

### Regularizers
Supported dictionary types:

- `l1` / `torch_l1` / `pytorch_l1`
- `lag_dependent_l1` / `lagdependentl1` / `torch_lag_dependent_l1` / `pytorch_lag_dependent_l1`

Simple example:

```python
reg = {"type": "pytorch_l1", "l1": 0.005}
```

### Constraints
Supported dictionary types:

- `mask` / `mask_constraint` / `torch_mask` / `pytorch_mask`
- `mask_and_min_abs_sum` / `mask_min_abs_sum` / `torch_mask_and_min_abs_sum` / `pytorch_mask_and_min_abs_sum`

Simple example:

```python
constraint = {
    "type": "pytorch_mask",
    "mask": [[1, 0, 1]],
}
```

## TensorFlow Backend
Backend names: `tensorflow` / `tf` / `keras`

### Callbacks (Keras callbacks)
Supported dictionary types:

- `early_stopping` / `keras_early_stopping` / `earlystopping`
- `reduce_lr_on_plateau` / `reduce_learning_rate` / `reduce_lr` / `reducelronplateau` / `keras_reduce_lr`
- `tensorboard` / `keras_tensorboard`
- `model_checkpoint` / `checkpoint` / `keras_checkpoint`
- `csv_logger` / `csvlogger` / `keras_csv_logger`
- `terminate_on_nan` / `keras_terminate_on_nan`

Simple example:

```python
callbacks = [
    {"type": "early_stopping", "monitor": "loss", "patience": 3},
    {"type": "tensorboard", "log_dir": "logs/demo"},
]
```

### Regularizers
Supported dictionary types:

- `l1` / `keras_l1` / `tensorflow_l1`
- `lag_dependent_l1` / `lagdependentl1` / `keras_lag_dependent_l1` / `tensorflow_lag_dependent_l1`

Simple example:

```python
reg = {"type": "keras_l1", "l1": 0.01}
```

### Constraints
Supported dictionary types:

- `mask` / `mask_constraint` / `keras_mask` / `tensorflow_mask`
- `mask_and_min_abs_sum` / `mask_min_abs_sum` / `keras_mask_and_min_abs_sum` / `tensorflow_mask_and_min_abs_sum`

Simple example:

```python
constraint = {
    "type": "tensorflow_mask",
    "mask": [[1, 1, 0, 0]],
}
```

## Verbose Loading
All strategies support internal loader logging via `loading_verbose=True`.

Simple example:

```python
strategy = BackendFactory.get_strategy("pytorch", loading_verbose=True)
```
