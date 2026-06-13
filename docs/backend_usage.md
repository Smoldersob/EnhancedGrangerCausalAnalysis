# Backend Usage

This document explains how to work with backends using `BackendFactory`, how to create components from dictionaries, and how to extend the system.

## 1. Get a Backend Strategy from Factory

Simple examples:

```python
from enhanced_granger_analysis import BackendFactory

# Explicit backend
strategy = BackendFactory.get_strategy("pytorch")

# Preferred available backend (priority: pytorch > tensorflow > sklearn)
strategy_auto = BackendFactory.get_strategy()

# Enable loading logs
strategy_verbose = BackendFactory.get_strategy("tensorflow", loading_verbose=True)
```

Useful aliases:

- TensorFlow: `tensorflow` / `tf` / `keras`
- PyTorch: `pytorch` / `torch`
- Scikit/Numpy: `sklearn` / `scikit` / `scikit-learn`

## 2. Create Components Using Strategy

### Regularizer from dict

```python
regularizer = strategy.build_regularizer({"type": "l1", "l1": 0.01})
```

### Constraint from dict

```python
constraint = strategy.build_constraint(
    {"type": "mask", "mask": [[1, 1, 0], [1, 0, 0]]}
)
```

### Callbacks from dict list

```python
callbacks = strategy.resolve_callbacks([
    {"type": "early_stopping", "patience": 5}
])
```

### Optimizer from spec

```python
optimizer = strategy.resolve_optimizer({"type": "adam", "learning_rate": 0.001})
```

Note: scikit backend does not support optimizer objects; configure `learning_rate`/`max_iter`/`tol` in `model_config`.

## 3. Validation with resolve_... and build_...

You can use `resolve_...` and `build_...` methods as a practical validation step for component compatibility.

- If a component spec is not supported by the selected backend, these methods raise an error early.
- This lets you validate configs before full model training.

Simple validation examples:

```python
# Validates regularizer spec for selected backend
strategy.build_regularizer({"type": "l1", "l1": 0.01})

# Validates callback specs for selected backend
strategy.resolve_callbacks([
    {"type": "early_stopping", "patience": 5}
])
```

Mixed callback format is also supported (dictionary specs + callback objects/classes in one list).

```python
from enhanced_granger_analysis.backends.callbacks.common_callbacks import EarlyStoppingCallback

mixed_callbacks = strategy.resolve_callbacks([
    {"type": "early_stopping", "patience": 4},
    EarlyStoppingCallback(patience=6),
])
```

## 4. Build Model Using Components

```python
model = strategy.build_model(
    n_features=20,
    n_outputs=3,
    regularizer=regularizer,
    constraint=constraint,
    callbacks=callbacks,
    optimizer=optimizer,
    epochs=20,
)
```

## 5. Build Constraint from Relation Mapping

Use backend-native conversion from relation rules:

```python
constraint = strategy.build_constraint_from_relations(
    relations={("y1", "x1"): 0, ("y1", "x2"): 1.0},
    predictor_names=["x1", "x2"],
    output_names=["y1"],
    col_offsets=[0, 3],
    n_features=6,
    base_mask=None,
)
```

## 6. How to Add New Components

## Callback

- Scikit/PyTorch shared callbacks: inherit backend callback base interface and implement hooks.
- TensorFlow callbacks: inherit `tf.keras.callbacks.Callback`.
- Register dictionary names in the corresponding object loader (`np_object_loader`, `torch_object_loader`, `tf_object_loader`).

## Constraint

- Provide callable behavior compatible with model training.
- For TensorFlow, implement a `tf.keras.constraints.Constraint`-compatible class.
- Register names in object loaders so dict specs can instantiate your class.

## Regularizer

- Scikit: callable object returning penalty/gradient-compatible output.
- PyTorch: `torch.nn.Module` or callable accepted by backend model.
- TensorFlow: `tf.keras.regularizers.Regularizer`.
- Register names in object loaders.

## 7. When You Need a New Backend (not only new components)

Create a new backend strategy when at least one of these is true:

- You use a different ML runtime/model lifecycle not compatible with current backends.
- Optimizer/callback/constraint semantics are fundamentally different.
- You need separate model build/train/predict logic and validation rules.

In practice, add:

- New strategy class implementing `BackendStrategy` contract.
- New backend model implementation.
- New object loader for dict-to-object resolution.
- Factory registration (with aliases) in `BackendFactory`.

## 8. Practical Recommendation

Start with new component classes first. Create a new backend only when component-level extension is no longer enough.
