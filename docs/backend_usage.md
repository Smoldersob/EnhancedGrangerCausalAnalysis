# Backend Usage

This document explains how to work with backends using `BackendFactory`, how to create components from dictionaries, and how to extend the system.

## 1. Get a Backend Strategy from Factory

Simple examples:

```python
from enhanced_granger_analysis import BackendFactory

# Explicit backend
strategy = BackendFactory.get_strategy("pytorch")

# Preferred available backend (priority: pytorch > keras/tensorflow > sklearn)
strategy_auto = BackendFactory.get_strategy()

# Enable loading logs
strategy_verbose = BackendFactory.get_strategy("tensorflow", loading_verbose=True)
```

Useful aliases:

- TensorFlow: `tensorflow` / `tf` / `keras`
- PyTorch: `pytorch` / `torch`
- Scikit: `sklearn` / `scikit` / `scikit-learn`

## 2. Create Components Using Strategy

### Regularizer from a selector and prepared data

Backend strategies receive a selector separately from regularizer data.
Do not pass a raw parameter dictionary directly to
`build_regularizer(...)`.

```python
import numpy as np

from enhanced_granger_analysis.core.components.regularizer_parsing import (
    parse_regularizer_spec,
)
from enhanced_granger_analysis.core.components.regularizer_preparation import (
    prepare_regularization_layout,
    prepare_regularizer_data,
)

layout = prepare_regularization_layout(
    cause_slices={
        "x1": slice(0, 2),
        "x2": slice(2, 4),
    },
    output_names=["y"],
    feature_lags=np.array(, dtype=np.int64),[1][2]
    base_mask=np.ones((1, 4), dtype=bool),
)

spec = parse_regularizer_spec({
    "type": "l1",
    "params": {"l1": 0.01},
})

data = prepare_regularizer_data(layout=layout, spec=spec)

regularizer = strategy.build_regularizer(
    spec.selector,
    data=data,
)
```

`strategy.build_regularizer(None)` disables regularization.
Non-disabled built-in regularizers require prepared data.

### Constraint from a specification and prepared data

```python
from enhanced_granger_analysis.core.components.constraint_parsing import (
    parse_relations_spec,
)
from enhanced_granger_analysis.core.components.constraint_preparation import (
    prepare_constraint_data,
)

relations = parse_relations_spec([
    {
        "cause": "x1",
        "effect": "y",
        "rule": {"type": "zero"},
    },
])

data = prepare_constraint_data(
    relations,
    cause_slices=layout.cause_slices,
    output_names=layout.output_names,
    n_features=layout.n_features,
    base_mask=layout.base_mask,
)

constraint = strategy.build_constraint(
    "mask_and_min_abs_sum",
    data=data,
)
```

These examples describe the version 2.6 construction interface.
Current backend import and construction defects must be resolved before
using them as end-to-end execution examples.

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

Constraint and regularizer validation is staged:

1. Parse the user specification.
2. Validate and prepare layout-dependent data.
3. Resolve the selected backend class.
4. Construct the component and validate implementation-specific parameters.

Registry resolution alone does not construct the component or validate
every constructor parameter. Successful component construction does not
establish that training or serialization works for the complete model.

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

## 5. Model Selection and Backend Default Fallback

The backend-supported configuration may include a `model` or `model_type` field. For now this is intentionally a lightweight compatibility hook and does not introduce a global model registry.

Current behavior:

- If `model` / `model_type` is missing, empty, or set to a default alias, the current backend model remains the active default.
- If `model` / `model_type` is a class that subclasses the current backend model class, that subclass is used instead.
- Other values fall back to the default model implementation for backward compatibility.

This keeps the system stable while still allowing a custom subclass to plug into the same model lifecycle when it inherits from the backend's existing base implementation.

### Custom model subclasses

Some strategy implementations accept a custom subclass of their default
model implementation.

Use the actual model class exposed by the selected strategy, and preserve
its initialization, fitting, prediction, weight-management, omission, and
callback-reset contracts.

This extension mechanism is not a general cross-backend model registry (maybe added in future).
Check the selected strategy's accepted `model` or `model_type` configuration
before using a custom subclass.

## 6. Compile Named Relations Before Backend Construction

The former `build_constraint_from_relations(...)` strategy method has been
removed.

Use `parse_relations_spec(...)`, then `prepare_constraint_data(...)`,
then `strategy.build_constraint(spec, data=...)`.

The preparation stage uses named `cause_slices` and effective output
ordering. It does not require callers to reconstruct `col_offsets`.

The builder exposes the same separation through
`prepare_constraint_data()` and `prepare_constraint()`.

## 7. How to Add New Components

## Adding new components

Use the interfaces accepted by the selected library strategy.

### Callbacks

For the native PyTorch and Scikit implementations, follow the callback
interface used by their models.

For the Keras-based implementation, use callbacks compatible with the Keras
API and verify their behavior under the selected runtime.

A callback must also be compatible with the per-run reset or cloning mechanism.
Constructor compatibility alone does not establish lifecycle compatibility.

Register dictionary names in the strategy's component loader when the
component should be available through configuration files.

### Constraints and regularizers

Implement a concrete subclass of the selected backend's library base
class and register it through that backend's component registry.

Package locations:

- `backends.scikit.constraints` and `backends.scikit.regularizers`;
- `backends.pytorch.constraints` and `backends.pytorch.regularizers`;
- `backends.keras.constraints` and `backends.keras.regularizers`.

Constraint implementations provide `enforce(...)` and
`is_satisfied(...)`. Their constructors receive `PreparedConstraintData`
and any implementation-specific parameters selected through
`constraint_spec.params`.

Regularizer constructors receive `PreparedRegularizerData`.
User parameters are available through `data.params`.

For custom regularizers:

- Scikit subclasses implement `_value_and_grad(...)`, returning a scalar
  penalty and a gradient matching the coefficient shape.
- PyTorch subclasses implement `forward(...)`, returning a tensor penalty.
- Keras subclasses implement `__call__(...)`, returning a backend-native
  penalty.

Register classes using `register_constraint(...)` or
`register_regularizer(...)` from the corresponding backend registry.
The first registered name is canonical; additional names are aliases.

Registries reject abstract classes, incompatible subclasses, instances,
and factory functions.

Additional serialized constructor parameters must be included in the
component's configuration methods..

### Optimizers

Optimizer constructor arguments are backend-specific.

A generic configuration shape does not imply that parameter names can be
forwarded unchanged to every optimizer implementation.

Document any translation, accepted aliases, and rejected fields in the
component configuration reference.

## 8. Model lifecycle requirements

The builder expects the working model to support:

- initialization with prepared inputs and targets;
- fitting and prediction;
- reading and assigning weights;
- setting omitted feature columns;
- resetting callbacks for a named run.

The builder restores base weights before restricted fitting.
The backend must define the treatment of optimizer state and other training
state across these runs.

Weight assignment should not unexpectedly change masks or omission settings.
Omission updates must not accumulate across independent restricted runs.

When implementing a custom model, verify snapshot independence and consistent
prediction shapes in addition to constructor compatibility.

## 9. Library strategy versus Keras runtime

The library backend strategy and the runtime used by Keras are separate
configuration choices.

The strategy determines which model implementation and component loaders
the library uses.

The Keras runtime determines the execution backend used internally by Keras.
Selecting the library's `"pytorch"` strategy is not the same as selecting
the PyTorch runtime for a Keras-based model.

Historical strategy names and aliases may retain TensorFlow-oriented naming.
Do not infer runtime requirements or component compatibility from an alias
alone.

Configure the intended Keras runtime before importing Keras or library modules
that import it.

Compatibility must be checked for the selected runtime, model, optimizer,
callbacks, constraints, and regularizers. A configurable runtime does not
by itself establish that every component combination is supported.

## 10. When You Need a New Backend (not only new components)

Create a new backend strategy when at least one of these is true:

- You use a different ML runtime/model lifecycle not compatible with current backends.
- Optimizer/callback/constraint semantics are fundamentally different.
- You need separate model build/train/predict logic and validation rules.

In practice, add:

- New strategy class implementing `BackendStrategy` contract.
- New backend model implementation.
- New object loader for dict-to-object resolution.
- Factory registration (with aliases) in `BackendFactory`.

## 11. Practical Recommendation

Start with new component classes first. Create a new backend only when component-level extension is no longer enough.
