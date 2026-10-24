# Components Loading

Component loading depends on the component family. There is no single
specification format shared unchanged by callbacks, optimizers,
constraints, regularizers, and initializers.

Version 2.6 introduces dedicated parsing, preparation, and class-resolution
paths for constraints and regularizers.

> Version 2.6 is experimental. The interfaces below describe the new
> component contract, but the current development commit still contains
> integration defects in builder and backend execution paths.

## Component families

| Component | Resolution/preparation | Strategy construction |
| --- | --- | --- |
| Model | Backend-local model registry | `build_model(spec, components=..., config=...)` |
| Callbacks | Backend-local resolver/registry | `build_callbacks(specs)` |
| Optimizer | Backend optimizer resolver | `build_optimizer(spec)` |
| Loss | Backend loss resolver | `build_loss(spec)` |
| Constraint | Relation parser, layout preparation, backend registry | `build_constraint(spec, data=...)` |
| Regularizer | Shared parser, layout preparation, backend registry | `build_regularizer(selector, data=...)` |

Registry resolution and object construction are separate stages.

## Regularizer specification

The shared parser accepts:

- `None`.
- A registered name.
- A concrete regularizer class.
- A normalized `RegularizerSpec`.
- A mapping containing `type` and optional `params`.

Ready regularizer instances and factory functions are not accepted by the
shared parser.

Strategy-level construction is a separate API and may accept an already
compatible backend regularizer instance. Do not infer parser acceptance from
strategy-level behavior.

Built-in names are `l1` and `lag_dependent_l1`. Their default strength is `0.0`,
so selecting a regularizer by name alone does not introduce a nonzero penalty.

## Component bases

| Backend | Constraint base | Regularizer base |
| --- | --- | --- |
| NumPy | `BaseNumpyConstraint` | `BaseNumpyRegularizer` |
| Native PyTorch | `BasePyTorchConstraint` | `BasePyTorchRegularizer` |
| Keras | `BaseKerasConstraint` | `BaseKerasRegularizer` |

NumPy and PyTorch use native coefficient matrices of shape
`(n_outputs, n_features)`. Keras Dense kernels use
`(n_features, n_outputs)`.

Prepared layout data remains canonical. Backend implementations handle native
orientation; built-in linear-model parameter export uses
`(n_features, n_outputs)`.

## Callback lifecycle

NumPy and PyTorch have separate callback classes and registries.
Similar hook names do not make callback objects interchangeable across backends.

Custom native-loop callbacks implement `clone_for_run()` to preserve
configuration while discarding previous runtime state.

Shared lifecycle utilities live in `core.components.callback_lifecycle`.
Keras callback construction and reset handling remain backend-specific.

Resetting callbacks must not restore predictor weights, alter input omission,
or reset optimizer state.

## Constraint configuration

Constraint configuration has two independent parts:

1. `relations` describes named cause/effect rules.
2. `constraint_spec` selects the backend implementation enforcing the
   prepared mask and rules.

### Named relation rules

Use a list of canonical declarations:

```python
relations = [
    {
        "cause": "x1",
        "effect": "y",
        "rule": {"type": "zero"},
    },
    {
        "cause": "x2",
        "effect": "y",
        "rule": {
            "type": "min_abs_sum",
            "params": {"value": 0.2},
        },
    },
]
```

The supported rules are:

- `zero`: masks all coefficients for the selected cause/effect relation.
- `min_abs_sum`: requires the sum of absolute values of the relation's
  active coefficients to be at least `value`.

`value` must be finite and non-negative. A lower bound of zero adds no
restriction; it is not equivalent to a `zero` rule.

Each cause/effect pair may appear only once.

The parser accepts a `RelationConstraintsSpec` instance or a list of
declarations. Legacy tuple-key mappings, string-key mappings, shorthand
numeric values, and `None` are not accepted.

Use `[]` or `RelationConstraintsSpec()` when no explicit rules are needed.
An empty relation specification preserves the preprocessing base mask.

### Selecting the constraint implementation

```python
constraint_spec = {
    "type": "mask_and_min_abs_sum",
    "params": {"eps": 1e-8},
}
```

Accepted selectors are:

- a registered name;
- a mapping with `type` and optional `params`;
- a concrete subclass of the selected backend's constraint base class.

The low-level component registries recognize existing instances of their
backend base classes. The constraint registry also recognizes `None`.

Registry-level acceptance is distinct from the public builder contract.
Builder configuration and strategy construction must be documented
according to their own accepted inputs; registry behavior alone does not
establish support for passing instances through the complete pipeline.

Built-in names shared by the backend registries:

| Implementation | Registered names |
| --- | --- |
| Mask only | `MaskConstraint`, `mask_constraint`, `mask` |
| Mask and minimum absolute sum | `MaskAndMinAbsSumConstraint`, `mask_and_min_abs_sum_constraint`, `mask_and_min_abs_sum` |

Use `mask_and_min_abs_sum` when positive minimum-sum rules are present.
The mask-only implementation rejects such rules.

Do not put `mask`, indexed `rules`, `output_names`, `n_features`, or
`cause_slices` inside `constraint_spec.params`. These values are prepared
from the final input layout and are not constructor tuning parameters.

### Preparation and construction

The shared preparation utility combines:

- named relation rules;
- effective cause slices;
- effective output ordering;
- the preprocessing base mask.

It produces `PreparedConstraintData`, whose mask uses the canonical shape
`(n_outputs, n_features)`.

Positive minimum-sum rules include only coefficients still active after
masking. If none remain, preparation raises an error by default. The
low-level utility supports `infeasible_policy="warn"`; the builder currently
uses the default error policy.

The backend then resolves the selected class and constructs it using
`build_constraint(constraint_spec, data=prepared_constraint_data)`.

## Regularizer configuration

Use `regularizer_spec` as the configuration source:

```python
regularizer_spec = {
    "type": "l1",
    "params": {"l1": 0.01},
}
```

Accepted forms are:

- `None`, which disables regularization;
- a registered name;
- a concrete backend regularizer class;
- a concrete backend regularizer instance;
- a mapping containing `type` and optional `params`;
- a normalized `RegularizerSpec`.

In Python mappings, `type` may also be a concrete class. JSON/YAML
configuration must use a registered string name.

Instances and factory functions are not accepted. The parser rejects
`name`, `kind`, and extra top-level parameter fields.

Built-in registered names are:

- `l1`;
- `lag_dependent_l1`.

Historical backend-prefixed regularizer aliases are not registered by
the new built-in implementations.

### Lag-dependent L1

```python
regularizer_spec = {
    "type": "lag_dependent_l1",
    "params": {
        "l1": 0.01,
        "lag_weights": [1.0, 2.0, 3.0, 4.0],
        "relative_lag_indexing": False,
    },
}
```

The regularizer uses `PreparedData.feature_lags`, not inferred block
positions.

With absolute indexing, each feature uses
`lag_weights[feature_lag]`. For example, lag 1 uses index 1, not index 0.

With `relative_lag_indexing=True`, the minimum lag across the complete
feature layout is subtracted before indexing. This origin is global,
not calculated separately for each cause.

`lag_weights` must be a non-empty one-dimensional sequence of finite,
non-negative numbers and must cover every required index.

The built-in default `l1` is `0.0`. Selecting a regularizer by name alone
therefore does not introduce a nonzero penalty.

Do not supply `max_lags_per_pred` or `col_offsets` as regularizer
parameters. The new preparation path derives layout metadata from
`cause_slices`, `output_names`, `feature_lags`, and `base_mask`.

### Preparation and construction

The shared preparation stage creates:

- `RegularizationLayout`, describing the canonical coefficient layout;
- `PreparedRegularizerData`, containing that layout and user parameters.

The backend resolves the class and constructs it using:

```python
regularizer = strategy.build_regularizer(
    parsed_spec.selector,
    data=prepared_regularizer_data,
)
```

Registry resolution selects and validates a class without constructing it.
Prepared data is supplied separately to the constructor.

Regularizer implementations do not automatically apply every model mask.
The model integration is responsible for supplying the appropriate
effective coefficients.

## Backend-specific component bases

| Backend | Constraint base | Regularizer base |
| --- | --- | --- |
| Scikit | `BaseScikitConstraint` | `BaseScikitRegularizer` |
| Native PyTorch | `BasePyTorchConstraint` | `BasePyTorchRegularizer` |
| Keras | `BaseKerasConstraint` | `BaseKerasRegularizer` |

Scikit and native PyTorch coefficient matrices use
`(n_outputs, n_features)`.

Keras Dense kernels use `(n_features, n_outputs)`. The prepared data
retains the canonical layout; backend implementations handle orientation.

Custom components must inherit from the selected backend's library base
class. Inheriting only from a native Keras regularizer or providing an
arbitrary callable is not sufficient for the new model validation rules.

## Serialization

Regularizer specification serialization stores a registered type name and
parameters. Serializing a class-based specification requires an explicit
`registered_name`.

Prepared regularizer serialization stores:

- `schema_version`;
- layout metadata;
- parameters.

Backend components expose configuration serialization methods. Custom
components must include any additional constructor parameters required
for reconstruction.

## Scikit Backend
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

### Optimizer and training settings

Optimizer configuration and training-loop configuration are related but
distinct.

A typical model configuration contains training settings together with an
optimizer selection:

```python
model_config = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 32,
}
```

For structured optimizer specifications, use:

```python
model_config = {
    "optimizer": {
        "type": "adam",
        "params": {
            # Parameters accepted by the selected optimizer loader.
        },
    },
}
```

Parameter names and accepted aliases depend on the selected backend loader.
Do not assume that one optimizer parameter dictionary is portable unchanged
between native PyTorch and Keras.

Training-loop options must not be forwarded to an optimizer constructor
unless that optimizer explicitly supports them.

### Gradient accumulation

Gradient accumulation has backend-specific configuration semantics.

For the native PyTorch implementation, it is a training-loop setting rather
than a standard optimizer constructor argument. It must remain outside the
optimizer's constructor parameters.

For Keras optimizers that support `gradient_accumulation_steps`, it belongs
to the optimizer configuration.

Compatibility aliases should be documented separately from the canonical
configuration location. An accepted alias must not be described as evidence
that the underlying optimizer accepts the same argument.

Check the selected loader before moving accumulation settings between
model-level and optimizer-level configuration.

## Keras-based strategy

This section describes components loaded for the library's Keras-based
strategy, including historical TensorFlow-oriented aliases.

The library strategy name and the Keras execution runtime are distinct.
Component aliases do not independently establish runtime compatibility.

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

## Callback lifecycle

Resolving a callback specification constructs or selects a callback.
It does not by itself establish that the callback can be safely reused across
base and restricted runs.

The model's callback-reset mechanism is responsible for run isolation.
Stateful callbacks and callbacks writing files may require reconstruction,
resetting, or run-specific output paths.

Custom callback objects should be checked against the reset mechanism used
by the selected model.

## Feature-block metadata

Component preparation uses the final effective feature layout:

- `PreparedData.cause_slices` maps each cause to its feature columns;
- `PreparedData.effects` defines output ordering;
- `PreparedData.feature_lags` gives the actual lag of each feature column;
- `PreparedData.base_mask` describes the preprocessing coefficient mask.

These values must agree with the prepared input and target dimensions.

The order of `tested_causes` is not a feature-layout specification.
Do not reconstruct regularizer lag metadata from tested-cause order or
legacy positional offsets.

## Verbose Loading
All strategies support internal loader logging via `loading_verbose=True`.

Simple example:

```python
strategy = BackendFactory.get_strategy("pytorch", loading_verbose=True)
```