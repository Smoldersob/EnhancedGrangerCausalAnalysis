# Component Loading and Construction

Component declarations are interpreted by their existing family-specific parsers, resolvers, and registries. Shared document loading is not a component factory, and there is no universal accepted input format for every family.

## Responsibility map

| Family | Existing interpretation | Construction |
| --- | --- | --- |
| Backend | `BackendFactory` | Strategy selection |
| Lag engine/selector | Preparation pipeline and lag resolvers | Preparation pipeline |
| Scaler | Existing scaler resolver | Preparation pipeline |
| Model | Backend-local model registry | Strategy `build_model` |
| Optimizer | Backend-local resolver | Strategy `build_optimizer` |
| Loss | Backend-local resolver | Strategy `build_loss` |
| Callbacks | Backend-local resolver/registry | Strategy `build_callbacks` |
| Relations | `parse_relations_spec` | Compile against final layout |
| Constraint | Backend constraint registry | Strategy `build_constraint(spec, data=...)` |
| Regularizer | `parse_regularizer_spec` and backend registry | Strategy `build_regularizer(selector, data=...)` |
| Initializer | `resolve_initializer_spec` | `ModelInitializer` constructs and executes it |

`ModelAssembler` coordinates these construction dependencies; it does not maintain another registry or reinterpret component aliases. `AnalysisRunner` does not replace these mechanisms.

## Constraint declarations

```python
relations = [
    {"cause": "x1", "effect": "y", "rule": {"type": "zero"}},
    {
        "cause": "x2",
        "effect": "y",
        "rule": {"type": "min_abs_sum", "params": {"value": 0.2}},
    },
]
constraint_spec = "mask_and_min_abs_sum"
```

Relations accept the canonical list or `RelationConstraintsSpec`, not arbitrary shorthand maps or `None`. Empty relations preserve the base mask. Duplicate cause/effect declarations are rejected. A zero minimum adds no lower-bound restriction; it is not a zero mask.

The assembler calls the existing relation parser, compiles `PreparedConstraintData` using effective cause slices/output order/base mask, and delegates component construction. Unknown effective variables and infeasible positive bounds fail according to existing preparation policy.

Do not supply layout metadata, masks, or indexed rules through constraint params. High-level analysis rejects `constraint_spec=None`, even though low-level registries/strategies may have broader acceptance.

## Regularization

```python
regularizer_spec = {"type": "l1", "params": {"l1": 0.01}}
```

The shared parser accepts `None`, a registered name, a concrete class, a normalized `RegularizerSpec`, or its supported `type`/`params` declaration. It does not accept ready instances or factory functions. Strategy-level support for a compatible instance is a separate, lower-level contract.

The assembler prepares `RegularizationLayout` and `PreparedRegularizerData`, then calls the backend strategy with selector and prepared data. There is no model-side reverse conversion from a ready regularizer to configuration.

`None` disables regularization. Built-in `l1` and `lag_dependent_l1` default to zero strength when no nonzero value is specified. An existing zero-strength regularizer object can still affect NumPy solver eligibility; it is not identical to absence of a regularizer.

Lag-dependent example:

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

Actual `feature_lags` determine indexing. With absolute indexing, lag 1 uses index 1. Relative indexing subtracts the global minimum feature lag, not a separate minimum for each cause. Weights must cover required indices. Do not supply legacy `col_offsets` or `max_lags_per_pred` as layout substitutes.

## Initializers

The existing resolver accepts `None`, a registered name, a supported mapping, or a class. Instances and factory functions are not accepted. An initializer receives `InitializationContext` with prepared arrays, copied current weights, and the preparation mask. It returns the weight-array sequence expected by the model.

`ModelInitializer.compute_weights()` does not assign weights. Its caller uses the model's existing `set_weights()` and captures the accepted representation. Shape/orientation validation remains the model's responsibility.

## Optimizers and losses

Use `model_config.optimizer` and `model_config.loss`. The assembler extracts them and passes ready components separately from constructor options.

PyTorch Adam uses `params.lr`; Keras Adam uses `params.learning_rate`. Do not infer cross-backend parameter portability from a shared typed-specification shape. PyTorch gradient accumulation is a model training-loop setting; supported Keras optimizer accumulation options belong to that optimizer's configuration.

NumPy does not gain arbitrary optimizer/loss support from these fields. Its existing resolvers and solver contracts govern accepted values.

## Callbacks

Callback declarations pass to the existing backend callback resolver. The loader and specification do not maintain a separate normalization/instantiation path.

NumPy/PyTorch callbacks use backend-local classes and run-reset infrastructure. Keras uses its backend-specific callback path. Custom callbacks must satisfy lifecycle requirements, not merely constructor syntax. Run resets must not accidentally restore model parameters or alter omission state.

## Backend bases and orientation

| Backend | Constraint base | Regularizer base |
| --- | --- | --- |
| NumPy | `BaseNumpyConstraint` | `BaseNumpyRegularizer` |
| PyTorch | `BasePyTorchConstraint` | `BasePyTorchRegularizer` |
| Keras | `BaseKerasConstraint` | `BaseKerasRegularizer` |

Prepared layouts are canonical `(n_outputs, n_features)`. Native NumPy/PyTorch coefficients and Keras Dense kernels can have different orientations; backend implementations handle this. `FitOutcome` retains the model's complete exported weight representation without trying to standardize it.

## Extensibility and serialization

Register project components in their existing registries. Do not duplicate registration in the document loader, builder, runner, or iterator. Accepted forms are family-specific; programmatic objects are not automatically JSON/YAML serializable.

Configuration serialization is separate from component resolution. Preserve existing specification/prepared-data serializers and include custom constructor requirements in component serialization contracts. Do not introduce reverse normalization merely to make assembly work.

## Devices

Device values are not a component family needing another resolver. Keras/PyTorch use existing managers. NumPy ignores `device` in its strategy's constructor dispatch. See [Backend usage](backend_usage.md).

Related: [Configuration](config_file_usage.md), [Architecture](execution_architecture.md), [Model contract](model_contract.md).
