# Backend Usage

## Selection and aliases

```python
from enhanced_granger_analysis.backends import BackendFactory

strategy = BackendFactory.get_strategy("pytorch")
automatic = BackendFactory.get_strategy()
verbose_strategy = BackendFactory.get_strategy("keras", loading_verbose=True)
```

Canonical names are `pytorch`, `keras`, and `numpy`. Aliases are maintained once in `BackendFactory`: `torch`; `tensorflow`/`tf`; and `sklearn`/`scikit`/`scikit-learn`, respectively. Automatic selection prefers PyTorch, Keras, then NumPy subject to availability.

`canonical_name(name)` normalizes a string without creating a strategy. `resolve_configuration(backend_spec, model_config=...)` additionally handles the retained backend mapping format and merges model defaults. It does not create native components or validate installed-runtime availability.

Explicit `model_config` entries override defaults in backend mapping params. Those params are not arbitrary strategy constructor arguments.

## Strategy construction interface

- `build_optimizer(spec=None, **context)`
- `build_loss(spec=None)`
- `build_callbacks(specs=None)`
- `build_constraint(spec=None, *, data=None)`
- `build_regularizer(spec=None, *, data=None)`
- `build_model(spec=None, *, components=None, config=None)`

Strategies invoke existing backend-local resolvers and registries. There is no additional generic component resolver in the loader, specification, or runner.

Models receive ready components and structural `ModelInputSpec`; training observations are supplied later to `fit(X, y, ...)`.

## NumPy low-level example

This example assumes an existing `PreparedData` object called `prepared`. It demonstrates protocol fitting, not the complete constraint-aware analysis assembly path.

```python
from enhanced_granger_analysis.backends import BackendFactory
from enhanced_granger_analysis.core.models.input_spec import ModelInputSpec

strategy = BackendFactory.get_strategy("numpy")
input_spec = ModelInputSpec(
    n_features=prepared.X_scaled.shape[1],
    n_outputs=prepared.y_scaled.shape[1],
    causes=tuple(prepared.causes),
    effects=tuple(prepared.effects),
    cause_slices=prepared.cause_slices,
    feature_lags=tuple(prepared.feature_lags),
)
model = strategy.build_model(
    "conditional_linear",
    config={"input_spec": input_spec, "epochs": 100},
)
training = model.fit(prepared.X_scaled, prepared.y_scaled, collect_history=True)
```

Use the normal runner/assembler when preparation masks and named rules must be enforced. A bare low-level model example does not automatically compile those rules.

NumPy has a built-in MSE objective and solver loop. Its `learning_rate` is relaxation/step control, not a neural optimizer learning rate. See [NumPy optimization](numpy_backend_optimization.md).

## Device handling

### PyTorch

The conditional adapter passes `device` to the existing native `DeviceManager`. It resolves placement, checks availability, and uses the resolved device for network and tensor creation. `auto` prefers CUDA:0 when available, otherwise CPU. Explicit MPS is supported by this manager. Explicit unavailable accelerator requests do not silently fall back to CPU.

### Keras

The conditional adapter passes `device` and the active Keras runtime to its existing `DeviceManager`. Placement uses the manager's scope during construction, fitting, prediction, and weight access. Runtime-specific resolution supports TensorFlow, JAX, and Torch as implemented by the manager.

Choosing library backend `pytorch` is not equivalent to choosing Torch as the Keras execution runtime. Configure Keras runtime before importing Keras.

### NumPy

NumPy does not select a training device. Its strategy declares `device` as an ignored constructor configuration name. The existing `_build_registered_model()` helper removes it from both resolved model params and outer config copies before constructor validation/dispatch.

This enables one device declaration across experiment cases, including automatic selection that eventually chooses NumPy. It does not suppress unrelated invalid parameters. Do not add a catch-all `**kwargs` to the NumPy adapter for this purpose.

### Placement in configuration

`AnalysisSpec` transfers top-level `compute_device` or `device` into `model_config.device` only when a device is not already present in the merged model configuration. It does not interpret device values. No second device whitelist belongs in the loader or backend factory.

The ordinary full-analysis path still rejects conflicting duplicated constructor parameters. Ignored NumPy device entries are intentionally removed from both constructor sources.

## Model registration

Backend-local model registries resolve omitted selection, registered names, strict typed mappings, classes, or compatible instances according to their existing contracts. The built-in conditional model has alias `conditional_linear`.

Unknown names do not silently fall back. Ready model instances cannot be combined with constructor overrides; standard assembly normally supplies its own structural config and components, so low-level ready-instance support is not general support through every high-level entry point.

Implement the existing `GrangerModelProtocol`, not the removed shared model base class. Model construction must not retain training observations.

## Components and lifecycle

Constraint and regularizer classes receive prepared layout-dependent data. Callbacks, optimizers, and losses are resolved by their backend-specific paths. Similar callback hook names do not make objects portable across backends.

Weight assignment, input omission, predictor rebuild, and component reset are separate operations. Standard full/restricted training explicitly resets optimizer and callbacks and then calls fit with `reset_components=False`. Weight restoration is not an optimizer-state restore.

## Extending backends

Add a new component to its existing resolver/registry when possible. Add a strategy only for genuinely different execution/component semantics. Register model and component implementations in the appropriate backend-local packages; do not recreate removed object loaders.

See [Components](components_loading.md), [Model contract](model_contract.md), and [Execution architecture](execution_architecture.md). Examples in this document are not reports of executed validation.
