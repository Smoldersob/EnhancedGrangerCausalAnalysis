# Configuration Files

The external analysis format remains a flat JSON/YAML mapping. Internal `AnalysisSpec.preparation`, `.roles`, and `.modeling` fields do not introduce matching nested document sections.

## Responsibilities

1. `AnalysisConfigLoader` reads documents and resolves relation-file references.
2. `AnalysisSpec.from_mapping()` constructs configuration sections and delegates backend declaration handling to `BackendFactory`.
3. Existing preprocessing and backend resolvers interpret component declarations.
4. `AnalysisRunner` executes the standard analysis.

The loader does not normalize backend aliases, parse relations, resolve callbacks, instantiate selectors, train models, or evaluate results.

## Minimal YAML

```yaml
backend: numpy
causes: [x, y]
effects: [y]
tested_causes: [x]
stationarity: null
lag_config:
  max_lag: 2
  use_lag_zero: false
relations: []
regularizer_spec: null
model_config:
  model: conditional_linear
  epochs: 100
```

```python
from enhanced_granger_analysis.configuration.loader import AnalysisConfigLoader
from enhanced_granger_analysis.configuration import AnalysisSpec
from enhanced_granger_analysis.execution import AnalysisRunner

mapping = AnalysisConfigLoader.load_file("analysis.yaml")
spec = AnalysisSpec.from_mapping(mapping)
results = AnalysisRunner().run_analysis(spec=spec, data=frame)
```

Or load through a builder:

```python
from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder

results = MultiTaskGrangerBuilder().from_file("analysis.yaml").data(frame).fit()
```

`from_config()` updates supplied top-level fields; `replace_config()` resets configuration and execution artifacts. Nested maps are replaced, not recursively merged.

## Loader methods

| Method | Result |
| --- | --- |
| `read_document(path)` | JSON/YAML value of any top-level shape |
| `load_raw_file(path)` | Mapping without resolving embedded references |
| `load_file(path)` | Mapping with supported references resolved |
| `resolve_references(mapping, base_dir=...)` | Mapping with supported references resolved |
| `load_relations_file(path)` | Raw relation document, not a parsed relation specification |

Relative relation paths are interpreted relative to the analysis file. For group files, references are resolved after each case is expanded, relative to the group file. Arbitrary strings elsewhere in component specifications are not interpreted as file paths.

```yaml
relations: ./rules/relations.json
```

Programmatic mappings have no implicit file base directory. `from_config()` does not read a relation path; load it through `load_file()` or call `resolve_references()` explicitly.

The compatibility name `BuilderConfigLoader` aliases `AnalysisConfigLoader`. Removed methods `normalize_builder_config()` and `normalize_relations()` must not be used. For explicit relation parsing, call the existing `parse_relations_spec()`.

## Backend declaration

Canonical names are `pytorch`, `keras`, and `numpy`. Alias resolution lives only in `BackendFactory`: `torch`; `tensorflow`/`tf`; and `sklearn`/`scikit`/`scikit-learn`, respectively.

```yaml
backend: pytorch
```

The retained mapping form supplies model defaults:

```yaml
backend:
  type: pytorch
  params:
    epochs: 50
model_config:
  epochs: 100
```

Here `epochs=100` wins. Backend mapping parameters are not automatically strategy-constructor arguments. In particular, placing `loading_verbose` there is not equivalent to calling `BackendFactory.get_strategy(..., loading_verbose=True)`.

Availability is checked during backend selection, not merely when parsing an `AnalysisSpec`. With no backend, runtime selection prefers PyTorch, Keras, then NumPy. Configure explicitly when recording a reproducible run.

## Device selection

Supported configuration locations, in precedence order:

1. `model_config.device` (also overriding defaults from the backend mapping).
2. Top-level `compute_device`.
3. Top-level `device`.

```yaml
device: cuda:0
model_config:
  epochs: 100
```

Device values are forwarded, not parsed against a second whitelist. PyTorch and Keras use their existing device managers. Explicit unavailable accelerators can raise errors; CPU fallback is not promised for explicit requests. The native PyTorch manager additionally supports explicit MPS; do not infer that every value is portable to Keras.

NumPy ignores `device` during model construction, including values in resolved model-specification parameters. This also works if automatic backend selection chooses NumPy. Unknown unrelated model parameters are not silently discarded.

Do not specify a constructor parameter both in `model.params` and the outer `model_config`; such duplication normally raises an error. NumPy's ignored device is an intentional exception.

## Variable roles and preparation

- `causes`: predictor variables; omitted/null uses available columns.
- `effects`: jointly predicted outputs; omitted/null uses available columns.
- `tested_causes`: subset of causes; omitted/null defaults to causes.
- `schema_policy`: analysis default is `union`; the low-level `DataPreparationConfig` default remains `error`.
- `stationarity`: declarative enforcer selection resolved by the preparation pipeline. Analysis entry points default to `adf`; `null`/`None` disables the stage, while the low-level `DataPreparationConfig` default remains `None` (stage off). Accepted declarations: `None`, a registered name, `{"type": ..., "params": {...}}`, an enforcer class, or a compatible instance. The prepared artifact is exposed as `stationarity_enforcer`.
- `x_scaler`, `y_scaler`: passed to existing scaler resolution.
- `missing_value_imputer`, `sampling_strategy`: preserve their existing accepted programmatic contracts; a config file does not automatically make arbitrary class mappings constructible.
- `backend_sample_fraction`, `backend_max_samples`, `random_state`: preparation workload reduction settings, not a general model random-seed contract.

Roles and preprocessing metadata are validated by the existing preparation pipeline.

### Lag preparation

```yaml
lag_config:
  max_lag: 4
  use_lag_zero: false
lag_selector: null
lag_n_jobs: 1
```

With a selector:

```yaml
lag_config:
  aggregation: union
  use_lag_zero: false
lag_selector:
  type: ic
  params:
    max_lag: 8
    use_bic: true
lag_n_jobs: -1
```

Alternative separate parameters use `lag_selector_params`. Do not duplicate a constructor parameter inside both locations. Selectors and lag-engine settings are resolved in preprocessing, not in the loader or builder.

Without a selector, `lag_config.max_lag` sets the fixed positive-lag range. With a selector, its own `max_lag` sets the search bound. `lag_n_jobs` is dataset-level lag-selection parallelism, not training-device selection.

Manual lag values are explicit sets, not endpoints:

```yaml
lag_config:
  use_lag_zero: true
  predictor_overrides:
    x1: [1, 3]
    x2: []
  pair_overrides:
    y:
      x1: [0, 2, 4]
```

Pair overrides take precedence. Own contemporaneous values are forbidden. A tested input with no final lags receives no restricted fit; restoration of its result column is a reporting convention, not an independent negative test.

Detailed selector families, scoring, aggregation, overrides, and migration from range-based settings are documented in [Lag selection](lag_selection.md).

## Model and components

```yaml
model_config:
  model: conditional_linear
  epochs: 100
  batch_size: 32
```

`model_type` is an alternative to `model`, not an additional setting. Model registries reject unknown names rather than silently selecting a default. Structural fields such as `input_spec`, `n_features`, and `n_outputs` are generated from prepared data and must not be supplied here.

PyTorch optimizer example:

```yaml
backend: pytorch
model_config:
  optimizer:
    type: adam
    params:
      lr: 0.001
  loss: mse
  epochs: 100
```

Keras Adam uses `learning_rate` in optimizer parameters, not PyTorch's `lr`. NumPy uses its own solver/objective and does not gain custom optimizer/loss support from these keys.

```yaml
constraint_spec: mask_and_min_abs_sum
relations:
  - cause: x1
    effect: y
    rule:
      type: zero
regularizer_spec:
  type: l1
  params:
    l1: 0.01
initializer: zeros
```

Use empty relations for no additional named rules. `relations: null` is not supported. `constraint_spec: null` is rejected; `regularizer_spec: null` disables regularization. Do not put constraints, regularizers, callbacks, or layout metadata inside `model_config`. Initializer instances/factories are not accepted by the existing initializer resolver.

See [Components](components_loading.md) and [Backend usage](backend_usage.md).

## Explicit groups of experiments

```json
{
  "device": "cpu",
  "base_config": {
    "backend": "numpy",
    "stationarity": null,
    "lag_config": {"max_lag": 2, "use_lag_zero": false},
    "model_config": {"epochs": 100}
  },
  "sweep": {
    "param_names": ["backend", "model_config.epochs"],
    "cases": [["numpy", 100], ["pytorch", 50]]
  }
}
```

```python
from enhanced_granger_analysis.experiments import TestGroupConfigIterator
from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder

for config in TestGroupConfigIterator.from_file("group.json"):
    result = MultiTaskGrangerBuilder().from_config(config).data(frame).fit()
```

Each row is one explicit case. There is no automatic Cartesian product. Top-level group settings are combined with `base_config`; base values win, followed by case overrides. Dotted paths traverse dictionaries, not lists. Replace whole callback/relation lists rather than using list indices.

The iterator returns mappings and does not normalize aliases or resolve components. It can resolve a different relation-file reference for each expanded case.

## Prepared data and reuse

DataFrame inputs and `PreparedData` are runtime inputs, not serializable fields of `AnalysisSpec`. The builder retains `data`/`prepared_data` compatibility handling, but direct runner calls supply them separately.

`reuse_data` does not create a runner cache. Explicitly supplied prepared data takes precedence and is not automatically checked against a newly changed preparation specification. The experiment script has its own preparation cache; its key must cover all settings affecting preparation before reusing data across differing cases.

## HPO status

`hiperoptimalization_state` and `hiperoptimalization_conf` are retained but inactive. No `search` document section is implemented by this refactor. Sweeping L1 here executes multiple experiments; it does not select L1 internally by model-quality validation.

## Migration

See [Migration guide](api_refactor_migration.md) for removed loader helpers, preferred imports, and the distinction between update and replacement.
