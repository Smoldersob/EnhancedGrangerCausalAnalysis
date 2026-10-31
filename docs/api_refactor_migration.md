# Migration to the Shared Analysis Runner

## Scope

The refactor extracts construction, initialization, single-fit execution, and standard analysis coordination from the builder. Public configuration remains flat. Automatic HPO and high-level execution sessions are not introduced.

## Preferred imports

```python
from enhanced_granger_analysis.configuration import AnalysisSpec
from enhanced_granger_analysis.configuration.loader import AnalysisConfigLoader
from enhanced_granger_analysis.execution import AnalysisRunner
from enhanced_granger_analysis.experiments import TestGroupConfigIterator
```

`api.config_loader.BuilderConfigLoader` remains an alias of the document loader, not the former normalizing implementation. `MultitaskGrangerBuilder` remains a builder-name compatibility alias.

## Loading and parsing

Before, loading could normalize backend/device fields and parse relations. Now document loading and interpretation are separate:

```python
mapping = AnalysisConfigLoader.load_file("analysis.yaml")
spec = AnalysisSpec.from_mapping(mapping)
results = AnalysisRunner().run_analysis(spec=spec, data=frame)
```

Removed helper replacements:

| Removed helper | Replacement |
| --- | --- |
| `normalize_builder_config(mapping)` | `AnalysisSpec.from_mapping(mapping)` when an analysis specification is needed |
| `normalize_relations(value)` | Existing `parse_relations_spec(value)` |
| Private `_read_mapping(...)` | Public `load_raw_file(...)` |
| Private `_resolve_config_references(...)` | Public `resolve_references(..., base_dir=...)` |

`load_relations_file()` now returns a raw document. Parse explicitly only when the caller needs a `RelationConstraintsSpec`; normal assembly already calls the parser.

## Updating versus replacing builder configuration

`from_config()` updates supplied top-level fields. It does not recursively merge model configuration or copy resolved runtime objects from another run.

`replace_config()` discards previous configuration and execution artifacts. Reattach data explicitly when not included. This method resets the concrete builder; do not treat it as a generic subclass reconstruction mechanism.

`to_spec()` excludes source data and prepared data. Standalone execution of that specification does not synchronize the originating builder.

## Extension points

Complete builder fit no longer relies on overridden `prepare_data()`, `fit_base()`, or `fit_restricted()` methods of a builder subclass. Existing backend resolvers, component registries, model protocols, and extracted services are the intended lower-level extension boundaries.

Do not add duplicate resolvers to loaders, specifications, runners, or experiment iterators.

## Backend and device configuration

`BackendFactory` is the sole source of backend aliases and handles mapping-form backend defaults. Explicit `model_config` entries win over those defaults.

Mapping backend params retain their historical role as model defaults; they are not automatically forwarded to the strategy constructor. Use explicit `get_strategy(..., loading_verbose=True)` for strategy loading verbosity.

Top-level `compute_device`/`device` map to `model_config.device` unless it is already supplied. Device values remain for the selected backend to interpret. NumPy ignores `device`, including in resolved model specification params, allowing common group configuration. Other unknown constructor parameters still fail normally.

## Experiment iteration

The iterator returns mappings, resolves references after expansion, and does not canonicalize component names. Sweep cases are rows, not a Cartesian grid. Code inspecting a returned backend declaration must not assume it is already a canonical string.

For canonical configuration reporting, create an `AnalysisSpec`. With backend `None`, this still records automatic selection rather than an actually selected runtime backend.

## Outputs

Public builder/orchestrator output remains `GrangerAnalysisResults`. `FitOutcome` is an internal single-fit artifact, not a replacement public return type. `TrainingResult` remains the model's diagnostics contract.

Predictions are inverse-scaled before snapshot storage. Snapshot coefficient weights retain the current first-weight-array convention. Fit diagnostics are not automatically exported as new summary columns or result metadata.

## Remaining limitations

- HPO settings are accepted but inactive.
- `reuse_data` alone does not create a runner cache.
- Frozen configuration does not deeply freeze nested objects.
- Unrelated configuration metadata can be ignored; this is not a strict unknown-key schema validator.
- Completed partial snapshots after an exception do not establish a successful full analysis.
- No session/resume API or interchangeable procedure registry is introduced.
- Device-manager accepted values are backend-specific; identical configuration structure does not make all component parameters portable.

See [Architecture](execution_architecture.md), [API](api_usage.md), and [Configuration](config_file_usage.md).
