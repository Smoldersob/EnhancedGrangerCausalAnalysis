# Project Structure

The installable source package is `enhanced_granger_analysis/`.

```text
enhanced_granger_analysis/
├── api/
│   ├── builder.py
│   ├── orchestrator.py
│   ├── simple_granger.py
│   └── config_loader.py
├── configuration/
│   ├── analysis_spec.py
│   ├── modeling_spec.py
│   └── loader.py
├── modeling/
│   ├── assembler.py
│   ├── initialization.py
│   ├── executor.py
│   └── artifacts.py
├── execution/
│   ├── runner.py
│   └── granger_trainer.py
├── experiments/
│   └── config_iterator.py
├── backends/
│   ├── backend_factory.py
│   ├── base_backend.py
│   ├── keras/
│   ├── pytorch/
│   └── numpy/
├── core/
├── initializers/
├── preprocessing/
├── results/
├── hiperopt/
├── scripts/
├── tests/
└── utilities/
```

## Responsibilities

| Package | Responsibility |
| --- | --- |
| `api` | Fluent builder, one-call orchestrator, independent baseline, compatibility imports |
| `configuration` | Configuration sections and document/reference reading; no backend component construction |
| `modeling` | Assembly, initializer execution, one fit, and its artifacts |
| `execution` | Full workflow coordination, standard Granger fitting, outcome-to-snapshot conversion |
| `experiments` | Explicit test-case expansion, separate from HPO |
| `backends` | Strategy selection, native model adapters/networks, existing resolvers and component registries |
| `core` | Model protocols, structural metadata, registry infrastructure, shared component data, exceptions |
| `initializers` | Existing initializer class registry and implementations |
| `preprocessing` | Alignment, imputation, cleanup, stationarity, lag preparation, scaling, sampling |
| `results` | Snapshots, preparation context, metrics, restoration, labelled matrices |
| `hiperopt` | Existing optimization subsystem; not active in the standard runner |
| `scripts` | Group experiments, export, timing, and ground-truth reporting |
| `utilities` | Shared helpers including ground-truth metric calculation |

`configuration/legacy.py` has been removed. `api/config_loader.py` only exposes compatibility imports. Removed shared `backends/models`, `backends/object_loaders`, and `backends/scikit` packages are not restored.

## Existing shared contracts

- `core/models/input_spec.py`: `ModelInputSpec`.
- `core/models/protocol.py`: `GrangerModelProtocol` and related types.
- `core/models/training_result.py`: `TrainingResult`.
- `core/initialization.py`: `InitializationContext`.
- `core/components/`: relation/regularizer parsing, preparation data, registry support, and serialization.
- `core/lag_config.py`: lag specifications, engine settings, preparation result metadata.
- `preprocessing/pipeline.py`: existing `DataPreparationConfig` and `DataPreparer`.
- `preprocessing/prepare_data.py`: `PreparedData`, `VariableRoles`, and removed-variable metadata.

Do not introduce parallel versions of these contracts in the new packages.

## Results and execution artifacts

`modeling/artifacts.py` defines `FitOutcome`; it is not a causal result container. `execution/runner.py` defines internal `AnalysisRunProgress`; it is not a public session. Final snapshot evaluation remains in `results/`.

## Repository-level documentation

`README.md` provides navigation and quick start; `Changelog.md` records changes. `docs/execution_architecture.md` describes responsibilities and data flow; `docs/api_refactor_migration.md` describes migration.

The detailed preprocessing, lag-selection, model-contract, result-metric, NumPy-solver, and ground-truth pages retain their subsystem scope. Future sessions/procedures are not part of the tree above.

## Backend-local Linear Model Families

Each built-in backend uses the following model-package organization:

```text
models/
├── conditional.py
├── classic.py
├── network_parts.py
├── registry.py
└── __init__.py
```

- `conditional.py` defines the unrestricted-baseline adapter and reusable
  training lifecycle.
- `classic.py` defines the restricted autoregressive-baseline adapter.
- `network_parts.py` defines predictor architectures and input-control
  helpers.
- `registry.py` uses the existing registry infrastructure for model
  selection.
- `__init__.py` imports model implementations so registration is executed.

Classical adapters specialize predictor construction through
`_build_network()`. NumPy additionally specializes training-design and
gradient calculations while preserving its existing solver framework.

No new backend strategy, component resolver, preprocessing contract, or
result-container type is introduced.

See [Classical Granger models](classical_granger_models.md).