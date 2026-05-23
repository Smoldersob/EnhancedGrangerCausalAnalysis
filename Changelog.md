## Version Changes 1.0 

This version is still available on branch version_1.0, but is considered to be out dated. It was the first version which was consisted of three classes of tests which corresponed with diffrent backends. It had regularizer, constraints and callback. The data preparation was automatic and had one precedure.

## Version Changes 2.0.0 

The previous README described the project using the older `granger_tests/...` entry points. The current codebase has moved to a new, modular API and added several practical capabilities.

Implemented additions and upgrades:

- **New public API layer**: `api/` now provides `MultitaskGrangerBuilder`, `MultiTaskGrangerAPI`, `SimpleGrangerAPI`, and config loaders, replacing legacy direct class usage from `granger_tests`.
- **Config-first workflows**: JSON/YAML loading (`BuilderConfigLoader`) and sweep expansion (`TestGroupConfigIterator`) are implemented for reproducible experiments.
- **Scripted benchmark runs**: `scripts/run_group_causality_tests.py` executes grouped experiments and writes per-case and summary outputs against ground truth.
- **Extended lag control**: in addition to automatic lag selection, the library now supports `custom_lags` and per-pair `custom_pair_lags` overrides through `LagConfiguration`.
- **Explicit preprocessing modules**: stationarity (`preprocessing/stationarity`) and scaling (`preprocessing/scaling`) are now separate reusable components.
- **Multiple deterministic scalers**: `standard`, `minmax`, `robust`, `maxabs`, and `identity` scalers are available in the core pipeline.
- **Structured output containers**: output dataclasses and `GrangerAnalysisResults` provide standardized access to p-values, F-tests, errors, signs, predictions, and weights.
- **Backend orchestration improvements**: backend load throttling (`backend_sample_fraction`, `backend_max_samples`) and optional hyperparameter optimization stages are available.
- **Run-scoped callback handling**: callbacks can be cloned and specialized per run (base/reference/hyperopt), improving logging and training loop isolation.
- **Expanded tests**: dedicated tests exist for builder API, config loader, custom lag behavior, backend factory, callbacks, regularizers, and result objects.

Items from the Vision that are still planned (not fully implemented yet):

- Native missing-value handling/imputation inside the pipeline.
- Distributed execution beyond current local parallelization.
- Full documentation generation and end-user CLI productization.

### Version Changes 2.1.0 (incremental improvements over 2.0.0)

The 2.1.0 line continues the 2.0.0 refactor with practical usability and workflow improvements, aligned with the current README and docs set.

Implemented and stabilized between 2.0.0 and 2.1.0:

- **Cleaner top-level project summary and navigation**: README now acts as a concise capability summary and points users directly to focused docs pages instead of duplicating long explanations.
- **Dedicated preprocessing documentation**: added explicit documentation for stationarity, lag preparation, and scaling in one place (`docs/data_preprocessing.md`), including stage order and example usage.
- **Stronger script-driven evaluation flow**: grouped experiment execution is documented and structured around list-of-DataFrame input, per-case result matrices, and a consolidated `summary.csv` with execution metrics.
- **More explicit API positioning**: distinction between orchestrator (`MultiTaskGrangerAPI`), fluent builder (`MultitaskGrangerBuilder`), and pairwise baseline (`SimpleGrangerAPI`) is now clearer in user-facing docs.
- **Improved config-first reproducibility story**: docs now better connect config loading, sweep expansion, and script execution into one reproducible pipeline.
- **Better discoverability of backend capabilities**: backend aliases and backend-specific component resolution are surfaced more clearly via direct links to backend/component documentation.

These improvements do not replace the long-term roadmap, but they significantly reduce onboarding friction and make the 2.x architecture easier to adopt in real testing workflows.


### Version Changes 2.2.0 (pip support and futher consitency updrades)

The 2.2.0 once again changes the folder structure of repository. Additional improvements in documentation and script usage. Changes allow:
- **Choosing computational unit in grouped experiments**: user can choose to calcutate on **GPU** or **CPU**
- **Easier contol of grouped experiment**: configuration of grouped experiment has been futher developt to allow better and more universal control
- **Development of saved parametes**: grouped experiments can be dane in two modes:
    - minimum: binary causality matrix + summary.csv
    - matrices: binary, p-values, F-test, sign + summary.csv
- **Pip support**: pyproject.toml has been added and module have been moved in to subfolder to allow pip install
- **Documentation update**: futher development of docs to allow easier introduction to new users and development of library
 
