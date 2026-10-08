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


### Version Changes 2.2.0 (pip support and further consistency upgrades)

The 2.2.0 once again changes the folder structure of repository. Additional improvements in documentation and script usage. Changes allow:
- **Choosing computational unit in grouped experiments**: user can choose to calculate on **GPU** or **CPU**
- **Easier control of grouped experiment**: configuration of grouped experiment has been further development to allow better and more universal control
- **Development of saved parameters**: grouped experiments can be dane in two modes:
    - minimum: binary causality matrix + summary.csv
    - matrices: binary, p-values, F-test, sign + summary.csv
- **Pip support**: pyproject.toml has been added and module have been moved in to subfolder to allow pip install
- **Documentation update**: further development of docs to allow easier introduction to new users and development of library
 
### Version Changes 2.3.0
The 2.3.0 versions has added 3 major changes:
- **Keras/TensorFlow backend and modeling update**: in new version tensorflow backend has been replaced
with fully keras backend, in which all constraints, regularizers and model are described implemented 
from backend used by keras. This allows to you keras implementation with any keras backend:
tensorflow, pytorch or jax. It has to be remembered that it has to be set before using the library.
- **Intelligent data preparation**: code responsible for data preparation has been partially rewritten.
The most important change and addition is checking usability of signal by checking if they have constant
values or are copies of other signals. These columns have to be handled in special way if they are 
part of analyzed causes or effects.
- **Results and metrics handling**: the way results of test are analyzed has been change and upgraded.
New implementation returns more comprehensive information about the values, model and analyzed signals.
New entities `metrics` have been added, so the in new implementation results are analyzed using them.
Metrics have their registry, but it is also possible now to add custom metrics and used them on test 
results, which is supposed to help expand future. 

There were also few smaller changes:
- hiper optimization method have been separated and slightly upgraded
- calculating constrained parameters have been updated and made a little customizable
- results saved by using script have changed - minimum option save only results after thresholding and summary table, 
while matrices also save matrices of metric and p-value. New parameter `with_sign` was added to the script to allow
saving and using unsigned relation matices.