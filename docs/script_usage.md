# Running Group Granger Experiments

`enhanced_granger_analysis/scripts/run_group_causality_tests.py` reads datasets and explicitly configured cases, runs each analysis, exports matrices, and compares completed analysis output with ground truth.

The script may still call `MultiTaskGrangerBuilder`; full builder execution now delegates to the shared runner. This documentation does not assume an additional script rewrite or an internal HPO stage.

## Invocation

From the repository root:

```bash
python enhanced_granger_analysis/scripts/run_group_causality_tests.py --config enhanced_granger_analysis/scripts/run_group_causality_tests.config.json --save matrices --test f
```

- `--save minimum`: causality matrix and summary.
- `--save matrices`: causality, selected test's p-value/statistic, sign matrix, and summary.
- `--test f` / `wald` / `lr`: select reporting/decision metric.
- `--sign`: include coefficient-derived signs in thresholded results.

For the current Wald path the script skips restricted-model fits. This is a property of that selected metric path, not an assertion that every future Wald variant requires no restricted fitting. Matrix mode saves the selected statistic, not all three test statistics at once.

## Script document

```json
{
  "output_dir": "./results",
  "ground_truth_path": "./ground_truth.csv",
  "group_config_path": "./group_config.json",
  "threshold": 0.01,
  "data": {
    "csv_paths": ["./data.csv"],
    "index_col": 0
  }
}
```

Script paths are resolved relative to the script configuration document. These path rules are implemented by the script; `AnalysisConfigLoader` does not recursively reinterpret every path-like string.

## Group document

```json
{
  "device": "cpu",
  "base_config": {
    "backend": "numpy",
    "stationarity": null,
    "lag_config": {"max_lag": 2, "use_lag_zero": false},
    "relations": [],
    "model_config": {"epochs": 100}
  },
  "sweep": {
    "param_names": ["model_config.epochs"],
    "cases": [[100], [200]]
  }
}
```

Each row defines one experiment, not a Cartesian expansion. Dotted paths traverse mappings only. Replace whole callback/relation lists rather than indexing them. Group relation-file references are resolved relative to the group file after expansion.

Preferred imports for script maintenance:

```python
from enhanced_granger_analysis.configuration.loader import AnalysisConfigLoader
from enhanced_granger_analysis.experiments import TestGroupConfigIterator
```

Old loader/iterator import paths remain compatibility exports. The loader reads documents; it does not turn initializer strings into classes or modify runtime environment variables for devices. Those responsibilities belong to existing execution/backend paths.

## Pipeline

1. Read script and group documents.
2. Load CSVs as ordered DataFrames.
3. Expand explicit cases and resolve each case's document references.
4. Execute one analysis per case.
5. Evaluate selected metric and threshold decisions.
6. Export the requested matrices.
7. Compare the exported decisions with ground truth and write summary rows.

The existing script aligns multiple input column sets to their union and fills missing columns with zero before library preparation. This script behavior is separate from the preparation pipeline's schema/imputation policy and must be considered when interpreting an experiment.

## Outputs

Typical matrix-mode files:

```text
results/
  summary.csv
  case_000_..._causality.csv
  case_000_..._p_value.csv
  case_000_..._f_test.csv
  case_000_..._sign.csv
```

The statistic suffix is `f_test`, `wald_test`, or `lr_test` according to selection. Filename details depend on case index, backend declaration, and swept parameter values.

Current summary fields include `case_id`, `backend`, `test_type`, `causality_file`, `signed_result`, `p_value_file`, `test_file`, `sign_file`, `execution_time_seconds`, `restricted_models_fitted`, swept values, and ground-truth metrics. Failed cases record an error and failed file markers; subsequent cases continue.

A reported backend may be the requested declaration rather than the canonical actually selected runtime. Do not infer actual device placement from the filename or case label alone.

Sign is based on sums of full-model coefficients in cause blocks, not the single strongest lag. A row's order in `summary.csv` is case order, not an automatic ranking by accuracy.

## Prepared-data cache and timing

The script maintains its own preparation cache when `reuse_data` is requested. The runner has no cross-call cache. The existing script signature is lightweight and does not constitute a complete fingerprint of data contents and all preparation settings. Do not rely on it for arbitrary changes of causes, tested roles, stationarity, imputation, lag-selector parameters, sampling, or source contents without extending the cache key.

Timing excludes matrix file writes in the measured analysis interval. On cache hits the script may add recorded preparation time for reporting. Read summary timing as the script's accounting convention, not necessarily pure wall time per cached call.

## Ground truth and HPO

Ground-truth scoring measures analysis output after fitting. It does not choose hyperparameters inside a case. Existing HPO configuration is inactive. Future model-quality tuning must not consume causal F1/accuracy as its objective.

## Data requirements and interpretation

CSV columns name variables; rows preserve time order. Ground-truth rows represent effects and columns represent causes. See [Ground-truth evaluation](ground_truth_evaluation.md) for alignment and unknown-value semantics.

Ensure requested variables exist, data and references point to intended files, and selected backend dependencies/devices are available. Starting with one small case is useful for checking configuration; successful execution does not establish statistical validity or cross-backend equivalence.

See [Configuration](config_file_usage.md), [API](api_usage.md), [Backend usage](backend_usage.md), and [HPO status](hipero_optimization_documentation.md).
