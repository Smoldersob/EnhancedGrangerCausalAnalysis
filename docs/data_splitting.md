# Data splitting — implementation on top of version_2.7

This document describes the data-splitting implementation with two groups of
methods: **before preprocessing** (source level, `stage="source"`) and
**after lag selection** (prepared-row level, `stage="prepared"`). The code is
complete and tested (164 tests: 144 pre-existing + 20 new).

## Naming

| Group | Registered name | Aliases | Description |
|---|---|---|---|
| before preprocessing | `dataset_level` | `by_dataset`, `whole_datasets` | whole datasets are assigned to one part each |
| before preprocessing | `temporal_blocks` | `source_temporal`, `per_series_blocks` | each dataset is divided into contiguous time blocks (training prefix + validation suffix), proportions in the order the parts are given |
| after lag selection | `random_rows` | `prepared_random` | global random assignment of completed rows; warns about information leakage |
| after lag selection | `stratified_random_rows` | `by_series_random`, `per_dataset_random` | random assignment within each series, preserving part proportions; warns about information leakage |

Both `prepared`-stage splitters emit a warning when constructed:

> `'<name>' splits completed lagged rows after stationarity and lag selection
> were fitted on the full sample. Validation under this split may be
> optimistic because preprocessing decisions already saw the validation rows
> (information leakage). Use a source-stage splitter for validation that is
> independent of preprocessing.`

## Contracts

- **Registry and decorators** — the new module `preprocessing/splitting/`
  uses `core.registry.Registry`, mirroring the scaler and lag-selector
  families: `register_data_splitter(*names)` (decorator, first name is
  canonical), `resolve_data_splitter(spec, **params)` (accepts `None`, a
  registered name, a `{type, params}` mapping, a class, or an instance —
  instances are deep-copied), `available_data_splitters()`,
  `get_data_splitter_class(name)`.
- **Part definition lives outside preprocessing** — part names, sizes
  (`parts`: fractions or counts of rows/datasets) and `random_state` are
  parameters of the splitter in the user configuration
  (`DataPreparationConfig.splitter`), not something the preprocessing
  pipeline computes. The pipeline never derives part boundaries from its own
  state; it only consumes the fitted splitter (`select_for_fitting` frames
  and the resulting `DataSplit`), plus the conventional `"training"` part
  name.
- **`select_for_fitting`** — `source`-stage splitters expose the frames used
  to fit preprocessing decisions (imputation, redundancy, stationarity, lag
  selection). Each contiguous training block is returned as its own frame;
  scattered rows are never concatenated into a new series. For `prepared`-stage
  splitters the training data for fitting is `DataSplit.training_indices`.
- **The splitter owns the final split** — `splitter.split(lagged)` returns a
  `DataSplit` (part → row indices into `X_raw`/`y_raw`); `DataPreparer` only
  stores the result in `PreparedData.split`. Splitting is not a method of
  `DataPreparer`.
- **Fit on training data, transform on everything** — `imput()`,
  `force_stationarity()` and `remove_redundant()` fit their decisions on
  training frames and apply them to the full series; `scale()`/`block_scale()`
  fit the scalers on training rows and transform all rows. Without a splitter
  (`None`) everything is fitted on the full data — the previous behavior is
  preserved.
- **Row correspondence** — `source`-stage splitters record the source index
  labels; `LagSegmentInfo.retained_index` maps prepared rows back to source
  rows, so the plan survives `dropna` after differencing. Unique index labels
  are required (otherwise `DataValidationError` with a `reset_index()` hint).
- **`reduce_workload()`** — with an active split it explicitly rejects actual
  row removal (which would invalidate part indices); the no-op case (keeping
  all rows) remains allowed.

## `LagEngine` split into fit and transform

- `LagEngine.fit(datasets, outputs=..., predictors=...)` — validation,
  (parallel, per-dataset) selection, aggregation, overrides, layout
  construction; returns a `LagFittedConfig` (no data) and stores it in
  `engine.fitted_config_`.
- `LagEngine.transform(datasets)` — builds `LaggedData` (`X`, `y`,
  `segments`) from the full series using a single layout; requires a prior
  `fit()`, validates the schema (columns required by the layout, finite
  values).
- `LagEngine.prepare(...)` — kept for compatibility: `fit` + `transform`
  returning the legacy `LagPreparationResult`.
- `LagPreparationResult` remains in `core/lag_config.py` as a compatibility
  container; the new contract is the pair `LagFittedConfig` + `LaggedData`.
- In `DataPreparer.lag()`: selection runs on the training frames
  (`select_for_fitting`), matrices are built from the full series
  (`transform`), and `PreparedData.lag_result` is replaced by
  `PreparedData.lag_fitted`.

## Split stability when preprocessing is repeated (HPO)

Hyperparameter optimization may re-run preprocessing — for example when the
scaler type changes — while the split must stay unchanged. The design
guarantees this as follows:

1. **The splitter is resolved from configuration, not from preprocessing
   state.** `DataPreparer.__init__` resolves `config.splitter` once; the
   instance (with its `parts`, `random_state`, etc.) is independent of any
   preprocessing parameters. Re-running the pipeline with a different scaler
   spec re-uses the same splitter configuration.
2. **The plan is fitted on the source data only.** For `source`-stage
   splitters the plan is recorded in `set_data()` from the raw frames
   (dataset geometry, index labels, block boundaries). Nothing in
   `imput()`, `remove_redundant()`, `force_stationarity()` or `lag()` can
   change it. A re-run with different preprocessing therefore reproduces an
   identical plan, because the plan depends only on the source data and the
   splitter parameters.
3. **`prepared`-stage splitters are reproducible.** `random_rows` and
   `stratified_random_rows` draw their assignment from
   `np.random.default_rng(random_state)`; with the same seed and the same
   row count they return identical `DataSplit`s. `fit(n_rows)` additionally
   guards against a silently changed row count.
4. **What can invalidate the split:** changing the source data, the splitter
   parameters, or — for `prepared`-stage splitters — anything that changes
   the number of prepared rows (e.g. a different lag selector or
   stationarity settings). Changing the scaler, model parameters, regularizers
   or callbacks does not affect the split: scalers are re-fitted on the same
   `split.training_indices`, and `X_scaled`/`y_scaled` are simply re-derived
   from the unchanged `X_raw`/`y_raw`. In an HPO trial loop the evaluator can
   therefore rebuild preprocessing per candidate and slice the same part
   indices every time.

Caveat: with an active split, `reduce_workload()` refuses to remove rows, so
workload reduction and splitting cannot be combined in the first version.

## From `X` and `y` to the final part assignment

The final artifact is index-based; no physical copies of `X_train`/
`X_validation` are created. The flow:

1. **Source plan** — `set_data()` calls `splitter.fit(data_list)`. The
   splitter records, per dataset, the index labels and the block plan
   (`temporal_blocks`: fractions of each dataset in part order;
   `dataset_level`: dataset positions per part).
2. **Fitting decisions on training rows** — each fitting stage calls
   `select_for_fitting(self.data_list)`, which returns the training blocks
   (frames sliced by label membership). The imputer, redundancy detection,
   the stationarity enforcer and lag selection are fitted on those frames;
   transforms are applied to all frames, preserving the index labels.
3. **Lagged matrices** — `engine.fit(selection_frames)` produces the layout;
   `engine.transform(self.data_list)` builds `X` `(n_rows × n_features)` and
   `y` `(n_rows × n_outputs)`, concatenated across datasets. Each
   `LaggedData.segments` entry records `prepared_row_slice` (the global row
   range of that dataset) and `retained_index` (the source labels of the
   prepared rows, after leading rows were dropped for lag history).
4. **Assignment** — `splitter.split(lagged)`:
   - `source`-stage: each prepared row is mapped back to its source position
     via `index.get_indexer(retained_index)`, then to its part via the block
     plan (`searchsorted` over block boundaries). Global row indices are
     collected per part and sorted.
   - `prepared`-stage: the assignment is drawn directly on the global row
     array (`random_rows`: one permutation; `stratified_random_rows`: one
     permutation per segment, merged and sorted).
5. **Result** — `PreparedData.split` is a `DataSplit`: a mapping such as
   `{"training": idx_train, "validation": idx_val}` where the arrays are
   strictly increasing positional indices into `X_raw`/`y_raw`. Downstream
   consumers (model fitting, an HPO evaluator) slice with those indices:
   training matrices are `X_raw[split.training_indices]` (or the scaled
   counterpart), the scalers are fitted on exactly those rows, and
   `X_scaled`/`y_scaled` are transforms of all rows.

A prepared row belongs to the part of its target row (`y`). Because lag
features only use past values, this keeps the training/validation boundary
clean: validation targets never enter training, while training rows near the
boundary may legitimately use source values that lie in the training part.
Test parts are supported by simply adding a third entry to `parts` (block
order = the order the parts are given).

## Integration

- `DataPreparationConfig` gains a `splitter: Any = None` field;
  `DataPreparer` resolves the splitter in `__init__` and fits it in
  `set_data()` (`source.fit(data_list)` for `stage="source"`;
  `prepared.fit(n_rows)` is called inside `lag()` before `split()`).
- `PreparedData` gains `lag_fitted: LagFittedConfig | None` and
  `split: DataSplit | None`; `DataPreparer.split` and
  `PreparationContext.split` expose the split downstream.
- The functional `prepare_data(...)` gains a `splitter` parameter.

Example declarations:

```yaml
splitter:
  type: temporal_blocks
  params:
    parts:
      training: 0.7
      validation: 0.3
```

```python
DataPreparationConfig(
    lag_config={"max_lag": 3},
    stationarity="adf",
    splitter={"type": "random_rows", "params": {
        "parts": {"training": 0.8, "validation": 0.2},
        "random_state": 42,
    }},
)
```

## File changes

### New files

- `enhanced_granger_analysis/preprocessing/splitting/__init__.py`
- `enhanced_granger_analysis/preprocessing/splitting/base.py` — `DataSplit`,
  `SourceBlock`, `BaseDataSplitter`, `SourceDataSplitter` (plan fit,
  `select_for_fitting`, `split` via label correspondence),
  `PreparedDataSplitter`, constant `TRAINING_PART = "training"`.
- `enhanced_granger_analysis/preprocessing/splitting/registry.py` — registry,
  decorator, resolver.
- `enhanced_granger_analysis/preprocessing/splitting/splitters.py` — the four
  implementations with `parts` (fractions or counts) and `random_state`.
- `enhanced_granger_analysis/tests/test_data_splitting_2_9.py` — 20 tests.

### Existing files — only the fragments to replace

The full fragments are in `podzial_danych_2_9.patch` (apply with
`git apply podzial_danych_2_9.patch`). In short:

1. `core/lag_config.py` — `LagFittedConfig` and `LaggedData` inserted before
   `LagPreparationResult`; the latter annotated as the compatibility
   container.
2. `preprocessing/lag/lag_engine.py` — `prepare` split into `fit`
   (selection + layout → `LagFittedConfig`, stored in `self.fitted_config_`)
   and `transform` (segment loop → `LaggedData`); `_validate_transform_inputs`
   added; compatible `prepare` kept.
3. `preprocessing/pipeline.py`:
   - import `DataSplit, resolve_data_splitter`; `splitter` field in
     `DataPreparationConfig`;
   - `__init__`: `self.splitter = resolve_data_splitter(self.config.splitter)`
     plus helpers `_source_splitter()` and `_splitter_training_frames()`;
   - `set_data()`: `source_splitter.fit(data_list)` for `stage="source"`;
   - `imput()` and `force_stationarity()`: `fit(training).transform(all)`;
   - `remove_redundant()`: detection on training frames
     (`detection_aligned`), application on full frames (`aligned_data`);
   - `lag()`: `engine.fit(selection_frames, ...)` +
     `engine.transform(self.data_list)`, `result.*` references replaced with
     `lagged.*`/`fitted.*`, commit of `prepared.lag_fitted = fitted` and
     `prepared.split = self.splitter.split(lagged)` (for the `prepared`
     stage: `splitter.fit(n_rows)` first);
   - `scale()`/`block_scale()`: shared `_fit_scale_pair()` — fit on training
     rows (`prepared.split.training_indices`), transform on all rows; without
     a splitter, legacy `fit_transform`;
   - `reduce_workload()`: rejection when `n_keep < n` and a split is active;
   - the `split` property.
4. `preprocessing/prepare_data.py` — `PreparedData`: `lag_result` →
   `lag_fitted`, `split` added (also in `empty()`); `prepare_data()` gains a
   `splitter` parameter.
5. `preprocessing/__init__.py` — `splitting` exported.
6. `results/granger_results.py` — `PreparationContext.split`.

## Validation

- 144 pre-existing tests (`test_prepare_data_regressions_2_3`,
  `test_staged_builder_pipeline_2_7`, `test_stationarity_2_8`,
  `test_scalers_2_7`) pass with unchanged no-splitter behavior.
- 20 new tests cover: registry resolution (None/name/mapping/class/instance,
  errors), `temporal_blocks` (prefix/suffix per dataset, stationarity fitted
  on training only, fraction sum), `dataset_level` (counts and explicit
  positions, assignment errors), `random_rows` and `stratified_random_rows`
  (leakage warning, `random_state` determinism, per-series proportions),
  scalers fitted only on training rows (training mean of `X_scaled` ≈ 0),
  `reduce_workload` rejection with an active split, the `LagEngine`
  `fit`/`transform` contract (`fit` required, schema validation, legacy
  `prepare`) and `DataSplit` validation.

## Relation to the HPO context

- The "after lagging, before scaling" path = `prepared`-stage splitters
  (stationarity and lag selection fitted on the full sample, warning about
  limited validation independence).
- The "before preprocessing is fitted" path = `source`-stage splitters
  (training prefix + validation suffix, decisions fitted on training,
  transforms applied to full series while preserving history; the lag engine
  selects on training fragments but builds matrices from the full series
  under one layout; the two parts are never lagged independently).
- `PreparedData.split` instead of physical copies of `X_train`/
  `X_validation` — indices suffice; existing `LagPreparationResult.segments`
  (now `LaggedData.segments`) and `sample_indices` are reused without
  duplicating provenance metadata.
- No second, conflicting `search.validation` configuration is introduced; its
  relation to the future HPO runner is to be settled during integration
  (planned path: the model-quality evaluator reads `PreparedData.split`).
