# Data Preprocessing

This document explains the components in the `preprocessing` frolder, their purpose,
and the data contract passed to the backend model.

## Module Scope

The `preprocessing` package contains:

- `preprocessing/prepare_data.py` and `preprocessing/pipeline.py`: the public
  preparation entry point and staged preparation pipeline.
- `preprocessing/stationarity`: stationarity testing and differencing.
- `preprocessing/lag`: lag selection and lagged feature matrix construction.
- `preprocessing/scaling`: deterministic scaling for inputs and targets.
- `preprocessing/missing` and `preprocessing/sampling`: optional imputation and
  sample-reduction strategies.

## Standard preparation order

`prepare_data(...)` creates a `DataPreparer` and executes:

1. Schema alignment.
2. Optional missing-value imputation.
3. Removal of constant and exactly duplicated variables.
4. Optional stationarity transformation.
5. Lagged input and target construction.
6. Input and target scaling.
7. Workload reduction through the configured sampling strategy.

Role assignment takes place before these stages.

The standard workflow fits scalers on the lagged arrays before workload
reduction. Consequently, reducing the workload does not reduce the data used
to fit those scalers.

The base model and all restricted runs use the same prepared observations and
effect ordering.

When the input is a sequence of DataFrames, each DataFrame is treated as an
independent time segment during lag preparation. Lagged features and target
values are aligned within that segment, the first rows lacking the maximum
required history are removed, and only then are rows concatenated in input
order. History is never carried from one segment into the next, and segment
sizes are not equalized. Missing values occurring inside a segment are handled
by the separate missing-value preprocessing stage.

This workflow does not automatically create training, validation, and test
partitions. Applications requiring held-out evaluation must define the split
and transformation-fitting policy explicitly.

## Defaults differ between entry points

| Setting | `prepare_data()` | `DataPreparationConfig` | Analysis builder |
| --- | --- | --- | --- |
| Schema policy | `error` | `error` | `union` |
| Stationarity transformer | `None` | `None` | Default `StationarityTransformer` |
| Sampling | `NoSampling` | `NoSampling` | `None`, resolved to `NoSampling` |
| Input scaler | `standard` | `standard` | `standard` |
| Target scaler | `standard` | `standard` | `standard` |

Use explicit settings when reproducing an experiment across entry points.

For schema alignment:

- `error` requires matching column names and order across segments.
- `union` aligns segments to the union of columns. Absent columns become
  missing values; alignment alone does not impute them.

## Optional transformations

For `DataPreparer`, omitting a stage argument uses its configuration.
Passing `None` explicitly disables the optional transformation:

```python
preparer.imput(None)
preparer.force_stationarity(None)
```

For example:

```python
from enhanced_granger_analysis.preprocessing.pipeline import (
    DataPreparer,
    DataPreparationConfig,
)

preparer = DataPreparer(
    DataPreparationConfig(
        stationarity_transformer=None,
        missing_value_imputer=None,
    )
)
```

The analysis builder stores a default `StationarityTransformer`.
Calling `.stationarity()` retains an enabled default transformation.

To disable stationarity explicitly:

```python
builder.stationarity(None)
```

The equivalent configuration setting is:

```python
builder.from_config({"stationarity_transformer": None})
```

For explicit no-op scaling, use `"identity"`:

```python
builder.scaling(x_scaler="identity", y_scaler="identity")
```

## Staged `DataPreparer` Example

Use `DataPreparer` when preprocessing should be inspected or modified before the
model is built. Its methods mutate one preparation object and return `self`, so
they can be chained or executed one at a time.

```python
import numpy as np
import pandas as pd

from enhanced_granger_analysis.preprocessing.missing import ConstantImputer
from enhanced_granger_analysis.preprocessing.pipeline import (
    DataPreparer,
    DataPreparationConfig,
)
from enhanced_granger_analysis.preprocessing.stationarity import (
    StationarityTransformer,
)

first = pd.DataFrame(
    {
        "cause": np.linspace(0.0, 1.0, 40),
        "context": np.sin(np.linspace(0.0, 4.0, 40)),
        "effect": np.linspace(1.0, 0.0, 40),
    }
)
second = first.copy()
second.loc[5, "context"] = np.nan

config = DataPreparationConfig(
    schema_policy="error",
    lag_config=LagConfiguration(max_lag=3, use_lag_zero=False),
    lag_config={
        "use_lag_zero": False,
        "aggregation": "union",
    },
    lag_selector={
        "type": "ic",
        "params": {
            "max_lag": 3,
            "use_bic": True,
        },
    },
    lag_n_jobs=-1,
    stationarity_transformer=StationarityTransformer(
        max_differencing_order=1,
        test_name="adf",
        alpha=0.05,
        dropna=True,
    ),
    missing_value_imputer=ConstantImputer(value=0.0),
    x_scaler="standard",
    y_scaler="standard",
)

preparer = DataPreparer(config=config)
preparer.set_data([first, second])
preparer.set_roles(
    causes=["cause", "context"],
    tested_causes=["cause"],
    effects=["effect"],
)

# These calls follow the same order as DataPreparer.fit().
preparer.align()
preparer.imput()
preparer.remove_redundant()
preparer.force_stationarity()
preparer.lag()
preparer.scale()
preparer.reduce_workload()

prepared = preparer.prepared_data_
print(preparer.get_report())
print(preparer.get_feature_map())
print(prepared.X_scaled.shape, prepared.y_scaled.shape)
print(prepared.cause_slices.get("cause"))
print(prepared.lag_result.removed_predictors)
```

Automatic lag preparation can remove an input entirely, so a requested
cause is not guaranteed to appear in `cause_slices`.

The complete lag-preparation result is available through
`prepared.lag_result`, including per-dataset selections, aggregated lags,
the final specification, the feature layout, and removed predictors.

`prepared_data_` can be passed directly to the analysis builder. This avoids
running preprocessing a second time:

```python
from enhanced_granger_analysis.api import MultiTaskGrangerBuilder

builder = MultiTaskGrangerBuilder(backend="sklearn").prepared_data(prepared)
builder.prepare_data()
builder.prepare_components()
builder.build_model({"max_iter": 100})
builder.fit_base()
builder.fit_restricted_models()
results = builder.get_results()
```

The builder keeps the prepared arrays, feature slices, masks, scalers, and
role metadata in the `PreparedData` object used by every model run.

Supplying `PreparedData` bypasses preprocessing inside the builder.
The supplied arrays, role metadata, feature mappings, masks, and fitted
scalers are therefore authoritative for that execution.

Changing preprocessing settings on the builder does not transform an already
supplied `PreparedData` object. Rebuild prepared data explicitly when changing
the preprocessing workflow.

## Lag-and-scaling example

The lag engine returns unscaled arrays and a structured feature layout.
Scaling is a separate operation.

```python
from enhanced_granger_analysis.preprocessing.lag import (
    LagEngine,
    LagEngineConfig,
    resolve_lag_selector,
)
from enhanced_granger_analysis.preprocessing.scaling import (
    StandardScaler,
)

selector = resolve_lag_selector(
    "ic",
    max_lag=12,
    use_bic=True,
)

engine = LagEngine(
    config=LagEngineConfig(
        use_lag_zero=False,
        aggregation="union",
    ),
    selector=selector,
    n_jobs=-1,
)

lagged = engine.prepare(
    data_stationary,
    predictors=["x1", "x2"],
    outputs=["x1", "x2"],
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

if lagged.X.shape:[1]
    X_scaled = x_scaler.fit_transform(lagged.X)
else:
    X_scaled = lagged.X.copy()

y_scaled = y_scaler.fit_transform(lagged.y)

print(X_scaled.shape, y_scaled.shape)
print(lagged.feature_predictorss)
print(lagged.feature_lags)
print(lagged.block_boundaries)
print(lagged.removed_predictors)
```

`data_stationary` is a nonempty sequence of numeric DataFrames prepared by
the preceding stages.

Direct engine callers supply a resolved selector and a `LagEngineConfig`.
The staged pipeline performs this resolution automatically.

### Structured lag-preparation result

`LagEngine.prepare()` returns `LagPreparationResult`.

| Field | Meaning |
| --- | --- |
| `X`, `y` | Unscaled lagged inputs and aligned targets |
| `output_names` | Target and mask-row order |
| `active_predictors` | Ordered inputs contributing features |
| `removed_predictors` | Inputs with no final lags |
| `feature_predictors` | Variable name for each feature column |
| `feature_lags` | Actual lag for each feature column |
| `block_boundaries` | Endpoints of active predictor blocks |
| `mask` | Output-by-feature coefficient eligibility |
| `final_lags` | Validated specification after overrides |
| `aggregated_lags` | Baseline before manual overrides |
| `selection_results` | Per-dataset lag specifications |
| `segments` | Source-dataset and prepared-row correspondence |

`PreparedData.lag_result` retains this result before subsequent scaling
and workload reduction. Its arrays and segment slices are not updated
by later pipeline sampling.

Use `PreparedData.sample_indices` to map sampled rows back to the
pre-sampling prepared rows.

Selector-specific diagnostic attributes belong to the selector instances.
The current engine result stores per-dataset lag specifications, not
complete diagnostic objects from its selector copies.

### Feature-lag metadata

The layout provides:

- `output_names`: output order used by `y` and mask rows;
- `predictor_names`: active predictor order;
- `feature_predictors`: variable name associated with each feature column;
- `feature_lags`: actual lag associated with each feature column;
- `block_boundaries`: endpoints of active predictor blocks;
- `mask`: allowed coefficients in output-by-feature order.

For example:

```python
feature_predictors = ("x1", "x1", "x2", "x2")
feature_lags = (1, 3, 0, 2)
block_boundaries = (0, 2, 4)
```

The first input contributes lags 1 and 3. The second contributes lags 0
and 2. Blocks are contiguous in the feature matrix even when their lag
values are not consecutive.

`feature_lags` must match the exact column order of both raw and scaled
inputs. Scaling changes values, not the feature layout.

The pipeline copies this metadata into `PreparedData.feature_lags`,
`cause_slices`, and `base_mask`. It also stores the complete engine result
as `PreparedData.lag_result`.

Externally supplied prepared data must contain consistent feature
metadata. Changing builder lag settings does not regenerate an externally
supplied `PreparedData` object.

### Feature-lag metadata

`feature_lags` records the actual lag of each column in `X_raw` and
`X_scaled`.

It must have shape `(n_features,)`, with entries matching the exact feature
column order.

Externally supplied prepared data used with regularization must include
valid `feature_lags` consistent with `cause_slices`, `base_mask`, and the
input dimensions.

The empty default field is not a substitute for valid feature metadata.
Changing builder lag settings does not repair or regenerate metadata in an
externally supplied `PreparedData` object.

## Stationarity

Files:

- `preprocessing/stationarity/transformer.py`
- `preprocessing/stationarity/tests.py`

Main class: `StationarityTransformer`

Key capabilities:

- Per-feature differencing order selection using:
  - ADF (`test_name="adf"`)
  - KPSS (`test_name="kpss"`)
- Search range: `0..max_differencing_order`.
- Optional exclusion of selected variables from differencing via `non_stationary`.
- Consistent application of fitted differencing orders across a list of DataFrames.
- `dropna=True/False` control after differencing.

API:

- `fit_stationarity(data_list)`
- `transform(data_list)`
- `fit_transform(data_list)`
- `prepare_static(...)` (backward-compatible path)

Helper functions:

- `static_adfuller_order(...)`
- `static_kpss_order(...)`
- `apply_differencing(series, order)`

## Lag processing

Lag selection is an auxiliary preprocessing heuristic. It proposes a
restricted set of lagged inputs for the final model, which may apply
different constraints, regularization, or architecture.

The available families are:

- Per-pair selection: `ICLagSelector` and `CVLagSelector`.
- Common-order selection: `CommonOrderICLagSelector` and
  `CommonOrderCVLagSelector`.
- Historical VAR selection: `NaiveVARLagSelector`, also available as
  `VARLagSelector` and registered as `"var"`.

Selectors return explicit `LagSpec` dictionaries. They do not construct
final data arrays or coefficient masks.

See [Lag Selection](lag_selection.md) for candidate construction,
scoring, pruning, numerical policies, historical compatibility, and
interpretation.

### Engine workflow

`LagEngine.prepare()`:

1. Validates input schema and configuration.
2. Runs selection independently for each dataset, optionally in parallel.
3. Validates each selector result.
4. Aggregates selections using union, intersection, or selection frequency.
5. Applies predictor-wide and pair-specific replacements.
6. Validates the final specification.
7. Identifies active and removed predictors.
8. Constructs one ordered set of feature identities.
9. Prepares each dataset independently and concatenates the aligned rows.

Without a selector, the baseline contains positive lags `1..max_lag`.
Cross-variable lag 0 is additionally included when enabled.

History never crosses dataset boundaries.

### Explicit lag sets

Override values contain actual lags:

- `(1, 3)` means exactly lags 1 and 3.
- `(0,)` means contemporaneous-only inclusion.
- `()` excludes the predictor.

Pair-specific overrides take precedence over predictor-wide overrides.
Own-variable lag 0 is forbidden.

### Feature construction

Each predictor contributes the union of its final lag sets across outputs.

Feature order follows requested predictor order and ascending lag order.
An output-by-feature mask determines which shared columns each equation
may use.

Inputs with no final lags for any output are omitted from the feature
matrix and reported as removed predictors. This decision occurs after
manual overrides.

A variable removed as a predictor can remain an output.

### Numerical policies

IC/CV auxiliary fitting permits rank-deficient designs. Candidate
admissibility depends on the scoring method.

The engine does not interpret numerical rank deficiency as a reason to
remove a signal or restore a negative causality result.

Historical VAR can still fail when none of its candidate orders produces
valid information criteria.

The engine represents zero-feature arrays, but scaling and backend
execution must separately support them.

## Scaling

File:

- `preprocessing/scaling/scalers.py`

All scalers implement:

- `fit_transform(data)`
- `transform(data)`
- `inverse_transform(data)`

Available implementations:

- `StandardScaler`: z-score (`(x-mean)/std`)
- `MinMaxScaler`: range scaling (default `[0,1]`)
- `RobustScaler`: median + IQR (outlier-robust)
- `MaxAbsScaler`: divides by max absolute value
- `IdentityScaler`: no-op transformation

Common behavior:

- Input validation to 2D arrays.
- Explicit fitted-state checks (`ScalerNotFittedError` when used before fit).
- `inverse_transform` to map outputs back to original scale.

## Standard and block-wise scaling

The standard `scale()` stage applies the configured scaler to the prepared
input array and independently scales the targets.

`block_scale()` is an alternative input-scaling stage. It constructs a
`BlockScaler` from `cause_slices`; target scaling remains separate.

```python
preparer.lag()
preparer.block_scale(x_scaler="standard", y_scaler="standard")
preparer.reduce_workload()
```

The standard `fit()` workflow calls `scale()`, not `block_scale()`.

## Workload reduction

`NoSampling` is the default strategy. To enable reduction, configure a
sampling strategy that selects a subset of observations.

For non-empty prepared data, the requested sample budget is calculated as:

```text
n_keep = max(1, floor(backend_sample_fraction * n_samples))
```

If `backend_max_samples` is set, it further limits that budget.
The configured strategy receives the requested budget and selects indices.

Selection must return valid, unique, strictly increasing integer indices.

Workload reduction is not a training/validation/test split.

The standard workflow samples after scaling. A manual workflow can sample
after lagging and before scaling, in which case the scalers are fitted on the
retained subset:

```python
preparer.lag()
preparer.reduce_workload()
preparer.scale()
```

Changing this order changes the data used to fit scaling parameters and should
be recorded as part of the experiment configuration.

## Integration with API

In the public preparation function and builder:

- `prepare_data(...)` returns a `PreparedData` object containing scaled and raw
  arrays, scalers, masks, role metadata, and feature slices;
- `MultiTaskGrangerBuilder.prepare_data()` stores that object and initializes
  the result context;
- `build_model()` passes the same scaled arrays to the backend's
  `initialize(X, targets=y)`;
- predictions are mapped back with
  `prepared_data.y_scaler.inverse_transform(...)`;
- `fit_base()` and `fit_restricted()` save snapshots directly to the result
  object.

This keeps preprocessing consistent across all supported backends (`pytorch`, `tensorflow`, `sklearn`).

## PreparedData contract

`PreparedData` connects preprocessing with model execution.

### Arrays and transforms

- `X_raw`: lagged input features before scaling.
- `y_raw`: aligned targets before scaling.
- `X_scaled`: model inputs after scaling.
- `y_scaled`: model targets after scaling.
- `x_scaler` and `y_scaler`: fitted scaling objects.
- `stationarity_transformer`: transformer used before lag construction,
  or `None`.
- `lag_engine`: engine used to prepare the data;
- `lag_result`: structured lag-preparation result, including the final
  specification, layout, selection provenance, and segment metadata;
- `base_mask`: final output-by-feature lag-activity mask.


`raw` means unscaled at this preparation stage. It does not necessarily mean
the original input values: stationarity transformation may already have been
applied.

Inverse target scaling reverses scaling, not differencing.

### Variable roles

- `requested_roles`: input, output, and tested-cause roles requested before
  redundancy removal.
- `effective_roles`: roles remaining after cleanup.
- `causes`: effective model inputs.
- `effects`: effective model outputs.
- `tested_causes`: effective inputs tested through restricted runs.

`tested_causes` must be a subset of `causes`.

An effect does not have to belong to `causes`. In that case, its lagged history
is not automatically included in the model inputs, and role assignment emits
a warning.

Omitted or empty role lists are currently interpreted as defaults during
initial role assignment. They should not be used to request an intentionally
empty role.

An empty effective role before lag selection remains an input/preparation
error under the staged pipeline's role checks.

Lag selection can subsequently exclude a predictor from every output.
The pipeline then removes it from effective `causes` and, where applicable,
effective `tested_causes`. Requested roles remain unchanged.

The variable is not removed from effective `effects` merely because it is
unused as an input.

If all tested causes are excluded, execution must skip restricted-model
fits and retain the preparation context for restoration. If all predictors
are excluded, the backend and scaling path must support a zero-feature
design; engine support alone does not guarantee end-to-end execution.

### Named input blocks

- `cause_slices`: maps each effective input name to its complete feature slice.
- `tested_cause_slices`: exposes mappings for effective tested causes.
- `predictor_slices`: compatibility alias; prefer `cause_slices` in new code.

Example:

```python
cause_slices = {
    "x1": slice(0, 2),
    "x2": slice(2, 4),
    "x3": slice(4, 6),
}
```

For three model inputs, this mapping contains three entries. Each slice
already stores both boundaries.

The equivalent positional boundary array is:

```python
block_boundaries =[0][2][4][6]
```

It contains four entries because adjacent boundaries define each block.

In general, a mapping for `n` inputs contains `n` slices, while a boundary
array contains `n + 1` endpoints.

Select an input block by name:

```python
block = prepared.cause_slices["x1"]
x1_features = prepared.X_scaled[:, block]
```

The order of `tested_causes` does not determine feature-block positions.

`PreparedData` does not expose `col_offsets` as its public mapping contract.
Lower-level lag-selection and component interfaces may still use positional
offsets internally.

### Removed variables

`removed_variables` maps names to `RemovedVariable` records:

- `name`;
- `reason`: `"constant"`, `"duplicate"`, or `"no_selected_lags"`;
- `removed_roles`;
- `retained_variable`, where applicable.

Constant and duplicate diagnostics operate on combined input segments.
Duplicate detection uses exact equality rather than a correlation threshold.

`"no_selected_lags"` records exclusion from the predictor role after
aggregation and manual overrides. Its removed roles include `"cause"`
and, when applicable, `"tested_cause"`, but not `"effect"`.

Lag removal metadata is merged with earlier redundancy records.

### Restoration of requested results

The result layer restores evaluated matrices to requested output and
tested-cause axes.

Lag-excluded tested causes receive the metric's `removed_signal_value`:
typically `1` for p-values, `0` for test statistics and signs, and `NaN`
when the metric does not define a restoration value.

Duplicate effects copy available representative rows. Duplicate tested
causes copy available representative columns, including a representative
restored as neutral after lag exclusion.

A representative that was never a requested/evaluated tested cause has
no result column to copy.

Restoration does not remove output rows for variables excluded only as
inputs. It does not fit additional models or perform an independent test
of the excluded relationships.

Inspect removal metadata to distinguish computed results from
structurally restored entries.

### How removal metadata is used by results

The preparation pipeline records removed variables but does not calculate
their causality results. The result layer uses this metadata when restoring
evaluated metric matrices to the requested roles.

Restoration is metric-dependent:

- newly added rows and columns use the metric's `removed_signal_value`;
- duplicate effect rows are copied from available fitted representatives;
- duplicate tested-cause columns are copied from available tested
  representatives.

A representative that was only a model input has no computed tested-cause
column to copy.

The sign metric uses the same restoration mechanism as other metrics, with
a fill value of `0`. Statistical metrics can define different fill values,
such as `1` for p-values or `0` for test statistics.

Restoration preserves the requested layout according to these rules.
It does not fit additional models or independently estimate signs for
removed variables.

### Observation tracking

`sample_indices` stores positions in the combined lagged arrays before
workload reduction. It is not a replacement for the original DataFrame index
and does not encode segment identifiers or timestamps.

Sampling updates raw arrays, already available scaled arrays, and
`sample_indices` using the same selected positions.

`prepared.lag_result.segments` preserves the connection between prepared
rows and the source datasets. Each segment records its original row count,
dropped leading history, prepared-row slice, and retained DataFrame index.

These slices describe the combined arrays before workload reduction.
After sampling, use `sample_indices` to map retained observations back
to those pre-sampling positions.
## When to Use Which Component

- `StationarityTransformer`: when data shows trend/non-stationarity and you need stable Granger inference.
- `LagEngine` + selector: when lag order should be data-adaptive.
- `LagEngine` without selector: when you need deterministic, fixed lag structure.
- `StandardScaler`: default choice for most workflows.
- `RobustScaler`: useful with strong outliers.
- `IdentityScaler`: when model/backend expects original scale.

## Related Documents

- [API Usage](api_usage.md)
- [Configuration File Usage](config_file_usage.md)
- [Script Usage](script_usage.md)
