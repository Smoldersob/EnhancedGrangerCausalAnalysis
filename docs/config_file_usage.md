# Configuration File Usage

This document explains how to load a builder configuration file and pass it into the builder/orchestrator pipeline.

For the generic component spec format used by callbacks, regularizers, and constraints, see [components loading](components_loading.md).

## What the loader does

`BuilderConfigLoader` is responsible for normalization only. It does not run training and does not create the final analysis result.

It can:
- load JSON or YAML files,
- normalize `lag_config` into `LagConfiguration`,
- normalize relation constraints,
- validate and normalize backend aliases,
- preserve backend-specific callback specs for later resolution,
- normalize relation-based constraint declarations into the `relations` mapping,
- move backend defaults from a backend spec into `model_config` when present.

## Supported backend forms

```json
{
  "backend": "pytorch"
}
```

or

```json
{
  "backend": {
    "type": "pytorch",
    "params": {
      "loading_verbose": true
    }
  }
}
```

The backend spec form is useful when you want to keep loader-level settings close to the config file while still letting the backend strategy build the concrete runtime objects.

## Example configuration

```json
{
  "backend": "tensorflow",
  "reuse_data": true,
  "lag_config": {
    "max_lag": 8,
    "use_lag_zero": false
  },
  "callbacks": [
    {
      "type": "early_stopping",
      "monitor": "loss",
      "patience": 5
    }
  ],
  "relations": {
    "y->x1": {
      "zero": true
    },
    "y->x2": {
      "min_abs_sum": 0.2
    }
  },
  "model_config": {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam"
  }
}
```

## How to load and use it

```python
from complex_granger_analysis.api import BuilderConfigLoader, MultitaskGrangerBuilder

cfg = BuilderConfigLoader.load_file("config.json")
output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

You can also load the raw mapping first and then adapt it yourself if needed:

```python
from complex_granger_analysis.api import BuilderConfigLoader

cfg = BuilderConfigLoader.load_file("config.json")
```

Then pass the normalized mapping to `MultitaskGrangerBuilder.from_config(...)`.

## Constrainty

Constrainty dla analiz Grangera podaje się przez `relations`. To jest mapowanie par `(effect, cause)` na reguły ograniczeń, a loader dopuszcza także JSON-friendly zapis stringowy lub listę obiektów.

Przykłady:

```json
{
  "relations": {
    "y->x1": { "zero": true },
    "y->x2": { "min_abs_sum": 0.2 }
  }
}
```

oraz:

```json
{
  "relations": [
    { "effect": "y", "cause": "x1", "zero": true },
    { "effect": "y", "cause": "x2", "min_abs_sum": 0.2 }
  ]
}
```

To jest spójne z opisem w [components loading](components_loading.md): `relations` zawiera reguły, a backend strategy zamienia je na backend-native constraint object.

## `reuse_data`

If `reuse_data` is set to `true`, the orchestrator can keep the prepared data object on the output when the builder/orchestrator decides to preserve it. This is useful when you want to inspect or reuse preprocessing artifacts across repeated runs.

## Recommended workflow

- Keep the file focused on configuration data.
- Let the loader normalize the shape.
- Let the builder orchestrator create and train the model.
- Put backend-specific runtime behavior into the backend strategy, not the loader.
