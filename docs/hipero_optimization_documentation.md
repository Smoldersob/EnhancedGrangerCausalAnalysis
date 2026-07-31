# Hyperoptimization Documentation

This document explains how to define hyperoptimization settings for the Granger analysis orchestrator.

Hyperoptimization is performed as a sweep over candidate configurations. The optimizer evaluates each trial by building backend objects from the strategy, training a short-run model, and selecting the candidate with the best score.

## Core Idea

Hyperoptimization does not train one fixed model and then patch it afterward. Instead, it treats each trial as a complete configuration candidate and builds the trial-specific objects through the backend strategy.

The sweep can cover three parameter groups:

- `model` - model configuration parameters, such as `epochs`, `batch_size`, `learning_rate`, `max_iter`, `tol`, `optimizer`, and `loss`
- `optimizer` - optimizer-specific settings, if the backend uses a structured optimizer spec
- `regularizer` - regularizer parameters, such as `l1` or lag-dependent regularizer settings

## Configuration Shape

The preferred format is a sectioned sweep definition:

```json
{
  "hiperoptimalization_conf": {
    "type": "model",
    "n_trials": 6,
    "trial_epochs": 5,
    "sections": {
      "model": {
        "learning_rate": [0.01, 0.001],
        "batch_size": [16, 32]
      },
      "optimizer": {
        "type": ["adam", "sgd"]
      },
      "regularizer": {
        "l1": [0.0001, 0.001]
      }
    }
  }
}
```

### Supported Fields

- `type` or `mode` - selects the optimization target. Current supported values are `model` and `regularization`.
- `n_trials` - maximum number of trial combinations to evaluate.
- `sections` - preferred sweep container with `model`, `optimizer`, and `regularizer` subsections.
- `param_grid` - backward-compatible flat sweep definition.
- `trial_epochs` - explicit number of epochs used for trial training.
- `trial_train_config` - optional mapping merged into the short-run training configuration.

## Backward-Compatible Flat Form

A flat `param_grid` is still accepted. In that form, values are mapped using dotted keys or the default section for the selected state.

```json
{
  "hiperoptimalization_conf": {
    "type": "regularization",
    "n_trials": 4,
    "param_grid": {
      "regularizer.l1": [0.0001, 0.001, 0.01],
      "model.learning_rate": [0.01, 0.001]
    }
  }
}
```

If the optimizer is running in `model` mode and no section is specified, flat keys are treated as model parameters by default. In `regularization` mode, flat keys default to the regularizer section.

## Trial Training Budget

Each trial uses a reduced training budget so that sweep evaluation stays cheaper than the final fit.

The trial budget can be controlled in two ways:

- `trial_epochs` - direct override for the trial epoch count
- `trial_train_config` - extra trial-only overrides, such as `batch_size`, `max_iter`, or `learning_rate`

If these values are omitted, the optimizer derives a short-run budget from the base `model_config`.

Example:

```json
{
  "hiperoptimalization_conf": {
    "type": "model",
    "n_trials": 8,
    "trial_epochs": 3,
    "trial_train_config": {
      "batch_size": 16
    },
    "sections": {
      "model": {
        "epochs": [20, 50],
        "learning_rate": [0.01, 0.001]
      }
    }
  }
}
```

## Model, Optimizer, and Regularizer Sweeps

### Model sweep

Use the `model` section for parameters forwarded into the backend model constructor.

Common examples:

```json
{
  "model": {
    "epochs": [20, 50],
    "batch_size": [16, 32],
    "learning_rate": [0.01, 0.001]
  }
}
```

### Optimizer sweep

Use the `optimizer` section for optimizer type or optimizer configuration fields when the backend supports them.

Example:

```json
{
  "optimizer": {
    "type": ["adam", "sgd"]
  }
}
```

### Regularizer sweep

Use the `regularizer` section for regularizer parameters.

Example:

```json
{
  "regularizer": {
    "type": ["l1"],
    "l1": [0.0001, 0.001, 0.01]
  }
}
```

For lag-dependent regularizers, you can also sweep parameters such as `max_lags_per_pred` or `col_offsets` when the backend spec accepts them.

## Result Handling

The optimizer returns the best configuration update and the best regularizer specification update. The orchestrator then applies these updates before building the final runtime objects.

This means hyperoptimization changes:

- the final `model_config`
- the final `regularizer_spec`

It does not store a prebuilt final regularizer object in the result.

## Callbacks

Callbacks can be part of the trial configuration if they are passed through the model configuration or constructed by the run configuration factory.

A practical note:

- If the orchestrator injects a fixed callback template, that template is cloned per run and will usually dominate trial-specific callback settings.
- If you want to sweep callback settings, keep the callback definition in the sweep configuration rather than hard-coding a fixed callback list.

## Practical Recommendation

Use hyperoptimization for parameters that materially affect training cost or model quality:

- learning rate
- number of epochs
- batch size
- optimizer type
- regularizer strength

Keep the sweep small and focused. The optimizer is designed to compare a manageable set of candidate configurations rather than perform large-scale automated search.
