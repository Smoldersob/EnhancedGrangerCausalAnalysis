# Builder Usage Guide (MultitaskGrangerBuilder)

Use the builder to compose the analysis pipeline step by step with a fluent interface.

## When to use the Builder
- When the pipeline has many optional stages and readability matters.
- When you frequently experiment with lag, scaling, regularization, or hyperoptimization settings.
- When you want to load most settings from a dictionary via from_config(...).

## Typical flow
1. Create a builder instance.
2. Provide data.
3. Configure optional stages.
4. Execute with fit() or run().

## Example
```python
from complex_granger_analysis.api import MultitaskGrangerBuilder

output = (
    MultitaskGrangerBuilder(backend="pytorch")
    .data(df)
    .variables(causes=["x1", "x2"], effects=["y"], tested_causes=["x1", "x2"])
    .lag(lag_config=my_lag_config, lag_selector=my_selector)
    .scaling(x_scaler="standard", y_scaler="standard")
    .backend_load(backend_sample_fraction=0.8, backend_max_samples=5000)
    .regularization(regularizer_spec={"type": "L1", "l1": 0.01})
    .hyperoptimization(
        state="model",
        config={
            "n_trials": 20,
            "param_grid": {
                "epochs": [20, 50],
                "learning_rate": [1e-3, 5e-4],
            },
        },
    )
    .model({"epochs": 100, "batch_size": 64})
    .fit()
)
```

## from_config(...)
The builder accepts a mapping with keys aligned to MultiTaskGrangerAPI.fit(...).

```python
cfg = {
    "backend": "pytorch",
    "x_scaler": "standard",
    "y_scaler": "standard",
    "model_config": {"epochs": 100, "batch_size": 64},
}

output = MultitaskGrangerBuilder().from_config(cfg).data(df).fit()
```

## Best practices
- Set variables(...) explicitly when analyzing only a subset of columns.
- Use backend_load(...) for fast exploratory runs.
- Start hyperoptimization with a small grid, then expand.
- Keep model(...) focused on backend-specific training parameters.

## Validation behavior
- Calling fit() without data(...) raises DataValidationError.
- The builder does not duplicate training logic; it forwards configuration to MultiTaskGrangerAPI.fit(...).
