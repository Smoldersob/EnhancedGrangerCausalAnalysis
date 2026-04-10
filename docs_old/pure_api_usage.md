# Pure API Usage Guide (MultiTaskGrangerAPI)

Pure API usage means calling MultiTaskGrangerAPI.fit(...) directly, without the builder layer.

## When to use pure API
- When you want all parameters in one explicit function call.
- When your pipeline configuration is stable and reused as-is.
- When writing wrappers or service functions with a single entry point.

## Typical flow
1. Create MultiTaskGrangerAPI with the selected backend.
2. Call fit(...) with data and configuration.
3. Read results from MultitaskGrangerOutput.

## Example
```python
from complex_granger_analysis.api import MultiTaskGrangerAPI

api = MultiTaskGrangerAPI(backend="pytorch")

output = api.fit(
    data=df,
    causes=["x1", "x2"],
    effects=["y"],
    tested_causes=["x1", "x2"],
    x_scaler="standard",
    y_scaler="standard",
    backend_sample_fraction=0.8,
    backend_max_samples=5000,
    regularizer_spec={"type": "L1", "l1": 0.01},
    hiperoptimalization_state="model",
    hiperoptimalization_conf={
        "n_trials": 20,
        "param_grid": {
            "epochs": [20, 50],
            "learning_rate": [1e-3, 5e-4],
        },
    },
    model_config={"epochs": 100, "batch_size": 64},
)

matrix = output.results.causality_matrix
```

## Output structure
- results: aggregated causality statistics and matrices.
- base_model: trained base model.
- reference_models: per-cause reference artifacts.
- stationarity_transformer, lag_engine, X_scaler, y_scaler: preprocessing objects used in the run.

## Best practices
- Provide causes/effects/tested_causes explicitly for partial analysis.
- Use hiperoptimalization_conf only when tuning is needed.
- Keep backend-specific parameters inside model_config.
- Control runtime cost with backend_sample_fraction and backend_max_samples.

## Pure API vs Builder
- Pure API: direct and compact for fixed configurations.
- Builder: clearer for iterative experimentation and staged setup.
