from collections.abc import Callable
from .granger_results import CausalityMatrix, GrangerAnalysisResults
from .statistics import f_test_value
from ..core.exceptions import ResultsError

import numpy as np

Metric = Callable[..., CausalityMatrix]

_METRICS: dict[str, Metric] = {}


def register_metric(*names: str):
    def decorator(func: Metric) -> Metric:
        for name in names:
            if name in _METRICS:
                raise ResultsError(f"Metric '{name}' is already registered.")
            _METRICS[name] = func
        return func
    return decorator


def resolve_metric(metric: str | Metric) -> Metric:
    if callable(metric):
        return metric

    normalized = metric.strip().lower()
    try:
        return _METRICS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_METRICS))
        raise ResultsError(
            f"Unknown metric '{metric}'. Available metrics: {available}."
        ) from exc

@register_metric("f_statistic","f")
def f_statistic(
    results: GrangerAnalysisResults,
) -> CausalityMatrix:
    base_rss = results.base_rss_per_effect()
    values = np.empty((len(results.effects), len(results.causes)))

    for cause_col, cause in enumerate(results.causes):
        ref_rss = results.reference_rss_per_effect(cause)
        q = results.n_restrictions_for(cause)
        k_u = results.n_unrestricted_parameters()
        values[:, cause_col] = f_test_value(
            error_ref=ref_rss,
            error_base=base_rss,
            n_restrictions=q,
            n_unrestricted_parameters=k_u,
            n_samples=results.n_samples,
        )

    return results.make_matrix(values, name="f_statistic")


@register_metric("base_mse")
def base_mse(results: GrangerAnalysisResults) -> CausalityMatrix:
    mse = results.base_mse_per_effect()
    values = np.repeat(mse[:, None], len(results.causes), axis=1)
    return results.make_matrix(values, name="base_mse")