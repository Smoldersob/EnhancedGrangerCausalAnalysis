from numbers import Integral

import numpy as np


def _nonnegative_finite(value: float, name: str) -> float:
    """Validate an absolute or relative non-negative tolerance."""
    result = float(value)

    if not np.isfinite(result) or result < 0:
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return result


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")

    return int(value)


def _augment_intercept(design: np.ndarray) -> np.ndarray:
    """Add an intercept, including for a zero-column design."""
    return np.column_stack(
        (np.ones(design.shape[0], dtype=float), design)
    )


def _fit_ols(
    y: np.ndarray,
    design: np.ndarray,
    *,
    rcond: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit an auxiliary OLS model, allowing redundant columns."""
    augmented = _augment_intercept(design)

    coefficients, _, rank, _ = np.linalg.lstsq(
        augmented,
        y,
        rcond=rcond,
    )

    predictions = augmented @ coefficients
    residuals = y - predictions

    if not (
        np.isfinite(coefficients).all()
        and np.isfinite(residuals).all()
    ):
        raise FloatingPointError(
            "Auxiliary OLS fit produced non-finite values."
        )

    return coefficients, residuals, int(rank)

def _integer(
    value: int,
    name: str,
    *,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}."
        )
    return int(value)

