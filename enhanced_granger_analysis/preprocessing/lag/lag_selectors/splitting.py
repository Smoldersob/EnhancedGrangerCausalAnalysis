"""Time-ordered cross-validation splitting without external dependencies."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _validate_integer(
    value: int,
    *,
    name: str,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to "
            f"{minimum}."
        )

    return int(value)


def time_series_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    test_size: int | None = None,
    gap: int = 0,
    max_train_size: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create forward-chaining training/validation splits.

    Parameters
    ----------
    n_samples
        Number of observations in the common candidate-comparison window.
    n_splits
        Number of validation folds. Must be at least 2.
    test_size
        Number of observations in each validation fold.
        If None, use n_samples // (n_splits + 1).
    gap
        Number of observations excluded immediately before each
        validation window.
    max_train_size
        Maximum training-window length. If None, training expands from
        the beginning. Otherwise, retain only the most recent training
        observations.

    Returns
    -------
    tuple
        Ordered (train_indices, validation_indices) pairs.

    Notes
    -----
    Validation windows are consecutive, non-overlapping, and anchored
    so that the final window ends at n_samples.

    Every training observation precedes its validation window.
    No randomization or shuffling is performed.

    This function validates split geometry, not regression feasibility.
    The scorer must check whether each training fold can fit a candidate.
    """
    n_samples = _validate_integer(
        n_samples,
        name="n_samples",
        minimum=1,
    )
    n_splits = _validate_integer(
        n_splits,
        name="n_splits",
        minimum=2,
    )
    gap = _validate_integer(
        gap,
        name="gap",
        minimum=0,
    )

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

        if test_size == 0:
            raise ValueError(
                "Too few observations for the requested number of "
                "CV folds."
            )
    else:
        test_size = _validate_integer(
            test_size,
            name="test_size",
            minimum=1,
        )

    if max_train_size is not None:
        max_train_size = _validate_integer(
            max_train_size,
            name="max_train_size",
            minimum=1,
        )

    first_validation_start = (
        n_samples - n_splits * test_size
    )

    if first_validation_start - gap <= 0:
        raise ValueError(
            "The requested CV configuration leaves no training "
            "observations in the first fold. Reduce n_splits, "
            "test_size, or gap, or provide more observations."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_index in range(n_splits):
        validation_start = (
            first_validation_start + fold_index * test_size
        )
        validation_stop = validation_start + test_size

        train_stop = validation_start - gap
        train_start = (
            0
            if max_train_size is None
            else max(0, train_stop - max_train_size)
        )

        train_indices = np.arange(
            train_start,
            train_stop,
            dtype=np.intp,
        )
        validation_indices = np.arange(
            validation_start,
            validation_stop,
            dtype=np.intp,
        )

        splits.append(
            (train_indices, validation_indices)
        )

    return tuple(splits)