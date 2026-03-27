from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

@dataclass
class LagSelectionResult:
    """
    Container for lag selection outputs.

    Parameters
    ----------
    ar_lags : ndarray of shape (n_features,)
        Selected autoregressive lag per variable. For selectors that do not
        distinguish AR lags explicitly (e.g. VAR-based), this is typically
        the common VAR order for all variables.
    pred_lag_matrix : ndarray of shape (n_features, n_features)
        Maximum lag per (target, predictor) pair. Element [i, j] gives the
        maximum lag L of predictor j used for target i, interpreted as:
        - 0  -> predictor j is not used for target i
        - L>0 -> lags 1..L of predictor j may be used for target i.
    max_lags_per_pred : ndarray of shape (n_features,)
        Maximum selected lag per predictor (column-wise maximum of
        pred_lag_matrix).
    col_offsets : ndarray of shape (n_features,)
        Starting column index of each predictor block in the lagged design
        matrix and mask. Block j has length:
        - max_lags_per_pred[j] (lags 1..L) if use_lag_zero is False
        - 1 + max_lags_per_pred[j] (lag0 + lags 1..L) if use_lag_zero is True.
    mask : ndarray of shape (n_features, n_total_features)
        Binary mask for linear weights A in a model Y = A @ X_lagged.T + B[:, None].
        Row i corresponds to target i, columns correspond to lagged features
        in the design matrix. mask[i, k] = 1 means that weight A[i, k] is
        allowed to be optimized; 0 means it should be constrained to zero.
    """

    ar_lags: np.ndarray
    pred_lag_matrix: np.ndarray
    max_lags_per_pred: np.ndarray
    col_offsets: np.ndarray
    mask: np.ndarray


@dataclass
class LagConfiguration:
    """
    Declarative configuration for the lag structure used by :class:`LagEngine`.

    This dataclass separates *what lags to use* from *how to compute them*.
    It is intentionally a plain value object with no behavior so that it can
    be serialized, logged, or passed between stages of a pipeline.

    Parameters
    ----------
    max_lag : int, default 12
        Global upper bound on lags.

        * When no selector is provided this value is used as the uniform
          maximum lag for every variable.
        * When a selector **is** provided this value serves as the upper
          search bound passed to the selector (``selector.max_lag``).

    use_lag_zero : bool, default False
        If ``True`` the design matrix includes the current (un-shifted)
        value of each predictor.  For the same-variable autoregressive
        component this is always forced to ``False`` (the target cannot
        appear on both sides of the equation without bias).

    custom_lags : dict[str, tuple[int, int]], default empty
        Per-variable lag range overrides applied **after** selection.
        Keys are column names, values are ``(min_lag, max_lag)`` tuples.
        A single-element tuple ``(max_lag,)`` is also accepted and
        interpreted as ``(default_min, max_lag)``.

        Example::

            {"temperature": (1, 8), "pressure": (3, 15)}

    custom_pair_lags : dict[tuple[str, str], tuple[int, int]], default empty
        Per-pair ``(target, predictor)`` lag overrides.  Same semantics as
        ``custom_lags`` but applied at the pair level.  This is the
        fine-grained control you mentioned - it is kept optional so that
        the common case stays simple.

        Example::

            {("y1", "x3"): (2, 10)}

    Notes
    -----
    Using a dataclass here (rather than ``__init__`` parameters on
    ``LagEngine``) has concrete advantages:

    * **Separation of concerns** — configuration is decoupled from the
      engine that consumes it; the same config can be shared across
      experiments or serialized to YAML/JSON.
    * **Immutability-friendly** — you can ``freeze`` the dataclass or
      simply treat it as a value object.
    * **IDE support** — auto-complete and type-checking work out of the box.

    If the configuration were inlined into ``LagEngine.__init__`` the
    constructor signature would grow with every new option, making the API
    harder to understand.
    """

    max_lag: int = 12
    use_lag_zero: bool = False
    custom_lags: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    custom_pair_lags: Dict[Tuple[str, str], Tuple[int, ...]] = field(
        default_factory=dict
    )