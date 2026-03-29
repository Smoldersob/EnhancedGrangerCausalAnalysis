import warnings
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .lag_selectors import BaseLagSelector, LagSelectionResult
from ...core.exceptions import (
    ColumnMismatchError,
    EmptyDataError,
    LagConfigurationError,
    LagPreparationError,
)
from ...core.lag_config import LagConfiguration
from ...utilities.validation import (
    validate_columns_present,
    validate_dataframe_list,
    validate_lag_bounds,
)

# ===========================================================================
# Free functions for parallel workers
# ===========================================================================

def _create_lagged_data(
    data: pd.DataFrame,
    min_lags: np.ndarray,
    max_lags: np.ndarray,
) -> np.ndarray:
    """
    Generate a lagged feature matrix from a single DataFrame.

    Each column of *data* is expanded into a contiguous block of delayed
    copies (``min_lags[k]`` … ``max_lags[k]``).  Rows with insufficient
    history (the first ``max(max_lags)`` rows) are dropped.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series (rows = time steps, columns = variables).
    min_lags : ndarray of shape (n_columns,)
        Minimum lag to include for each variable.
    max_lags : ndarray of shape (n_columns,)
        Maximum lag to include for each variable.

    Returns
    -------
    X : ndarray of shape (n_usable_rows, sum(max_lags - min_lags + 1))
        Lagged feature matrix.  The caller is responsible for aligning
        target values by dropping the first ``max(max_lags)`` rows.
    """
    x = data.values.copy()
    min_lags = np.asarray(min_lags, dtype=int)
    max_lags = np.asarray(max_lags, dtype=int)

    validate_lag_bounds(min_lags, max_lags)

    n_rows, n_vars = x.shape
    total_cols = int((max_lags - min_lags + 1).sum())
    X = np.empty((n_rows, total_cols), dtype=x.dtype)

    col_offsets = np.concatenate(
        [[0], (max_lags - min_lags + 1).cumsum(dtype=int)]
    )

    for k in range(n_vars):
        cols = []
        for lag in range(int(min_lags[k]), int(max_lags[k]) + 1):
            col = np.roll(x[:, k], lag)
            if lag > 0:
                col[:lag] = np.nan
            elif lag == 0:
                pass  # no shift needed
            cols.append(col)
        X[:, col_offsets[k]:col_offsets[k + 1]] = np.column_stack(cols)

    # Drop rows that contain NaNs (first max(max_lags) rows)
    valid_start = int(max_lags.max())
    return X[valid_start:]


def _drop_unusable_targets(
    data: pd.DataFrame,
    effects: List[str],
    n_drop: int,
) -> np.ndarray:
    """
    Align target variables with lagged features by dropping leading rows.

    Parameters
    ----------
    data : pd.DataFrame
        Full (un-lagged) time series.
    effects : list of str
        Column names used as targets.
    n_drop : int
        Number of leading rows to remove (equal to ``max(max_lags)``).

    Returns
    -------
    y : ndarray of shape (n_usable_rows, n_effects)
    """
    return data[effects].iloc[n_drop:].values


def _fit_selector_on_single(
    selector: BaseLagSelector,
    data: pd.DataFrame,
) -> LagSelectionResult:
    """
    Fit a lag selector on a single DataFrame (worker function).

    Parameters
    ----------
    selector : BaseLagSelector
        A lag selector instance (IC, CV, or VAR-based).
    data : pd.DataFrame
        Multivariate time series.

    Returns
    -------
    result : LagSelectionResult
    """
    return selector.fit(data.values)


# ===========================================================================
# LagEngine
# ===========================================================================

class LagEngine:
    """
    Orchestrates lag selection and lagged-feature construction for
    multivariate causal time series models.

    The engine follows a three-phase workflow:

    1. **Determine lag structure** — either use a fixed ``max_lag`` from
       ``config`` or delegate to a :class:`BaseLagSelector` that returns
       per-pair optimal lags.
    2. **Apply overrides** — ``custom_lags`` and ``custom_pair_lags`` from
       ``config`` modify the lag structure after selection.
    3. **Build features** — each DataFrame in *data_list* is expanded into
       lagged columns in parallel, then all are concatenated.

    Parameters
    ----------
    config : LagConfiguration, default LagConfiguration()
        Lag structure configuration (max_lag, zero-lag flag, overrides).
    selector : BaseLagSelector or None, default None
        If provided, lags are determined automatically by this selector.
        If ``None``, the fixed ``config.max_lag`` is used uniformly.
    n_jobs : int, default -1
        Number of parallel jobs for feature construction and selector
        fitting.  ``-1`` means all available cores.

    Attributes
    ----------
    lag_order_ : dict
        Fitted lag order with keys ``"min"`` and ``"max"``, each an
        ndarray of shape ``(n_features,)``.
    selection_result_ : LagSelectionResult or None
        The raw selector output (available only when a selector was used).
    mask_ : ndarray or None
        Binary weight mask from the selector (shape depends on selector).

    Examples
    --------
    Fixed uniform lag (no selector):

    >>> cfg = LagConfiguration(max_lag=6, use_lag_zero=True)
    >>> engine = LagEngine(config=cfg)
    >>> X, y, col_idx = engine.prepare(data_list, effects=["y1"])

    Automatic per-pair selection:

    >>> from lag_selectors import ICLagSelector
    >>> sel = ICLagSelector(max_lag=20, use_bic=True, n_jobs=-1)
    >>> cfg = LagConfiguration(max_lag=20, custom_lags={"x3": (2, 8)})
    >>> engine = LagEngine(config=cfg, selector=sel)
    >>> X, y, col_idx = engine.prepare(data_list, effects=["y1", "y2"])
    """

    def __init__(
        self,
        config: LagConfiguration = LagConfiguration(),
        selector: Optional[BaseLagSelector] = None,
        n_jobs: int = -1,
    ) -> None:
        self.config = config
        self.selector = selector
        self.n_jobs = n_jobs

        # Fitted state
        self.lag_order_: Optional[Dict[str, np.ndarray]] = None
        self.selection_result_: Optional[LagSelectionResult] = None
        self.mask_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        data_list: List[pd.DataFrame],
        effects: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Determine lags, build lagged features, and align targets.

        Parameters
        ----------
        data_list : list of pd.DataFrame
            One or more DataFrames sharing the same column schema.  Each
            DataFrame represents an independent segment (e.g. different
            training runs or time windows).
        effects : list of str
            Column names that serve as prediction targets.

        Returns
        -------
        X : ndarray of shape (n_total_usable, n_lagged_features)
            Concatenated lagged feature matrix.
        y : ndarray of shape (n_total_usable, n_effects)
            Concatenated aligned target matrix.
        col_index : ndarray of shape (n_features + 1,)
            Cumulative column offsets.  ``X[:, col_index[k]:col_index[k+1]]``
            contains the lagged block for the *k*-th variable.

        Raises
        ------
        ValueError
            If ``data_list`` is empty or DataFrames have inconsistent
            columns.
        """

        # Validation
        data_list, columns = validate_dataframe_list(
            data_list,
            require_same_columns=True,
            require_same_shape=True,
            allow_superset_columns=False,
            copy=False,
        )

        n_vars = len(columns)

        # Determine lag structure
        min_lags, max_lags = self._determine_lags(data_list, columns, effects)

        # Apply overrides from config
        min_lags, max_lags = self._apply_overrides(
            min_lags, max_lags, columns, effects
        )

        # Ensure  mask has correct dimensions
        total_cols = int((max_lags - min_lags + 1).sum())
        if self.mask_ is None:
            # No selector and no overrides triggered rebuild.
            # Build mask properly to handle autoregression with use_lag_zero.
            pred_lag_matrix = np.zeros((n_vars, n_vars), dtype=int)
            for i in range(n_vars):
                for j in range(n_vars):
                    pred_lag_matrix[i, j] = max_lags[j]
            self.mask_ = self._rebuild_mask(pred_lag_matrix, min_lags, max_lags)
        elif self.mask_.shape[1] != total_cols:
            # Safety net: if dimensions stil disagree after overrides
            # (should not happen with correct _apply_overrides, but
            # guards against future regressions).
            warnings.warn(f"Mask has {self.mask_.shape[1]} columns but expected {total_cols} after applying overrides.  Rebuilding mask with correct dimensions.",
                          stacklevel=2)
            if self._pred_lag_matrix is not None:
                self.mask_ = self._rebuild_mask(
                    self._pred_lag_matrix, min_lags, max_lags
                )
            else:
                self.mask_ = np.ones((n_vars, total_cols), dtype=int)

        # Store fitted lag order
        self.lag_order_ = {"min": min_lags.copy(), "max": max_lags.copy()}

        # Build lagged features in parallel
        X = self._build_features(data_list, min_lags, max_lags)

        # Align targets
        y = self._align_targets(data_list, effects, int(max_lags.max()))

        # Column index for downstream usage
        col_index = np.concatenate(
            [[0], (max_lags - min_lags + 1).cumsum(dtype=int)]
        )

        return X, y, col_index

    def _determine_lags(
        self,
        data_list: List[pd.DataFrame],
        columns: List[str],
        effects: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute min/max lag arrays depending on whether a selector is set.

        When a selector is used, the aggregated ``pred_lag_matrix`` is
        stored in ``self._pred_lag_matrix`` for later use by
        :meth:`_rebuild_mask`.

        Returns
        -------
        min_lags : ndarray of shape (n_vars,)
        max_lags : ndarray of shape (n_vars,)
        """
        n_vars = len(columns)

        # Default min lags: 1 if lag_zero is off, 0 if lag_zero is on
        if self.config.use_lag_zero:
            min_lags = np.zeros(n_vars, dtype=int)
        else:
            min_lags = np.ones(n_vars, dtype=int)

        self._pred_lag_matrix: Optional[np.ndarray] = None

        if self.selector is None:
            max_lags = np.full(n_vars, self.config.max_lag, dtype=int)
            return min_lags, max_lags

        # Configure the selector with engine-level settings
        self.selector.max_lag = self.config.max_lag
        if hasattr(self.selector, "n_jobs"):
            self.selector.n_jobs = self.n_jobs

        # Set target indices on the selector
        validate_columns_present(columns, effects, context="effects")
        target_indices = [columns.index(e) for e in effects]
        self.selector.target_indices = target_indices

        # Fit selector on each DataFrame in parallel
        results: List[LagSelectionResult] = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_selector_on_single)(self.selector, df)
            for df in data_list
        )

        # Aggregate across segments (take element-wise maximum)
        self.selection_result_ = results[0]  # keep first as reference

        if hasattr(results[0], "pred_lag_matrix"):
            # Per-pair selector (IC / CV) — aggregate pred_lag_matrix
            agg_pred_matrix = np.max(
                [r.pred_lag_matrix for r in results], axis=0
            )
            self._pred_lag_matrix = agg_pred_matrix.copy()
            max_lags = agg_pred_matrix.max(axis=0).astype(int)
        else:
            # Common-lag selector (VAR) — single scalar per predictor
            max_lags = np.max(
                [r.max_lags_per_pred for r in results], axis=0
            ).astype(int)
            # Build a uniform pred_lag_matrix for VAR case
            self._pred_lag_matrix = np.full(
                (n_vars, n_vars), int(max_lags.max()), dtype=int
            )

        # Build initial mask (will be rebuilt after overrides)
        self.mask_ = self._rebuild_mask(
            self._pred_lag_matrix, min_lags, max_lags
        )

        return min_lags, max_lags

    def _apply_overrides(
        self,
        min_lags: np.ndarray,
        max_lags: np.ndarray,
        columns: List[str],
        effects: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ``custom_lags`` and ``custom_pair_lags`` from configuration.

        Per-variable overrides are applied first, then per-pair overrides
        (which adjust the mask if available).

        Parameters
        ----------
        min_lags, max_lags : ndarray of shape (n_vars,)
            Current lag bounds.
        columns : list of str
            Column names.
        effects : list of str
            Target column names.

        Returns
        -------
        min_lags, max_lags : ndarray of shape (n_vars,)
            Updated lag bounds.
        """
        col_to_idx = {name: i for i, name in enumerate(columns)}

        # per-variable overrides
        for name, lags in self.config.custom_lags.items():
            if name not in col_to_idx:
                warnings.warn(
                    f"custom_lags key '{name}' not found in columns; skipping.",
                    stacklevel=2,
                )
                continue

            idx = col_to_idx[name]
            if len(lags) == 1:
                max_lags[idx] = int(lags[0])
            elif len(lags) == 2:
                min_lags[idx] = int(lags[0])
                max_lags[idx] = int(lags[1])
            else:
                raise LagConfigurationError(
                    f"custom_lags['{name}'] must have 1 (max_lag,) or "
                    f"2 (min_lag, max_lag) elements, got {len(lags)}"
                )

        # per-pair overrides
        # These only affect the mask (if available) — they narrow or widen
        # the lag window for a specific (target, predictor) pair.
        if self.config.custom_pair_lags and self.mask_ is not None:
            for (target_name, pred_name), lags in (
                self.config.custom_pair_lags.items()
            ):
                if target_name not in col_to_idx:
                    warnings.warn(
                        f"custom_pair_lags target '{target_name}' "
                        f"not in columns; skipping.",
                        stacklevel=2,
                    )
                    continue
                if pred_name not in col_to_idx:
                    warnings.warn(
                        f"custom_pair_lags predictor '{pred_name}' "
                        f"not in columns; skipping.",
                        stacklevel=2,
                    )
                    continue

                t_idx = col_to_idx[target_name]
                p_idx = col_to_idx[pred_name]

                pair_min = int(lags[0]) if len(lags) >= 2 else int(min_lags[p_idx])
                pair_max = int(lags[-1])

                # Ensure the global max_lag for this predictor covers the
                # pair-level requirement.
                if pair_max > max_lags[p_idx]:
                    max_lags[p_idx] = pair_max

                # Update the mask for this (target, predictor) pair.
                # The mask layout follows col_index offsets.
                col_index = np.concatenate(
                    [[0], (max_lags - min_lags + 1).cumsum(dtype=int)]
                )
                block_start = int(col_index[p_idx])
                block_end = int(col_index[p_idx + 1])
                block_len = block_end - block_start

                # Zero out the pair's block and re-enable the valid range
                if t_idx < self.mask_.shape[0]:
                    self.mask_[t_idx, block_start:block_end] = 0
                    # lags in the block are ordered min_lag .. max_lag
                    rel_start = max(0, pair_min - int(min_lags[p_idx]))
                    rel_end = min(block_len, pair_max - int(min_lags[p_idx]) + 1)
                    self.mask_[t_idx, block_start + rel_start:block_start + rel_end] = 1
        
        elif self.config.custom_pair_lags and self.mask_ is None:
            warnings.warn(
                "custom_pair_lags provided but no mask is available "
                "(no selector was used).  Per-pair overrides are ignored "
                "when operating in fixed-lag mode.",
                stacklevel=2,
            )

        return min_lags, max_lags
    
    def _apply_overrides(
        self,
        min_lags: np.ndarray,
        max_lags: np.ndarray,
        columns: List[str],
        effects: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ``custom_lags`` and ``custom_pair_lags`` from configuration.

        The method works in three passes:

        1. **Collect** all changes to ``min_lags`` / ``max_lags`` from both
           ``custom_lags`` (per-variable) and ``custom_pair_lags``
           (per-pair).  At this stage the mask is **not** touched.
        2. **Rebuild** the mask from scratch using the final
           ``min_lags`` / ``max_lags`` (via :meth:`_rebuild_mask`).
        3. **Apply per-pair restrictions** on the freshly built mask so
           that only the requested lag window is enabled for each
           ``(target, predictor)`` pair.

        Parameters
        ----------
        min_lags, max_lags : ndarray of shape (n_vars,)
            Current lag bounds (from Phase 1).
        columns : list of str
            Column names.
        effects : list of str
            Target column names.

        Returns
        -------
        min_lags, max_lags : ndarray of shape (n_vars,)
            Updated lag bounds.
        """
        has_custom_lags = bool(self.config.custom_lags)
        has_custom_pairs = bool(self.config.custom_pair_lags)

        if not has_custom_lags and not has_custom_pairs:
            return min_lags, max_lags

        col_to_idx = {name: i for i, name in enumerate(columns)}
        needs_mask_rebuild = False

        # per-variable overrides  (custom_lags)
        for name, lags in self.config.custom_lags.items():
            if name not in col_to_idx:
                warnings.warn(
                    f"custom_lags key '{name}' not found in columns; "
                    f"skipping.",
                    stacklevel=2,
                )
                continue

            idx = col_to_idx[name]
            if len(lags) == 1:
                max_lags[idx] = int(lags[0])
            elif len(lags) == 2:
                min_lags[idx] = int(lags[0])
                max_lags[idx] = int(lags[1])
            else:
                raise LagConfigurationError(
                    f"custom_lags['{name}'] must have 1 (max_lag,) or "
                    f"2 (min_lag, max_lag) elements, got {len(lags)}"
                )
            needs_mask_rebuild = True

        # per-pair overrides — update min/max lags only
        _pair_overrides: List[Tuple[int, int, int, int]] = []  # (t, p, lo, hi)

        if has_custom_pairs:
            for (target_name, pred_name), lags in (
                self.config.custom_pair_lags.items()
            ):
                if target_name not in col_to_idx:
                    warnings.warn(
                        f"custom_pair_lags target '{target_name}' "
                        f"not in columns; skipping.",
                        stacklevel=2,
                    )
                    continue
                if pred_name not in col_to_idx:
                    warnings.warn(
                        f"custom_pair_lags predictor '{pred_name}' "
                        f"not in columns; skipping.",
                        stacklevel=2,
                    )
                    continue

                t_idx = col_to_idx[target_name]
                p_idx = col_to_idx[pred_name]

                if len(lags) == 1:
                    pair_min = int(min_lags[p_idx])
                    pair_max = int(lags[0])
                elif len(lags) == 2:
                    pair_min = int(lags[0])
                    pair_max = int(lags[1])
                else:
                    raise LagConfigurationError(
                        f"custom_pair_lags[{(target_name, pred_name)!r}] "
                        f"must have 1 or 2 elements, got {len(lags)}"
                    )

                # Widen the global bounds if the pair requires it
                if pair_max > max_lags[p_idx]:
                    max_lags[p_idx] = pair_max
                    needs_mask_rebuild = True
                if pair_min < min_lags[p_idx]:
                    min_lags[p_idx] = pair_min
                    needs_mask_rebuild = True

                _pair_overrides.append((t_idx, p_idx, pair_min, pair_max))

        # ==============================================================
        # Rebuild mask with final min/max lags
        # ==============================================================
        # The mask must be (re)built whenever:
        #   - min_lags / max_lags changed (needs_mask_rebuild), OR
        #   - pair overrides exist but the mask doesn't yet (no-selector
        #     mode where _determine_lags didn't create one).
        needs_mask_for_pairs = bool(_pair_overrides) and self.mask_ is None

        if (needs_mask_rebuild or needs_mask_for_pairs):
            n_vars = len(columns)
            if self._pred_lag_matrix is not None:
                # Update pred_lag_matrix columns to match new max_lags.
                plm = self._pred_lag_matrix.copy()
                for j in range(plm.shape[1]):
                    plm[:, j] = np.where(
                        plm[:, j] > 0,
                        np.minimum(plm[:, j], int(max_lags[j])),
                        plm[:, j],
                    )
                self.mask_ = self._rebuild_mask(plm, min_lags, max_lags)
            else:
                # No selector was used - build mask with proper autoregression handling.
                pred_lag_matrix = np.zeros((n_vars, n_vars), dtype=int)
                for i in range(n_vars):
                    for j in range(n_vars):
                        pred_lag_matrix[i, j] = max_lags[j]
                self.mask_ = self._rebuild_mask(pred_lag_matrix, min_lags, max_lags)

        # ==============================================================
        #  Per-pair mask restrictions implementation
        # ==============================================================
        if _pair_overrides and self.mask_ is not None:
            col_index = np.concatenate(
                [[0], (max_lags - min_lags + 1).cumsum(dtype=int)]
            )

            for t_idx, p_idx, pair_min, pair_max in _pair_overrides:
                block_start = int(col_index[p_idx])
                block_end = int(col_index[p_idx + 1])
                block_len = block_end - block_start

                if t_idx >= self.mask_.shape[0]:
                    continue

                # Zero out the entire block for this
                self.mask_[t_idx, block_start:block_end] = 0

                # Re-enable only the requested sub-range.
                # Columns in the block correspond to lags
                # min_lags[p_idx] .. max_lags[p_idx] (left to right).
                global_min = int(min_lags[p_idx])
                rel_start = max(0, pair_min - global_min)
                rel_end = min(block_len, pair_max - global_min + 1)
                self.mask_[
                    t_idx,
                    block_start + rel_start : block_start + rel_end,
                ] = 1

        min_lags = np.minimum(min_lags, max_lags)

        return min_lags, max_lags

    
    def _rebuild_mask(
        self,
        pred_lag_matrix: np.ndarray,
        min_lags: np.ndarray,
        max_lags: np.ndarray,
    ) -> np.ndarray:
        """
        Build (or rebuild) the binary weight mask from scratch.

        This is the single source of truth for mask construction inside
        ``LagEngine``.  It is called both after initial selector fitting
        and after any override that changes ``min_lags`` / ``max_lags``.

        The mask layout matches the lagged design matrix produced by
        :func:`_create_lagged_data`: for each predictor *j* there is a
        contiguous block of ``max_lags[j] - min_lags[j] + 1`` columns
        ordered from ``min_lags[j]`` to ``max_lags[j]``.

        When ``use_lag_zero`` is True (and ``min_lags[j] == 0``), the first
        column of each block is the lag0 (current) value. For autoregression
        (i == j), this column is set to 0 (forbidden); for external predictors
        (i != j), it is set to 1 (allowed).

        Parameters
        ----------
        pred_lag_matrix : ndarray of shape (n_targets, n_features)
            Maximum lag per (target, predictor) pair.  ``plm[i, j] = L``
            means predictor *j* is used for target *i* with lags up to
            *L*.  ``plm[i, j] = 0`` means predictor *j* is unused for
            target *i*.
        min_lags : ndarray of shape (n_features,)
            Per-predictor minimum lag (inclusive).
        max_lags : ndarray of shape (n_features,)
            Per-predictor maximum lag (inclusive).

        Returns
        -------
        mask : ndarray of shape (n_targets, total_cols)
            Binary mask.  ``mask[i, k] = 1`` means the corresponding
            weight may be optimised; ``0`` means it must stay zero.
        """
        min_lags = np.asarray(min_lags, dtype=int)
        max_lags = np.asarray(max_lags, dtype=int)
        pred_lag_matrix = np.asarray(pred_lag_matrix, dtype=int)

        n_targets, n_features = pred_lag_matrix.shape
        block_widths = max_lags - min_lags + 1
        total_cols = int(block_widths.sum())
        col_offsets = np.concatenate([[0], block_widths.cumsum(dtype=int)])

        mask = np.zeros((n_targets, total_cols), dtype=int)

        for j in range(n_features):
            blk_start = int(col_offsets[j])
            blk_width = int(block_widths[j])
            mn_j = int(min_lags[j])

            for i in range(n_targets):
                L = int(pred_lag_matrix[i, j])
                if L <= 0:
                    continue

                # When use_lag_zero is True and min_lags[j] == 0,
                # the block starts with lag0 (current value).
                # For autoregression (i == j), lag0 must be forbidden.
                # For external predictors (i != j), lag0 is allowed.
                if self.config.use_lag_zero and mn_j == 0:
                    # First column in block is lag0
                    if i != j:
                        # External predictor: allow lag0
                        mask[i, blk_start] = 1
                    # For i == j (autoregression), lag0 stays 0 (forbidden)

                    # Enable lagged terms (lag1, lag2, ..., lagL)
                    lag_lo = max(1, mn_j)  # start from lag1 at minimum
                    lag_hi = min(L, int(max_lags[j]))
                    
                    if lag_hi >= lag_lo:
                        rel_lo = lag_lo - mn_j
                        rel_hi = lag_hi - mn_j + 1
                        mask[i, blk_start + rel_lo : blk_start + rel_hi] = 1
                else:
                    # Standard case (use_lag_zero=False or min_lags[j] > 0)
                    lag_lo = mn_j
                    lag_hi = min(L, int(max_lags[j]))

                    if lag_hi < lag_lo:
                        continue

                    rel_lo = lag_lo - mn_j
                    rel_hi = lag_hi - mn_j + 1
                    mask[i, blk_start + rel_lo : blk_start + rel_hi] = 1

        return mask

    def _build_features(
        self,
        data_list: List[pd.DataFrame],
        min_lags: np.ndarray,
        max_lags: np.ndarray,
    ) -> np.ndarray:
        """
        Build the lagged design matrix from all segments in parallel.

        Parameters
        ----------
        data_list : list of pd.DataFrame
        min_lags, max_lags : ndarray of shape (n_vars,)

        Returns
        -------
        X : ndarray of shape (n_total_usable, n_lagged_features)
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_create_lagged_data)(df, min_lags, max_lags)
            for df in data_list
        )

        non_empty = [r for r in results if r.size > 0]
        if non_empty:
            return np.concatenate(non_empty, axis=0)
        return np.empty((0, 0))

    def _align_targets(
        self,
        data_list: List[pd.DataFrame],
        effects: List[str],
        n_drop: int,
    ) -> np.ndarray:
        """
        Drop leading rows from targets so they align with lagged features.

        Parameters
        ----------
        data_list : list of pd.DataFrame
        effects : list of str
        n_drop : int
            Number of rows to remove (= max(max_lags)).

        Returns
        -------
        y : ndarray of shape (n_total_usable, n_effects)
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_drop_unusable_targets)(df, effects, n_drop)
            for df in data_list
        )

        non_empty = [r for r in results if r.size > 0]
        if non_empty:
            return np.concatenate(non_empty, axis=0)
        return np.empty((0, 0))

    def get_feature_names(
        self,
        columns: List[str],
    ) -> List[str]:
        """
        Generate human-readable names for the lagged feature columns.

        Requires that :meth:`prepare` has been called first.

        Parameters
        ----------
        columns : list of str
            Original column names.

        Returns
        -------
        names : list of str
            Names like ``"temperature_lag3"``, ``"pressure_lag0"``, etc.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`.
        """
        if self.lag_order_ is None:
            raise LagPreparationError(
                "get_feature_names() requires a fitted engine; "
                "call prepare() first."
            )

        min_lags = self.lag_order_["min"]
        max_lags = self.lag_order_["max"]
        names: List[str] = []
        for k, col in enumerate(columns):
            for lag in range(int(min_lags[k]), int(max_lags[k]) + 1):
                names.append(f"{col}_lag{lag}")
        return names

    def summary(self, columns: List[str]) -> pd.DataFrame:
        """
        Return a summary table of the fitted lag structure.

        Parameters
        ----------
        columns : list of str
            Original column names.

        Returns
        -------
        summary : pd.DataFrame
            Table with columns ``variable``, ``min_lag``, ``max_lag``,
            ``n_features``.

        Raises
        ------
        RuntimeError
            If called before :meth:`prepare`.
        """
        if self.lag_order_ is None:
            raise LagPreparationError("Call prepare() first.")

        rows = []
        for k, col in enumerate(columns):
            mn = int(self.lag_order_["min"][k])
            mx = int(self.lag_order_["max"][k])
            rows.append({
                "variable": col,
                "min_lag": mn,
                "max_lag": mx,
                "n_features": mx - mn + 1,
            })
        return pd.DataFrame(rows)


# ===========================================================================
# Usage test
# ===========================================================================

if __name__ == "__main__":
    # Example 1: fixed uniform lag
    np.random.seed(42)
    n, d = 200, 4
    col_names = ["y1", "x1", "x2", "x3"]
    df1 = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    df2 = pd.DataFrame(np.random.randn(n, d), columns=col_names)

    cfg = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_lags={"x3": (0, 8)},
        custom_pair_lags={("y1", "x3"): (1, 3)},
    )
    engine = LagEngine(config=cfg, n_jobs=1)
    X, y, col_idx = engine.prepare([df1, df2], effects=["y1"])

    print("=== Fixed lag mode ===")
    print(f"X shape:  {X.shape}")
    print(f"y shape:  {y.shape}")
    print(f"col_idx:  {col_idx}")
    print(engine.summary(col_names))
    print(engine.mask_)
    print()

    # Example 2: with a selector
    from .lag_selectors import ICLagSelector
    sel = ICLagSelector(max_lag=20, use_bic=True, n_jobs=-1)
    cfg2 = LagConfiguration(
        max_lag=20,
        custom_lags={"x3": (2, 10)},
        custom_pair_lags={("y1", "x2"): (1, 6)},
    )
    engine2 = LagEngine(config=cfg2, selector=sel, n_jobs=-1)
    X2, y2, ci2 = engine2.prepare([df1, df2], effects=["y1"])
    print(engine2.summary(col_names))
    print(engine2.mask_)
    print(engine2.selection_result_)