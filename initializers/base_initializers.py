import numpy as np
from numpy.linalg import lstsq
from typing import Tuple

from ..core.exceptions import DataShapeError


class LinearARXInitializer:
    """
    Initializer for linear ARX models of the form:

        Y = A @ X_lagged.T + B[:, None]

    where:
        - Y has shape (n_targets, T_eff)
        - X_lagged has shape (T_eff, n_features_eff)
        - A has shape (n_targets, n_features_eff)
        - B has shape (n_targets,)

    The initializer uses:
        - zeros
        - random normal (Keras-style std based on fan_in)
        - OLS based on provided (Y, X_lagged) and an optional mask.

    Parameters
    ----------
    n_targets : int
        Number of target variables (rows of A).
    n_features_eff : int
        Number of effective input features (columns of X_lagged, columns of A).
    use_lag_zero : bool, default False
        If True, the first block column for each predictor corresponds to lag=0.
        This class does not build X_lagged itself, but when using OLS init we
        will explicitly zero-out weights for lag=0 columns after fitting.
    lag0_indices : array-like of int, optional
        Indices of columns in X_lagged (and A) that correspond to lag=0
        (current values) and should be forced to zero in OLS init.
        If None and use_lag_zero=True, you should provide them externally.
    """

    def __init__(self,
                 n_targets: int,
                 n_features_eff: int,
                 use_lag_zero: bool = False,
                 lag0_indices=None):
        self.n_targets = n_targets
        self.n_features_eff = n_features_eff
        self.use_lag_zero = use_lag_zero
        if lag0_indices is None:
            self.lag0_indices = None
        else:
            self.lag0_indices = np.asarray(lag0_indices, dtype=int)

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass
# ------------------------------------------------------------------
# ZEROS INITIALIZER
# ------------------------------------------------------------------
class ZerosInitializer(LinearARXInitializer):
    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize A and B with zeros.

        Returns
        -------
        A : ndarray of shape (n_targets, n_features_eff)
        B : ndarray of shape (n_targets,)
        """
        A = np.zeros((self.n_targets, self.n_features_eff), dtype=float)
        B = np.zeros(self.n_targets, dtype=float)
        return A, B

# ------------------------------------------------------------------
# RANDOM NORMAL INITIALIZER (KERAS-STYLE)
# ------------------------------------------------------------------
class RandomNormalInitializer(LinearARXInitializer):
    def __init__(self,*args, mean: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean

    def __call__(self,
                 mask: np.ndarray | None = None,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize A with random normal weights, Keras-style:

            std = sqrt(1 / fan_in)

        where fan_in = n_features_eff. B is initialized to zeros.

        Parameters
        ----------
        mean : float, default 0.0
            Mean of the normal distribution.

        Returns
        -------
        A : ndarray of shape (n_targets, n_features_eff)
        B : ndarray of shape (n_targets,)
        """
        fan_in = self.n_features_eff
        if fan_in <= 0:
            std = 1.0
        else:
            std = np.sqrt(1.0 / fan_in)

        A = np.random.normal(loc=self.mean, scale=std,
                             size=(self.n_targets, self.n_features_eff))
        if mask is not None:
            A = A * mask  # Apply mask to zero out disallowed weights
        B = np.zeros(self.n_targets, dtype=float)
        return A, B

# ------------------------------------------------------------------
# OLS-BASED INITIALIZER
# ------------------------------------------------------------------
class OLSInitializer(LinearARXInitializer):
    def __call__(self,
                 Y: np.ndarray,
                 X_lagged: np.ndarray,
                 mask: np.ndarray | None = None,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize A and B using ordinary least squares (OLS) for each target
        independently, optionally respecting a binary mask on weights, and
        then force lag=0 weights to zero if requested.

        The regression model per target i is:

            y_i = X_lagged @ a_i + b_i + noise

        with a_i being the i-th row of A and b_i the i-th element of B.

        Parameters
        ----------
        Y : ndarray of shape (T_eff, n_targets)
            Target matrix in time-major form (rows = time, columns = targets).
        X_lagged : ndarray of shape (T_eff, n_features_eff)
            Design matrix of lagged inputs (rows = time, columns = features).
        mask : ndarray of shape (n_targets, n_features_eff), optional
            Binary mask indicating which weights are allowed to be non-zero.
            - mask[i, j] = 1 -> weight A[i, j] can be estimated.
            - mask[i, j] = 0 -> weight A[i, j] is forced to zero.

            If None, all weights are allowed.

        Returns
        -------
        A : ndarray of shape (n_targets, n_features_eff)
        B : ndarray of shape (n_targets,)
        """
        Y = np.asarray(Y, dtype=float)
        X_lagged = np.asarray(X_lagged, dtype=float)
        T_eff, n_features = X_lagged.shape
        if n_features != self.n_features_eff:
            raise DataShapeError(
                f"X_lagged has {n_features} features, expected {self.n_features_eff}"
            )
        if Y.shape[0] != T_eff:
            raise DataShapeError("Y and X_lagged must have the same number of rows (time).")
        if Y.shape[1] != self.n_targets:
            raise DataShapeError(
                f"Y has {Y.shape[1]} targets, expected {self.n_targets}"
            )

        if mask is not None:
            mask = np.asarray(mask, dtype=int)
            if mask.shape != (self.n_targets, self.n_features_eff):
                raise DataShapeError(
                    f"mask shape {mask.shape} does not match "
                    f"(n_targets, n_features_eff)=({self.n_targets}, {self.n_features_eff})"
                )

        A = np.zeros((self.n_targets, self.n_features_eff), dtype=float)
        B = np.zeros(self.n_targets, dtype=float)

        # Perform OLS independently for each target i
        for i in range(self.n_targets):
            y_i = Y[:, i]

            if mask is None:
                # All features allowed
                X_i = X_lagged
                active_idx = None
            else:
                active = mask[i] == 1
                if not np.any(active):
                    # No active features: intercept-only model
                    B[i] = float(y_i.mean())
                    A[i, :] = 0.0
                    continue
                X_i = X_lagged[:, active]
                active_idx = np.where(active)[0]

            # Fit linear model with intercept: y_i = [1, X_i] @ beta
            n = X_i.shape[0]
            X_aug = np.column_stack([np.ones(n), X_i])
            beta, _, _, _ = lstsq(X_aug, y_i, rcond=None)

            b_i = beta[0]
            w_i = beta[1:]

            B[i] = float(b_i)

            if mask is None:
                # All features correspond to w_i in order
                A[i, :] = w_i
            else:
                # Only active features were fitted; map them back
                A_i = np.zeros(self.n_features_eff, dtype=float)
                A_i[active_idx] = w_i
                A[i, :] = A_i

        # Enforce zero weights for lag=0 columns if requested
        if self.use_lag_zero and self.lag0_indices is not None:
            A[:, self.lag0_indices] = 0.0

        return A, B
