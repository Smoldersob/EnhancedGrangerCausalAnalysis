"""Degrees-of-freedom contracts and registered strategies.

A strategy receives a DofContext and returns a DofResult. Method-specific
numerical options (for example ``rank_rcond`` or ``active_support_tolerance``)
travel inside ``DofContext.params``; the context has no dedicated fields for
individual built-in strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import ResultsError
from ..core.registry import Registry
from .statistics import nested_design_geometry


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DofContext:
    """Inputs for one cause comparison.

    Arrays follow the effective output and feature order. The design loader
    is evaluated only when a strategy needs it. ``params`` carries arbitrary
    strategy options supplied by the caller.
    """

    n_samples: int
    effects: tuple[str, ...]
    cause: str
    unrestricted_mask: BoolArray
    restricted_mask: BoolArray
    bias_enabled: bool
    unrestricted_weights: FloatArray | None
    load_features: Callable[[], FloatArray]
    params: dict[str, Any] = field(default_factory=dict)
    unrestricted_metadata: dict[str, Any] = field(default_factory=dict)
    restricted_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DofResult:
    """Per-output dimensions and diagnostics.

    Unknown dimensions may be NaN. Finite dimensions must be nonnegative.
    Residual degrees of freedom may be nonpositive. Values are not coerced
    to integers, so regularized or fractional-dimension strategies remain
    expressible.
    """

    model_df_u: FloatArray
    model_df_r: FloatArray
    restriction_df: FloatArray
    residual_df: FloatArray
    status: tuple[str, ...]
    heuristic: bool = False
    unrestricted_mask: BoolArray | None = None
    restricted_mask: BoolArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


DofStrategy = Callable[[DofContext], DofResult]
DofMethod = str | DofStrategy

_DOF: Registry[DofStrategy] = Registry(label="DOF method")


def register_dof(*names: str):
    """Register a strategy under a canonical name and optional aliases."""

    def decorate(strategy: DofStrategy) -> DofStrategy:
        if isinstance(strategy, type) or not callable(strategy):
            raise ResultsError(
                "Register a DOF function or ready callable instance, not a class"
            )
        return _DOF.decorator(*names)(strategy)

    return decorate


def resolve_dof(method: DofMethod) -> DofStrategy:
    """Resolve a registered name or retain a ready strategy instance."""
    if isinstance(method, str):
        return _DOF.resolve(method)

    if isinstance(method, type) or not callable(method):
        raise ResultsError(
            "DOF method must be a registered name or ready callable"
        )

    return method


def dof_name(method: DofMethod) -> str:
    """Return the canonical registry name or a callable label."""
    if isinstance(method, str):
        return _DOF.canonical_name(method)

    resolve_dof(method)
    return getattr(method, "__name__", type(method).__name__)


def available_dof(*, include_aliases: bool = False) -> tuple[str, ...]:
    """List registered methods."""
    return _DOF.available(include_aliases=include_aliases)


def validate_rank_rcond(value: float | None) -> float | None:
    """Validate the relative singular-value cutoff."""
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ResultsError(
            "rank_rcond must be None or a finite number in [0, 1)"
        )

    return float(value)


def validate_support_tolerance(value: float) -> float:
    """Validate the absolute coefficient-support threshold."""
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ResultsError(
            "active_support_tolerance must be finite and nonnegative"
        )

    return float(value)


def _validated_masks(context: DofContext) -> tuple[BoolArray, BoolArray]:
    """Return independent nested masks in output-feature order."""
    u = np.array(context.unrestricted_mask, dtype=bool, copy=True)
    r = np.array(context.restricted_mask, dtype=bool, copy=True)

    if (
        u.ndim != 2
        or r.shape != u.shape
        or u.shape[0] != len(context.effects)
    ):
        raise ResultsError("DOF masks must match the output-feature layout")

    if np.any(r & ~u):
        raise ResultsError("Restricted mask must be a subset of U")

    return u, r


def validate_dof_result(
    result: DofResult,
    context: DofContext,
) -> DofResult:
    """Validate and copy a strategy result without integer coercion."""
    if not isinstance(result, DofResult):
        raise ResultsError("A DOF strategy must return DofResult")

    shape = (len(context.effects),)
    arrays = {}

    for name in (
        "model_df_u",
        "model_df_r",
        "restriction_df",
        "residual_df",
    ):
        value = np.array(getattr(result, name), dtype=float, copy=True)

        if value.shape != shape:
            raise ResultsError(
                f"{name} must have shape {shape}, got {value.shape}"
            )

        if np.isinf(value).any():
            raise ResultsError(f"{name} cannot contain infinity")

        if name != "residual_df" and np.any(value < 0):
            raise ResultsError(f"{name} cannot contain negative values")

        arrays[name] = value

    status = tuple(result.status)

    if len(status) != shape[0] or any(
        not isinstance(item, str) or not item for item in status
    ):
        raise ResultsError("DOF status must contain one string per output")

    u = result.unrestricted_mask
    r = result.restricted_mask

    if (u is None) != (r is None):
        raise ResultsError("Return both effective masks or neither")

    if u is not None:
        u = np.array(u, dtype=bool, copy=True)
        r = np.array(r, dtype=bool, copy=True)
        structural_u, structural_r = _validated_masks(context)

        if u.shape != structural_u.shape or r.shape != structural_r.shape:
            raise ResultsError("Effective DOF masks have invalid shapes")

        if (
            np.any(u & ~structural_u)
            or np.any(r & ~structural_r)
            or np.any(r & ~u)
        ):
            raise ResultsError("Effective DOF masks are not nested subsets")

    return DofResult(
        **arrays,
        status=status,
        heuristic=bool(result.heuristic),
        unrestricted_mask=u,
        restricted_mask=r,
        metadata=dict(result.metadata),
    )


def evaluate_dof(
    method: DofMethod,
    context: DofContext,
) -> DofResult:
    """Execute a resolved strategy and validate its result."""
    return validate_dof_result(resolve_dof(method)(context), context)


def _count_masks(
    context: DofContext,
    u: BoolArray,
    r: BoolArray,
    *,
    heuristic: bool,
) -> DofResult:
    """Count enabled slopes and a shared intercept."""
    k_u = u.sum(axis=1).astype(float) + int(context.bias_enabled)
    k_r = r.sum(axis=1).astype(float) + int(context.bias_enabled)
    q = (u & ~r).sum(axis=1).astype(float)
    residual_df = float(context.n_samples) - k_u

    status = tuple(
        "no_restrictions"
        if qi == 0
        else "insufficient_residual_df"
        if df <= 0
        else "ok"
        for qi, df in zip(q, residual_df)
    )

    return DofResult(
        model_df_u=k_u,
        model_df_r=k_r,
        restriction_df=q,
        residual_df=residual_df,
        status=status,
        heuristic=heuristic,
        unrestricted_mask=u,
        restricted_mask=r,
    )


@register_dof("structural")
def structural_dof(context: DofContext) -> DofResult:
    """Count structurally enabled coefficients per output."""
    u, r = _validated_masks(context)
    return _count_masks(context, u, r, heuristic=False)


@register_dof("active_support")
def active_support_dof(context: DofContext) -> DofResult:
    """Count coefficients retained by the unrestricted support."""
    u, r = _validated_masks(context)
    tolerance = validate_support_tolerance(
        context.params.get("active_support_tolerance", 1e-12)
    )

    if context.unrestricted_weights is None:
        raise ResultsError("Active-support DOF requires U weights")

    weights = np.asarray(context.unrestricted_weights, dtype=float)

    if weights.shape != u.shape:
        raise ResultsError("U weights and masks must have matching shapes")

    if not np.isfinite(weights[u]).all():
        raise ResultsError("Enabled U weights must be finite")

    support = np.abs(weights) > tolerance

    return _count_masks(
        context,
        u & support,
        r & support,
        heuristic=True,
    )


def rank_geometries(context: DofContext) -> list[dict[str, Any]]:
    """Compute nested-design geometry independently of the DOF choice."""
    u, r = _validated_masks(context)
    rcond = validate_rank_rcond(context.params.get("rank_rcond"))
    x = np.asarray(context.load_features(), dtype=float)

    if x.shape != (context.n_samples, u.shape[1]):
        raise ResultsError("Feature matrix does not match the DOF context")

    if not np.isfinite(x).all():
        raise ResultsError("Rank geometry requires finite features")

    geometries = []

    for output in range(len(context.effects)):
        design_u = x[:, u[output]]
        design_r = x[:, r[output]]

        if context.bias_enabled:
            ones = np.ones((context.n_samples, 1))
            design_u = np.column_stack((ones, design_u))
            design_r = np.column_stack((ones, design_r))

        geometries.append(
            nested_design_geometry(design_u, design_r, rcond=rcond)
        )

    return geometries


@register_dof("rank")
def rank_dof(context: DofContext) -> DofResult:
    """Count retained design directions using nested SVD geometry."""
    u, r = _validated_masks(context)
    geometries = rank_geometries(context)

    k_u = np.asarray([g["rank_u"] for g in geometries], dtype=float)
    k_r = np.asarray([g["rank_r"] for g in geometries], dtype=float)
    q = np.asarray([g["q"] for g in geometries], dtype=float)
    residual_df = float(context.n_samples) - k_u

    status = tuple(
        g["geometry_status"]
        if not g["is_nested"]
        else "no_restrictions"
        if qi == 0
        else "insufficient_residual_df"
        if df <= 0
        else "ok"
        for g, qi, df in zip(geometries, q, residual_df)
    )

    return DofResult(
        model_df_u=k_u,
        model_df_r=k_r,
        restriction_df=q,
        residual_df=residual_df,
        status=status,
        unrestricted_mask=u,
        restricted_mask=r,
        metadata={"rank_geometries": geometries},
    )
