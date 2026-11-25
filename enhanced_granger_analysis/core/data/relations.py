"""Non-executable signal relations, observations and policy outputs."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np

from ..lags.specification import StructuralLagConstraints
from .feature_layout import FeatureIdentity, validate_names


def _finite_scalar(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real scalar.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite real scalar.")
    return result


@dataclass(frozen=True)
class AlgebraicTerm:
    """A coefficient of one original signal evaluated at time t - lag."""

    signal: str
    coefficient: float
    lag: int = 0

    def __post_init__(self) -> None:
        reference = FeatureIdentity(self.signal, self.lag)
        coefficient = _finite_scalar(self.coefficient, "Term coefficient")
        if coefficient == 0:
            raise ValueError("A term coefficient must be nonzero.")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "lag", reference.lag)


@dataclass(frozen=True)
class AlgebraicRelation:
    """An affine identity on original signals: target[t] = offset + sum(terms)."""

    target: str
    terms: tuple[AlgebraicTerm, ...]
    offset: float = 0

    def __post_init__(self) -> None:
        validate_names((self.target,), "Relation target")
        terms = tuple(self.terms)
        if not terms or not all(isinstance(term, AlgebraicTerm) for term in terms):
            raise ValueError("A relation requires at least one AlgebraicTerm.")
        keys = {(term.signal, term.lag) for term in terms}
        if len(keys) != len(terms):
            raise ValueError("Relation terms must be distinct signal/lag pairs.")
        if (self.target, 0) in keys:
            raise ValueError("A relation cannot reference its contemporaneous target.")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "offset", _finite_scalar(self.offset, "Relation offset"))


@dataclass(frozen=True)
class ProcessRelation:
    """Named process association with an opaque kind and ordered participants.

    No graph, feedback semantics, or Granger restrictions are implied.
    """

    name: str
    kind: str
    signals: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_names((self.name,), "Process relation name")
        validate_names((self.kind,), "Process relation kind")
        signals = validate_names(self.signals, "Process relation signals")
        if len(signals) < 2:
            raise ValueError("A process relation requires at least two signals.")
        object.__setattr__(self, "signals", signals)


@dataclass(frozen=True)
class AlgebraicRelationEvidence:
    """An observation about a relation; `source` identifies the evidence producer."""

    relation: AlgebraicRelation
    source: str
    method: Literal["detection", "verification"]
    status: Literal["supported", "contradicted", "inconclusive"]

    def __post_init__(self) -> None:
        if not isinstance(self.relation, AlgebraicRelation):
            raise TypeError("Evidence requires an AlgebraicRelation.")
        validate_names((self.source,), "Evidence source")
        if self.method not in {"detection", "verification"}:
            raise ValueError("Unknown algebraic evidence method.")
        if self.status not in {"supported", "contradicted", "inconclusive"}:
            raise ValueError("Unknown algebraic evidence status.")


@dataclass(frozen=True)
class RelationPolicyDecision:
    """Optional policy output, not a fitted action or a modification of declarations."""

    relation: AlgebraicRelation | ProcessRelation
    policy: str
    action: Literal["keep", "exclude_predictor", "restrict_lags"]
    reason: str
    strength: Literal["suggestion", "hard"] = "suggestion"
    predictor: str | None = None
    lag_constraints: StructuralLagConstraints | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, (AlgebraicRelation, ProcessRelation)):
            raise TypeError("A policy decision requires a supported relation.")
        validate_names((self.policy,), "Decision policy")
        validate_names((self.reason,), "Decision reason")
        if self.strength not in {"suggestion", "hard"}:
            raise ValueError("Decision strength must be suggestion or hard.")
        if self.action == "keep":
            if self.predictor is not None or self.lag_constraints is not None:
                raise ValueError("A keep decision carries no restriction.")
        elif self.action == "exclude_predictor":
            signals = (
                (self.relation.target, *(term.signal for term in self.relation.terms))
                if isinstance(self.relation, AlgebraicRelation) else self.relation.signals
            )
            if self.predictor not in signals or self.lag_constraints is not None:
                raise ValueError("Predictor exclusion requires a participating signal and no lag constraints.")
        elif self.action == "restrict_lags":
            if self.predictor is not None or not isinstance(self.lag_constraints, StructuralLagConstraints):
                raise ValueError("Lag restriction requires structural lag constraints and no predictor exclusion.")
        else:
            raise ValueError(f"Unknown relation policy action: {self.action!r}.")
