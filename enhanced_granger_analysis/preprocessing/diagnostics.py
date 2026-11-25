"""Fit-scoped observations and explicit representation reduction.

Diagnostics report facts only.  They never select a representative or alter
the input frames; representation changes require a separate policy decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from ..core.data.analysis_roles import VariableRoles
from ..core.data.provenance import RemovedVariable
from ..core.data.relations import AlgebraicRelation, AlgebraicTerm
from ..core.data.variables import ProcessRole, VariableDescription
from ..core.exceptions import DataValidationError
from .contracts import (
    DiagnosticFact,
    DiagnosticsReport,
    RepresentationDecision,
)
from .splitting.scope import FitScope


def _validate_tolerance(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite non-negative number.")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return normalized


@dataclass(frozen=True)
class ControlErrorDifferenceSpec:
    """Explicit candidate equations for ``error = first - second``.

    Candidate pairs are not discovered through regression or correlation.
    Numerical support requires every selected residual to satisfy
    ``abs(error - (first - second)) <= atol + rtol * scale``.  A relation
    outside this strict numerical tolerance is not reported as a match.
    """

    candidate_pairs: Mapping[str, tuple[tuple[str, str], ...]]
    atol: float = 1e-10
    rtol: float = 1e-8
    operand_roles: tuple[ProcessRole, ...] = (
        ProcessRole.SETPOINT,
        ProcessRole.CONTROLLED_VARIABLE,
        ProcessRole.PROCESS_VARIABLE,
    )

    def __post_init__(self) -> None:
        candidates: dict[str, tuple[tuple[str, str], ...]] = {}
        for target, pairs in self.candidate_pairs.items():
            if not isinstance(target, str) or not target:
                raise ValueError("Difference targets must be non-empty signal names.")
            normalized_pairs: list[tuple[str, str]] = []
            for pair in pairs:
                if (
                    not isinstance(pair, (tuple, list))
                    or len(pair) != 2
                    or any(not isinstance(signal, str) or not signal for signal in pair)
                    or pair[0] == pair[1]
                    or target in pair
                ):
                    raise ValueError(
                        "Each difference candidate must be a distinct (first, second) "
                        "pair that excludes its target."
                    )
                normalized_pairs.append((pair[0], pair[1]))
            if len(set(normalized_pairs)) != len(normalized_pairs):
                raise ValueError(f"Duplicate difference candidates for {target!r}.")
            candidates[target] = tuple(normalized_pairs)
        roles: list[ProcessRole] = []
        for role in self.operand_roles:
            normalized = ProcessRole(role) if isinstance(role, str) else role
            if not isinstance(normalized, ProcessRole):
                raise TypeError("operand_roles must contain ProcessRole values.")
            if normalized not in roles:
                roles.append(normalized)
        object.__setattr__(self, "candidate_pairs", MappingProxyType(candidates))
        object.__setattr__(self, "atol", _validate_tolerance(self.atol, "atol"))
        object.__setattr__(self, "rtol", _validate_tolerance(self.rtol, "rtol"))
        object.__setattr__(self, "operand_roles", tuple(roles))


def _role_axes(roles: VariableRoles) -> dict[str, tuple[str, ...]]:
    return {
        "effect": tuple(roles.effects),
        "cause": tuple(roles.causes),
        "tested_cause": tuple(roles.tested_causes),
    }


def _ordered_role_names(roles: VariableRoles) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        name
        for role_names in _role_axes(roles).values()
        for name in role_names
    ))


def _is_missing_scalar(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _same_constant_value(left: object, right: object) -> bool:
    if _is_missing_scalar(left) or _is_missing_scalar(right):
        return _is_missing_scalar(left) and _is_missing_scalar(right)
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _is_constant(selected: Sequence[pd.DataFrame], name: str) -> bool:
    found = False
    first: object = None
    for frame in selected:
        for value in frame[name].array:
            if not found:
                first = value
                found = True
            elif not _same_constant_value(first, value):
                return False
    return found


def _exactly_equal_across_frames(
    selected: Sequence[pd.DataFrame],
    left: str,
    right: str,
) -> bool:
    return all(frame[left].reset_index(drop=True).equals(
        frame[right].reset_index(drop=True)
    ) for frame in selected)


def _numeric_values(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    series = frame[name]
    if (
        is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
        or not is_numeric_dtype(series.dtype)
    ):
        return None
    try:
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(values).all():
        return None
    return values


def _check_difference(
    selected: Sequence[pd.DataFrame],
    target: str,
    first: str,
    second: str,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float | None, float | None, str | None]:
    absolute_residuals: list[np.ndarray] = []
    tolerance_ratios: list[np.ndarray] = []
    for frame in selected:
        error = _numeric_values(frame, target)
        first_values = _numeric_values(frame, first)
        second_values = _numeric_values(frame, second)
        if error is None or first_values is None or second_values is None:
            return False, None, None, "non_numeric_or_non_finite"
        expected = first_values - second_values
        residual = np.abs(error - expected)
        scale = np.maximum(np.abs(error), np.abs(expected))
        tolerance = atol + rtol * scale
        if not np.isfinite(residual).all() or not np.isfinite(tolerance).all():
            return False, None, None, "non_finite_residual"
        absolute_residuals.append(residual)
        ratio = np.divide(
            residual,
            tolerance,
            out=np.where(residual == 0, 0.0, np.inf),
            where=tolerance > 0,
        )
        tolerance_ratios.append(ratio)
    if not absolute_residuals:
        return False, None, None, "empty_scope"
    residuals = np.concatenate(absolute_residuals)
    ratios = np.concatenate(tolerance_ratios)
    return (
        bool(np.all(ratios <= 1.0)),
        float(np.max(residuals)),
        float(np.max(ratios)),
        None,
    )


class SignalDiagnostics:
    """Diagnose constants, exact duplicates, and configured error differences."""

    def __init__(
        self,
        roles: VariableRoles,
        *,
        variable_descriptions: Mapping[str, VariableDescription] | None = None,
        control_error_difference: ControlErrorDifferenceSpec | None = None,
    ) -> None:
        if not isinstance(roles, VariableRoles):
            raise TypeError("roles must be VariableRoles.")
        descriptions = dict(variable_descriptions or {})
        for name, description in descriptions.items():
            if not isinstance(description, VariableDescription):
                raise TypeError("variable_descriptions values must be VariableDescription.")
            if name != description.name:
                raise ValueError("Variable description keys must match description names.")
        if control_error_difference is not None and not isinstance(
            control_error_difference, ControlErrorDifferenceSpec
        ):
            raise TypeError(
                "control_error_difference must be ControlErrorDifferenceSpec or None."
            )
        self.roles = roles
        self.variable_descriptions = MappingProxyType(descriptions)
        self.control_error_difference = control_error_difference

    def diagnose(
        self,
        frames: Sequence[pd.DataFrame],
        *,
        scope: FitScope,
    ) -> DiagnosticsReport:
        """Return observations computed only from rows selected by ``scope``."""
        if not isinstance(scope, FitScope):
            raise TypeError("scope must be an explicit FitScope.")
        frame_list = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        if not frame_list or not all(isinstance(frame, pd.DataFrame) for frame in frame_list):
            raise DataValidationError("Diagnostics require one or more DataFrames.")
        if any(frame.columns.has_duplicates for frame in frame_list):
            raise DataValidationError("Diagnostics input contains duplicate column names.")
        names = _ordered_role_names(self.roles)
        for index, frame in enumerate(frame_list):
            missing = [name for name in names if name not in frame.columns]
            if missing:
                raise DataValidationError(
                    f"Diagnostics input {index} is missing role variables: {missing!r}."
                )
        selected = scope.select_frames(frame_list)

        facts: list[DiagnosticFact] = []
        for name in names:
            if _is_constant(selected, name):
                facts.append(DiagnosticFact(
                    fact_id=f"constant:{name}",
                    kind="constant",
                    variables=(name,),
                    evidence={"n_rows": sum(len(frame) for frame in selected)},
                ))

        representatives: list[str] = []
        for name in names:
            match = next(
                (
                    retained for retained in representatives
                    if _exactly_equal_across_frames(selected, name, retained)
                ),
                None,
            )
            if match is None:
                representatives.append(name)
                continue
            facts.append(DiagnosticFact(
                fact_id=f"duplicate:{name}:{match}",
                kind="duplicate",
                variables=(name, match),
                evidence={
                    "duplicate": name,
                    "retained_variable": match,
                    "comparison": "pandas.Series.equals",
                    "n_rows": sum(len(frame) for frame in selected),
                },
            ))

        spec = self.control_error_difference
        if spec is not None:
            facts.extend(self._difference_facts(selected, spec))
        return DiagnosticsReport(scope=scope, facts=tuple(facts))

    def _difference_facts(
        self,
        selected: Sequence[pd.DataFrame],
        spec: ControlErrorDifferenceSpec,
    ) -> list[DiagnosticFact]:
        facts: list[DiagnosticFact] = []
        for target in sorted(spec.candidate_pairs):
            target_description = self.variable_descriptions.get(target)
            configured_pairs = sorted(spec.candidate_pairs[target])
            matches: list[tuple[tuple[str, str], AlgebraicRelation, dict[str, float | str]]] = []
            considered: list[dict[str, object]] = []
            for first, second in configured_pairs:
                left_description = self.variable_descriptions.get(first)
                right_description = self.variable_descriptions.get(second)
                target_is_error = (
                    target_description is not None
                    and ProcessRole.CONTROL_ERROR in target_description.process_roles
                )
                operands_are_eligible = all(
                    description is not None
                    and bool(set(description.process_roles).intersection(spec.operand_roles))
                    for description in (left_description, right_description)
                )
                if not target_is_error or not operands_are_eligible:
                    considered.append({
                        "pair": (first, second),
                        "status": "ineligible_roles",
                    })
                    continue
                if any(
                    name not in frame.columns
                    for frame in selected
                    for name in (target, first, second)
                ):
                    considered.append({
                        "pair": (first, second),
                        "status": "missing_signal",
                    })
                    continue
                supported, max_residual, max_ratio, reason = _check_difference(
                    selected,
                    target,
                    first,
                    second,
                    atol=spec.atol,
                    rtol=spec.rtol,
                )
                if reason is not None:
                    considered.append({
                        "pair": (first, second),
                        "status": reason,
                    })
                    continue
                assert max_residual is not None and max_ratio is not None
                considered.append({
                    "pair": (first, second),
                    "status": "supported" if supported else "outside_tolerance",
                    "max_abs_residual": max_residual,
                    "max_tolerance_ratio": max_ratio,
                })
                if supported:
                    relation = AlgebraicRelation(
                        target,
                        (
                            AlgebraicTerm(first, 1),
                            AlgebraicTerm(second, -1),
                        ),
                    )
                    matches.append((
                        (first, second),
                        relation,
                        {
                            "max_abs_residual": max_residual,
                            "max_tolerance_ratio": max_ratio,
                            "match_kind": (
                                "exact"
                                if max_residual == 0.0
                                else "within_numeric_tolerance"
                            ),
                        },
                    ))

            ambiguous = len(matches) > 1
            status = "ambiguous" if ambiguous else "supported" if matches else "no_match"
            variables = tuple(dict.fromkeys((
                target,
                *(signal for first, second in configured_pairs for signal in (first, second)),
            )))
            facts.append(DiagnosticFact(
                fact_id=f"control-error-difference:{target}",
                kind="control_error_difference",
                variables=variables,
                evidence={
                    "status": status,
                    "tolerance": {"atol": spec.atol, "rtol": spec.rtol},
                    "considered": tuple(considered),
                    "matches": tuple({
                        "pair": pair,
                        "relation": relation,
                        **metrics,
                    } for pair, relation, metrics in matches),
                    "relation": matches[0][1] if len(matches) == 1 else None,
                },
            ))
        return facts


@dataclass(frozen=True)
class AppliedRemoval:
    """Removal provenance plus the fact and optional defining equation."""

    removal: RemovedVariable
    fact_ids: tuple[str, ...]
    decision_source: str
    defining_relation: AlgebraicRelation | None = None


@dataclass(frozen=True)
class AppliedRepresentation:
    """The result of applying an existing decision, without redetection."""

    frames: tuple[pd.DataFrame, ...]
    effective_roles: VariableRoles
    removals: tuple[AppliedRemoval, ...]


def apply_representation_decision(
    frames: Sequence[pd.DataFrame],
    roles: VariableRoles,
    report: DiagnosticsReport,
    decision: RepresentationDecision,
) -> AppliedRepresentation:
    """Apply only explicit actions to new frames and return removal provenance.

    This function does not inspect frame values or run diagnostics.  Pass the
    already resolved roles, facts, and policy decision from the fit phase.
    """
    if not isinstance(roles, VariableRoles):
        raise TypeError("roles must be VariableRoles.")
    if not isinstance(report, DiagnosticsReport):
        raise TypeError("report must be DiagnosticsReport.")
    if not isinstance(decision, RepresentationDecision):
        raise TypeError("decision must be RepresentationDecision.")
    frame_list = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
    if not frame_list or not all(isinstance(frame, pd.DataFrame) for frame in frame_list):
        raise DataValidationError("Representation application requires one or more DataFrames.")
    if any(frame.columns.has_duplicates for frame in frame_list):
        raise DataValidationError("Representation input contains duplicate column names.")

    facts_by_id = {fact.fact_id: fact for fact in report.facts}
    if len(facts_by_id) != len(report.facts):
        raise ValueError("Diagnostics report contains duplicate fact ids.")
    removals: list[AppliedRemoval] = []
    removals_by_name: dict[str, AppliedRemoval] = {}
    action_by_name: dict[str, str] = {}
    axes = _role_axes(roles)
    effective = {name: list(names) for name, names in axes.items()}

    for action in decision.actions:
        if action.variable in action_by_name:
            if action_by_name[action.variable] != action.action:
                raise ValueError(
                    f"Conflicting representation actions for {action.variable!r}."
                )
            raise ValueError(f"Duplicate representation action for {action.variable!r}.")
        action_by_name[action.variable] = action.action
        if action.action == "keep":
            continue
        if not action.fact_ids:
            raise ValueError(
                f"Removal action for {action.variable!r} must cite diagnostic fact ids."
            )
        try:
            cited_facts = tuple(facts_by_id[fact_id] for fact_id in action.fact_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown diagnostic fact id: {exc.args[0]!r}.") from exc
        if any(action.variable not in fact.variables for fact in cited_facts):
            raise ValueError(
                f"Every cited fact for {action.variable!r} must concern that variable."
            )

        present_roles = tuple(
            role for role, names in effective.items() if action.variable in names
        )
        if action.action == "remove_predictor":
            removed_roles = tuple(
                role for role in ("cause", "tested_cause")
                if action.variable in effective[role]
            )
        else:
            removed_roles = present_roles
        if not removed_roles:
            continue

        fact_kinds = {fact.kind for fact in cited_facts}
        if len(fact_kinds) != 1:
            raise ValueError("A removal must cite facts with one consistent kind.")
        fact_kind = next(iter(fact_kinds))
        relation: AlgebraicRelation | None = None
        retained_variable: str | None = None
        if fact_kind == "constant":
            reason = "constant"
        elif fact_kind == "duplicate":
            reason = "duplicate"
            duplicate_facts = [
                fact for fact in cited_facts if fact.evidence.get("duplicate") == action.variable
            ]
            if len(duplicate_facts) != len(cited_facts):
                raise ValueError(
                    "Duplicate reduction must cite facts identifying the removed signal."
                )
            representatives = {
                fact.evidence.get("retained_variable") for fact in duplicate_facts
            }
            if len(representatives) != 1:
                raise ValueError("Duplicate facts disagree about their representative.")
            retained_variable = next(iter(representatives))
            if not isinstance(retained_variable, str):
                raise ValueError("Duplicate fact has no retained representative.")
        elif fact_kind == "control_error_difference":
            if action.action != "remove_variable":
                raise ValueError(
                    "A defining difference relation can only justify explicit "
                    "whole-variable removal."
                )
            relations = []
            for fact in cited_facts:
                if fact.evidence.get("status") != "supported":
                    raise ValueError(
                        "Only a uniquely supported difference fact can justify reduction."
                    )
                candidate_relation = fact.evidence.get("relation")
                if not isinstance(candidate_relation, AlgebraicRelation):
                    raise ValueError("Supported difference fact has no unique relation.")
                if candidate_relation.target != action.variable:
                    raise ValueError("The reduced signal must be the relation target.")
                relations.append(candidate_relation)
            if len(set(relations)) != 1:
                raise ValueError("Cited difference facts disagree about the defining relation.")
            relation = relations[0]
            reason = "algebraic_redundancy"
        else:
            raise ValueError(f"Unsupported diagnostic fact kind for reduction: {fact_kind!r}.")

        for role in removed_roles:
            effective[role].remove(action.variable)
        provenance = AppliedRemoval(
            removal=RemovedVariable(
                action.variable,
                reason,
                removed_roles,
                retained_variable,
            ),
            fact_ids=tuple(action.fact_ids),
            decision_source=action.source,
            defining_relation=relation,
        )
        removals.append(provenance)
        removals_by_name[action.variable] = provenance

    resulting_roles = VariableRoles(
        effects=tuple(effective["effect"]),
        causes=tuple(effective["cause"]),
        tested_causes=tuple(effective["tested_cause"]),
    )
    still_used = set((*resulting_roles.effects, *resulting_roles.causes))
    removed_columns = {
        name for name in removals_by_name if name not in still_used
    }
    output_frames: list[pd.DataFrame] = []
    for index, frame in enumerate(frame_list):
        missing = [
            name for name in (*resulting_roles.effects, *resulting_roles.causes)
            if name not in frame.columns
        ]
        if missing:
            raise DataValidationError(
                f"Representation input {index} is missing effective role variables: "
                f"{list(dict.fromkeys(missing))!r}."
            )
        output_frames.append(
            frame.drop(columns=[name for name in removed_columns if name in frame.columns]).copy(deep=True)
        )
    return AppliedRepresentation(
        frames=tuple(output_frames),
        effective_roles=resulting_roles,
        removals=tuple(removals),
    )


__all__ = [
    "ControlErrorDifferenceSpec",
    "SignalDiagnostics",
    "AppliedRemoval",
    "AppliedRepresentation",
    "apply_representation_decision",
]
