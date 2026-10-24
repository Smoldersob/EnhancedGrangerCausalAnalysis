from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..exceptions import ConstraintConfigurationError


def validate_run_name(run_name: str) -> None:
    if (
        not isinstance(run_name, str)
        or not run_name.strip()
        or run_name in {".", ".."}
        or any(char in run_name for char in ("/", "\\", "\x00"))
    ):
        raise ConstraintConfigurationError(
            "run_name must be a non-empty, single directory name."
        )


def resolve_callback_updates(
    callbacks: Sequence[Any],
    callback_updates: Mapping | None,
) -> list[dict[str, Any]]:
    if callback_updates is None:
        callback_updates = {}
    if not isinstance(callback_updates, Mapping):
        raise ConstraintConfigurationError(
            "callback_updates must be a mapping or None."
        )

    resolved = [{} for _ in callbacks]

    for selector, values in callback_updates.items():
        if (
            isinstance(selector, bool)
            or not isinstance(selector, (int, str, type))
        ):
            raise ConstraintConfigurationError(
                f"Invalid callback selector: {selector!r}."
            )
        if not isinstance(values, Mapping):
            raise ConstraintConfigurationError(
                "Each callback update must be a mapping."
            )
        if any(
            not isinstance(name, str) or name.startswith("_")
            for name in values
        ):
            raise ConstraintConfigurationError(
                "Callback updates must use public configuration names."
            )

        matches = [
            index
            for index, callback in enumerate(callbacks)
            if (
                (isinstance(selector, int) and selector == index)
                or (
                    isinstance(selector, str)
                    and selector == callback.__class__.__name__
                )
                or (
                    isinstance(selector, type)
                    and isinstance(callback, selector)
                )
            )
        ]

        if not matches:
            raise ConstraintConfigurationError(
                f"Callback selector {selector!r} matches no callback."
            )

        for index in matches:
            resolved[index].update(copy.deepcopy(dict(values)))

    return resolved


def clone_custom_callback(
    callback: Any,
    run_name: str,
    updates: Mapping[str, Any],
) -> Any:
    clone = getattr(callback, "clone_for_run", None)
    if not callable(clone):
        raise ConstraintConfigurationError(
            f"{callback.__class__.__name__} must implement "
            "clone_for_run(run_name)."
        )

    if updates:
        try:
            inspect.signature(clone).bind(
                run_name,
                callback_updates=dict(updates),
            )
        except (TypeError, ValueError) as exc:
            raise ConstraintConfigurationError(
                f"{callback.__class__.__name__}.clone_for_run() "
                "must accept callback_updates to apply updates."
            ) from exc

        fresh = clone(
            run_name,
            callback_updates=copy.deepcopy(dict(updates)),
        )
    else:
        fresh = clone(run_name)

    if fresh is callback:
        raise ConstraintConfigurationError(
            "clone_for_run() must return a new callback instance."
        )

    return fresh


def reset_callback_list(
    callbacks: Sequence[Any],
    run_name: str,
    callback_updates: Mapping | None,
    *,
    expected_type: type,
    clone_callback: Callable[[Any, str, Mapping[str, Any]], Any],
) -> list[Any]:
    validate_run_name(run_name)

    if not isinstance(callbacks, list):
        raise ConstraintConfigurationError(
            "Callbacks must be stored as a list."
        )
    if any(not isinstance(cb, expected_type) for cb in callbacks):
        raise ConstraintConfigurationError(
            f"All callbacks must inherit from {expected_type.__name__}."
        )

    updates = resolve_callback_updates(callbacks, callback_updates)
    fresh_callbacks = []

    for callback, callback_changes in zip(callbacks, updates):
        fresh = clone_callback(callback, run_name, callback_changes)

        if fresh is callback or not isinstance(fresh, expected_type):
            raise ConstraintConfigurationError(
                "Reset must return a new callback of the expected type."
            )

        fresh_callbacks.append(fresh)

    return fresh_callbacks