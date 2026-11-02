from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....core.components.callback_lifecycle import (
    clone_custom_callback,
    reset_callback_list,
)
from .base import Callback


CallbackUpdates = Mapping[
    int | str | type,
    Mapping[str, Any],
]


def reset_callbacks_for_run(
    callbacks: list[Callback],
    *,
    run_name: str,
    callback_updates: CallbackUpdates | None = None,
) -> list[Callback]:
    """Create fresh callbacks for an independent training run.

    Preserve configuration unless explicitly updated.
    Do not modify source callbacks or the source list.
    Do not copy runtime state from previous training.
    """
    return reset_callback_list(
        callbacks,
        run_name,
        callback_updates,
        expected_type=Callback,
        clone_callback=clone_custom_callback,
    )