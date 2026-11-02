from collections.abc import Mapping
from typing import Any

from ....core.components.callback_lifecycle import (
    clone_custom_callback,
    reset_callback_list,
)
from .base import Callback


def reset_callbacks_for_run(
    callbacks: list[Callback],
    *,
    run_name: str,
    callback_updates: Mapping[
        int | str | type,
        Mapping[str, Any],
    ] | None = None,
) -> list[Callback]:
    return reset_callback_list(
        callbacks,
        run_name,
        callback_updates,
        expected_type=Callback,
        clone_callback=clone_custom_callback,
    )