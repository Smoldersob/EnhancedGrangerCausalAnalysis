from typing import Any

# Currently, we don't have any optimizers to register for numpy backend.
# It is included here for consistency with othe backends and to allow for future expansion.
def resolve_optimizer(spec: Any) -> Any:
    return None