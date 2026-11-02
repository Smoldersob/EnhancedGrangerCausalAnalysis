"""Data-splitter class registration and instance resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, TypeVar

from ...core.registry import Registry
from .base import BaseDataSplitter, SplitStage

DataSplitterClass = TypeVar(
    "DataSplitterClass",
    bound=type[BaseDataSplitter],
)

_DATA_SPLITTERS: Registry[type[BaseDataSplitter]] = Registry(
    label="data splitter",
)

_VALID_STAGES = ("source", "prepared")


def _validate_splitter_class(cls: Any) -> None:
    """Reject non-classes and classes missing the splitter contract."""
    if not isinstance(cls, type):
        raise TypeError(
            "Only data-splitter classes can be registered, not "
            "instances or factory functions."
        )

    if not issubclass(cls, BaseDataSplitter):
        raise TypeError(
            f"Data-splitter class {cls.__name__!r} must inherit "
            "BaseDataSplitter."
        )

    stage = getattr(cls, "stage", None)

    if stage not in _VALID_STAGES:
        raise TypeError(
            f"Data-splitter class {cls.__name__!r} must declare "
            f"stage as one of {_VALID_STAGES}."
        )

    if not callable(getattr(cls, "split", None)):
        raise TypeError(
            f"Data-splitter class {cls.__name__!r} must provide "
            "a callable split method."
        )


def register_data_splitter(
    *names: str,
) -> Callable[[DataSplitterClass], DataSplitterClass]:
    """Register a splitter class; the first name is canonical."""
    if not names:
        raise ValueError(
            "Data-splitter registration requires at least one name."
        )

    def decorator(cls: DataSplitterClass) -> DataSplitterClass:
        _validate_splitter_class(cls)

        _DATA_SPLITTERS.register(
            names[0],
            cls,
            aliases=names[1:],
        )
        return cls

    return decorator


def get_data_splitter_class(name: str) -> type[BaseDataSplitter]:
    """Return a registered splitter class without constructing it."""
    return _DATA_SPLITTERS.resolve(name)


def available_data_splitters(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List registered splitter names without constructing instances."""
    return _DATA_SPLITTERS.available(include_aliases=include_aliases)


def resolve_data_splitter(
    spec: Any = None,
    **params: Any,
) -> BaseDataSplitter | None:
    """Resolve a data-splitter specification into an instance.

    Supported inputs:
        None:
            Disable data splitting; parameters must be empty.
        Registered name:
            Construct the corresponding class with params.
        ``{'type': ..., 'params': ...}`` mapping:
            Resolve the embedded name and parameters. Embedded and
            separate parameters cannot share keys.
        Splitter class:
            Construct it with params without requiring registration.
        Splitter instance:
            Return an independent deepcopy; params must be empty.

    Constructor and factory exceptions propagate unchanged.
    """
    if spec is None:
        if params:
            raise ValueError(
                "Data-splitter parameters cannot be supplied when "
                "data splitting is disabled."
            )
        return None

    if isinstance(spec, Mapping):
        unknown = set(spec).difference({"type", "params"})
        if unknown:
            raise ValueError(
                "Unknown data-splitter specification fields: "
                f"{sorted(unknown, key=repr)!r}."
            )

        if "type" not in spec:
            raise ValueError(
                "A data-splitter specification requires 'type'."
            )

        splitter_name = spec["type"]
        if (
            not isinstance(splitter_name, str)
            or not splitter_name.strip()
        ):
            raise TypeError(
                "splitter.type must be a non-empty registered name."
            )

        embedded_params = spec.get("params", {})
        if not isinstance(embedded_params, Mapping):
            raise TypeError(
                "splitter.params must be a mapping."
            )

        if not all(
            isinstance(name, str) for name in embedded_params
        ):
            raise TypeError(
                "splitter.params keys must be strings."
            )

        duplicate_params = set(embedded_params).intersection(params)
        if duplicate_params:
            raise ValueError(
                "Data-splitter parameters were supplied both inside "
                "the specification and as separate parameters: "
                f"{sorted(duplicate_params)!r}."
            )

        params = {**dict(embedded_params), **params}
        spec = splitter_name

    if isinstance(spec, str):
        cls = get_data_splitter_class(spec)
        return cls(**params)

    if isinstance(spec, type):
        _validate_splitter_class(spec)
        return spec(**params)

    if isinstance(spec, BaseDataSplitter):
        if params:
            raise ValueError(
                "Constructor parameters cannot be supplied with an "
                "existing data-splitter instance. Pass a class or "
                "registered name instead."
            )
        return deepcopy(spec)

    raise TypeError(
        "Data splitter must be None, a registered name, a "
        "{'type': ..., 'params': ...} specification, a splitter class, "
        "or a splitter instance."
    )
