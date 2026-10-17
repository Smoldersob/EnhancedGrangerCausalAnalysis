import pytest
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enhanced_granger_analysis.preprocessing.scaling import (
    IdentityScaler,
    MinMaxScaler,
    StandardScaler,
    get_scaler_class,
    register_scaler,
    resolve_scaler,
)


def test_registered_names_resolve_to_classes():
    assert get_scaler_class("standard") is StandardScaler
    assert get_scaler_class(" ZSCORE ") is StandardScaler


def test_resolver_constructs_independent_instances():
    first = resolve_scaler("standard")
    second = resolve_scaler("zscore")

    assert isinstance(first, StandardScaler)
    assert isinstance(second, StandardScaler)
    assert first is not second


def test_none_constructs_identity_instance():
    scaler = resolve_scaler(None)

    assert isinstance(scaler, IdentityScaler)


def test_constructor_parameters_are_forwarded():
    scaler = resolve_scaler(
        "minmax",
        feature_range=(-1.0, 1.0),
    )

    assert isinstance(scaler, MinMaxScaler)
    assert scaler.feature_range == (-1.0, 1.0)


def test_direct_class_is_supported():
    scaler = resolve_scaler(StandardScaler, eps=1e-8)

    assert isinstance(scaler, StandardScaler)
    assert scaler.eps == 1e-8


def test_existing_instance_is_not_recreated():
    scaler = StandardScaler()

    assert resolve_scaler(scaler) is scaler

    with pytest.raises(ValueError):
        resolve_scaler(scaler, eps=1e-8)


def test_factory_cannot_be_registered():
    with pytest.raises(TypeError):
        register_scaler("__invalid_factory__")(
            lambda: StandardScaler()
        )


def test_direct_factory_remains_supported():
    scaler = resolve_scaler(
        lambda: StandardScaler(eps=1e-8)
    )

    assert isinstance(scaler, StandardScaler)
    assert scaler.eps == 1e-8


def test_constructor_key_error_is_not_masked():
    class BrokenScaler(StandardScaler):
        def __init__(self):
            raise KeyError("constructor failure")

    with pytest.raises(KeyError, match="constructor failure"):
        resolve_scaler(BrokenScaler)