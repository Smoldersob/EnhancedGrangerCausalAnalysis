import importlib
import importlib.util
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SkipTest(Exception):
    pass


PROFILE_BLOCKS = {
    "tf-only": {"torch", "sklearn"},
    "torch-only": {"tensorflow", "sklearn"},
    "sklearn-only": {"tensorflow", "torch"},
}

PROFILE_PRIMARY = {
    "tf-only": "tensorflow",
    "torch-only": "torch",
    "sklearn-only": "sklearn",
}


def _clear_modules(prefixes):
    to_remove = []
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            to_remove.append(name)
    for name in to_remove:
        sys.modules.pop(name, None)


@contextmanager
def _blocked_imports(blocked_roots):
    old_find_spec = importlib.util.find_spec

    def guarded_find_spec(name, package=None):
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            return None
        return old_find_spec(name, package)

    importlib.util.find_spec = guarded_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = old_find_spec


def _run_profile_smoke(profile_name):
    blocked = PROFILE_BLOCKS[profile_name]
    primary = PROFILE_PRIMARY[profile_name]

    # A profile is only executable if its main backend is actually installed.
    if importlib.util.find_spec(primary) is None:
        raise SkipTest(f"{profile_name}: primary backend '{primary}' is not installed")

    _clear_modules(
        [
            "complex_granger_analysis",
            "tensorflow",
            "torch",
            "sklearn",
        ]
    )

    with _blocked_imports(blocked):
        backends_pkg = importlib.import_module("complex_granger_analysis.backends")
        factory_module = importlib.import_module("complex_granger_analysis.backends.backend_factory")

        assert hasattr(backends_pkg, "BackendFactory")

        BackendFactory = factory_module.BackendFactory
        BackendFactory.reset_cache()

        available = BackendFactory.list_available_backends()
        assert len(available) > 0, f"{profile_name}: no available backends"

        # The profile should expose only one canonical backend.
        if profile_name == "tf-only":
            assert available == ["tensorflow"], f"{profile_name}: unexpected available={available}"
            strategy = BackendFactory.get_strategy("tensorflow")
            assert strategy.is_available()
        elif profile_name == "torch-only":
            assert available == ["pytorch"], f"{profile_name}: unexpected available={available}"
            strategy = BackendFactory.get_strategy("pytorch")
            assert strategy.is_available()
        elif profile_name == "sklearn-only":
            assert available == ["sklearn"], f"{profile_name}: unexpected available={available}"
            strategy = BackendFactory.get_strategy("sklearn")
            assert strategy.is_available()


def _run_top_level_import_smoke():
    """Top-level package import should not require optional ML/runtime dependencies."""
    blocked = {"tensorflow", "torch", "sklearn", "joblib", "pandas"}

    _clear_modules(
        [
            "complex_granger_analysis",
            "tensorflow",
            "torch",
            "sklearn",
            "joblib",
            "pandas",
        ]
    )

    with _blocked_imports(blocked):
        pkg = importlib.import_module("complex_granger_analysis")

        # Import itself should succeed without touching optional stacks.
        assert hasattr(pkg, "__all__")
        assert "callbacks" in pkg.__all__

        # Lazy alias should still resolve and expose callback symbols.
        callbacks_mod = getattr(pkg, "callbacks")
        assert hasattr(callbacks_mod, "Callback")
        assert hasattr(callbacks_mod, "EarlyStopping")

        # Lazy submodule should resolve on demand.
        backends_mod = getattr(pkg, "backends")
        assert hasattr(backends_mod, "BackendFactory")

        # Convenient top-level symbols should resolve lazily.
        backend_factory_cls = getattr(pkg, "BackendFactory")
        assert backend_factory_cls.__name__ == "BackendFactory"


def _run_top_level_convenient_api_exports_smoke():
    """Top-level convenient API exports should resolve when API deps are available."""
    if importlib.util.find_spec("pandas") is None:
        pytest.skip("pandas is not installed")
    if importlib.util.find_spec("joblib") is None:
        pytest.skip("joblib is not installed")

    _clear_modules(["complex_granger_analysis"])

    pkg = importlib.import_module("complex_granger_analysis")
    builder_cls = getattr(pkg, "MultitaskGrangerBuilder")
    api_cls = getattr(pkg, "MultiTaskGrangerAPI")
    loader_cls = getattr(pkg, "BuilderConfigLoader")

    assert builder_cls.__name__ == "MultitaskGrangerBuilder"
    assert api_cls.__name__ == "MultiTaskGrangerAPI"
    assert loader_cls.__name__ == "BuilderConfigLoader"


def test_import_profile_tf_only_smoke():
    _run_profile_smoke("tf-only")


def test_import_profile_torch_only_smoke():
    _run_profile_smoke("torch-only")


def test_import_profile_sklearn_only_smoke():
    _run_profile_smoke("sklearn-only")


def test_top_level_lazy_import_smoke():
    _run_top_level_import_smoke()


def test_top_level_convenient_api_exports_smoke():
    _run_top_level_convenient_api_exports_smoke()


if __name__ == "__main__":
    tests = [
        lambda: _run_profile_smoke("tf-only"),
        lambda: _run_profile_smoke("torch-only"),
        lambda: _run_profile_smoke("sklearn-only"),
        _run_top_level_import_smoke,
        _run_top_level_convenient_api_exports_smoke,
    ]

    print("\n" + "=" * 80)
    print("IMPORT SMOKE PROFILE TESTS")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for idx, test_fn in enumerate(tests, start=1):
        name = [
            "tf-only",
            "torch-only",
            "sklearn-only",
            "top-level-lazy-import",
            "top-level-convenient-api-exports",
        ][idx - 1]
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP: {name} -> {exc}")
            skipped += 1
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=2)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(
        f"Summary: {passed}/{total} passed, {failed}/{total} failed, {skipped}/{total} skipped"
    )
    print("=" * 80 + "\n")
