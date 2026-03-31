import json
import sys
import tempfile
from pathlib import Path

# Allow running this file directly from its nested location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.api import BuilderConfigLoader, TestGroupConfigIterator
from complex_granger_analysis.core.lag_config import LagConfiguration
from complex_granger_analysis.preprocessing.lag.lag_selectors import ICLagSelector
from complex_granger_analysis.callbacks import ConvergenceCheck, EarlyStopping


def test_builder_config_loader_json_and_lag_config_conversion():
    cfg = {
        "backend": "pytorch",
        "lag_config": {
            "max_lag": 7,
            "use_lag_zero": False,
        },
        "model_config": {
            "epochs": 10,
            "optimizer": "adam",
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")

        loaded = BuilderConfigLoader.load_file(p)
        assert loaded["backend"] == "pytorch"
        assert isinstance(loaded["lag_config"], LagConfiguration)
        assert loaded["lag_config"].max_lag == 7
        assert loaded["model_config"]["epochs"] == 10


def test_test_group_iterator_next_and_has_next():
    template_path = PROJECT_ROOT / "complex_granger_analysis" / "memories" / "group_config.json"
    it = TestGroupConfigIterator.from_file(template_path)

    assert it.has_next() is True
    c1 = it.next()
    assert c1["backend"] == "pytorch"
    assert c1["model_config"]["optimizer"] == "adam"
    assert c1["model_config"]["learning_rate"] == 0.001
    assert c1["model_config"]["epochs"] == 50
    assert isinstance(c1["lag_config"], LagConfiguration)
    assert c1["lag_config"].max_lag == 8

    assert it.has_next() is True
    c2 = it.next()
    assert c2["model_config"]["optimizer"] == "adam"
    assert c2["model_config"]["learning_rate"] == 0.0005
    assert c2["model_config"]["epochs"] == 100
    assert isinstance(c2["lag_config"], LagConfiguration)
    assert c2["lag_config"].max_lag == 10

    assert it.has_next() is True
    c3 = it.next()
    assert c3["model_config"]["optimizer"] == "sgd"
    assert c3["model_config"]["learning_rate"] == 0.01
    assert c3["model_config"]["epochs"] == 80
    assert isinstance(c3["lag_config"], LagConfiguration)
    assert c3["lag_config"].max_lag == 12

    assert it.has_next() is False


def test_test_group_iterator_without_sweep_produces_one_config():
    group = {
        "base_config": {
            "backend": "pytorch",
            "lag_config": {"max_lag": 6, "use_lag_zero": False},
            "model_config": {"epochs": 12},
        }
    }

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "single_group.json"
        p.write_text(json.dumps(group), encoding="utf-8")

        it = TestGroupConfigIterator.from_file(p)

        assert it.has_next() is True
        cfg = it.next()
        assert cfg["backend"] == "pytorch"
        assert isinstance(cfg["lag_config"], LagConfiguration)
        assert cfg["lag_config"].max_lag == 6
        assert it.has_next() is False


def test_builder_config_loader_builds_supported_objects_from_json_specs():
    cfg = {
        "backend": "pytorch",
        "lag_config": {
            "max_lag": 9,
            "use_lag_zero": False,
        },
        "lag_selector": {
            "type": "ic",
            "use_bic": True,
        },
        "callbacks": [
            {"type": "early_stopping", "patience": 5, "min_delta": 0.001},
            {"type": "convergence_check", "relative_change_threshold": 1e-5},
        ],
        "regularizer": {
            "type": "l1",
            "l1": 0.02,
        },
        "relations": [
            {"effect": "y", "cause": "x1", "zero": True},
            {"effect": "y", "cause": "x2", "min_abs_sum": 0.4},
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cfg_objects.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")

        loaded = BuilderConfigLoader.load_file(p)

    assert isinstance(loaded["lag_config"], LagConfiguration)
    assert isinstance(loaded["lag_selector"], ICLagSelector)
    assert loaded["lag_selector"].max_lag == 9
    assert loaded["lag_selector"].use_bic is True

    assert isinstance(loaded["callbacks"], list)
    assert len(loaded["callbacks"]) == 2
    assert isinstance(loaded["callbacks"][0], EarlyStopping)
    assert isinstance(loaded["callbacks"][1], ConvergenceCheck)

    assert "regularizer" not in loaded
    assert loaded["regularizer_spec"]["type"] == "l1"
    assert loaded["regularizer_spec"]["l1"] == 0.02

    assert ("y", "x1") in loaded["relations"]
    assert ("y", "x2") in loaded["relations"]
    assert loaded["relations"][("y", "x1")]["zero"] is True
    assert loaded["relations"][("y", "x2")]["min_abs_sum"] == 0.4


def test_builder_config_loader_rejects_unsupported_types():
    bad_cfg = {
        "lag_selector": {"type": "unknown_selector"},
    }

    try:
        BuilderConfigLoader.normalize_builder_config(bad_cfg)
        assert False, "Unsupported lag selector should raise DataValidationError"
    except Exception as exc:
        assert "Unsupported lag_selector type" in str(exc)


def test_builder_config_loader_keeps_tensorflow_callback_specs_raw_for_backend_resolution():
    cfg = {
        "backend": "tensorflow",
        "callbacks": [
            {"type": "early_stopping", "monitor": "loss", "patience": 3},
            {"type": "tensorboard", "log_dir": "runs/tf"},
        ],
    }

    loaded = BuilderConfigLoader.normalize_builder_config(cfg)
    assert isinstance(loaded["callbacks"], list)
    assert isinstance(loaded["callbacks"][0], dict)
    assert loaded["callbacks"][0]["type"] == "early_stopping"
    assert loaded["callbacks"][1]["type"] == "tensorboard"


def test_builder_config_loader_rejects_unknown_callback_for_non_tensorflow_backend():
    cfg = {
        "backend": "pytorch",
        "callbacks": [
            {"type": "model_checkpoint"},
        ],
    }

    try:
        BuilderConfigLoader.normalize_builder_config(cfg)
        assert False, "Unknown callback type should be rejected for non-TensorFlow backends"
    except Exception as exc:
        assert "Unsupported callback type" in str(exc)


if __name__ == "__main__":
    tests = [
        test_builder_config_loader_json_and_lag_config_conversion,
        test_test_group_iterator_next_and_has_next,
        test_test_group_iterator_without_sweep_produces_one_config,
        test_builder_config_loader_builds_supported_objects_from_json_specs,
        test_builder_config_loader_rejects_unsupported_types,
        test_builder_config_loader_keeps_tensorflow_callback_specs_raw_for_backend_resolution,
        test_builder_config_loader_rejects_unknown_callback_for_non_tensorflow_backend,
    ]

    ok = 0
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
        ok += 1

    print(f"Summary: {ok}/{len(tests)} passed")
