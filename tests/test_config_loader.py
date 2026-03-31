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


if __name__ == "__main__":
    tests = [
        test_builder_config_loader_json_and_lag_config_conversion,
        test_test_group_iterator_next_and_has_next,
        test_test_group_iterator_without_sweep_produces_one_config,
    ]

    ok = 0
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
        ok += 1

    print(f"Summary: {ok}/{len(tests)} passed")
