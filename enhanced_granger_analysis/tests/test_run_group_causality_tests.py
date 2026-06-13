import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def test_run_group_causality_script_with_two_learning_rates():
    project_root = Path(__file__).resolve().parents[2]
    script_path = (
        project_root
        / "complex_granger_analysis"
        / "scripts"
        / "run_group_causality_tests.py"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Sample dataset with x1 and x2
        data = pd.DataFrame(
            {
                "x1": [0.0, 1.0, 0.5, 1.5, 1.2, 0.7, 1.8, 1.1, 0.9, 1.4],
                "x2": [1.0, 0.5, 1.3, 0.2, 0.8, 1.1, 0.4, 0.9, 1.2, 0.6],
            }
        )
        data_path = tmp_path / "data.csv"
        data.to_csv(data_path)

        # Ground truth matrix [[1,0],[0,1]]
        gt = pd.DataFrame(
            [[1, 0], [0, 1]],
            index=["x1", "x2"],
            columns=["x1", "x2"],
        )
        gt_path = tmp_path / "gt.csv"
        gt.to_csv(gt_path)

        # Group config with two learning-rate cases
        group_cfg = {
            "base_config": {
                "backend": "sklearn",
                "causes": ["x1", "x2"],
                "effects": ["x1", "x2"],
                "tested_causes": ["x1", "x2"],
                "lag_config": {"max_lag": 1, "use_lag_zero": False},
                "model_config": {
                    "max_iter": 50,
                    "batch_size": None,
                    "learning_rate": 0.01,
                    "verbose": 0,
                },
            },
            "sweep": {
                "param_names": ["model_config.learning_rate"],
                "cases": [[0.01], [0.001]],
            },
        }
        group_cfg_path = tmp_path / "group_config.json"
        group_cfg_path.write_text(json.dumps(group_cfg), encoding="utf-8")

        # Script config
        results_dir = tmp_path / "results"
        script_cfg = {
            "output_dir": str(results_dir),
            "ground_truth_path": str(gt_path),
            "group_config_path": str(group_cfg_path),
            "threshold": 0.01,
            "data": {
                "csv_paths": [str(data_path)],
                "index_col": 0,
            },
        }
        script_cfg_path = tmp_path / "script_config.json"
        script_cfg_path.write_text(json.dumps(script_cfg), encoding="utf-8")

        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(script_cfg_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(project_root))

        assert results_dir.exists(), "results folder was not created"

        matrix_files = list(results_dir.glob("*.csv"))
        matrix_files = [p for p in matrix_files if p.name != "summary.csv"]
        assert len(matrix_files) == 2, "Expected two result matrix files"
        for path in matrix_files:
            assert "learning_rate_" in path.name
            assert "model_config_learning_rate" not in path.name
            assert "wartosc" not in path.name

        summary_path = results_dir / "summary.csv"
        assert summary_path.exists(), "summary.csv was not created"

        summary_df = pd.read_csv(summary_path)
        assert len(summary_df) == 2, "Expected two rows in summary for two tests"
        assert "execution_time_seconds" in summary_df.columns


def test_run_group_causality_script_uses_default_config_when_not_provided():
    project_root = Path(__file__).resolve().parents[2]
    script_path = (
        project_root
        / "complex_granger_analysis"
        / "scripts"
        / "run_group_causality_tests.py"
    )

    # Verify CLI can be called without --config and defaults to sample config.
    cmd = [sys.executable, str(script_path)]
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr

    summary_path = (
        project_root
        / "complex_granger_analysis"
        / "scripts"
        / "results"
        / "summary.csv"
    )
    assert summary_path.exists(), "Default config run did not create summary.csv"


if __name__ == "__main__":
    test_run_group_causality_script_with_two_learning_rates()
    print("PASS: test_run_group_causality_script_with_two_learning_rates")
