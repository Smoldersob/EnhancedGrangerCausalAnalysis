from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complex_granger_analysis.api import MultitaskGrangerBuilder, TestGroupConfigIterator
from complex_granger_analysis.api.config_loader import BuilderConfigLoader
from complex_granger_analysis.utilities.metric_calculator import MetricCalculator


def _sanitize_token(value: Any) -> str:
    s = str(value)
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]", "_", s)
    return s


def _short_param_name(param_name: str) -> str:
    return param_name.split(".")[-1]


def _build_result_filename(backend: str, param_names: List[str], case_values: List[Any]) -> str:
    parts = [_sanitize_token(backend)]
    for p, v in zip(param_names, case_values):
        parts.append(_sanitize_token(_short_param_name(p)))
        parts.append(_sanitize_token(v))
    return "_".join(parts) + ".csv"


def _load_dataframes(cfg: Dict[str, Any]) -> List[pd.DataFrame]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("script config must contain 'data' mapping")

    csv_paths = data_cfg.get("csv_paths")
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    if not isinstance(csv_paths, list) or not csv_paths:
        raise ValueError("script config data.csv_paths must be a non-empty list")

    index_col = data_cfg.get("index_col", None)
    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        csv_path = Path(p)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV input file not found: {csv_path}. "
                "Update 'data.csv_paths' in script config or provide --config with valid paths."
            )
        frames.append(pd.read_csv(csv_path, index_col=index_col))
    return frames


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def run_from_config(script_config_path: str | Path) -> Path:
    config_path = Path(script_config_path).resolve()
    config_dir = config_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Script config file not found: {config_path}")

    script_cfg = BuilderConfigLoader.load_raw_file(config_path)

    output_dir = _resolve_path(config_dir, script_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = _resolve_path(config_dir, script_cfg["ground_truth_path"])
    threshold = float(script_cfg.get("threshold", 0.01))
    group_config_path = _resolve_path(config_dir, script_cfg["group_config_path"])

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {ground_truth_path}. "
            "Update 'ground_truth_path' in script config."
        )
    if not group_config_path.exists():
        raise FileNotFoundError(
            f"Group config file not found: {group_config_path}. "
            "Update 'group_config_path' in script config."
        )

    data_cfg = script_cfg.get("data")
    if isinstance(data_cfg, dict):
        csv_paths = data_cfg.get("csv_paths")
        if isinstance(csv_paths, str):
            data_cfg["csv_paths"] = [str(_resolve_path(config_dir, csv_paths))]
        elif isinstance(csv_paths, list):
            data_cfg["csv_paths"] = [str(_resolve_path(config_dir, p)) for p in csv_paths]

    data_frames = _load_dataframes(script_cfg)

    group_raw = BuilderConfigLoader.load_raw_file(group_config_path)
    sweep = group_raw.get("sweep", {})
    param_names = list(sweep.get("param_names", []))
    cases = list(sweep.get("cases", []))

    iterator = TestGroupConfigIterator.from_file(group_config_path)

    summary_rows: List[Dict[str, Any]] = []
    case_idx = 0

    while iterator.has_next():
        cfg = iterator.next()
        case_values = cases[case_idx] if case_idx < len(cases) else []

        backend = str(cfg.get("backend", "unknown"))
        filename = _build_result_filename(backend, param_names, case_values)
        pred_path = output_dir / filename

        start = time.time()
        out = MultitaskGrangerBuilder().from_config(cfg).data(data_frames).fit()
        pred_df = out.results.result(threshold=threshold)
        pred_df.to_csv(pred_path)
        elapsed_s = int(round(time.time() - start))

        metrics = MetricCalculator(str(ground_truth_path), str(pred_path)).evaluate()

        row: Dict[str, Any] = {
            "backend": backend,
            "prediction_file": filename,
            "execution_time_seconds": elapsed_s,
        }
        for p, v in zip(param_names, case_values):
            row[p] = v
        row.update(metrics)
        summary_rows.append(row)

        case_idx += 1

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Granger test sweeps from JSON/YAML configuration files."
    )
    default_config = Path(__file__).with_name("run_group_causality_tests.config.json")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to script configuration JSON/YAML (default: sample config next to this script)",
    )
    args = parser.parse_args()

    summary_path = run_from_config(args.config)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
