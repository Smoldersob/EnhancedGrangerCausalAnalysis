"""
Script for running Granger causality test sweeps from configuration files.

This script:
1. Loads multiple datasets (DataFrames) from CSV files.
2. Iterates through group configurations (sweep of parameter variations).
3. For each configuration, runs Granger analysis via MultitaskGrangerBuilder.
4. Saves results according to --save mode:
    - minimum: binary causality matrix + summary.csv
    - matrices: binary, p-values, F-test, sign + summary.csv
5. Computes metrics against ground truth (TP, FP, accuracy, F1, etc.).
6. Produces a summary CSV with timing and metrics for each test configuration.

Typical workflow:
    python run_group_causality_tests.py --config script_config.json --save minimum

Config structure (scripts/run_group_causality_tests.config.json):
    {
        "output_dir": "./results",  # folder for result matrices and summary.csv
        "ground_truth_path": "./ground_truth.csv",  # reference causality matrix
        "group_config_path": "./group_config.json",  # test sweep config
        "threshold": 0.01,  # p-value threshold for binary causality
        "data": {
            "csv_paths": ["./data.csv"],  # list of CSV files to load
            "index_col": 0  # column index (or name) for row index
        }
    }

Results in output_dir:
    - minimum mode: case_XXX_causality.csv + summary.csv
    - matrices mode: case_XXX_causality.csv, case_XXX_p_value.csv,
      case_XXX_f_test.csv, case_XXX_sign.csv + summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..api import MultiTaskGrangerBuilder, TestGroupConfigIterator
from ..api.config_loader import BuilderConfigLoader
from ..utilities.metric_calculator import MetricCalculator
from .. import initializers as init_initializers


def _sanitize_token(value: Any) -> str:
    """Convert arbitrary value into a safe filename token (alphanumeric, -, _)."""
    s = str(value)
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]", "_", s)
    return s


def _short_param_name(param_name: str) -> str:
    """Extract last segment from dotted parameter name (e.g., 'model_config.epochs' -> 'epochs')."""
    return param_name.split(".")[-1]


def _freeze_for_signature(value: Any) -> Any:
    """Convert nested config values into hashable signature components."""
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_for_signature(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_for_signature(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_for_signature(item) for item in value)
    if hasattr(value, "__dict__"):
        public_items = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public_items:
            return (type(value).__name__, _freeze_for_signature(public_items))
    return value


def _data_signature(data_frames: List[pd.DataFrame]) -> Tuple[Any, ...]:
    """Build a lightweight signature for the loaded inputs."""
    return tuple((tuple(frame.columns), frame.shape) for frame in data_frames)


def _preparation_signature(cfg: Dict[str, Any], data_frames: List[pd.DataFrame]) -> Tuple[Any, ...]:
    """Build a cache key for prepared data that changes with lag or selector settings."""
    return (
        _data_signature(data_frames),
        _freeze_for_signature(cfg.get("effects")),
        _freeze_for_signature(cfg.get("lag_config")),
        _freeze_for_signature(cfg.get("lag_selector")),
        _freeze_for_signature(cfg.get("x_scaler")),
        _freeze_for_signature(cfg.get("y_scaler")),
        _freeze_for_signature(cfg.get("backend_sample_fraction", 1.0)),
        _freeze_for_signature(cfg.get("backend_max_samples")),
    )


def _apply_compute_device(cfg: Dict[str, Any]) -> None:
    """Apply a CPU/GPU preference from group config before model creation."""
    device_spec = cfg.get("compute_device", cfg.get("device"))
    if device_spec is None:
        return

    backend = str(cfg.get("backend", "")).strip().lower()
    device_text = str(device_spec).strip().lower()
    if not device_text:
        return

    if backend == "tensorflow":
        if device_text in {"cpu", "cpu-only"}:
            os.environ["CGA_TF_FORCE_CPU"] = "1"
            os.environ["CGA_TF_USE_GPU"] = "0"
        elif device_text in {"gpu", "cuda"} or device_text.startswith("cuda"):
            os.environ["CGA_TF_FORCE_CPU"] = "0"
            os.environ["CGA_TF_USE_GPU"] = "1"
        elif device_text == "auto":
            os.environ.pop("CGA_TF_FORCE_CPU", None)
            os.environ.pop("CGA_TF_USE_GPU", None)
        return

    if backend == "pytorch":
        model_cfg = cfg.get("model_config")
        if not isinstance(model_cfg, dict):
            model_cfg = {}
            cfg["model_config"] = model_cfg
        if "device" not in model_cfg:
            if device_text in {"gpu", "cuda"}:
                model_cfg["device"] = "cuda"
            elif device_text.startswith("cuda"):
                model_cfg["device"] = device_spec
            elif device_text == "cpu":
                model_cfg["device"] = "cpu"
            elif device_text != "auto":
                model_cfg["device"] = device_spec


def _build_result_filename(case_idx: int, backend: str, param_names: List[str], case_values: List[Any], suffix: str = "causality") -> str:
    """
    Build descriptive result filename for a single test case.
    
    Args:
        case_idx: numeric index of the test case (e.g., 0, 1, 2)
        backend: backend name (pytorch, tensorflow, sklearn)
        param_names: list of swept parameter names
        case_values: list of values for this specific case
        suffix: type of result (causality, p_value, f_test, sign)
    
    Returns:
        filename like "case_000_causality.csv" or "case_001_pytorch_adam_0.001_causality.csv"
    """
    # Base: case index
    parts = [f"case_{case_idx:03d}"]
    
    # Optionally add backend and param values for richer naming
    if backend and backend.lower() not in {"none", "unknown", "auto"}:
        parts.append(_sanitize_token(backend))
    
    for p, v in zip(param_names, case_values):
        parts.append(_sanitize_token(_short_param_name(p)))
        parts.append(_sanitize_token(v))
    
    # Suffix (causality, p_value, f_test, sign)
    parts.append(suffix)
    
    return "_".join(parts) + ".csv"


def _load_dataframes(cfg: Dict[str, Any]) -> List[pd.DataFrame]:
    """
    Load list of DataFrames from CSV paths specified in script config.
    
    Args:
        cfg: script config dict with 'data' section containing 'csv_paths' and optional 'index_col'
    
    Returns:
        List of loaded DataFrames in order of csv_paths
    
    Raises:
        ValueError: if config structure is invalid
        FileNotFoundError: if any CSV file does not exist
    """
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
        frames.append(pd.read_csv(csv_path, sep=';', index_col=index_col))
    
    return frames


def _align_dataframe_columns(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align all frames to the union of columns and fill missing values with zeros."""
    if not frames:
        return frames

    all_columns: List[str] = []
    seen: set[str] = set()
    changed = False
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                seen.add(column)
                all_columns.append(column)
        if list(frame.columns) != all_columns:
            changed = True

    aligned: List[pd.DataFrame] = []
    for frame in frames:
        aligned_frame = frame.reindex(columns=all_columns, fill_value=0)
        aligned.append(aligned_frame)

    return aligned if changed else frames


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    """
    Resolve relative path against base_dir, or return absolute path as-is.
    
    Args:
        base_dir: reference directory for relative paths
        raw_path: relative or absolute path
    
    Returns:
        Resolved absolute Path
    """
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def run_from_config(script_config_path: str | Path, save_mode: str = "minimum") -> Path:
    """
    Run Granger analysis test sweep from configuration file.

    Args:
        script_config_path: path to script configuration JSON/YAML file
        save_mode: save strategy for per-case matrices:
            - "minimum": save only binary causality matrix (+ summary.csv at the end)
            - "matrices": save all matrices (binary, p-value, f-test, sign) (+ summary.csv)

    Returns:
        Path to generated summary.csv file

    Raises:
        FileNotFoundError: if required config files or data files not found
        ValueError: if config structure is invalid
    """
    config_path = Path(script_config_path).resolve()
    config_dir = config_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Script config file not found: {config_path}")

    print(f"\n{'='*70}")
    print(f"Loading script config from: {config_path}")
    script_cfg = BuilderConfigLoader.load_raw_file(config_path)

    # Resolve paths relative to config directory
    output_dir = _resolve_path(config_dir, script_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

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

    # Resolve CSV paths
    data_cfg = script_cfg.get("data")
    if isinstance(data_cfg, dict):
        csv_paths = data_cfg.get("csv_paths")
        if isinstance(csv_paths, str):
            data_cfg["csv_paths"] = [str(_resolve_path(config_dir, csv_paths))]
        elif isinstance(csv_paths, list):
            data_cfg["csv_paths"] = [str(_resolve_path(config_dir, p)) for p in csv_paths]

    # Load data
    print(f"Loading data from CSV files...")
    data_frames = _load_dataframes(script_cfg)
    aligned_frames = _align_dataframe_columns(data_frames)
    if aligned_frames is not data_frames:
        print("  ✓ Aligned DataFrame columns across all inputs; missing values filled with 0")
    data_frames = aligned_frames
    print(f"  ✓ Loaded {len(data_frames)} DataFrame(s)")

    # Load group config and extract sweep parameters
    print(f"Loading group config from: {group_config_path}")
    group_raw = BuilderConfigLoader.load_raw_file(group_config_path)
    sweep = group_raw.get("sweep", {})
    param_names = list(sweep.get("param_names", []))
    cases = list(sweep.get("cases", []))
    
    if cases:
        print(f"  Sweep: {len(cases)} configuration case(s)")
        for k, v in zip(param_names, cases[0] if cases else []):
            print(f"    - {_short_param_name(k)}")
    else:
        print(f"  No sweep parameters; running single configuration")

    # Iterate through test group configurations
    iterator = TestGroupConfigIterator.from_file(group_config_path)
    prepared_data_cache: Dict[Tuple[Any, ...], Any] = {}
    
    summary_rows: List[Dict[str, Any]] = []
    case_idx = 0
    total_time = 0.0

    print(f"\n{'='*70}")
    print(f"Running test cases...")
    print(f"{'='*70}\n")

    while iterator.has_next():
        cfg = iterator.next()
        _apply_compute_device(cfg)
        case_values = cases[case_idx] if case_idx < len(cases) else []
        
        backend = str(cfg.get("backend", "unknown"))
        reuse_data = bool(cfg.get("reuse_data"))
        prep_signature = _preparation_signature(cfg, data_frames)
        cached_prepared_data_entry = prepared_data_cache.get(prep_signature) if reuse_data else None
        cached_prepared_data = None
        cached_preparation_time = 0.0
        if isinstance(cached_prepared_data_entry, dict):
            cached_prepared_data = cached_prepared_data_entry.get("prepared_data")
            cached_preparation_time = float(cached_prepared_data_entry.get("preparation_time_seconds", 0.0))
        
        # Build filenames for all result matrices
        base_filename = _build_result_filename(case_idx, backend, param_names, case_values)
        causality_path = output_dir / base_filename
        p_value_path = output_dir / _build_result_filename(case_idx, backend, param_names, case_values, "p_value")
        f_test_path = output_dir / _build_result_filename(case_idx, backend, param_names, case_values, "f_test")
        sign_path = output_dir / _build_result_filename(case_idx, backend, param_names, case_values, "sign")
        
        # Print case header
        case_desc = ", ".join(
            f"{_short_param_name(k)}={v}" for k, v in zip(param_names, case_values)
        ) if case_values else "default"
        print(f"[Case {case_idx:3d}] {backend:15s} | {case_desc}")
        
        # Run Granger analysis
        start = time.perf_counter()
        try:
            # Resolve initializer string (from JSON) into actual initializer class when provided
            init_spec = cfg.get("initializer")
            if isinstance(init_spec, str):
                name = init_spec.strip().lower()
                if name in {"olsinitializer", "ols"}:
                    cfg["initializer"] = init_initializers.OLSInitializer
                elif name in {"zerosinitializer", "zeros", "zero"}:
                    cfg["initializer"] = init_initializers.ZerosInitializer
                elif name in {"randomnormalinitializer", "randomnormal", "random_normal", "random"}:
                    cfg["initializer"] = init_initializers.RandomNormalInitializer

            builder = MultiTaskGrangerBuilder().from_config(cfg).data(data_frames)
            if cached_prepared_data is not None:
                builder.prepared_data(cached_prepared_data)
            out = builder.fit()
            
            # Extract and save all result matrices
            causality_df = out.results.result(threshold=threshold, with_sign=True)
            p_value_df = out.results.p_value
            f_test_df = out.results.F_test
            sign_df = out.results.sign

            # Measure per-case execution time excluding file-save operations.
            elapsed_s = time.perf_counter() - start
            if cached_prepared_data is not None:
                elapsed_s += cached_preparation_time
            total_time += elapsed_s
            
            causality_df.to_csv(causality_path)
            if save_mode == "matrices":
                p_value_df.to_csv(p_value_path)
                f_test_df.to_csv(f_test_path)
                sign_df.to_csv(sign_path)

            if reuse_data and out.prepared_data is not None:
                prepared_data_cache[prep_signature] = {
                    "prepared_data": out.prepared_data,
                    "preparation_time_seconds": out.preparation_time_seconds,
                }
            
            # Compute metrics against ground truth
            metrics = MetricCalculator(str(ground_truth_path), str(causality_path)).evaluate()
            
            # Build summary row
            row: Dict[str, Any] = {
                "case_id": case_idx,
                "backend": backend,
                "causality_file": causality_path.name,
                "p_value_file": p_value_path.name if save_mode == "matrices" else "NOT_SAVED",
                "f_test_file": f_test_path.name if save_mode == "matrices" else "NOT_SAVED",
                "sign_file": sign_path.name if save_mode == "matrices" else "NOT_SAVED",
                "execution_time_seconds": round(elapsed_s, 2),
            }
            for p, v in zip(param_names, case_values):
                row[p] = v
            row.update(metrics)
            summary_rows.append(row)
            
            # Print timing
            print(f"        ✓ Completed in {elapsed_s:.2f}s | Accuracy: {metrics.get('accuracy', 0):.3f} | F1: {metrics.get('f1', 0):.3f}")
        
        except Exception as e:
            elapsed_s = time.perf_counter() - start
            print(f"        ✗ FAILED after {elapsed_s:.2f}s: {e}")
            # Still create a row but mark as failed
            row: Dict[str, Any] = {
                "case_id": case_idx,
                "backend": backend,
                "causality_file": "FAILED",
                "p_value_file": "FAILED",
                "f_test_file": "FAILED",
                "sign_file": "FAILED",
                "execution_time_seconds": round(elapsed_s, 2),
                "error": str(e),
            }
            for p, v in zip(param_names, case_values):
                row[p] = v
            summary_rows.append(row)

        case_idx += 1

    # Save summary
    print(f"\n{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Saved summary to: {summary_path}")
    print(f"Total execution time: {total_time:.2f}s across {case_idx} case(s)")
    print(f"Average time per case: {total_time/case_idx if case_idx > 0 else 0:.2f}s")
    print(f"{'='*70}\n")
    
    return summary_path


def main() -> None:
    """
    Main entry point: parse command-line arguments and run test sweep.
    
    Usage:
        python run_group_causality_tests.py --config scripts/run_group_causality_tests.config.json
    
    If --config is not provided, defaults to run_group_causality_tests.config.json in the script directory.
    """
    parser = argparse.ArgumentParser(
        description="Run Granger test sweeps from JSON/YAML configuration files. "
        "Loads data, iterates through parameter sweeps, runs Granger analysis, "
        "computes metrics against ground truth, and saves summary."
    )
    default_config = Path(__file__).with_name("run_group_causality_tests.config.json")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to script configuration JSON/YAML (default: sample config next to this script)",
    )
    parser.add_argument(
        "--save",
        default="minimum",
        choices=["minimum", "matrices"],
        help=(
            "Saving mode for per-case outputs: "
            "'minimum' saves only binary causality matrix; "
            "'matrices' saves binary, p-value, f-test, and sign matrices"
        ),
    )
    args = parser.parse_args()

    try:
        summary_path = run_from_config(args.config, save_mode=args.save)
        print(f"\n✓ SUCCESS: Results saved to {summary_path}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
