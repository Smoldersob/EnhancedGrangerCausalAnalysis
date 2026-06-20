from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import sys
import numpy as np
import pandas as pd
try:
    from ..utilities.metric_calculator import MetricCalculator
except ImportError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from enhanced_granger_analysis.utilities.metric_calculator import MetricCalculator


def binarize_signed_matrix(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, index_col=0)
    df = df.astype(float).fillna(0)
    df = (df != 0).astype(int)
    df.to_csv(output_path)


def recalculate_summary(
    output_dir: Path,
    ground_truth_path: Path,
    summary_name: str = "summary.csv",
    output_name: str = "summary_recalculated.csv",
    fixed_suffix: str = "_binary_fixed",
    save_fixed_matrices: bool = True,
) -> Path:
    summary_path = output_dir / summary_name
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    summary_df = pd.read_csv(summary_path)
    repaired_rows: List[Dict[str, Any]] = []

    metric_columns = [
        "tp", "fp", "tn", "fn",
        "fdr", "tpr", "fpr", "shd",
        "accuracy", "precision", "recall", "f1"
    ]

    fixed_dir = output_dir / "recalculated_binary_matrices"
    if save_fixed_matrices:
        fixed_dir.mkdir(parents=True, exist_ok=True)

    for _, row in summary_df.iterrows():
        row_dict = row.to_dict()
        causality_file = row_dict.get("causality_file")

        if not isinstance(causality_file, str) or causality_file == "FAILED":
            repaired_rows.append(row_dict)
            continue

        source_matrix_path = output_dir / causality_file
        if not source_matrix_path.exists():
            row_dict["error"] = f"Missing causality file: {causality_file}"
            repaired_rows.append(row_dict)
            continue

        if save_fixed_matrices:
            fixed_matrix_path = fixed_dir / f"{source_matrix_path.stem}{fixed_suffix}.csv"
        else:
            fixed_matrix_path = output_dir / f"__tmp__{source_matrix_path.stem}{fixed_suffix}.csv"

        try:
            binarize_signed_matrix(source_matrix_path, fixed_matrix_path)
            metrics = MetricCalculator(str(ground_truth_path), str(fixed_matrix_path)).evaluate()

            for col in metric_columns:
                row_dict[col] = metrics[col]

            if save_fixed_matrices:
                row_dict["recalculated_causality_file"] = fixed_matrix_path.relative_to(output_dir).as_posix()

        except Exception as e:
            row_dict["error"] = str(e)

        finally:
            if not save_fixed_matrices and fixed_matrix_path.exists():
                fixed_matrix_path.unlink()

        repaired_rows.append(row_dict)

    repaired_df = pd.DataFrame(repaired_rows)
    output_path = output_dir / output_name
    repaired_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate summary metrics from signed causality matrices using the provided MetricCalculator."
    )
    parser.add_argument("--output-dir", required=True, help="Directory containing summary.csv and causality matrices")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV")
    parser.add_argument("--summary-name", default="summary.csv", help="Input summary filename")
    parser.add_argument("--output-name", default="summary_recalculated.csv", help="Output summary filename")
    parser.add_argument(
        "--no-save-fixed-matrices",
        action="store_true",
        help="Do not keep binarized repaired matrices"
    )
    args = parser.parse_args()

    output_path = recalculate_summary(
        output_dir=Path(args.output_dir).resolve(),
        ground_truth_path=Path(args.ground_truth).resolve(),
        summary_name=args.summary_name,
        output_name=args.output_name,
        save_fixed_matrices=not args.no_save_fixed_matrices,
    )
    print(output_path)


if __name__ == "__main__":
    main()