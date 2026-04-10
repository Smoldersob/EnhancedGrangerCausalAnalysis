# Running Group Granger Tests (Script Guide)

This document explains how to use `run_group_causality_tests.py` to run multiple Granger analysis configurations and generate summarized results with metrics.

## Overview

The script performs the following workflow:

1. **Load Configuration** — reads script config file specifying data paths, ground truth, and output directory
2. **Load Data** — reads CSV files into a list of DataFrames
3. **Iterate Test Cases** — expands parameter sweep from group_config.json file
4. **Run Analysis** — for each configuration, executes Granger analysis via `MultitaskGrangerBuilder`
5. **Save Results** — stores causality matrices, p-values, F-test statistics, and sign information for each case
6. **Compute Metrics** — evaluates predictions against ground truth (TP, FP, FN, accuracy, F1, precision, recall, etc.)
7. **Generate Summary** — outputs summary.csv with timing and metrics across all test cases

## Configuration Structure

### Script Configuration File

**Location:** `scripts/run_group_causality_tests.config.json`

**Structure:**
```json
{
  "output_dir": "./results",
  "ground_truth_path": "./ground_truth.csv",
  "group_config_path": "./group_config.json",
  "threshold": 0.01,
  "data": {
    "csv_paths": ["./data.csv"],
    "index_col": 0
  }
}
```

**Parameters:**

| Field | Type | Description |
|-------|------|-------------|
| `output_dir` | string (path) | Folder where results are saved (causality matrices, summary.csv) |
| `ground_truth_path` | string (path) | CSV file with reference causality matrix (for metric computation) |
| `group_config_path` | string (path) | Group test configuration file (with base config + sweep params) |
| `threshold` | float | P-value threshold for binary causality classification (default: 0.01) |
| `data.csv_paths` | array of paths | List of CSV files to load as input data |
| `data.index_col` | int or null | Column index/name to use as row index (null = integer index) |

**Path Resolution:**
- All paths are resolved relative to the script config directory.
- Absolute paths are used as-is.
- Example: if config is in `scripts/`, then `"./data.csv"` resolves to `scripts/data.csv`

### Group Test Configuration File

**Location:** Specified via `group_config_path` in script config (default: `scripts/group_config.json`)

**Structure:**
```json
{
  "base_config": {
    "backend": "pytorch",
    "lag_config": { "max_lag": 8 },
    "model_config": { "epochs": 50, "batch_size": 32 },
    "callbacks": [ ... ],
    "relations": [ ... ]
  },
  "sweep": {
    "param_names": ["model_config.epochs", "lag_config.max_lag"],
    "cases": [
      [5, 8],
      [10, 10],
      [20, 12]
    ]
  }
}
```

**Components:**

- **`base_config`** — Starting configuration merged with each sweep case. Supports all builder parameters:
  - `backend`, `lag_config`, `lag_selector`, `model_config`, `regularizer`, `callbacks`, `relations`, etc.
  - See [Configuration File Usage](config_file_usage.md) for full format.

- **`sweep.param_names`** — List of dotted parameter names to vary (e.g., `"model_config.epochs"`)

- **`sweep.cases`** — Array of cases; each case is an array of values matching `param_names` order

**Example Interpretation:**
```
base_config: epochs=50, max_lag=8
sweep cases:
  [1] epochs=5,  max_lag=8   (vary only epochs from base)
  [2] epochs=10, max_lag=10  (vary both)
  [3] epochs=20, max_lag=12  (vary both)
```

For details on sweep expansion, see [Test Group Configuration Usage](test_group_config_usage.md).

## Usage

### Basic Invocation

```bash
cd scripts
python run_group_causality_tests.py --config run_group_causality_tests.config.json
```

### With Custom Config

```bash
python run_group_causality_tests.py --config my_custom_config.json
```

### Default Behavior

If `--config` is omitted, the script looks for `run_group_causality_tests.config.json` in the same directory:

```bash
python run_group_causality_tests.py
```

## Output

### Directory Structure

After running, the output directory contains:

```
results/
├── summary.csv                          # Aggregated metrics for all test cases
├── case_000_causality.csv               # Binary causality matrix (threshold: 0.01)
├── case_000_p_value.csv                 # P-value matrix
├── case_000_f_test.csv                  # F-test statistics matrix
├── case_000_sign.csv                    # Sign of strongest coefficient
├── case_001_pytorch_adam_0.001_causality.csv
├── case_001_pytorch_adam_0.001_p_value.csv
├── case_001_pytorch_adam_0.001_f_test.csv
├── case_001_pytorch_adam_0.001_sign.csv
└── ... (additional cases)
```

**Filename Pattern:**

Each result set has four matrix files:
- `case_NNN_*_causality.csv` — Binary/signed causality (1/-1/0)
- `case_NNN_*_p_value.csv` — P-value for each (effect, cause) pair
- `case_NNN_*_f_test.csv` — F-test statistic
- `case_NNN_*_sign.csv` — Sign of the strongest lag coefficient

Optional suffix in filename:
- `case_000_causality.csv` — Case 0, no backend/param info (when sweep is empty or minimal)
- `case_001_pytorch_adam_0.001_causality.csv` — Case 1, pytorch backend, adam optimizer, learning_rate=0.001

### Summary CSV

**Location:** `results/summary.csv`

**Columns:**
- `case_id` — Numeric index of the test case (0, 1, 2, ...)
- `backend` — Backend used (pytorch, tensorflow, sklearn)
- `causality_file` — Filename of the binary causality matrix
- `p_value_file` — Filename of the p-value matrix
- `f_test_file` — Filename of the F-test matrix
- `sign_file` — Filename of the sign matrix
- `execution_time_seconds` — Wall-clock time for this configuration
- `param_names...` — Parameter values for this case (e.g., `model_config.epochs`, `lag_config.max_lag`)
- `tp`, `fp`, `tn`, `fn` — Confusion matrix components (True Positive, False Positive, etc.)
- `accuracy` — $(TP + TN) / N$
- `precision` — $TP / (TP + FP)$
- `recall` (TPR) — $TP / (TP + FN)$
- `f1` — Harmonic mean of precision and recall
- `tpr` — True Positive Rate = recall
- `fpr` — False Positive Rate
- `fdr` — False Discovery Rate
- `shd` — Structural Hamming Distance (number of differing edges)

Example row:
```
case_id,backend,causality_file,...,epochs,max_lag,tp,fp,accuracy,f1
0,pytorch,case_000_causality.csv,...,5,8,8,2,0.833,0.89
1,pytorch,case_001_causality.csv,...,10,10,9,1,0.909,0.92
```

## Data Format Requirements

### Input CSV Files

- **Format:** Standard pandas DataFrame CSV (comma-separated by default)
- **Index Column:** Specified in config via `data.index_col` (default: 0 = first column)
- **Columns:** Variable names (causes, effects, etc.)
- **Values:** Numeric (float or int)
- **Rows:** Time-ordered observations (first row = earliest time)

Example `data.csv`:
```
time,x1,x2,y
0,1.0,0.5,1.5
1,1.1,0.6,1.6
2,1.2,0.7,1.7
...
```

### Ground Truth CSV

- **Format:** Same as input CSVs (comma-separated)
- **Rows:** Effect variables
- **Columns:** Cause variables
- **Values:** Causality indicators (typically 0 = no relation, 1 = causal relation)
- **Note:** Script compares binarized predictions (0/1) against this matrix

Example `ground_truth.csv`:
```
,x1,x2
y,1,0
x1,0,1
```

Meaning:
- y is caused by x1 ✓ and not by x2 ✗
- x1 is caused by x2 ✓

## Example Workflow

### Step 1: Prepare Data

```bash
cd scripts
# Ensure these files exist:
# - data.csv (time series with x1, x2, y columns)
# - ground_truth.csv (reference causality matrix)
# - group_config.json (test sweep configuration)
```

### Step 2: Configure Test Sweep

Edit `group_config.json`:
```json
{
  "base_config": {
    "backend": "sklearn",
    "causes": ["x1", "x2"],
    "effects": ["y"],
    "tested_causes": ["x1", "x2"],
    "lag_config": { "max_lag": 5 },
    "model_config": { "max_iter": 100 }
  },
  "sweep": {
    "param_names": ["model_config.max_iter"],
    "cases": [[50], [100], [200]]
  }
}
```

### Step 3: Run Script

```bash
python run_group_causality_tests.py --config run_group_causality_tests.config.json
```

Output:
```
======================================================================
Loading script config from: /home/user/my_pro/complex_granger_analysis/scripts/run_group_causality_tests.config.json
Output directory: /home/user/my_pro/complex_granger_analysis/scripts/results
Loading data from CSV files...
  ✓ Loaded 1 DataFrame(s)
Loading group config from: /home/user/my_pro/complex_granger_analysis/scripts/group_config.json
  Sweep: 3 configuration case(s)
    - max_iter

======================================================================
Running test cases...
======================================================================

[Case   0] sklearn         | max_iter=50
        ✓ Completed in 0.85s | Accuracy: 0.800 | F1: 0.857
[Case   1] sklearn         | max_iter=100
        ✓ Completed in 1.23s | Accuracy: 0.850 | F1: 0.889
[Case   2] sklearn         | max_iter=200
        ✓ Completed in 1.95s | Accuracy: 0.900 | F1: 0.923

======================================================================
Saved summary to: /home/user/my_pro/complex_granger_analysis/scripts/results/summary.csv
Total execution time: 4.03s across 3 case(s)
Average time per case: 1.34s
======================================================================

✓ SUCCESS: Results saved to results/summary.csv
```

### Step 4: Analyze Results

```bash
# View summary
cat results/summary.csv

# Examine best-performing case
head -1 results/summary.csv  # header
head -2 results/summary.csv | tail -1  # best case by accuracy
cat results/case_001_causality.csv  # causality matrix for case 1
```

## Error Handling

### Failed Cases

If a test case fails:
- The script prints `✗ FAILED` with the error message.
- A row is still added to summary.csv with `causality_file = "FAILED"` and an error description.
- Execution continues to the next test case.

Example:
```
[Case   3] pytorch         | epochs=50, lr=0.0001
        ✗ FAILED after 2.15s: CUDA out of memory
```

In `summary.csv`:
```
case_id,backend,causality_file,...
3,pytorch,FAILED,...,error="CUDA out of memory"
```

### Missing Config/Data

If required files are not found:
- Script prints error message to stderr and exits with status 1.
- Example:

```
✗ ERROR: CSV input file not found: ./data.csv
```

## Troubleshooting

### "FileNotFoundError: Ground-truth CSV not found"

Check:
1. `ground_truth_path` in script config is correct and file exists.
2. Path is relative to script config directory.

### "Unknown cause/effect" errors

Check:
1. Variable names in `base_config.causes`/`effects` match column names in data CSV.
2. Data CSV has the required columns.

### "Unsupported backend"

Check:
1. `backend` in base config is one of: `pytorch`, `tensorflow`, `sklearn`.
2. Required dependencies are installed (tensorflow, torch, scikit-learn).

### Very slow execution

Check:
1. `model_config.epochs` or `model_config.max_iter` is set very high.
2. Dataset is very large; consider reducing `lag_config.max_lag` or upsampling less frequently.
3. Backend is using CPU; for GPU-based backends, ensure CUDA/GPU is configured.

## Tips & Best Practices

1. **Start Small** — Begin with a minimal group config (one or few sweep cases) to validate configuration.

2. **Monitor Timing** — Check the execution timings in summary.csv. If they increase significantly across cases, investigate the backend or data size.

3. **Validate Ground Truth** — Ensure `ground_truth.csv` matches the true causal structure. Inaccurate ground truth leads to misleading metrics.

4. **Use Stable Backend** — For reproducible results, fix `random_seed` or equivalent in `model_config` if your backend supports it.

5. **Organize Results** — Keep dated or tagged output directories to avoid overwriting previous runs:
   ```bash
   python run_group_causality_tests.py --config configs/exp_2025_04_10.json
   ```

6. **Inspect Matrices** — Examine individual causality/p_value matrices in addition to summary metrics:
   ```python
   import pandas as pd
   df = pd.read_csv("results/case_000_causality.csv", index_col=0)
   print(df)
   ```

## References

- [API Usage Guide](api_usage.md) — Detailed API documentation
- [Configuration File Usage](config_file_usage.md) — How to structure builder configs
- [Test Group Configuration Usage](test_group_config_usage.md) — Sweep parameter expansion
- [Backend Usage](backend_usage.md) — Backend-specific setup and components

