"""
Analysis of dimensions and mask generation logic for use_lag_zero=True.
"""

import numpy as np
import pandas as pd
from ..preprocessing.lag.lag_selectors import ICLagSelector, CVLagSelector
from ..core.lag_config import LagConfiguration
from ..preprocessing.lag.lag_engine import LagEngine

def test_mask_dimensions_with_lag_zero():
    """Test whether the mask dimensions are correct for use_lag_zero=True."""
    print("\n" + "="*80)
    print("TEST 1: Wymiary maski dla use_lag_zero=True")
    print("="*80)
    
    # Test data
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Test 1a: Selektor IC z use_lag_zero=True
    print("\nTest 1a: ICLagSelector z use_lag_zero=True")
    sel = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nInput data shape: {df.values.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"col_offsets: {result.col_offsets}")
    print(f"Mask shape: {result.mask.shape}")
    print(f"Mask:\n{result.mask}")
    
    # Compute expected dimensions
    # For each variable: if use_lag_zero=True and max_lag[j]>0, then 1 + max_lag[j] columns
    expected_cols = 0
    for j in range(d):
        if result.max_lags_per_pred[j] > 0:
            expected_cols += 1 + result.max_lags_per_pred[j]  # lag0 + lag1..max_lag
        else:
            expected_cols += 0
    
    print(f"\nExpected columns: {expected_cols}")
    print(f"Actual columns: {result.mask.shape[1]}")
    print(f"Correct dimensions: {expected_cols == result.mask.shape[1]}")
    
    # Test 1b: LagEngine z use_lag_zero=True
    print("\n\nTest 1b: LagEngine with use_lag_zero=True")
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"X shape: {X.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Mask shape: {engine.mask_.shape}")
    print(f"Mask:\n{engine.mask_}")
    
    # col_idx should account for lag0
    print(f"\nCol_idx: {col_idx}")
    

def test_lag_zero_autoregression():
    """Test whether lag0 is zeroed out for autoregression."""
    print("\n" + "="*80)
    print("TEST 2: Is lag0 zeroed out for autoregression (i==j)?")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Selector with use_lag_zero=True
    sel = ICLagSelector(max_lag=5, use_lag_zero=True, target_indices=[0])
    result = sel.fit(df.values)
    
    print(f"\npred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"col_offsets: {result.col_offsets}")
    print(f"\nMask[\n{result.mask}")
    
    # Analysis: for target 0, predictor 0 (autoregression)
    # The mask should have structure: [lag0, lag1, lag2, ..., lag_max]
    # lag0 should be 0 for autoregression!
    
    # Find the block for variable 0
    j = 0  # predictor 0
    block_start = result.col_offsets[j]
    max_lag_j = result.max_lags_per_pred[j]
    
    if max_lag_j > 0:
        block_end = (result.col_offsets[j+1] if j+1 < len(result.col_offsets) 
                     else result.mask.shape[1])
        
        print(f"\nBlock for variable 0: columns {block_start}..{block_end-1}")
        print(f"use_lag_zero=True, so the structure is: [lag0_col, lag1_col, ..., lag{max_lag_j}_col]")
        
        for i in range(result.mask.shape[0]):
            mask_block = result.mask[i, block_start:block_end]
            print(f"\nTarget {i}, Predictor 0: {mask_block}")
            if i == 0:  # autoregresja
                if max_lag_j > 0:
                    first_col_value = result.mask[i, block_start]
                    print(f"  -> lag0 value: {first_col_value}")
                    if first_col_value == 0:
                        print("  -> OK: lag0 is correctly zeroed out for autoregression")
                    else:
                        print(f"  -> PROBLEM: lag0 should be 0 for autoregression! It is: {first_col_value}")


def test_lag_zero_with_lag_engine():
    """Test lag_zero in LagEngine with a selector."""
    print("\n" + "="*80)
    print("TEST 3: use_lag_zero in LagEngine with a selector")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Engine with selector and use_lag_zero
    sel = CVLagSelector(max_lag=5, use_lag_zero=True, cv_folds=3)
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, selector=sel, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Mask shape: {engine.mask_.shape}")
    print(f"Mask:\n{engine.mask_}")
    
    # Dimension analysis
    print("\nDimension analysis:")
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    print(f"min_lags: {min_lags}")
    print(f"max_lags: {max_lags}")
    
    # Expected columns
    expected_cols = 0
    for j in range(d):
        n_cols_j = max_lags[j] - min_lags[j] + 1
        expected_cols += n_cols_j
    
    print(f"Expected columns: {expected_cols}")
    print(f"Actual columns (from col_idx): {col_idx[-1]}")
    print(f"Actual columns (mask): {engine.mask_.shape[1]}")
    print(f"Dimension consistency: {expected_cols == engine.mask_.shape[1]}")


if __name__ == "__main__":
    test_mask_dimensions_with_lag_zero()
    test_lag_zero_autoregression()
    test_lag_zero_with_lag_engine()
