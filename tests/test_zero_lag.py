"""
Analiza wymiarów i logiki generowania maski dla use_lag_zero=True
"""
import sys
from pathlib import Path

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/test_zero_lag.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from complex_granger_analysis.preprocessing.lag.lag_selectors import ICLagSelector, CVLagSelector
from complex_granger_analysis.core.lag_config import LagConfiguration
from complex_granger_analysis.preprocessing.lag.lag_engine import LagEngine

def test_mask_dimensions_with_lag_zero():
    """Test czy wymiary maski są prawidłowe dla use_lag_zero=True"""
    print("\n" + "="*80)
    print("TEST 1: Wymiary maski dla use_lag_zero=True")
    print("="*80)
    
    # Dane testowe
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Test 1a: Selektor IC z use_lag_zero=True
    print("\nTest 1a: ICLagSelector z use_lag_zero=True")
    sel = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nDane wejściowe: shape={df.values.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"col_offsets: {result.col_offsets}")
    print(f"Maska shape: {result.mask.shape}")
    print(f"Maska:\n{result.mask}")
    
    # Oblicz oczekiwane wymiary
    # Dla każdej zmiennej: jeśli use_lag_zero=True i max_lag[j]>0, to 1 + max_lag[j] kolumn
    expected_cols = 0
    for j in range(d):
        if result.max_lags_per_pred[j] > 0:
            expected_cols += 1 + result.max_lags_per_pred[j]  # lag0 + lag1..max_lag
        else:
            expected_cols += 0
    
    print(f"\nOczekiwane kolumny: {expected_cols}")
    print(f"Rzeczywiste kolumny: {result.mask.shape[1]}")
    print(f"Poprawne wymiary: {expected_cols == result.mask.shape[1]}")
    
    # Test 1b: LagEngine z use_lag_zero=True
    print("\n\nTest 1b: LagEngine z use_lag_zero=True")
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"X shape: {X.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Maska shape: {engine.mask_.shape}")
    print(f"Maska:\n{engine.mask_}")
    
    # Col_idx powinny uwzględniać lag0
    print(f"\nCol_idx: {col_idx}")
    

def test_lag_zero_autoregression():
    """Test czy lag0 jest wyzerowany dla autoregresji"""
    print("\n" + "="*80)
    print("TEST 2: Czy lag0 jest wyzerowany dla autoregresji (i==j)?")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Selektor z use_lag_zero=True
    sel = ICLagSelector(max_lag=5, use_lag_zero=True, target_indices=[0])
    result = sel.fit(df.values)
    
    print(f"\npred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"col_offsets: {result.col_offsets}")
    print(f"\nMaska[\n{result.mask}")
    
    # Analiza: dla target 0, predictor 0 (autoregresja)
    # Maska powinna mieć strukturę: [lag0, lag1, lag2, ..., lag_max]
    # lag0 powinno być 0 dla autoregresji!
    
    # Znajdź blok dla zmiennej 0
    j = 0  # predictor 0
    block_start = result.col_offsets[j]
    max_lag_j = result.max_lags_per_pred[j]
    
    if max_lag_j > 0:
        block_end = (result.col_offsets[j+1] if j+1 < len(result.col_offsets) 
                     else result.mask.shape[1])
        
        print(f"\nBlok dla zmiennej 0: kolumny {block_start}..{block_end-1}")
        print(f"use_lag_zero=True, więc struktura: [lag0_col, lag1_col, ..., lag{max_lag_j}_col]")
        
        for i in range(result.mask.shape[0]):
            mask_block = result.mask[i, block_start:block_end]
            print(f"\nTarget {i}, Predictor 0: {mask_block}")
            if i == 0:  # autoregresja
                if max_lag_j > 0:
                    first_col_value = result.mask[i, block_start]
                    print(f"  -> lag0 wartość: {first_col_value}")
                    if first_col_value == 0:
                        print("  -> OK: lag0 jest poprawnie wyzerowane dla autoregresji")
                    else:
                        print(f"  -> PROBLEM: lag0 powinno być 0 dla autoregresji! Jest: {first_col_value}")


def test_lag_zero_with_lag_engine():
    """Test lag_zero w LagEngine ze selektorem"""
    print("\n" + "="*80)
    print("TEST 3: use_lag_zero w LagEngine ze selektorem")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Engine z selektorem i use_lag_zero
    sel = CVLagSelector(max_lag=5, use_lag_zero=True, cv_folds=3)
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, selector=sel, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Maska shape: {engine.mask_.shape}")
    print(f"Maska:\n{engine.mask_}")
    
    # Analiza wymiarów
    print("\nAnaliza wymiarów:")
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    print(f"min_lags: {min_lags}")
    print(f"max_lags: {max_lags}")
    
    # Oczekiwane kolumny
    expected_cols = 0
    for j in range(d):
        n_cols_j = max_lags[j] - min_lags[j] + 1
        expected_cols += n_cols_j
    
    print(f"Oczekiwane kolumny: {expected_cols}")
    print(f"Rzeczywiste kolumny (z col_idx): {col_idx[-1]}")
    print(f"Rzeczywiste kolumny (maska): {engine.mask_.shape[1]}")
    print(f"Spójność wymiarów: {expected_cols == engine.mask_.shape[1]}")


if __name__ == "__main__":
    test_mask_dimensions_with_lag_zero()
    test_lag_zero_autoregression()
    test_lag_zero_with_lag_engine()
