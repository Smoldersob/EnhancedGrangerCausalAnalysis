import sys
import traceback
from pathlib import Path

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/tests_lag_selectors.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from complex_granger_analysis.core.lag_config import LagConfiguration
from complex_granger_analysis.preprocessing.lag.lag_selectors import ICLagSelector, CVLagSelector
from complex_granger_analysis.preprocessing.lag.lag_engine import LagEngine

# ===========================================================================
# Testy dla mechanizmu lag_zero
# ===========================================================================

def test_lag_zero_selector_autoregression():
    """
    Test 1: Weryfikuje, że przy use_lag_zero=True w selektora,
    lag0 (bieżąca wartość) jest wyzerowana dla autoregresji (i==j)
    ale dozwolona dla zewnętrznych predyktorów (i!=j).
    """
    print("\n" + "="*80)
    print("TEST 1: Lag0 dla autoregresji w selektora")
    print("="*80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Selektor IC z use_lag_zero=True
    sel = ICLagSelector(max_lag=8, use_lag_zero=True, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nDane: shape={df.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"Col_offsets: {result.col_offsets}")
    print(f"Maska shape: {result.mask.shape}")
    
    # Sprawdzenie: dla każdej zmiennej j, lag0 (pierwsza kolumna bloku)
    # powinna być 0 dla autoregresji (i==j)
    all_correct = True
    for j in range(d):
        max_lag_j = result.max_lags_per_pred[j]
        if max_lag_j <= 0:
            continue
        
        block_start = result.col_offsets[j]
        lag0_col = block_start  # Pierwsza kolumna to lag0
        
        for i in range(d):
            lag0_value = result.mask[i, lag0_col]
            is_autoregression = (i == j)
            
            is_correct = (lag0_value == 0) if is_autoregression else (lag0_value == 1)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {i}, Predictor {j}: lag0={lag0_value} (autoregression={is_autoregression})")
            
            if not is_correct:
                all_correct = False
    
    print(f"\n✅ SUKCES" if all_correct else "❌ FAIL - Lag0 nie jest prawidłowo obsługiwane dla autoregresji")
    assert all_correct


def test_lag_zero_engine_without_selector():
    """
    Test 2: Weryfikuje, że przy use_lag_zero=True w LagEngine bez selektora,
    lag0 dla autoregresji jest wyzerowana w masce.
    """
    print("\n" + "="*80)
    print("TEST 2: Lag0 w LagEngine bez selektora (fixed lag)")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 4
    col_names = ["y1", "x1", "x2", "x3"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nDane: shape={df.shape}")
    print(f"X shape: {X.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Maska shape: {engine.mask_.shape}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Sprawdzenie lag0 dla każdej zmiennej
    all_correct = True
    for j in range(d):
        if max_lags[j] <= 0:
            continue
            
        # Blok dla zmiennej j
        block_start = col_idx[j]
        lag0_col = block_start  # Pierwsza kolumna to lag0 (bo min_lags=0)
        
        # Sprawdź czy lag0 jest 0 dla autoregresji (target=j)
        # i 1 dla pozostałych targetów
        for target_idx in range(d):
            if target_idx >= engine.mask_.shape[0]:
                continue
            
            lag0_value = engine.mask_[target_idx, lag0_col]
            is_autoregression = (target_idx == j)
            
            is_correct = (lag0_value == 0) if is_autoregression else (lag0_value == 1)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {col_names[target_idx]}, Predictor {col_names[j]}: lag0={lag0_value} (autoregression={is_autoregression})")
            
            if not is_correct:
                all_correct = False
    
    print(f"\n✅ SUKCES" if all_correct else "❌ FAIL - Lag0 nie jest prawidłowo obsługiwane")
    assert all_correct


def test_lag_zero_engine_with_selector():
    """
    Test 3: Weryfikuje, że przy use_lag_zero=True w LagEngine z selektorem,
    wymiary maski są prawidłowe i lag0 dla autoregresji jest wyzerowana.
    """
    print("\n" + "="*80)
    print("TEST 3: Lag0 w LagEngine z selektorem (automatyczna selekcja)")
    print("="*80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    sel = CVLagSelector(max_lag=8, use_lag_zero=True, cv_folds=3)
    cfg = LagConfiguration(max_lag=8, use_lag_zero=True)
    engine = LagEngine(config=cfg, selector=sel, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nDane: shape={df.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Maska shape: {engine.mask_.shape}")
    print(f"Selekcja: pred_lag_matrix:\n{engine.selection_result_.pred_lag_matrix if engine.selection_result_ else 'N/A'}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Sprawdzenie: wymiary
    expected_cols = int((max_lags - min_lags + 1).sum())
    actual_cols = engine.mask_.shape[1]
    
    dim_check = expected_cols == actual_cols
    print(f"\nSprawdzenie wymiarów:")
    print(f"  Oczekiwane kolumny: {expected_cols}")
    print(f"  Rzeczywiste kolumny: {actual_cols}")
    print(f"  {'✓' if dim_check else '✗'} Wymiary są prawidłowe: {dim_check}")
    
    # Sprawdzenie lag0
    all_correct = True
    print(f"\nSprawdzenie lag0 dla autoregresji:")
    for j in range(d):
        if max_lags[j] <= 0:
            continue
            
        block_start = col_idx[j]
        lag0_col = block_start
        
        # Dla target y1 (index 0), sprawdzenie lag0 dla każdego prediktora
        for target_idx in range(min(1, d)):  # Sprawdzamy tylko dla target=0
            lag0_value = engine.mask_[target_idx, lag0_col]
            is_autoregression = (target_idx == j)
            
            is_correct = (lag0_value == 0) if is_autoregression else (lag0_value == 1)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {col_names[target_idx]}, Predictor {col_names[j]}: lag0={lag0_value}")
            
            if not is_correct:
                all_correct = False
    
    result = dim_check and all_correct
    print(f"\n✅ SUKCES" if result else "❌ FAIL")
    assert result


def test_lag_zero_disabled():
    """
    Test 4: Weryfikuje, że gdy use_lag_zero=False, maska nie zawiera kolumn lag0.
    """
    print("\n" + "="*80)
    print("TEST 4: use_lag_zero=False - brak lag0 w masce")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Selektor BEZ use_lag_zero
    sel = ICLagSelector(max_lag=5, use_lag_zero=False, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nDane: shape={df.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"Col_offsets: {result.col_offsets}")
    print(f"Maska shape: {result.mask.shape}")
    
    # Sprawdzenie: gdy use_lag_zero=False liczba kolumn maski
    # musi być sumą maksymalnych lagów na predyktor.
    expected_cols = int(result.max_lags_per_pred.sum())
    assert result.mask.shape[1] == expected_cols

    print(f"\n✅ SUKCES - use_lag_zero=False pracuje prawidłowo")


def test_mask_consistency_across_segments():
    """
    Test 5: Weryfikuje, że maski są spójne dla wielokrotnych segmentów danych
    (np. wiele datasów połączonych).
    
    Ważne: Wiersze maski mogą być różne dla każdego targetu (bo różne zmienne
    są autoregresyjne dla różnych targetów).
    """
    print("\n" + "="*80)
    print("TEST 5: Spójność maski dla wielokrotnych segmentów")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    
    # Dwa niezależne segmenty czasowe
    df1 = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    df2 = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, n_jobs=1)
    
    X, y, col_idx = engine.prepare([df1, df2], effects=["y1"])
    
    print(f"\nSegmenty: 2 x {n} samples")
    print(f"Połączone X shape: {X.shape}")
    print(f"Maska shape: {engine.mask_.shape}")
    print(f"Col_idx: {col_idx}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Sprawdzenie: dla każdego widersza (targetu), sprawdzić czy lag0 dla autoregresji
    # jest wyzerowana
    all_correct = True
    for target_idx in range(d):
        lag0_for_self = engine.mask_[target_idx, col_idx[target_idx]]
        if max_lags[target_idx] > 0:  # Jeśli ta zmienna ma jakieś lagi
            is_correct = (lag0_for_self == 0)  # lag0 dla autoregresji powinno być 0
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {col_names[target_idx]}: lag0 dla autoregresji = {lag0_for_self} (oczekiwane: 0)")
            if not is_correct:
                all_correct = False
    
    print(f"\n✅ SUKCES" if all_correct else "❌ FAIL")
    assert all_correct


def test_custom_pair_lags_with_lag_zero():
    """
    Test 6: Weryfikuje, że custom_pair_lags pracują prawidłowo z use_lag_zero=True.
    """
    print("\n" + "="*80)
    print("TEST 6: custom_pair_lags z use_lag_zero=True")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_pair_lags={
            ("y1", "x1"): (1, 3),  # Tylko lagi 1-3 dla pary (y1, x1)
        }
    )
    engine = LagEngine(config=cfg, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nDane: shape={df.shape}")
    print(f"custom_pair_lags: {cfg.custom_pair_lags}")
    print(f"Maska shape: {engine.mask_.shape}")
    print(f"Col_idx: {col_idx}")
    
    # Target 0 (y1), Predictor 1 (x1)
    # Blok dla x1 powinien mieć strukturę: [lag0, lag1, lag2, lag3, lag4, lag5]
    # custom_pair_lags ogranicza do lag1-lag3, więc:
    # - lag0 powinno być 0 (dodatkowo bo to może być prawidłem)
    # - lag1, lag2, lag3 powinny być 1
    # - lag4, lag5 powinny być 0
    
    block_start_x1 = col_idx[1]
    
    mask_y1 = engine.mask_[0, :]  # Target y1
    block_x1 = mask_y1[block_start_x1:col_idx[2]]
    
    print(f"\nTarget y1, Predictor x1:")
    print(f"  Block lags: {block_x1}")
    print(f"  Oczekiwane: [0, 1, 1, 1, 0, 0] (lag0 wyzerowana, lag1-3 aktywne)")
    
    # Sprawdzenie struktury
    expected = np.array([0, 1, 1, 1, 0, 0])
    
    # Jeśli mamy mniej kolumn, sprawdzić górny zakres
    if len(block_x1) >= len(expected):
        matches = np.array_equal(block_x1[:len(expected)], expected)
    else:
        matches = False
    
    print(f"  Czy prawidłowe: {matches}")
    print(f"\n✅ SUKCES" if matches else "❌ FAIL")
    assert matches


# ===========================================================================
# Main: Uruchomienie wszystkich testów
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SERIA TESTÓW DLA MECHANIZMU LAG_ZERO")
    print("="*80)

    tests = [
        test_lag_zero_selector_autoregression,
        test_lag_zero_engine_without_selector,
        test_lag_zero_engine_with_selector,
        test_lag_zero_disabled,
        test_mask_consistency_across_segments,
        test_custom_pair_lags_with_lag_zero,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=1)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed")
    print("="*80 + "\n")

