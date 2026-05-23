import traceback

import numpy as np
import pandas as pd
from ..core.lag_config import LagConfiguration
from ..preprocessing.lag.lag_selectors import ICLagSelector, CVLagSelector
from ..preprocessing.lag.lag_engine import LagEngine

# ===========================================================================
# Tests for the delta_min and backward pruning mechanism
# ===========================================================================

def test_iclaagselector_default_delta_min_and_prune():
    """
    Test: Verifies the default parameter values in ICLagSelector:
    - delta_min_ic should be 2.0 (minimum improvement threshold enabled)
    - prune_lags should be True (backward pruning enabled)
    - delta_prune_ic should be 2.0 (pruning tolerance threshold)
    """
    print("\n" + "="*80)
    print("TEST 1: Default ICLagSelector parameters")
    print("="*80)
    
    sel = ICLagSelector(max_lag=10)
    
    print(f"\nDefault values:")
    print(f"  delta_min_ic: {sel.delta_min_ic} (expected: 2.0)")
    print(f"  prune_lags: {sel.prune_lags} (expected: True)")
    print(f"  delta_prune_ic: {sel.delta_prune_ic} (expected: 2.0)")
    
    assert sel.delta_min_ic == 2.0, f"delta_min_ic should be 2.0, got {sel.delta_min_ic}"
    assert sel.prune_lags == True, f"prune_lags should be True, got {sel.prune_lags}"
    assert sel.delta_prune_ic == 2.0, f"delta_prune_ic should be 2.0, got {sel.delta_prune_ic}"
    
    print(f"\n✅ SUCCESS - All default values are correct")


def test_cvlagselector_default_parameters():
    """
    Test: Verifies the default parameter values in CVLagSelector:
    - delta_min_rel_cv should be None (threshold disabled)
    - prune_lags should be False (backward pruning disabled)
    - delta_prune_rel_cv should be 0.02 (for future use)
    """
    print("\n" + "="*80)
    print("TEST 2: Default CVLagSelector parameters")
    print("="*80)
    
    sel = CVLagSelector(max_lag=10)
    
    print(f"\nDefault values:")
    print(f"  delta_min_rel_cv: {sel.delta_min_rel_cv} (expected: None)")
    print(f"  prune_lags: {sel.prune_lags} (expected: False)")
    print(f"  delta_prune_rel_cv: {sel.delta_prune_rel_cv} (expected: 0.02)")
    
    assert sel.delta_min_rel_cv is None, f"delta_min_rel_cv should be None, got {sel.delta_min_rel_cv}"
    assert sel.prune_lags == False, f"prune_lags should be False, got {sel.prune_lags}"
    assert sel.delta_prune_rel_cv == 0.02, f"delta_prune_rel_cv should be 0.02, got {sel.delta_prune_rel_cv}"
    
    print(f"\n✅ SUCCESS - All default values are correct")


def test_iclaagselector_delta_min_ic_threshold():
    """
    Test: Verifies that delta_min_ic controls the minimum required improvement
    when selecting lags for predictors.
    
    When delta_min_ic=None, all strict improvements are accepted.
    When delta_min_ic>0, an improvement of at least that value is required.
    """
    print("\n" + "="*80)
    print("TEST 3: delta_min_ic threshold in ICLagSelector")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    X = df.values
    
    # Test 1: delta_min_ic=None - should accept every strict improvement
    sel_no_threshold = ICLagSelector(max_lag=8, delta_min_ic=None, prune_lags=False)
    result_no_threshold = sel_no_threshold.fit(X)
    
    # Test 2: delta_min_ic=2.0 (default) - requires a larger improvement
    sel_with_threshold = ICLagSelector(max_lag=8, delta_min_ic=2.0, prune_lags=False)
    result_with_threshold = sel_with_threshold.fit(X)
    
    print(f"\nData: shape={df.shape}")
    print(f"\nWithout delta_min_ic threshold (None):")
    print(f"  pred_lag_matrix:\n{result_no_threshold.pred_lag_matrix}")
    print(f"  max_lags_per_pred: {result_no_threshold.max_lags_per_pred}")
    
    print(f"\nWith delta_min_ic=2.0 threshold:")
    print(f"  pred_lag_matrix:\n{result_with_threshold.pred_lag_matrix}")
    print(f"  max_lags_per_pred: {result_with_threshold.max_lags_per_pred}")
    
    # Check: with a threshold, lags should be smaller or equal
    all_correct = True
    for i in range(d):
        if result_with_threshold.max_lags_per_pred[i] <= result_no_threshold.max_lags_per_pred[i]:
            print(f"✓ Target {i}: max_lag with threshold ({result_with_threshold.max_lags_per_pred[i]}) "
                f"<= without threshold ({result_no_threshold.max_lags_per_pred[i]})")
        else:
            print(f"✗ Target {i}: max_lag with threshold ({result_with_threshold.max_lags_per_pred[i]}) "
                f"> without threshold ({result_no_threshold.max_lags_per_pred[i]})")
            all_correct = False
    
    print(f"\n✅ SUCCESS" if all_correct else "❌ FAIL")
    assert all_correct


def test_iclaagselector_pruning_effect():
    """
    Test: Verifies that backward pruning in ICLagSelector reduces the number of
    selected lags when delta_prune_ic allows it.
    """
    print("\n" + "="*80)
    print("TEST 4: Backward pruning effect in ICLagSelector")
    print("="*80)
    
    np.random.seed(42)
    n, d = 120, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    X = df.values
    
    # Test 1: With pruning (default)
    sel_with_prune = ICLagSelector(max_lag=10, prune_lags=True, delta_prune_ic=2.0)
    result_with_prune = sel_with_prune.fit(X)
    
    # Test 2: Without pruning
    sel_no_prune = ICLagSelector(max_lag=10, prune_lags=False)
    result_no_prune = sel_no_prune.fit(X)
    
    print(f"\nData: shape={df.shape}")
    print(f"\nWithout pruning:")
    print(f"  ar_lags: {result_no_prune.ar_lags}")
    print(f"  max_lags_per_pred: {result_no_prune.max_lags_per_pred}")
    print(f"  Sum of non-zero lags: {np.count_nonzero(result_no_prune.pred_lag_matrix)}")
    
    print(f"\nWith pruning (delta_prune_ic=2.0):")
    print(f"  ar_lags: {result_with_prune.ar_lags}")
    print(f"  max_lags_per_pred: {result_with_prune.max_lags_per_pred}")
    print(f"  Sum of non-zero lags: {np.count_nonzero(result_with_prune.pred_lag_matrix)}")
    
    # Check: pruning should reduce the number of lags
    nonzero_no_prune = np.count_nonzero(result_no_prune.pred_lag_matrix)
    nonzero_with_prune = np.count_nonzero(result_with_prune.pred_lag_matrix)
    
    print(f"\nCheck:")
    print(f"  Non-zero lags without pruning: {nonzero_no_prune}")
    print(f"  Non-zero lags with pruning: {nonzero_with_prune}")
    print(f"  Pruning reduced the number of lags: {nonzero_with_prune <= nonzero_no_prune}")
    
    print(f"\n✅ SUCCESS - Pruning correctly reduces lags" if nonzero_with_prune <= nonzero_no_prune else "❌ FAIL")
    assert nonzero_with_prune <= nonzero_no_prune


def test_cvlagselector_with_delta_min_rel_cv():
    """
    Test: Verifies that delta_min_rel_cv for CVLagSelector controls the
    minimum relative improvement (based on CV error).
    
    delta_min_rel_cv=None means no threshold (accept every strict improvement).
    delta_min_rel_cv=0.05 requires at least a 5% improvement in CV error.
    """
    print("\n" + "="*80)
    print("TEST 5: delta_min_rel_cv in CVLagSelector")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    X = df.values
    
    # Test 1: delta_min_rel_cv=None (no threshold)
    sel_no_threshold = CVLagSelector(max_lag=8, delta_min_rel_cv=None, 
                                     cv_folds=3, prune_lags=False)
    result_no_threshold = sel_no_threshold.fit(X)
    
    # Test 2: delta_min_rel_cv=0.05 (requires 5% improvement)
    sel_with_threshold = CVLagSelector(max_lag=8, delta_min_rel_cv=0.05, 
                                       cv_folds=3, prune_lags=False)
    result_with_threshold = sel_with_threshold.fit(X)
    
    print(f"\nData: shape={df.shape}")
    print(f"\nWithout delta_min_rel_cv threshold (None):")
    print(f"  max_lags_per_pred: {result_no_threshold.max_lags_per_pred}")
    
    print(f"\nWith delta_min_rel_cv=0.05 threshold:")
    print(f"  max_lags_per_pred: {result_with_threshold.max_lags_per_pred}")
    
    # Check: with a stricter threshold, there should be fewer or equal lags
    all_correct = True
    for i in range(d):
        if result_with_threshold.max_lags_per_pred[i] <= result_no_threshold.max_lags_per_pred[i]:
            print(f"✓ Target {i}: max_lag with threshold ({result_with_threshold.max_lags_per_pred[i]}) "
                f"<= without threshold ({result_no_threshold.max_lags_per_pred[i]})")
        else:
            print(f"✗ Target {i}: max_lag with threshold ({result_with_threshold.max_lags_per_pred[i]}) "
                f"> without threshold ({result_no_threshold.max_lags_per_pred[i]})")
            all_correct = False
    
    print(f"\n✅ SUCCESS" if all_correct else "❌ FAIL")
    assert all_correct


def test_cvlagselector_pruning_disabled_by_default():
    """
    Test: Verifies that CVLagSelector has pruning=False by default
    (unlike ICLagSelector, which defaults to pruning=True).
    """
    print("\n" + "="*80)
    print("TEST 6: CVLagSelector without pruning by default")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    X = df.values
    
    # CV selector with default settings
    sel_default = CVLagSelector(max_lag=10)
    result_default = sel_default.fit(X)
    
    # CV selector with explicit pruning=True
    sel_with_prune = CVLagSelector(max_lag=10, prune_lags=True)
    result_with_prune = sel_with_prune.fit(X)
    
    print(f"\nData: shape={df.shape}")
    print(f"\nCVLagSelector by default (prune_lags=False):")
    print(f"  max_lags_per_pred: {result_default.max_lags_per_pred}")
    print(f"  Non-zero lags: {np.count_nonzero(result_default.pred_lag_matrix)}")
    
    print(f"\nCVLagSelector with pruning=True:")
    print(f"  max_lags_per_pred: {result_with_prune.max_lags_per_pred}")
    print(f"  Non-zero lags: {np.count_nonzero(result_with_prune.pred_lag_matrix)}")
    
    # By default there should be more lags (or equal, because pruning may have a small effect)
    nonzero_default = np.count_nonzero(result_default.pred_lag_matrix)
    nonzero_with_prune = np.count_nonzero(result_with_prune.pred_lag_matrix)
    
    print(f"\nCheck:")
    print(f"  No pruning by default: {sel_default.prune_lags} (expected: False)")
    print(f"  Non-zero lags by default >= with pruning: {nonzero_default >= nonzero_with_prune}")
    
    result = (sel_default.prune_lags == False) and (nonzero_default >= nonzero_with_prune)
    print(f"\n✅ SUCCESS" if result else "❌ FAIL")
    assert result


# ===========================================================================
# Tests for the lag_zero mechanism
# ===========================================================================

def test_lag_zero_selector_autoregression():
    """
    Test 1: Verifies that with use_lag_zero=True in the selector,
    lag0 (current value) is zeroed out for autoregression (i==j)
    but allowed for external predictors (i!=j).
    """
    print("\n" + "="*80)
    print("TEST 1: Lag0 for autoregression in the selector")
    print("="*80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # IC selector with use_lag_zero=True
    sel = ICLagSelector(max_lag=8, use_lag_zero=True, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nData: shape={df.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"Col_offsets: {result.col_offsets}")
    print(f"Mask shape: {result.mask.shape}")
    
    # Check: for each variable j, lag0 (the first column in the block)
    # should be 0 for autoregression (i==j)
    all_correct = True
    for j in range(d):
        max_lag_j = result.max_lags_per_pred[j]
        if max_lag_j <= 0:
            continue
        
        block_start = result.col_offsets[j]
        lag0_col = block_start  # The first column is lag0
        
        for i in range(d):
            lag0_value = result.mask[i, lag0_col]
            is_autoregression = (i == j)
            
            is_correct = (lag0_value == 0) if is_autoregression else (lag0_value == 1)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {i}, Predictor {j}: lag0={lag0_value} (autoregression={is_autoregression})")
            
            if not is_correct:
                all_correct = False
    
    print(f"\n✅ SUCCESS" if all_correct else "❌ FAIL - Lag0 is not handled correctly for autoregression")
    assert all_correct


def test_lag_zero_engine_without_selector():
    """
    Test 2: Verifies that with use_lag_zero=True in LagEngine without a selector,
    lag0 for autoregression is zeroed out in the mask.
    """
    print("\n" + "="*80)
    print("TEST 2: Lag0 in LagEngine without a selector (fixed lag)")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 4
    col_names = ["y1", "x1", "x2", "x3"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nData: shape={df.shape}")
    print(f"X shape: {X.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Mask shape: {engine.mask_.shape}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Check lag0 for each variable
    all_correct = True
    for j in range(d):
        if max_lags[j] <= 0:
            continue
            
        # Block for variable j
        block_start = col_idx[j]
        lag0_col = block_start  # The first column is lag0 (because min_lags=0)
        
        # Check whether lag0 is 0 for autoregression (target=j)
        # and 1 for the remaining targets
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
    
    print(f"\n✅ SUCCESS" if all_correct else "❌ FAIL - Lag0 is not handled correctly")
    assert all_correct


def test_lag_zero_engine_with_selector():
    """
    Test 3: Verifies that with use_lag_zero=True in LagEngine with a selector,
    mask dimensions are correct and lag0 for autoregression is zeroed out.
    """
    print("\n" + "="*80)
    print("TEST 3: Lag0 in LagEngine with a selector (automatic selection)")
    print("="*80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    sel = CVLagSelector(max_lag=8, use_lag_zero=True, cv_folds=3)
    cfg = LagConfiguration(max_lag=8, use_lag_zero=True)
    engine = LagEngine(config=cfg, selector=sel, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nData: shape={df.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Lag order: {engine.lag_order_}")
    print(f"Col_idx: {col_idx}")
    print(f"Mask shape: {engine.mask_.shape}")
    print(f"Selection: pred_lag_matrix:\n{engine.selection_result_.pred_lag_matrix if engine.selection_result_ else 'N/A'}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Check: dimensions
    expected_cols = int((max_lags - min_lags + 1).sum())
    actual_cols = engine.mask_.shape[1]
    
    dim_check = expected_cols == actual_cols
    print(f"\nDimension check:")
    print(f"  Expected columns: {expected_cols}")
    print(f"  Actual columns: {actual_cols}")
    print(f"  {'✓' if dim_check else '✗'} Dimensions are correct: {dim_check}")
    
    # Check lag0
    all_correct = True
    print(f"\nLag0 check for autoregression:")
    for j in range(d):
        if max_lags[j] <= 0:
            continue
            
        block_start = col_idx[j]
        lag0_col = block_start
        
        # For target y1 (index 0), check lag0 for each predictor
        for target_idx in range(min(1, d)):  # Only check target=0
            lag0_value = engine.mask_[target_idx, lag0_col]
            is_autoregression = (target_idx == j)
            
            is_correct = (lag0_value == 0) if is_autoregression else (lag0_value == 1)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {col_names[target_idx]}, Predictor {col_names[j]}: lag0={lag0_value}")
            
            if not is_correct:
                all_correct = False
    
    result = dim_check and all_correct
    print(f"\n✅ SUCCESS" if result else "❌ FAIL")
    assert result


def test_lag_zero_disabled():
    """
    Test 4: Verifies that when use_lag_zero=False, the mask does not contain lag0 columns.
    """
    print("\n" + "="*80)
    print("TEST 4: use_lag_zero=False - no lag0 in the mask")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Selector WITHOUT use_lag_zero
    sel = ICLagSelector(max_lag=5, use_lag_zero=False, use_bic=False)
    result = sel.fit(df.values)
    
    print(f"\nData: shape={df.shape}")
    print(f"pred_lag_matrix:\n{result.pred_lag_matrix}")
    print(f"max_lags_per_pred: {result.max_lags_per_pred}")
    print(f"Col_offsets: {result.col_offsets}")
    print(f"Mask shape: {result.mask.shape}")
    
    # Check: when use_lag_zero=False, the number of mask columns
    # must equal the sum of maximum lags per predictor.
    expected_cols = int(result.max_lags_per_pred.sum())
    assert result.mask.shape[1] == expected_cols

    print(f"\n✅ SUCCESS - use_lag_zero=False works correctly")


def test_mask_consistency_across_segments():
    """
    Test 5: Verifies that masks are consistent across multiple data segments
    (e.g. several concatenated datasets).
    
    Important: Mask rows can differ for each target (because different variables
    are autoregressive for different targets).
    """
    print("\n" + "="*80)
    print("TEST 5: Mask consistency across multiple segments")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    
    # Two independent time segments
    df1 = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    df2 = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    engine = LagEngine(config=cfg, n_jobs=1)
    
    X, y, col_idx = engine.prepare([df1, df2], effects=["y1"])
    
    print(f"\nSegments: 2 x {n} samples")
    print(f"Combined X shape: {X.shape}")
    print(f"Mask shape: {engine.mask_.shape}")
    print(f"Col_idx: {col_idx}")
    
    min_lags = engine.lag_order_["min"]
    max_lags = engine.lag_order_["max"]
    
    # Check: for each row (target), verify whether lag0 for autoregression
    # is zeroed out
    all_correct = True
    for target_idx in range(d):
        lag0_for_self = engine.mask_[target_idx, col_idx[target_idx]]
        if max_lags[target_idx] > 0:  # Jeśli ta zmienna ma jakieś lagi
            is_correct = (lag0_for_self == 0)  # lag0 for autoregression should be 0
            status = "✓" if is_correct else "✗"
            print(f"{status} Target {col_names[target_idx]}: lag0 for autoregression = {lag0_for_self} (expected: 0)")
            if not is_correct:
                all_correct = False
    
    print(f"\n✅ SUCCESS" if all_correct else "❌ FAIL")
    assert all_correct


def test_custom_pair_lags_with_lag_zero():
    """
    Test 6: Verifies that custom_pair_lags work correctly with use_lag_zero=True.
    """
    print("\n" + "="*80)
    print("TEST 6: custom_pair_lags with use_lag_zero=True")
    print("="*80)
    
    np.random.seed(42)
    n, d = 100, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    cfg = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_pair_lags={
            ("y1", "x1"): (1, 3),  # Only lags 1-3 for the pair (y1, x1)
        }
    )
    engine = LagEngine(config=cfg, n_jobs=1)
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nData: shape={df.shape}")
    print(f"custom_pair_lags: {cfg.custom_pair_lags}")
    print(f"Mask shape: {engine.mask_.shape}")
    print(f"Col_idx: {col_idx}")
    
    # Target 0 (y1), Predictor 1 (x1)
    # The x1 block should have the structure: [lag0, lag1, lag2, lag3, lag4, lag5]
    # custom_pair_lags restricts it to lags 1-3, so:
    # - lag0 should be 0 (also because this may be the rule)
    # - lag1, lag2, lag3 should be 1
    # - lag4, lag5 should be 0
    
    block_start_x1 = col_idx[1]
    
    mask_y1 = engine.mask_[0, :]  # Target y1
    block_x1 = mask_y1[block_start_x1:col_idx[2]]
    
    print(f"\nTarget y1, Predictor x1:")
    print(f"  Block lags: {block_x1}")
    print(f"  Expected: [0, 1, 1, 1, 0, 0] (lag0 zeroed, lag1-3 active)")
    
    # Structure check
    expected = np.array([0, 1, 1, 1, 0, 0])
    
    # If we have fewer columns, check the upper range
    if len(block_x1) >= len(expected):
        matches = np.array_equal(block_x1[:len(expected)], expected)
    else:
        matches = False
    
    print(f"  Correct: {matches}")
    print(f"\n✅ SUCCESS" if matches else "❌ FAIL")
    assert matches


# ===========================================================================
# Main: Run all tests
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAG SELECTORS TEST SUITE")
    print("="*80)

    tests = [
        # Tests for delta_min and backward pruning
        test_iclaagselector_default_delta_min_and_prune,
        test_cvlagselector_default_parameters,
        test_iclaagselector_delta_min_ic_threshold,
        test_iclaagselector_pruning_effect,
        test_cvlagselector_with_delta_min_rel_cv,
        test_cvlagselector_pruning_disabled_by_default,
        # Tests for the lag_zero mechanism
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

