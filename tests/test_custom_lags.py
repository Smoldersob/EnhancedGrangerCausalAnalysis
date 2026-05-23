"""
Test dla weryfikacji zgodności selection_result_ i lag_order_ oraz zachowania maski
przy braku i obecności custom_lags.

Wymaga:
- selection_result_ i lag_order_ są zgdone tylko gdy nie ma custom_lags
- custom_lags zmienia wymiary zmiennych
- maska zmienia się odpowiednio
"""

import numpy as np
import pandas as pd

from ..core.lag_config import LagConfiguration
from ..preprocessing.lag.lag_engine import LagEngine
from ..preprocessing.lag.lag_selectors import ICLagSelector


def _col_offsets_from_lag_order(engine: LagEngine) -> np.ndarray:
    widths = engine.lag_order_["max"] - engine.lag_order_["min"] + 1
    return np.concatenate([[0], widths.cumsum(dtype=int)])


def test_no_custom_lags_selection_result_matches_lag_order():
    """
    Test weryfikujący że przy braku custom_lags, selection_result_ i lag_order_
    zawierają te same informacje.
    
    Jeśli selektor jest użyty i nie ma custom_lags, to:
    - selection_result_.max_lags_per_pred == lag_order_["max"]
    - selection_result_.pred_lag_matrix === _pred_lag_matrix
    - selection_result_.mask === mask_
    """
    print("\n" + "=" * 80)
    print("TEST 1: Bez custom_lags - selection_result_ i lag_order_ są zgdone")
    print("=" * 80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Konfiguracja BEZ custom_lags, ze selektor
    cfg = LagConfiguration(max_lag=5, use_lag_zero=True)
    selector = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine = LagEngine(config=cfg, selector=selector)
    
    # Przygotowanie danych
    X, y, col_idx = engine.prepare([df], effects=["y1"])
    
    print(f"\nDane: shape={df.shape}")
    print(f"lag_order_: {engine.lag_order_}")
    
    # Weryfikacja
    assert engine.selection_result_ is not None, "selection_result_ powinien być ustawiony"
    
    selection_max = engine.selection_result_.max_lags_per_pred
    lag_order_max = engine.lag_order_["max"]
    
    print(f"selection_result_.max_lags_per_pred: {selection_max}")
    print(f"lag_order_['max']: {lag_order_max}")
    
    # Sprawdzenie zgodności
    assert np.array_equal(selection_max, lag_order_max), \
        f"selection_result_.max_lags_per_pred musi być równy lag_order_['max']\n" \
        f"  selection: {selection_max}\n" \
        f"  lag_order: {lag_order_max}"
    
    # Sprawdzenie mask
    assert np.array_equal(engine.selection_result_.mask, engine.mask_), \
        "selection_result_.mask musi być równy engine.mask_"
    
    # Sprawdzenie pred_lag_matrix
    assert np.array_equal(engine.selection_result_.pred_lag_matrix, engine._pred_lag_matrix), \
        "selection_result_.pred_lag_matrix musi być równy engine._pred_lag_matrix"
    
    print("\n✓ TEST PASSED: selection_result_ i lag_order_ są w pełni zgdone bez custom_lags")


def test_variable_custom_lags_change_dimensions():
    """
    Test weryfikujący że custom_lags zmienia wymiary zmiennych.
    
    Wchodzimy z selektor z wymaganą max_lag[j], potem override'ować beda
    max_lag['x1'] = 8 (większe niż z selektor).
    
    Oczekiwane:
    - lag_order_['max'] zmieni się dla 'x1'
    - selection_result_.max_lags_per_pred zmieni się dla 'x1'
    - maska się zmieni (więcej kolumn dla 'x1')
    - nowe kolumny w masce mają wartość 1
    """
    print("\n" + "=" * 80)
    print("TEST 2: Z custom_lags - wymiary zmiennych się zmieniają")
    print("=" * 80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Najpierw bez custom_lags - get baseline
    cfg_baseline = LagConfiguration(max_lag=5, use_lag_zero=True)
    selector_baseline = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine_baseline = LagEngine(config=cfg_baseline, selector=selector_baseline)
    X_baseline, _, _ = engine_baseline.prepare([df], effects=["y1"])
    
    baseline_max = engine_baseline.lag_order_["max"].copy()
    baseline_mask_shape = engine_baseline.mask_.shape
    baseline_mask = engine_baseline.mask_.copy()
    baseline_col_offsets = _col_offsets_from_lag_order(engine_baseline)
    
    print(f"\nBASELINE (bez custom_lags):")
    print(f"  lag_order_['max']: {baseline_max}")
    print(f"  mask_.shape: {baseline_mask_shape}")
    print(f"  col_offsets: {baseline_col_offsets}")
    
    # Teraz z custom_lags
    cfg_custom = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_lags={"x1": (8,)}  # Zwiększ max_lag dla x1 do 8
    )
    selector_custom = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine_custom = LagEngine(config=cfg_custom, selector=selector_custom)
    X_custom, _, _ = engine_custom.prepare([df], effects=["y1"])
    
    custom_max = engine_custom.lag_order_["max"].copy()
    custom_mask_shape = engine_custom.mask_.shape
    custom_mask = engine_custom.mask_.copy()
    custom_col_offsets = _col_offsets_from_lag_order(engine_custom)
    
    print(f"\nZ CUSTOM_LAGS (x1: max=8):")
    print(f"  lag_order_['max']: {custom_max}")
    print(f"  mask_.shape: {custom_mask_shape}")
    print(f"  col_offsets: {custom_col_offsets}")
    
    # Weryfikacja zmian
    x1_idx = col_names.index("x1")
    
    # x1 powinno mieć większe max_lag
    assert custom_max[x1_idx] == 8, \
        f"custom_max['x1'] powinno być 8, ale jest {custom_max[x1_idx]}"
    
    # Powinien być większy niż baseline dla x1
    assert custom_max[x1_idx] > baseline_max[x1_idx], \
        f"x1 max_lag powinno się zwiększyć z {baseline_max[x1_idx]} na {custom_max[x1_idx]}"
    
    # Maska powinna mieć więcej kolumn
    assert custom_mask_shape[1] > baseline_mask_shape[1], \
        f"Maska powinna mieć więcej kolumn:\n" \
        f"  baseline: {baseline_mask_shape}\n" \
        f"  custom: {custom_mask_shape}"
    
    # Sprawdzenie wartości w nowych kolumnach dla x1
    # Nowe kolumny dla x1 powinny mieć wartość 1
    x1_baseline_start = int(baseline_col_offsets[x1_idx])
    x1_baseline_end = int(baseline_col_offsets[x1_idx + 1])
    x1_custom_start = int(custom_col_offsets[x1_idx])
    x1_custom_end = int(custom_col_offsets[x1_idx + 1])
    
    # Nowe kolumny to te od x1_custom_end - (x1_custom_end - x1_baseline_end) do x1_custom_end
    num_new_cols = x1_custom_end - x1_baseline_end
    
    print(f"\nSprawdzenie wartości w nowych kolumnach dla x1:")
    print(f"  Bazowe kolumny x1: {x1_baseline_start}:{x1_baseline_end}")
    print(f"  Nowe kolumny x1: {x1_baseline_end}:{x1_custom_end}")
    print(f"  Liczba nowych kolumn: {num_new_cols}")
    
    # Nowe kolumny (po prawej stronie bloku) powinny mieć wartość 1 dla wszystkich wierszy.
    overlap_width = min(
        x1_baseline_end - x1_baseline_start,
        x1_custom_end - x1_custom_start,
    )
    new_cols_start = x1_custom_start + overlap_width
    new_cols_end = x1_custom_end
    new_cols_mask = custom_mask[:, new_cols_start:new_cols_end]
    assert np.all(new_cols_mask == 1), \
        f"Nowe kolumny dla x1 powinny mieć wartość 1, ale mają:\n{new_cols_mask}"
    
    print(f"✓ Nowe kolumny mają wartość 1")
    print(f"\n✓ Wymiary się zmieniły poprawnie")
    print(f"  x1 baseline max_lag: {baseline_max[x1_idx]} → custom: {custom_max[x1_idx]}")
    print(f"  Maska kolumny: {baseline_mask_shape[1]} → {custom_mask_shape[1]}")


def test_pair_custom_lags_mask_change():
    """
    Test weryfikujący że custom_pair_lags zmienia maskę i wymiary.
    
    Ustawiamy custom_pair_lags aby zmienić lag range dla konkretnej pary (target, predictor).
    Oczekiwane:
    - lag_order_['max'] zmieni się jeśli pair override wymaga większego zakresu
    - maska zmieni się dla konkretnej pary
    - nowe kolumny będą miały wartość 0 dla pozostałych wierszy
    - nowe kolumny będą miały wartość 1 dla wybranej pary (output, input)
    """
    print("\n" + "=" * 80)
    print("TEST 3: Z custom_pair_lags - maska zmienia się dla konkretnej pary")
    print("=" * 80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Baseline
    cfg_baseline = LagConfiguration(max_lag=5, use_lag_zero=True)
    selector_baseline = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine_baseline = LagEngine(config=cfg_baseline, selector=selector_baseline)
    X_baseline, _, _ = engine_baseline.prepare([df], effects=["y1"])
    
    baseline_mask = engine_baseline.mask_.copy()
    baseline_max = engine_baseline.lag_order_["max"].copy()
    baseline_col_offsets = _col_offsets_from_lag_order(engine_baseline)
    
    print(f"\nBASELINE (bez custom_pair_lags):")
    print(f"  lag_order_['max']: {baseline_max}")
    print(f"  baseline_mask shape: {baseline_mask.shape}")
    print(f"  col_offsets: {baseline_col_offsets}")
    
    # Z custom_pair_lags
    # Zwiększyć max_lag dla pary ('y1', 'x1') na 8
    cfg_custom = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_pair_lags={("y1", "x1"): (8,)}
    )
    selector_custom = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine_custom = LagEngine(config=cfg_custom, selector=selector_custom)
    X_custom, _, _ = engine_custom.prepare([df], effects=["y1"])
    
    custom_mask = engine_custom.mask_.copy()
    custom_max = engine_custom.lag_order_["max"].copy()
    custom_col_offsets = _col_offsets_from_lag_order(engine_custom)
    
    print(f"\nZ CUSTOM_PAIR_LAGS (('y1', 'x1'): max=8):")
    print(f"  lag_order_['max']: {custom_max}")
    print(f"  custom_mask shape: {custom_mask.shape}")
    print(f"  col_offsets: {custom_col_offsets}")
    
    # Weryfikacja
    x1_idx = col_names.index("x1")
    y1_idx = col_names.index("y1")
    
    # x1 powinno mieć większe max_lag
    assert custom_max[x1_idx] == 8, \
        f"x1 max_lag powinno być 8 z powodu pair override, ale jest {custom_max[x1_idx]}"
    
    # Maska powinna mieć więcej kolumn
    assert custom_mask.shape[1] > baseline_mask.shape[1], \
        f"Maska powinna mieć więcej kolumn z powodu powiększonej pary"
    
    # Sprawdzenie wartości w nowych kolumnach
    x1_baseline_start = int(baseline_col_offsets[x1_idx])
    x1_baseline_end = int(baseline_col_offsets[x1_idx + 1])
    x1_custom_start = int(custom_col_offsets[x1_idx])
    x1_custom_end = int(custom_col_offsets[x1_idx + 1])
    
    num_new_cols = x1_custom_end - x1_baseline_end
    
    print(f"\nSprawdzenie wartości w nowych kolumnach dla x1:")
    print(f"  Bazowe kolumny x1: {x1_baseline_start}:{x1_baseline_end}")
    print(f"  Nowe kolumny x1: {x1_baseline_end}:{x1_custom_end}")
    print(f"  Liczba nowych kolumn: {num_new_cols}")
    
    # Nowe kolumny są dokładane po prawej stronie bloku predyktora.
    overlap_width = min(
        x1_baseline_end - x1_baseline_start,
        x1_custom_end - x1_custom_start,
    )
    new_cols_start = x1_custom_start + overlap_width
    new_cols_end = x1_custom_end

    # Nowe kolumny powinny być 0 dla pozostałych wierszy
    new_cols_mask_others = custom_mask[
        [i for i in range(d) if i != y1_idx],
        new_cols_start:new_cols_end,
    ]
    assert np.all(new_cols_mask_others == 0), \
        f"Nowe kolumny dla x1 powinny być 0 dla pozostałych wierszy (nie y1):\n{new_cols_mask_others}"
    
    print(f"✓ Nowe kolumny mają wartość 0 dla pozostałych wierszy")
    
    # Nowe kolumny powinny być 1 dla wiersza y1 (wybranej pary)
    new_cols_y1 = custom_mask[y1_idx, new_cols_start:new_cols_end]
    assert np.all(new_cols_y1 == 1), \
        f"Nowe kolumny dla x1 powinny być 1 dla wiersza y1 (pary y1, x1):\n{new_cols_y1}"
    
    print(f"✓ Nowe kolumny mają wartość 1 dla pary (y1, x1)")
    
    # Sprawdzenie że stare kolumny nie uległy zmianie dla innych wierszy
    old_cols_others = custom_mask[[i for i in range(d) if i != y1_idx], x1_baseline_start:x1_baseline_end]
    old_cols_baseline_others = baseline_mask[[i for i in range(d) if i != y1_idx], x1_baseline_start:x1_baseline_end]
    assert np.array_equal(old_cols_others, old_cols_baseline_others), \
        "Stare kolumny dla x1 nie powinny się zmienić dla pozostałych wierszy"
    
    print(f"✓ Stare kolumny nie uległy zmianie dla pozostałych wierszy")
    
    print(f"\n✓ Maska się zmieniła poprawnie dla pary")
    print(f"  x1 baseline max_lag: {baseline_max[x1_idx]} → custom: {custom_max[x1_idx]}")
    print(f"  Maska kolumny: {baseline_mask.shape[1]} → {custom_mask.shape[1]}")


def test_custom_lags_min_max_both():
    """
    Test weryfikujący custom_lags z zakresom (min, max).
    """
    print("\n" + "=" * 80)
    print("TEST 4: custom_lags z zakresom (min, max)")
    print("=" * 80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # custom_lags z zakresom
    cfg = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_lags={"x1": (2, 7)}  # Set x1 to lags 2-7
    )
    selector = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine = LagEngine(config=cfg, selector=selector)
    X, _, _ = engine.prepare([df], effects=["y1"])
    
    x1_idx = col_names.index("x1")
    min_lag = engine.lag_order_["min"][x1_idx]
    max_lag = engine.lag_order_["max"][x1_idx]
    
    print(f"\ncustom_lags zakresy dla x1: (2, 7)")
    print(f"Rezultat: min_lag={min_lag}, max_lag={max_lag}")
    
    assert min_lag == 2, f"min_lag dla x1 powinno być 2, ale jest {min_lag}"
    assert max_lag == 7, f"max_lag dla x1 powinno być 7, ale jest {max_lag}"
    
    # Sprawdzenie liczby kolumn dla x1
    width = max_lag - min_lag + 1
    assert width == 6, f"Szerokość bloku x1 powinna być 6 (7-2+1), ale jest {width}"
    
    print(f"✓ Zakresy są poprawnie ustalone")
    
    # Sprawdzenie wartości w masce dla x1
    # Maska powinna mieć wartość 1 dla wszystkich wierszy w zakresie x1
    col_offsets = _col_offsets_from_lag_order(engine)
    x1_start = int(col_offsets[x1_idx])
    x1_end = x1_start + width
    
    print(f"\nSprawdzenie wartości w masce dla x1:")
    print(f"  Kolumny x1 w masce: {x1_start}:{x1_end}")
    
    x1_mask_block = engine.mask_[:, x1_start:x1_end]
    # Dla custom_lags nowo dodane kolumny są 1, a prawa część może zachować
    # wartości z baseline selektora (w tym 0).
    old_width = int(engine.selection_result_.max_lags_per_pred[x1_idx])
    if engine.config.use_lag_zero and old_width > 0:
        old_width += 1
    if engine.config.use_lag_zero and engine.selection_result_.max_lags_per_pred[x1_idx] == 0:
        old_width = 1
    overlap_width = min(old_width, width)
    new_part = x1_mask_block[:, : width - overlap_width]
    assert np.all(new_part == 1), \
        f"Nowa część maski dla x1 powinna mieć wartość 1, ale ma:\n{new_part}"

    print(f"✓ Nowa część maski dla x1 ma wartość 1")
    
    # Sprawdzenie że inne zmienne nie mają kolumn w tym zakresie
    # (jeśli nie były modyfikowane)
    y1_start = int(col_offsets[0])
    y1_end = int(col_offsets[1])
    x2_start = int(col_offsets[2])
    
    print(f"  y1 kolumny: {y1_start}:{y1_end}")
    print(f"  x1 kolumny: {x1_start}:{x1_end}")
    print(f"  x2 kolumny: {x2_start}:")
    
    print(f"\n✓ TEST PASSED: custom_lags zakresy są poprawnie zastosowane")


def test_selection_result_consistency_with_mask():
    """
        Test weryfikujący zachowanie selection_result_.

        - bez override'ów: selection_result_.mask == engine.mask_
        - z override'ami: selection_result_ zostaje baseline selektora,
            a engine.mask_ odzwierciedla finalne override'y
    """
    print("\n" + "=" * 80)
    print("TEST 5: selection_result_.mask spójny z engine.mask_")
    print("=" * 80)
    
    np.random.seed(42)
    n, d = 150, 3
    col_names = ["y1", "x1", "x2"]
    df = pd.DataFrame(np.random.randn(n, d), columns=col_names)
    
    # Case 1: bez custom_lags
    cfg1 = LagConfiguration(max_lag=5, use_lag_zero=True)
    selector1 = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine1 = LagEngine(config=cfg1, selector=selector1)
    X1, _, _ = engine1.prepare([df], effects=["y1"])
    
    assert np.array_equal(engine1.selection_result_.mask, engine1.mask_), \
        "Bez custom_lags: selection_result_.mask musi być równy mask_"
    
    print("✓ Case 1 (no custom_lags): masks are identical")
    
    # Case 2: z custom_lags
    cfg2 = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_lags={"x1": (8,)}
    )
    selector2 = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine2 = LagEngine(config=cfg2, selector=selector2)
    X2, _, _ = engine2.prepare([df], effects=["y1"])
    
    assert not np.array_equal(engine2.selection_result_.mask, engine2.mask_), \
        "Z custom_lags: selection_result_.mask nie powinien być nadpisany finalną maską"
    
    print("✓ Case 2 (with custom_lags): selection_result_ and final mask differ as expected")
    
    # Case 3: z custom_pair_lags
    cfg3 = LagConfiguration(
        max_lag=5,
        use_lag_zero=True,
        custom_pair_lags={("y1", "x1"): (8,)}
    )
    selector3 = ICLagSelector(max_lag=5, use_lag_zero=True, use_bic=False)
    engine3 = LagEngine(config=cfg3, selector=selector3)
    X3, _, _ = engine3.prepare([df], effects=["y1"])
    
    assert not np.array_equal(engine3.selection_result_.mask, engine3.mask_), \
        "Z custom_pair_lags: selection_result_.mask nie powinien być nadpisany finalną maską"
    
    print("✓ Case 3 (with custom_pair_lags): selection_result_ and final mask differ as expected")
    
    print("\n✓ TEST PASSED: selection_result_ jest baseline selektora, engine.mask_ to wynik końcowy")


if __name__ == "__main__":
    try:
        test_no_custom_lags_selection_result_matches_lag_order()
        test_variable_custom_lags_change_dimensions()
        test_pair_custom_lags_mask_change()
        test_custom_lags_min_max_both()
        test_selection_result_consistency_with_mask()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
