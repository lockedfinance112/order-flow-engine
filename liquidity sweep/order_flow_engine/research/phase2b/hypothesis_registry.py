import json

def float_val(v):
    if v is None or v == "" or v == "None" or v == "N/A":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0

def get_events(sig: dict) -> list:
    evs_str = sig.get("recent_events", "[]")
    try:
        return json.loads(evs_str)
    except Exception:
        return []

# H01
def h01_cvd_alignment(sig: dict) -> bool:
    direction = sig.get("direction")
    cvd = float_val(sig.get("session_cvd_usdt"))
    if direction == "LONG":
        return cvd > 0
    elif direction == "SHORT":
        return cvd < 0
    return False

# H02
def h02_full_delta_alignment(sig: dict) -> bool:
    direction = sig.get("direction")
    d1 = float_val(sig.get("delta_1m_usdt"))
    d5 = float_val(sig.get("delta_5m_usdt"))
    d15 = float_val(sig.get("delta_15m_usdt"))
    if direction == "LONG":
        return d1 > 0 and d5 > 0 and d15 > 0
    elif direction == "SHORT":
        return d1 < 0 and d5 < 0 and d15 < 0
    return False

# H03
def h03_15m_delta_alignment(sig: dict) -> bool:
    direction = sig.get("direction")
    d15 = float_val(sig.get("delta_15m_usdt"))
    if direction == "LONG":
        return d15 > 0
    elif direction == "SHORT":
        return d15 < 0
    return False

# H04
def h04_stronger_book_20(sig: dict) -> bool:
    direction = sig.get("direction")
    imb = float_val(sig.get("imbalance"))
    if direction == "LONG":
        return imb >= 0.20
    elif direction == "SHORT":
        return imb <= -0.20
    return False

# H05
def h05_stronger_book_30(sig: dict) -> bool:
    direction = sig.get("direction")
    imb = float_val(sig.get("imbalance"))
    if direction == "LONG":
        return imb >= 0.30
    elif direction == "SHORT":
        return imb <= -0.30
    return False

# H06
def h06_active_directional_sweep(sig: dict) -> bool:
    direction = sig.get("direction")
    sweep_active = sig.get("sweep_active") == "True" or sig.get("sweep_active") is True
    sweep_dir = String(sig.get("sweep_direction", "")).upper()
    if direction == "LONG":
        return sweep_active and "BULLISH" in sweep_dir
    elif direction == "SHORT":
        return sweep_active and "BEARISH" in sweep_dir
    return False

# H07
def h07_high_sweep_score(sig: dict) -> bool:
    score = float_val(sig.get("sweep_score"))
    return score >= 8

# H08
def h08_oi_rising(sig: dict) -> bool:
    oi_chg = float_val(sig.get("open_interest_change_pct"))
    return oi_chg > 0

# H09
def h09_trend_5m_aligned(sig: dict) -> bool:
    direction = sig.get("direction")
    trend = String(sig.get("trend_5m", "")).lower()
    if direction == "LONG":
        return trend == "up"
    elif direction == "SHORT":
        return trend == "down"
    return False

# H10
def h10_trend_15m_aligned(sig: dict) -> bool:
    direction = sig.get("direction")
    trend = String(sig.get("trend_15m", "")).lower()
    if direction == "LONG":
        return trend == "up"
    elif direction == "SHORT":
        return trend == "down"
    return False

# H11
def h11_trend_5m_and_15m_aligned(sig: dict) -> bool:
    direction = sig.get("direction")
    t5 = String(sig.get("trend_5m", "")).lower()
    t15 = String(sig.get("trend_15m", "")).lower()
    if direction == "LONG":
        return t5 == "up" and t15 == "up"
    elif direction == "SHORT":
        return t5 == "down" and t15 == "down"
    return False

# H12
def h12_binance_context_confirmed(sig: dict) -> bool:
    confirm = String(sig.get("binance_context_confirm", "")).upper()
    return confirm == "CONFIRMED"

# H13
def h13_low_book_drift(sig: dict) -> bool:
    drift = float_val(sig.get("book_drift"))
    return drift < 0.30

# H14
def h14_no_opposing_divergence(sig: dict) -> bool:
    direction = sig.get("direction")
    events = get_events(sig)
    if direction == "LONG":
        return "BEARISH_DIVERGENCE" not in events
    elif direction == "SHORT":
        return "BULLISH_DIVERGENCE" not in events
    return False

# H15
def h15_no_opposing_absorption(sig: dict) -> bool:
    direction = sig.get("direction")
    events = get_events(sig)
    if direction == "LONG":
        return "BEARISH_ABSORPTION" not in events and "CONFIRMED_BEARISH_ABSORPTION" not in events and "POSSIBLE_BEARISH_ABSORPTION" not in events
    elif direction == "SHORT":
        return "BULLISH_ABSORPTION" not in events and "CONFIRMED_BULLISH_ABSORPTION" not in events and "POSSIBLE_BULLISH_ABSORPTION" not in events
    return False

# H16
def h16_directional_aggression_present(sig: dict) -> bool:
    direction = sig.get("direction")
    events = get_events(sig)
    if direction == "LONG":
        return "BUY_AGGRESSION" in events
    elif direction == "SHORT":
        return "SELL_AGGRESSION" in events
    return False

def String(v):
    if v is None:
        return ""
    return str(v)

HYPOTHESES = {
    "H01_CVD_ALIGNMENT": h01_cvd_alignment,
    "H02_FULL_DELTA_ALIGNMENT": h02_full_delta_alignment,
    "H03_15M_DELTA_ALIGNMENT": h03_15m_delta_alignment,
    "H04_STRONGER_BOOK_20": h04_stronger_book_20,
    "H05_STRONGER_BOOK_30": h05_stronger_book_30,
    "H06_ACTIVE_DIRECTIONAL_SWEEP": h06_active_directional_sweep,
    "H07_HIGH_SWEEP_SCORE": h07_high_sweep_score,
    "H08_OI_RISING": h08_oi_rising,
    "H09_TREND_5M_ALIGNED": h09_trend_5m_aligned,
    "H10_TREND_15M_ALIGNED": h10_trend_15m_aligned,
    "H11_TREND_5M_AND_15M_ALIGNED": h11_trend_5m_and_15m_aligned,
    "H12_BINANCE_CONTEXT_CONFIRMED": h12_binance_context_confirmed,
    "H13_LOW_BOOK_DRIFT": h13_low_book_drift,
    "H14_NO_OPPOSING_DIVERGENCE": h14_no_opposing_divergence,
    "H15_NO_OPPOSING_ABSORPTION": h15_no_opposing_absorption,
    "H16_DIRECTIONAL_AGGRESSION_PRESENT": h16_directional_aggression_present
}
