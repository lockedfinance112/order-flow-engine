from typing import List, Dict, Any, Tuple

def evaluate_timeframe_evidence(feats: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes trend_up, trend_down, range, breakout_up, breakout_down evidence
    for a single timeframe. Returns values in range 0.0 to 1.0.
    """
    # Defaults
    trend_up = 0.0
    trend_down = 0.0
    range_score = 0.0
    breakout_up = 0.0
    breakout_down = 0.0

    # Ensure required features exist and are not None
    ema_gap = feats.get("ema_gap_bps")
    ema_slope = feats.get("ema_slope_atr")
    adx14 = feats.get("adx14")
    er20 = feats.get("er20")
    realized_vol = feats.get("realized_vol20")
    breakout_dist = feats.get("breakout_dist")
    vol_zscore = feats.get("vol_zscore")
    
    if (ema_gap is None or ema_slope is None or adx14 is None or 
        er20 is None or realized_vol is None or breakout_dist is None):
        return {
            "trend_up": 0.0,
            "trend_down": 0.0,
            "range": 0.0,
            "breakout_up": 0.0,
            "breakout_down": 0.0
        }

    # --- Trend Up ---
    if ema_gap > 0:
        trend_up += 0.25
    trend_up += min(max(ema_slope, 0.0) / 0.5, 1.0) * 0.25
    trend_up += er20 * 0.25
    trend_up += min(adx14 / 50.0, 1.0) * 0.25

    # --- Trend Down ---
    if ema_gap < 0:
        trend_down += 0.25
    trend_down += min(max(-ema_slope, 0.0) / 0.5, 1.0) * 0.25
    trend_down += er20 * 0.25
    trend_down += min(adx14 / 50.0, 1.0) * 0.25

    # --- Range ---
    # Range is stronger when ADX is low, ER is low, EMA gap is small, and slope is flat
    range_score += max(0.0, 1.0 - adx14 / 25.0) * 0.30
    range_score += (1.0 - er20) * 0.30
    range_score += max(0.0, 1.0 - abs(ema_gap) / 20.0) * 0.20
    range_score += max(0.0, 1.0 - abs(ema_slope) / 0.1) * 0.20

    # --- Breakout Up / Down ---
    if breakout_dist > 0.0:
        # Minimum baseline breakout evidence when exit occurs
        breakout_up = 0.40
        breakout_up += min(breakout_dist / 2.0, 0.40)
        if vol_zscore is not None:
            breakout_up += min(max(vol_zscore, 0.0) / 2.0, 1.0) * 0.20
    elif breakout_dist < 0.0:
        breakout_down = 0.40
        breakout_down += min(-breakout_dist / 2.0, 0.40)
        if vol_zscore is not None:
            breakout_down += min(max(vol_zscore, 0.0) / 2.0, 1.0) * 0.20

    return {
        "trend_up": min(max(trend_up, 0.0), 1.0),
        "trend_down": min(max(trend_down, 0.0), 1.0),
        "range": min(max(range_score, 0.0), 1.0),
        "breakout_up": min(max(breakout_up, 0.0), 1.0),
        "breakout_down": min(max(breakout_down, 0.0), 1.0)
    }

def classify_regime(
    tf_features: Dict[str, Dict[str, Any]], 
    config: Dict[str, Any]
) -> Tuple[str, float, str, str, Dict[str, float], List[str]]:
    """
    Combines multi-timeframe evidence using alignment weights.
    Returns:
      (primary_regime, confidence, structure, direction, scores, reasons)
    """
    timeframes = ["1m", "5m", "15m", "1h"]
    weights = config.get("REGIME_TIMEFRAME_WEIGHTS", {
        "1m": 0.10,
        "5m": 0.25,
        "15m": 0.35,
        "1h": 0.30
    })

    # Weighted sum of evidence scores
    scores = {
        "trend_up": 0.0,
        "trend_down": 0.0,
        "range": 0.0,
        "breakout_up": 0.0,
        "breakout_down": 0.0
    }

    tf_scores = {}
    for tf in timeframes:
        feats = tf_features.get(tf, {})
        tf_ev = evaluate_timeframe_evidence(feats)
        tf_scores[tf] = tf_ev
        for k in scores:
            scores[k] += tf_ev[k] * weights[tf]

    # Resolve primary regime
    trend_up = scores["trend_up"]
    trend_down = scores["trend_down"]
    range_score = scores["range"]
    breakout_up = scores["breakout_up"]
    breakout_down = scores["breakout_down"]

    # Precedence resolution
    primary_regime = "UNKNOWN"
    structure = "UNKNOWN"
    direction = "FLAT"
    reasons = []

    # 1. Breakout check
    if breakout_up >= 0.70:
        primary_regime = "BREAKOUT_UP"
        structure = "BREAKOUT"
        direction = "UP"
        reasons.append("Multi-timeframe breakout up detected")
    elif breakout_down >= 0.70:
        primary_regime = "BREAKOUT_DOWN"
        structure = "BREAKOUT"
        direction = "DOWN"
        reasons.append("Multi-timeframe breakout down detected")
    # 2. Trend check
    elif trend_up >= 0.65 and trend_up > trend_down:
        primary_regime = "TREND_UP"
        structure = "TREND"
        direction = "UP"
        reasons.append("Aligned multi-timeframe upward trend")
    elif trend_down >= 0.65 and trend_down > trend_up:
        primary_regime = "TREND_DOWN"
        structure = "TREND"
        direction = "DOWN"
        reasons.append("Aligned multi-timeframe downward trend")
    # 3. Range check
    elif range_score >= 0.65:
        primary_regime = "RANGE"
        structure = "RANGE"
        direction = "FLAT"
        reasons.append("Low volatility rolling range consolidation")
        
    # Check for conflict / transition (runner-up margin check)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner, winner_val = sorted_scores[0]
    runner_up, runner_up_val = sorted_scores[1]

    if primary_regime != "UNKNOWN":
        # Check if winner score is not dominant enough over runner-up
        if (winner_val - runner_up_val) < 0.10:
            primary_regime = "TRANSITION"
            structure = "TRANSITION"
            reasons.append(f"Transition conflict: {winner} ({winner_val:.2f}) close to {runner_up} ({runner_up_val:.2f})")

    # If no score is >= 0.50, it is UNKNOWN
    if winner_val < 0.50:
        primary_regime = "UNKNOWN"
        structure = "UNKNOWN"
        direction = "UNKNOWN"
        reasons.append("Insufficient classification evidence")

    # Confidence calculation
    # confidence = 0.40 * winner_score + 0.25 * normalized_margin + 0.20 * timeframe_agreement + ...
    # Timeframe agreement: fraction of TFs where direction matches the winner
    tf_agreement = 0.0
    matching_tfs = 0
    for tf in timeframes:
        tf_winner = sorted(tf_scores[tf].items(), key=lambda x: x[1], reverse=True)[0][0]
        if tf_winner == winner:
            matching_tfs += 1
    tf_agreement = matching_tfs / len(timeframes)

    margin = winner_val - runner_up_val
    confidence = (
        0.40 * winner_val +
        0.25 * margin +
        0.20 * tf_agreement +
        0.15 * (1.0 if winner_val >= 0.65 else 0.5)
    )
    confidence = min(max(confidence, 0.0), 1.0)

    # Specific reasons extraction for explainability
    p15 = tf_features.get("15m", {})
    if p15:
        if p15.get("ema20") is not None and p15.get("ema50") is not None:
            if p15["ema20"] > p15["ema50"]:
                reasons.append("15m EMA20 above EMA50")
            else:
                reasons.append("15m EMA20 below EMA50")
        if p15.get("adx14") is not None:
            reasons.append(f"15m ADX is {p15['adx14']:.0f}")
        if p15.get("er20") is not None:
            reasons.append(f"15m ER20 is {p15['er20']:.2f}")

    return primary_regime, confidence, structure, direction, scores, reasons
