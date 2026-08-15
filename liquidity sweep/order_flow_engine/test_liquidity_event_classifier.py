"""Unit and edge-case test suite for deterministic price outcome classifier."""

from decimal import Decimal
import pytest

from liquidity_event.models import (
    AggressorSide,
    EventClassification,
    LiquiditySide,
    MarketTrade,
)
from liquidity_event.policy import LiquidityClassificationPolicy
from liquidity_event.classifier import (
    CandidateProgress,
    PriceDecision,
    PriceOutcomeTracker,
)


def trade(price: float | Decimal, time_ms: int, seq: int = 1, side: str = "BUY", symbol: str = "BTCUSDT") -> MarketTrade:
    aggressor = AggressorSide.BUY if side.upper() == "BUY" else AggressorSide.SELL
    return MarketTrade(
        symbol=symbol,
        price=price,
        quantity=1.0,
        aggressor_side=aggressor,
        exchange_time_ms=time_ms,
        sequence_id=seq,
    )


def feed_prices(
    side: LiquiditySide,
    prices: list[float | Decimal],
    times: list[int],
    level: float | Decimal = 100.0,
    policy: LiquidityClassificationPolicy | None = None,
    symbol: str = "BTCUSDT",
) -> PriceOutcomeTracker:
    pol = policy or LiquidityClassificationPolicy()
    tracker = PriceOutcomeTracker(level=level, side=side, policy=pol, symbol=symbol)
    for idx, (p, t) in enumerate(zip(prices, times), start=1):
        tracker.on_trade(trade(price=p, time_ms=t, seq=idx, symbol=symbol))
    return tracker


def tracker_with_sufficient_times(
    reclaim: int | None = None,
    acceptance: int | None = None,
    side: LiquiditySide = LiquiditySide.SELL_SIDE,
    level: float | Decimal = 100.0,
    policy: LiquidityClassificationPolicy | None = None,
    has_penetration: bool = True,
) -> PriceOutcomeTracker:
    pol = policy or LiquidityClassificationPolicy()
    tracker = PriceOutcomeTracker(level=level, side=side, policy=pol)
    if has_penetration:
        penetration_price = 99.98 if side is LiquiditySide.SELL_SIDE else 100.02
        tracker.on_trade(trade(price=penetration_price, time_ms=0, seq=0))
    if reclaim is not None:
        tracker._reclaim_progress = CandidateProgress.sufficient_at(reclaim)
    if acceptance is not None:
        tracker._acceptance_progress = CandidateProgress.sufficient_at(acceptance)
    return tracker


# --- 1. Core Truth Table Tests (All 4 Outcomes) ---

def test_sell_side_reclaim_is_failed_breakdown():
    # Sell-side sweep at 100.0:
    # Penetration: 99.98 (< 100)
    # Reclaim threshold: 100.0 * (1 + 1/10000) = 100.01
    # 3 trades > 100.01 over 3,000ms (1,000 to 4,000)
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_000])
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.FAILED_BREAKDOWN
    assert decision.reason_code == "RECLAIM_CONFIRMED"
    assert decision.market_resolution_time_ms == 4_000
    assert decision.has_penetration is True


def test_sell_side_acceptance_is_bearish_continuation():
    # Sell-side sweep at 100.0:
    # Penetration: 99.98 (< 100)
    # Acceptance threshold: 100.0 * (1 - 1/10000) = 99.99
    # 3 trades < 99.99 over 3,000ms (1,000 to 4,000)
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 99.98, 99.98, 99.98], [0, 1_000, 2_000, 4_000])
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.BEARISH_CONTINUATION
    assert decision.reason_code == "ACCEPTANCE_CONFIRMED"
    assert decision.market_resolution_time_ms == 4_000
    assert decision.has_penetration is True


def test_buy_side_reclaim_is_failed_breakout():
    # Buy-side sweep at 100.0:
    # Penetration: 100.02 (> 100)
    # Reclaim threshold: 100.0 * (1 - 1/10000) = 99.99
    # 3 trades < 99.99 over 3,000ms (1,000 to 4,000)
    tracker = feed_prices(LiquiditySide.BUY_SIDE, [100.02, 99.98, 99.98, 99.98], [0, 1_000, 2_000, 4_000])
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.FAILED_BREAKOUT
    assert decision.reason_code == "RECLAIM_CONFIRMED"
    assert decision.market_resolution_time_ms == 4_000
    assert decision.has_penetration is True


def test_buy_side_acceptance_is_bullish_continuation():
    # Buy-side sweep at 100.0:
    # Penetration: 100.02 (> 100)
    # Acceptance threshold: 100.0 * (1 + 1/10000) = 100.01
    # 3 trades > 100.01 over 3,000ms (1,000 to 4,000)
    tracker = feed_prices(LiquiditySide.BUY_SIDE, [100.02, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_000])
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.BULLISH_CONTINUATION
    assert decision.reason_code == "ACCEPTANCE_CONFIRMED"
    assert decision.market_resolution_time_ms == 4_000
    assert decision.has_penetration is True


# --- 2. Penetration Strictness Tests ---

def test_strict_sell_side_penetration_boundary():
    # 99.999 < 100.0 -> penetration confirmed
    tracker_pen = feed_prices(LiquiditySide.SELL_SIDE, [99.999], [1_000])
    assert tracker_pen.has_penetration is True
    assert tracker_pen.penetration_bps > 0

    # 100.0 == 100.0 -> no penetration
    tracker_no_pen = feed_prices(LiquiditySide.SELL_SIDE, [100.000], [1_000])
    assert tracker_no_pen.has_penetration is False
    assert tracker_no_pen.decision_at(60_000, 60_000).classification is EventClassification.INVALID
    assert tracker_no_pen.decision_at(60_000, 60_000).reason_code == "PENETRATION_NOT_CONFIRMED"


def test_strict_buy_side_penetration_boundary():
    # 100.001 > 100.0 -> penetration confirmed
    tracker_pen = feed_prices(LiquiditySide.BUY_SIDE, [100.001], [1_000])
    assert tracker_pen.has_penetration is True
    assert tracker_pen.penetration_bps > 0

    # 100.0 == 100.0 -> no penetration
    tracker_no_pen = feed_prices(LiquiditySide.BUY_SIDE, [100.000], [1_000])
    assert tracker_no_pen.has_penetration is False
    assert tracker_no_pen.decision_at(60_000, 60_000).classification is EventClassification.INVALID
    assert tracker_no_pen.decision_at(60_000, 60_000).reason_code == "PENETRATION_NOT_CONFIRMED"


def test_reclaim_pattern_without_penetration_is_invalid():
    # Price trades above reclaim threshold immediately without ever penetrating below swept level
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [100.02, 100.02, 100.02], [1_000, 2_000, 4_000])
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.PENDING_SWEEP  # Still within window waiting for penetration
    final_decision = tracker.decision_at(watermark_ms=60_000, expiry_ms=60_000)
    assert final_decision.classification is EventClassification.INVALID
    assert final_decision.reason_code == "PENETRATION_NOT_CONFIRMED"


# --- 3. Hold Duration & Minimum Trade Count Boundaries ---

def test_hold_2999_ms_is_insufficient():
    # 3 trades over 2,999ms (1,000 to 3,999) -> hold < 3,000ms -> insufficient
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 3_999])
    assert tracker.reclaim_sufficient_time_ms is None
    assert tracker.reclaim_qualifying_duration_ms == 2_999
    assert tracker.reclaim_trade_count == 3


def test_hold_3000_ms_and_3_trades_is_sufficient():
    # 3 trades over exactly 3,000ms (1,000 to 4,000) -> hold = 3,000ms -> sufficient
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_000])
    assert tracker.reclaim_sufficient_time_ms == 4_000
    assert tracker.reclaim_qualifying_duration_ms == 3_000
    assert tracker.reclaim_trade_count == 3


def test_hold_3001_ms_is_sufficient():
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_001])
    assert tracker.reclaim_sufficient_time_ms == 4_001
    assert tracker.reclaim_qualifying_duration_ms == 3_001


def test_two_trades_over_long_duration_is_insufficient():
    # 2 trades over 10,000ms (1,000 to 11,000) -> trade count < 3 -> insufficient
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02], [0, 1_000, 11_000])
    assert tracker.reclaim_sufficient_time_ms is None
    assert tracker.reclaim_qualifying_duration_ms == 10_000
    assert tracker.reclaim_trade_count == 2


def test_ten_trades_under_2900_ms_is_insufficient():
    # 10 trades over 2,900ms -> duration < 3,000ms -> insufficient
    times = [0] + [1_000 + i * 290 for i in range(10)]
    prices = [99.98] + [100.02] * 10
    tracker = feed_prices(LiquiditySide.SELL_SIDE, prices, times)
    assert tracker.reclaim_sufficient_time_ms is None
    assert tracker.reclaim_trade_count == 10
    assert tracker.reclaim_qualifying_duration_ms == 2_610


# --- 4. Neutral Zone and Pause Semantics ---

def test_neutral_zone_pauses_qualifying_duration_without_accumulating_gap():
    # Reclaim trades at 100, 1,100 (duration = 1,000ms)
    # Neutral trade at 1,200 (pauses segment)
    # Long gap until 9,100
    # Reclaim trades at 9,100, 10,100 (resumes segment; duration = 1,000ms + 1,000ms = 2,000ms)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 100.02, 100.02, 100.00, 100.02, 100.02],
        [0, 100, 1_100, 1_200, 9_100, 10_100],
    )
    assert tracker.reclaim_qualifying_duration_ms == 2_000
    assert tracker.reclaim_trade_count == 4
    assert tracker.reclaim_sufficient_time_ms is None  # 2,000ms < 3,000ms


def test_exact_threshold_trades_are_neutral():
    # For sell-side at 100.0:
    # Reclaim threshold = 100.01
    # Acceptance threshold = 99.99
    # Penetration at 99.995 is below 100.0 (penetration), but above 99.99 (neutral, not acceptance)
    # Trades exactly on 100.01 or 99.99 or between are NEUTRAL
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.995, 100.01, 99.99, 100.00],
        [0, 1_000, 2_000, 3_000],
    )
    assert tracker.has_penetration is True
    assert tracker.reclaim_trade_count == 0
    assert tracker.acceptance_trade_count == 0
    assert tracker.reclaim_qualifying_duration_ms == 0
    assert tracker.acceptance_qualifying_duration_ms == 0


# --- 5. Opposite Threshold Resets Competing Branch ---

def test_opposite_threshold_resets_reclaim_branch_and_starts_acceptance():
    # Reclaim active from 100 to 2,000 (duration 1,900ms, count 2)
    # Opposite trade < 99.99 at 2,100 -> Resets reclaim, starts acceptance (count 1, duration 0)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 100.02, 100.02, 99.98],
        [0, 100, 2_000, 2_100],
    )
    assert tracker.reclaim_qualifying_duration_ms == 0
    assert tracker.reclaim_trade_count == 0
    assert tracker.acceptance_trade_count == 1
    assert tracker.acceptance_qualifying_duration_ms == 0


def test_opposite_threshold_resets_acceptance_branch_and_starts_reclaim():
    # Acceptance active from 100 to 2,000 (duration 1,900ms, count 2)
    # Opposite trade > 100.01 at 2,100 -> Resets acceptance, starts reclaim (count 1, duration 0)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 99.98, 99.98, 100.02],
        [0, 100, 2_000, 2_100],
    )
    assert tracker.acceptance_qualifying_duration_ms == 0
    assert tracker.acceptance_trade_count == 0
    assert tracker.reclaim_trade_count == 1
    assert tracker.reclaim_qualifying_duration_ms == 0


# --- 6. Earlier Sufficient Time & Conflicting Confirmation ---

def test_earlier_reclaim_sufficient_time_wins_over_later_acceptance():
    tracker = tracker_with_sufficient_times(reclaim=5_000, acceptance=7_000)
    decision = tracker.decision_at(watermark_ms=8_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.FAILED_BREAKDOWN
    assert decision.market_resolution_time_ms == 5_000


def test_earlier_acceptance_sufficient_time_wins_over_later_reclaim():
    tracker = tracker_with_sufficient_times(reclaim=8_000, acceptance=4_000)
    decision = tracker.decision_at(watermark_ms=9_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.BEARISH_CONTINUATION
    assert decision.market_resolution_time_ms == 4_000


def test_equal_sufficient_times_produce_conflicting_confirmation_indeterminate():
    tracker = tracker_with_sufficient_times(reclaim=5_000, acceptance=5_000)
    decision = tracker.decision_at(watermark_ms=7_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.INDETERMINATE
    assert decision.reason_code == "CONFLICTING_CONFIRMATION"
    assert decision.market_resolution_time_ms == 5_000


# --- 7. Watermark and Timeout Settlement ---

def test_watermark_must_settle_sufficient_time_before_decision():
    tracker = tracker_with_sufficient_times(reclaim=5_000)
    # Watermark at 4,999 (behind sufficient time) -> cannot finalize yet
    assert tracker.decision_at(watermark_ms=4_999, expiry_ms=60_000).classification is EventClassification.PENDING_SWEEP
    # Watermark at 5,000 (settles sufficient time) -> resolves
    decision = tracker.decision_at(watermark_ms=5_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.FAILED_BREAKDOWN
    assert decision.market_resolution_time_ms == 5_000


def test_timeout_settlement_at_expiry():
    # Only 1 qualifying trade, never sufficient
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02], [0, 1_000])
    # Watermark before expiry
    assert tracker.decision_at(watermark_ms=59_999, expiry_ms=60_000).classification is EventClassification.PENDING_SWEEP
    # Watermark at expiry -> confirmation window expired
    timeout_dec = tracker.decision_at(watermark_ms=60_000, expiry_ms=60_000)
    assert timeout_dec.classification is EventClassification.INDETERMINATE
    assert timeout_dec.reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    assert timeout_dec.market_resolution_time_ms == 60_000


# --- 8. Determinism and Symbol Isolation ---

def test_canonical_trades_produce_identical_classification_independent_of_feed_mode():
    trades_list = [
        trade(price=99.98, time_ms=0, seq=1),
        trade(price=100.02, time_ms=1_000, seq=2),
        trade(price=100.02, time_ms=2_000, seq=3),
        trade(price=100.02, time_ms=4_000, seq=4),
    ]
    tracker_fwd = PriceOutcomeTracker(level=100.0, side=LiquiditySide.SELL_SIDE, policy=LiquidityClassificationPolicy())
    for t in trades_list:
        tracker_fwd.on_trade(t)

    tracker_sorted = PriceOutcomeTracker(level=100.0, side=LiquiditySide.SELL_SIDE, policy=LiquidityClassificationPolicy())
    for t in sorted(reversed(trades_list), key=lambda x: x.canonical_key):
        tracker_sorted.on_trade(t)

    assert tracker_fwd.decision_at(5_000, 60_000) == tracker_sorted.decision_at(5_000, 60_000)


def test_tracker_ignores_trades_from_different_symbols():
    tracker = PriceOutcomeTracker(level=100.0, side=LiquiditySide.SELL_SIDE, symbol="BTCUSDT")
    # Feed ETHUSDT trades
    tracker.on_trade(trade(price=99.98, time_ms=0, seq=1, symbol="ETHUSDT"))
    tracker.on_trade(trade(price=100.02, time_ms=1_000, seq=2, symbol="ETHUSDT"))
    assert tracker.has_penetration is False
    assert tracker.reclaim_trade_count == 0


# --- 9. Final Hardening Tests (Pre-penetration Gating, Sufficiency Preservation, Expiry Boundaries) ---

def test_pre_penetration_reclaim_progress_is_not_reused():
    # Sell-side sweep at 100.0:
    # 3 reclaim trades at 100.02 before any penetration
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [100.02, 100.02, 100.02], [1_000, 2_000, 4_000])
    assert tracker.has_penetration is False
    assert tracker.reclaim_trade_count == 0
    assert tracker.reclaim_sufficient_time_ms is None

    # Late penetration at 5_000 (neutral trade 99.995 < 100.0)
    tracker.on_trade(trade(price=99.995, time_ms=5_000, seq=4))
    assert tracker.has_penetration is True
    assert tracker.reclaim_trade_count == 0
    assert tracker.reclaim_sufficient_time_ms is None
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.PENDING_SWEEP


def test_pre_penetration_acceptance_progress_is_not_reused():
    # Buy-side sweep at 100.0:
    # 3 reclaim trades at 99.98 (< 99.99 reclaim threshold) before any buy-side penetration (> 100.0)
    tracker = feed_prices(LiquiditySide.BUY_SIDE, [99.98, 99.98, 99.98], [1_000, 2_000, 4_000])
    assert tracker.has_penetration is False
    assert tracker.reclaim_trade_count == 0
    assert tracker.reclaim_sufficient_time_ms is None

    # Late penetration at 5_000 (neutral trade 100.005 > 100.0)
    tracker.on_trade(trade(price=100.005, time_ms=5_000, seq=4))
    assert tracker.has_penetration is True
    assert tracker.reclaim_trade_count == 0
    assert tracker.reclaim_sufficient_time_ms is None
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.PENDING_SWEEP


def test_sufficient_reclaim_survives_opposite_branch_before_watermark():
    # Sell-side sweep at 100.0:
    # 1. Penetration at 0
    # 2. Reclaim reaches sufficiency at 4_000
    # 3. Acceptance trade arrives at 5_000 before watermark reaches 4_000
    # 4. Watermark advances to 6_000 -> Earlier reclaim sufficiency must still win!
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 100.02, 100.02, 100.02, 99.98],
        [0, 1_000, 2_000, 4_000, 5_000],
    )
    assert tracker.reclaim_sufficient_time_ms == 4_000
    # Watermark behind 4_000
    assert tracker.decision_at(watermark_ms=3_500, expiry_ms=60_000).classification is EventClassification.PENDING_SWEEP
    # Watermark settles 4_000
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.FAILED_BREAKDOWN
    assert decision.market_resolution_time_ms == 4_000
    assert decision.reason_code == "RECLAIM_CONFIRMED"


def test_sufficient_acceptance_survives_opposite_branch_before_watermark():
    # Sell-side sweep at 100.0:
    # 1. Penetration at 0
    # 2. Acceptance reaches sufficiency at 4_000
    # 3. Reclaim trade arrives at 5_000 before watermark reaches 4_000
    # 4. Watermark advances to 6_000 -> Earlier acceptance sufficiency must still win!
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 99.98, 99.98, 99.98, 100.02],
        [0, 1_000, 2_000, 4_000, 5_000],
    )
    assert tracker.acceptance_sufficient_time_ms == 4_000
    # Watermark behind 4_000
    assert tracker.decision_at(watermark_ms=3_500, expiry_ms=60_000).classification is EventClassification.PENDING_SWEEP
    # Watermark settles 4_000
    decision = tracker.decision_at(watermark_ms=6_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.BEARISH_CONTINUATION
    assert decision.market_resolution_time_ms == 4_000
    assert decision.reason_code == "ACCEPTANCE_CONFIRMED"


def test_post_expiry_reclaim_confirmation_cannot_resolve():
    # Sell-side sweep with penetration at 0, but reclaim only reaches sufficiency at 63_000 (> expiry 60_000)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 100.02, 100.02, 100.02],
        [0, 60_000, 61_000, 63_000],
    )
    assert tracker.reclaim_sufficient_time_ms == 63_000
    decision = tracker.decision_at(watermark_ms=65_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.INDETERMINATE
    assert decision.reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    assert decision.market_resolution_time_ms == 60_000


def test_post_expiry_acceptance_confirmation_cannot_resolve():
    # Sell-side sweep with penetration at 0, but acceptance only reaches sufficiency at 61_000 (> expiry 60_000)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [99.98, 99.98, 99.98, 99.98],
        [0, 60_000, 61_000, 63_000],
    )
    assert tracker.acceptance_sufficient_time_ms == 61_000
    decision = tracker.decision_at(watermark_ms=65_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.INDETERMINATE
    assert decision.reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    assert decision.market_resolution_time_ms == 60_000


def test_post_expiry_penetration_remains_penetration_not_confirmed():
    # Sweep with no penetration until 60_001 (> expiry 60_000)
    tracker = feed_prices(
        LiquiditySide.SELL_SIDE,
        [100.02, 99.98],
        [1_000, 60_001],
    )
    assert tracker.has_penetration is True
    assert tracker.penetration_time_ms == 60_001
    decision = tracker.decision_at(watermark_ms=65_000, expiry_ms=60_000)
    assert decision.classification is EventClassification.INVALID
    assert decision.reason_code == "PENETRATION_NOT_CONFIRMED"
    assert decision.market_resolution_time_ms == 60_000
