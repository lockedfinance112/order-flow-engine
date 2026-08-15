from dataclasses import replace
from decimal import Decimal
from typing import Any, Mapping
import pytest

from liquidity_event.models import (
    AggressorSide,
    ConfidenceType,
    EventClassification,
    EventState,
    EvidenceAvailability,
    EvidenceValue,
    LiquidityEvent,
    LiquidityEvidence,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    SweepSource,
)
from liquidity_event.evidence import (
    EventFlowAccumulator,
    EventFlowSnapshot,
    LegacyLiquidityContextAdapter,
    LiquidityEvidenceBuilder,
)


def trade(
    exchange_time_ms: int,
    side: str = "BUY",
    notional: float = 100.0,
    sequence_id: int | str = 1,
    symbol: str = "BTCUSDT",
    price: float = 100.0,
) -> MarketTrade:
    aggressor = AggressorSide.BUY if side.upper() == "BUY" else AggressorSide.SELL
    quantity = notional / price
    return MarketTrade(
        symbol=symbol,
        price=price,
        quantity=quantity,
        aggressor_side=aggressor,
        exchange_time_ms=exchange_time_ms,
        sequence_id=sequence_id,
    )


def accumulate(trades: list[MarketTrade], event_time_ms: int = 1_000, canonical_sort: bool = False) -> EventFlowSnapshot:
    flow = EventFlowAccumulator(event_time_ms=event_time_ms)
    feed_trades = sorted(trades, key=lambda t: t.canonical_key) if canonical_sort else trades
    for t in feed_trades:
        flow.on_trade(t)
    return flow.snapshot()


def sweep_observation(
    event_id: str = "evt-1",
    symbol: str = "BTCUSDT",
    side: LiquiditySide = LiquiditySide.SELL_SIDE,
    level: float = 100.0,
    event_time_ms: int = 1_000,
    detection_time_ms: int = 2_000,
) -> LiquiditySweepObservation:
    return LiquiditySweepObservation(
        event_id=event_id,
        source_event_id="src-1",
        source_level_id="lvl-1",
        symbol=symbol,
        liquidity_side=side,
        swept_level=level,
        event_time_ms=event_time_ms,
        detection_time_ms=detection_time_ms,
        source_sweep_price=None,
        source_penetration_bps=None,
        source=SweepSource.SWEEPS_MONITOR_CSV,
        source_file_id="SWEEPS_MONITOR_CSV",
        source_row_hash=None,
        source_observation_hash="a" * 64,
        detector_version="1.0.0",
    )


# --- 1. EventFlowAccumulator Tests ---

def test_event_flow_accumulates_only_point_in_time_trades():
    flow = EventFlowAccumulator(event_time_ms=1_000, end_time_ms=61_000)
    flow.on_trade(trade(900, "BUY", 100, sequence_id=1))   # before event_time_ms -> ignored
    flow.on_trade(trade(1_100, "SELL", 40, sequence_id=2))
    flow.on_trade(trade(1_200, "BUY", 70, sequence_id=3))
    flow.on_trade(trade(62_000, "BUY", 500, sequence_id=4)) # after end_time_ms -> ignored

    snap = flow.snapshot(1_200)
    assert snap.buy_volume_usdt == 70.0
    assert snap.sell_volume_usdt == 40.0
    assert snap.signed_delta_usdt == 30.0
    assert snap.cumulative_delta_usdt == 30.0
    assert snap.post_sweep_cvd_usdt == 30.0
    assert snap.cvd_min_usdt == -40.0
    assert snap.cvd_max_usdt == 30.0


def test_live_and_replay_trade_order_produce_identical_flow_snapshot():
    chronological = [
        trade(1_000, "SELL", 50, sequence_id=1),
        trade(1_100, "BUY", 80, sequence_id=2),
        trade(1_200, "SELL", 20, sequence_id=3),
    ]
    snap_live = accumulate(chronological)
    snap_replay = accumulate(list(reversed(chronological)), canonical_sort=True)

    assert snap_live == snap_replay
    assert snap_live.buy_volume_usdt == 80.0
    assert snap_live.sell_volume_usdt == 70.0
    assert snap_live.signed_delta_usdt == 10.0
    assert snap_live.cvd_min_usdt == -50.0
    assert snap_live.cvd_max_usdt == 30.0


def test_event_flow_deduplicates_by_symbol_and_sequence_id():
    flow = EventFlowAccumulator(event_time_ms=1_000)
    t1 = trade(1_000, "BUY", 100, sequence_id=100)
    t1_dup = trade(1_000, "BUY", 100, sequence_id=100)
    t2 = trade(1_100, "SELL", 50, sequence_id=101)

    flow.on_trade(t1)
    flow.on_trade(t1_dup)
    flow.on_trade(t2)

    snap = flow.snapshot()
    assert snap.buy_volume_usdt == 100.0
    assert snap.sell_volume_usdt == 50.0
    assert len(flow.trades()) == 2


def test_event_flow_starts_cvd_at_zero():
    flow = EventFlowAccumulator(event_time_ms=1_000)
    snap = flow.snapshot()
    assert snap.buy_volume_usdt == 0.0
    assert snap.sell_volume_usdt == 0.0
    assert snap.signed_delta_usdt == 0.0
    assert snap.post_sweep_cvd_usdt == 0.0
    assert snap.cvd_min_usdt == 0.0
    assert snap.cvd_max_usdt == 0.0
    assert snap.cvd_recovery_usdt == 0.0


# --- 2. LegacyLiquidityContextAdapter Tests ---

def test_optional_context_is_typed_and_never_reimplemented():
    adapter = LegacyLiquidityContextAdapter()
    assert adapter.absorption(None, as_of_ms=2_000).availability is EvidenceAvailability.UNAVAILABLE
    assert adapter.replenishment(None, as_of_ms=2_000).availability is EvidenceAvailability.UNAVAILABLE
    assert adapter.stacking_pulling(None, as_of_ms=2_000).availability is EvidenceAvailability.UNAVAILABLE

    bullish_abs = adapter.absorption({"event_type": "BULLISH_ABSORPTION"}, as_of_ms=2_000)
    assert bullish_abs.availability is EvidenceAvailability.AVAILABLE
    assert bullish_abs.value == 1.0

    bearish_abs = adapter.absorption({"event_type": "BEARISH_ABSORPTION"}, as_of_ms=2_000)
    assert bearish_abs.availability is EvidenceAvailability.AVAILABLE
    assert bearish_abs.value == -1.0


def test_context_adapter_handles_unsafe_flag():
    adapter = LegacyLiquidityContextAdapter()
    unsafe_obs = adapter.absorption({"event_type": "BULLISH_ABSORPTION", "unsafe": True}, as_of_ms=2_000)
    assert unsafe_obs.availability is EvidenceAvailability.UNSAFE
    assert unsafe_obs.value is None


# --- 3. LiquidityEvidenceBuilder Tests ---

def test_evidence_builder_failed_breakdown_all_aligned():
    # Failed breakdown is BULLISH outcome (support sweep reclaimed)
    obs = sweep_observation(side=LiquiditySide.SELL_SIDE)
    event = LiquidityEvent(observation=obs)
    builder = LiquidityEvidenceBuilder()

    flow_snap = EventFlowSnapshot(
        buy_volume_usdt=100.0,
        sell_volume_usdt=20.0,
        signed_delta_usdt=80.0,        # aligned (+delta)
        cumulative_delta_usdt=80.0,
        post_sweep_cvd_usdt=80.0,      # aligned (+cvd)
        cvd_min_usdt=-10.0,
        cvd_max_usdt=80.0,
        cvd_recovery_usdt=90.0,
        as_of_ms=5_000,
    )

    evidence = builder.build(
        event=event,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=5_000,
        flow_snapshot=flow_snap,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000),      # aligned (+1.0)
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.AVAILABLE, 0.5, 5_000), # aligned (+0.5)
    )

    # 4 support groups aligned (+0.10 * 4 = +0.40) + base 0.60 = 1.00
    assert evidence.confidence == 1.0
    assert evidence.evidence_strength == 1.0
    assert evidence.contradiction_strength == 0.0
    assert evidence.context_coverage == 1.0
    assert len(evidence.reasons) >= 4
    assert evidence.contradictions == ()


def test_evidence_builder_failed_breakdown_with_contradictions():
    # Failed breakdown (bullish outcome), but negative delta, negative CVD, negative absorption
    obs = sweep_observation(side=LiquiditySide.SELL_SIDE)
    event = LiquidityEvent(observation=obs)
    builder = LiquidityEvidenceBuilder()

    flow_snap = EventFlowSnapshot(
        buy_volume_usdt=20.0,
        sell_volume_usdt=100.0,
        signed_delta_usdt=-80.0,       # opposed (-delta)
        cumulative_delta_usdt=-80.0,
        post_sweep_cvd_usdt=-80.0,     # opposed (-cvd)
        cvd_min_usdt=-80.0,
        cvd_max_usdt=0.0,
        cvd_recovery_usdt=0.0,
        as_of_ms=5_000,
    )

    evidence = builder.build(
        event=event,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=5_000,
        flow_snapshot=flow_snap,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, 5_000),     # opposed (-1.0)
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None),
    )

    # Base 0.60, 0 support groups, 3 contradiction groups (-0.10 * 3 = -0.30)
    # confidence = 0.60 - 0.30 = 0.30
    assert round(evidence.confidence, 4) == 0.30
    assert round(evidence.evidence_strength, 4) == 0.60
    assert round(evidence.contradiction_strength, 4) == 0.30
    assert round(evidence.context_coverage, 4) == 0.75  # 3 of 4 groups available
    assert len(evidence.contradictions) == 3


def test_evidence_builder_indeterminate_and_invalid_have_zero_confidence():
    obs = sweep_observation()
    event = LiquidityEvent(observation=obs)
    builder = LiquidityEvidenceBuilder()

    indet = builder.build(
        event=event,
        outcome_direction=EventClassification.INDETERMINATE,
        as_of_ms=60_000,
    )
    assert indet.confidence == 0.0
    assert indet.evidence_strength == 0.0

    inv = builder.build(
        event=event,
        outcome_direction=EventClassification.INVALID,
        as_of_ms=60_000,
    )
    assert inv.confidence == 0.0
    assert inv.evidence_strength == 0.0


def test_cvd_recovery_calculation_positive_and_negative_paths():
    # Path 1: CVD dips negative, then recovers to positive
    # trades: -50, +120 -> final CVD = +70, min = -50, max = +70, recovery = 70 - (-50) = 120
    flow1 = EventFlowAccumulator(event_time_ms=1_000)
    flow1.on_trade(trade(1_000, "SELL", 50, sequence_id=1))
    flow1.on_trade(trade(1_100, "BUY", 120, sequence_id=2))
    snap1 = flow1.snapshot()
    assert snap1.post_sweep_cvd_usdt == 70.0
    assert snap1.cvd_min_usdt == -50.0
    assert snap1.cvd_max_usdt == 70.0
    assert snap1.cvd_recovery_usdt == 120.0

    # Path 2: CVD peaks positive, then rejects to negative
    # trades: +50, -120 -> final CVD = -70, min = -70, max = +50, recovery = 50 - (-70) = 120
    flow2 = EventFlowAccumulator(event_time_ms=1_000)
    flow2.on_trade(trade(1_000, "BUY", 50, sequence_id=1))
    flow2.on_trade(trade(1_100, "SELL", 120, sequence_id=2))
    snap2 = flow2.snapshot()
    assert snap2.post_sweep_cvd_usdt == -70.0
    assert snap2.cvd_min_usdt == -70.0
    assert snap2.cvd_max_usdt == 50.0
    assert snap2.cvd_recovery_usdt == 120.0


def test_evidence_builder_all_outcomes_and_directions():
    builder = LiquidityEvidenceBuilder()

    # 1. FAILED_BREAKOUT (Bearish outcome: buy-side sweep reversed down)
    obs_buy = sweep_observation(side=LiquiditySide.BUY_SIDE)
    evt_breakout_fail = LiquidityEvent(observation=obs_buy)
    flow_bearish = EventFlowSnapshot(
        buy_volume_usdt=10.0,
        sell_volume_usdt=90.0,
        signed_delta_usdt=-80.0,       # aligned (negative for bearish)
        cumulative_delta_usdt=-80.0,
        post_sweep_cvd_usdt=-80.0,     # aligned
        cvd_min_usdt=-80.0,
        cvd_max_usdt=0.0,
        cvd_recovery_usdt=0.0,
        as_of_ms=5_000,
    )
    ev_breakout_fail = builder.build(
        event=evt_breakout_fail,
        outcome_direction=EventClassification.FAILED_BREAKOUT,
        as_of_ms=5_000,
        flow_snapshot=flow_bearish,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, 5_000),      # aligned (negative for bearish)
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.AVAILABLE, -0.5, 5_000), # aligned
    )
    assert ev_breakout_fail.confidence == 1.0
    assert ev_breakout_fail.evidence_strength == 1.0
    assert ev_breakout_fail.contradiction_strength == 0.0
    assert ev_breakout_fail.contradictions == ()

    # 2. BEARISH_CONTINUATION (Bearish outcome: sell-side sweep broken through)
    obs_sell = sweep_observation(side=LiquiditySide.SELL_SIDE)
    evt_bear_cont = LiquidityEvent(observation=obs_sell)
    ev_bear_cont = builder.build(
        event=evt_bear_cont,
        outcome_direction=EventClassification.BEARISH_CONTINUATION,
        as_of_ms=5_000,
        flow_snapshot=flow_bearish,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, 5_000),
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.AVAILABLE, -0.5, 5_000),
    )
    assert ev_bear_cont.confidence == 1.0

    # 3. BULLISH_CONTINUATION (Bullish outcome: buy-side sweep broken through)
    evt_bull_cont = LiquidityEvent(observation=obs_buy)
    flow_bullish = EventFlowSnapshot(
        buy_volume_usdt=90.0,
        sell_volume_usdt=10.0,
        signed_delta_usdt=80.0,        # aligned (positive for bullish)
        cumulative_delta_usdt=80.0,
        post_sweep_cvd_usdt=80.0,      # aligned
        cvd_min_usdt=0.0,
        cvd_max_usdt=80.0,
        cvd_recovery_usdt=80.0,
        as_of_ms=5_000,
    )
    ev_bull_cont = builder.build(
        event=evt_bull_cont,
        outcome_direction=EventClassification.BULLISH_CONTINUATION,
        as_of_ms=5_000,
        flow_snapshot=flow_bullish,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000),
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.AVAILABLE, 0.5, 5_000),
    )
    assert ev_bull_cont.confidence == 1.0


def test_neutral_and_unavailable_values_do_not_contradict():
    builder = LiquidityEvidenceBuilder()
    evt = LiquidityEvent(observation=sweep_observation())
    
    # Delta is 0.0 (neutral), CVD is 0.0 (neutral), absorption is UNAVAILABLE, depth is UNSAFE
    flow_neutral = EventFlowSnapshot(
        buy_volume_usdt=50.0,
        sell_volume_usdt=50.0,
        signed_delta_usdt=0.0,
        cumulative_delta_usdt=0.0,
        post_sweep_cvd_usdt=0.0,
        cvd_min_usdt=-10.0,
        cvd_max_usdt=10.0,
        cvd_recovery_usdt=20.0,
        as_of_ms=5_000,
    )
    ev = builder.build(
        event=evt,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=5_000,
        flow_snapshot=flow_neutral,
        absorption=EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None),
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.UNSAFE, None, 5_000),
    )
    # Neutral/unavailable do NOT subtract from base 0.60
    assert ev.confidence == 0.60
    assert ev.evidence_strength == 0.60
    assert ev.contradiction_strength == 0.0
    assert ev.contradictions == ()
    assert ev.context_coverage == 0.50  # 2 groups (delta & cvd) available out of 4


def test_optional_context_reasons_do_not_alter_confidence_score():
    builder = LiquidityEvidenceBuilder()
    evt = LiquidityEvent(observation=sweep_observation())
    flow_snap = EventFlowSnapshot(
        buy_volume_usdt=100.0,
        sell_volume_usdt=20.0,
        signed_delta_usdt=80.0,
        cumulative_delta_usdt=80.0,
        post_sweep_cvd_usdt=80.0,
        cvd_min_usdt=0.0,
        cvd_max_usdt=80.0,
        cvd_recovery_usdt=80.0,
        as_of_ms=5_000,
    )
    
    # Add replenishment, stacking/pulling, displacement, volume expansion
    ev = builder.build(
        event=evt,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=5_000,
        flow_snapshot=flow_snap,
        absorption=EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000),
        depth_weighted_imbalance=EvidenceValue(EvidenceAvailability.AVAILABLE, 0.5, 5_000),
        replenishment=EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000),
        stacking_pulling=EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000),
        displacement_bps=12.5,
        volume_expansion_ratio=2.4,
    )

    # Max confidence remains 1.00 (base 0.60 + 4 * 0.10)
    assert ev.confidence == 1.0
    assert "REPLENISHMENT_OBSERVED" in ev.reasons
    assert "STACKING_PULLING_ALIGNED" in ev.reasons
    assert "DISPLACEMENT_RECORDED" in ev.reasons
    assert "VOLUME_EXPANSION_RECORDED" in ev.reasons
    assert ev.values["displacement_bps"].value == 12.5
    assert ev.values["volume_expansion_ratio"].value == 2.4


def test_conflicting_duplicate_trade_sequence_resolves_canonically_independent_of_feed_order():
    # Two records with the same symbol and sequence_id, but conflicting contents
    trade_a = trade(1_000, "BUY", 100, sequence_id=500, price=100.0)
    trade_b = trade(1_000, "SELL", 100, sequence_id=500, price=101.0)

    snap_ab = accumulate([trade_a, trade_b])
    snap_ba = accumulate([trade_b, trade_a])

    assert snap_ab == snap_ba
    # Verify the winner is canonically deterministic (min by canonical_key)
    canonical_winner = min(trade_a, trade_b, key=lambda t: t.canonical_key)
    expected_buy = 100.0 if canonical_winner.aggressor_side is AggressorSide.BUY else 0.0
    assert snap_ab.buy_volume_usdt == expected_buy


def test_incompatible_context_types_and_strings_become_unavailable():
    adapter = LegacyLiquidityContextAdapter()
    incompatible_inputs = [
        {"value": "not-a-number"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"score": "invalid_string"},
        "not_a_mapping",
        12345,
        None,
    ]

    for raw in incompatible_inputs:
        abs_val = adapter.absorption(raw, as_of_ms=2_000)
        rep_val = adapter.replenishment(raw, as_of_ms=2_000)
        sp_val = adapter.stacking_pulling(raw, as_of_ms=2_000)

        assert abs_val.availability is EvidenceAvailability.UNAVAILABLE
        assert abs_val.value is None
        assert abs_val.as_of_ms is None

        assert rep_val.availability is EvidenceAvailability.UNAVAILABLE
        assert rep_val.value is None
        assert rep_val.as_of_ms is None

        assert sp_val.availability is EvidenceAvailability.UNAVAILABLE
        assert sp_val.value is None
        assert sp_val.as_of_ms is None


def test_optional_context_availability_preserved_on_all_outcomes_including_indeterminate_and_invalid():
    builder = LiquidityEvidenceBuilder()
    evt = LiquidityEvent(observation=sweep_observation())

    rep_unsafe = EvidenceValue(EvidenceAvailability.UNSAFE, None, 5_000)
    sp_available = EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 5_000)
    abs_available = EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, 5_000)
    depth_unavail = EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)

    for outcome in (EventClassification.INDETERMINATE, EventClassification.INVALID):
        ev = builder.build(
            event=evt,
            outcome_direction=outcome,
            as_of_ms=5_000,
            absorption=abs_available,
            depth_weighted_imbalance=depth_unavail,
            replenishment=rep_unsafe,
            stacking_pulling=sp_available,
            displacement_bps=15.0,
            volume_expansion_ratio=1.8,
        )

        # Confidence and evidence strength strictly 0.0 for unconfirmed outcomes
        assert ev.confidence == 0.0
        assert ev.evidence_strength == 0.0
        assert ev.contradiction_strength == 0.0
        assert "OUTCOME_UNCONFIRMED" in ev.reasons

        # Availability and raw values are preserved in values dictionary
        assert ev.values["wall_replenishment"] == rep_unsafe
        assert ev.values["stacking_pulling"] == sp_available
        assert ev.values["bid_ask_absorption"] == abs_available
        assert ev.values["depth_weighted_imbalance"] == depth_unavail
        assert ev.values["displacement_bps"].value == 15.0
        assert ev.values["volume_expansion_ratio"].value == 1.8


def test_no_flow_metrics_imported_in_liquidity_event():
    import sys
    import liquidity_event.evidence
    import liquidity_event

    assert "FlowMetrics" not in dir(liquidity_event.evidence)
    with open(liquidity_event.evidence.__file__, "r", encoding="utf-8") as f:
        content = f.read()
    assert "FlowMetrics" not in content
