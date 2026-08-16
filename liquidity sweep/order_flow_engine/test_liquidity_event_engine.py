from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import pytest

from liquidity_event.classifier import PriceOutcomeTracker
from liquidity_event.evidence import LiquidityEvidenceBuilder
from liquidity_event.event_store import LiquidityEventStore
from liquidity_event.identity_authority import (
    IdentityClaimOutcome,
    IdentityClaimResult,
    PersistedIdentity,
    SQLiteIdentityAuthority,
)
from liquidity_event.models import (
    AggressorSide,
    DepthObservation,
    EventClassification,
    EventState,
    LifecycleTransition,
    LiquidityEvent,
    LiquidityEventResult,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    RejectedSweepInput,
    SweepSource,
    TradeCoverage,
    TradeCoverageProvider,
    canonical_hash,
)
from liquidity_event.policy import LiquidityClassificationPolicy

# Engine import (will fail RED before engine.py is implemented)
from liquidity_event.engine import (
    CoverageSegment,
    EngineSnapshot,
    EngineTelemetry,
    LiquidityEventEngine,
    SymbolRuntime,
)


class StaticTradeCoverageProvider:
    def __init__(self, coverage_value: TradeCoverage):
        self._coverage = coverage_value

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> TradeCoverage:
        return self._coverage


class DynamicTradeCoverageProvider:
    def __init__(self, default_coverage: TradeCoverage):
        self._default = default_coverage
        self._overrides: list[tuple[str, int, int, TradeCoverage]] = []

    def add_override(self, symbol: str, start_ms: int, end_ms: int, cov: TradeCoverage) -> None:
        self._overrides.append((symbol, start_ms, end_ms, cov))

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> TradeCoverage:
        for s, st, et, c in self._overrides:
            if s == symbol and st == start_ms and et == end_ms:
                return c
        return self._default


def safe_cov() -> TradeCoverage:
    return TradeCoverage(
        feed_safe=True,
        known_gap=False,
        buffer_overflow=False,
        unresolved_sequence=False,
        interval_retained=True,
    )


def make_trade(
    price: float | Decimal = 100.0,
    time_ms: int = 0,
    seq: int = 1,
    quantity: float = 1.0,
    aggressor_side: AggressorSide = AggressorSide.BUY,
    symbol: str = "BTCUSDT",
) -> MarketTrade:
    return MarketTrade(
        symbol=symbol,
        price=price,
        quantity=quantity,
        aggressor_side=aggressor_side,
        exchange_time_ms=time_ms,
        sequence_id=seq,
    )


def make_obs(
    event_id: str = "sweep-1",
    symbol: str = "BTCUSDT",
    side: LiquiditySide = LiquiditySide.SELL_SIDE,
    level: float | Decimal = 100.0,
    event_time_ms: int = 0,
    detection_time_ms: int = 1_000,
    source_file_id: str = "sweeps.csv",
    source_level_id: str | None = None,
) -> LiquiditySweepObservation:
    row_hash = canonical_hash({"raw": event_id})
    obs_hash = canonical_hash({"event_id": event_id, "level": str(level)})
    return LiquiditySweepObservation(
        event_id=event_id,
        source_event_id=event_id,
        source_level_id=source_level_id,
        symbol=symbol,
        liquidity_side=side,
        swept_level=level,
        event_time_ms=event_time_ms,
        detection_time_ms=detection_time_ms,
        source_sweep_price=None,
        source_penetration_bps=None,
        source=SweepSource.SWEEPS_MONITOR_CSV,
        source_file_id=source_file_id,
        source_row_hash=row_hash,
        source_observation_hash=obs_hash,
        detector_version="1.0.0",
    )


def setup_engine(policy: LiquidityClassificationPolicy | None = None) -> tuple[LiquidityEventEngine, LiquidityEventStore, SQLiteIdentityAuthority]:
    pol = policy or LiquidityClassificationPolicy()
    store = LiquidityEventStore(pol)
    authority = SQLiteIdentityAuthority(":memory:")
    engine = LiquidityEventEngine(
        policy=pol,
        store=store,
        authority=authority,
        evidence_builder=LiquidityEvidenceBuilder(),
    )
    return engine, store, authority


# --- 1. Watermark and Timeout Tests ---

def test_advance_time_at_expiry_does_not_timeout_before_reorder_settlement():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Advance to 60,000 (watermark = 58,000 < expiry 60,000) -> Event must remain unresolved
    engine.advance_time("BTCUSDT", 60_000, safe_cov())
    assert store.active_count("BTCUSDT") == 1
    assert len(store.recent("BTCUSDT")) == 0

    authority.close()


def test_timeout_occurs_only_after_expiry_plus_reorder_tolerance():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Penetration at 500ms so it's not PENETRATION_NOT_CONFIRMED
    engine.on_trade(make_trade(price=99.98, time_ms=500, seq=1), safe_cov())

    # Advance to 61,999 (watermark = 59,999 < 60,000) -> still active
    engine.advance_time("BTCUSDT", 61_999, safe_cov())
    assert store.active_count("BTCUSDT") == 1

    # Advance to 62,000 (watermark = 60,000 == expiry) -> settled timeout!
    results = engine.advance_time("BTCUSDT", 62_000, safe_cov())
    assert len(results) == 1
    res = results[0]
    assert res.classification is EventClassification.INDETERMINATE
    assert res.reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    assert res.market_resolution_time_ms == 60_000
    assert store.active_count("BTCUSDT") == 0
    assert len(store.recent("BTCUSDT")) == 1

    authority.close()


# --- 2. Market Deduplication and Late Data Quarantine Tests ---

def test_duplicate_market_trade_does_not_double_confirm():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Feed penetration
    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())

    # Feed trade 2 at 1000
    t2 = make_trade(price=100.02, time_ms=1_000, seq=2)
    engine.on_trade(t2, safe_cov())
    # Duplicate feed of trade 2
    engine.on_trade(t2, safe_cov())

    # Feed trade 3 at 2000
    t3 = make_trade(price=100.02, time_ms=2_000, seq=3)
    engine.on_trade(t3, safe_cov())
    engine.on_trade(t3, safe_cov())

    # Only 2 unique reclaim trades have arrived so far. Need 3 unique trades to confirm.
    engine.advance_time("BTCUSDT", 4_000, safe_cov())
    assert store.active_count("BTCUSDT") == 1  # Not confirmed yet because trade count is only 2

    # Feed 3rd unique reclaim trade at 4000
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    assert store.active_count("BTCUSDT") == 0
    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.FAILED_BREAKDOWN
    assert recent[0].market_resolution_time_ms == 4_000

    authority.close()


def test_provisional_conflicting_trade_duplicate_resolves_canonically():
    engine, store, authority = setup_engine()
    # Feed two trades with same (symbol, seq) but different prices while provisional
    t1 = make_trade(price=100.50, time_ms=1_000, seq=10)
    t2 = make_trade(price=99.50, time_ms=1_000, seq=10)

    engine.on_trade(t1, safe_cov())
    engine.on_trade(t2, safe_cov())

    runtime = engine.symbol_runtime("BTCUSDT")
    # Buffer should contain exactly 1 entry for seq 10 (the canonical minimum)
    seq_trades = [t for t in runtime.trade_buffer if t.sequence_id == 10]
    assert len(seq_trades) == 1
    assert seq_trades[0].price == min(t1, t2, key=lambda x: x.canonical_key).price

    authority.close()


def test_record_at_previous_watermark_is_quarantined():
    engine, store, authority = setup_engine()
    # Advance watermark to 10,000 (max_seen = 12,000, watermark = 10,000)
    engine.on_trade(make_trade(price=100.0, time_ms=12_000, seq=1), safe_cov())
    assert engine.watermark("BTCUSDT") == 10_000

    # Arriving trade at exchange_time_ms <= 10,000 (e.g. 10,000) must be quarantined
    late_trade = make_trade(price=101.0, time_ms=10_000, seq=2)
    engine.on_trade(late_trade, safe_cov())

    assert engine.telemetry.late_data_count == 1
    runtime = engine.symbol_runtime("BTCUSDT")
    assert not any(t.sequence_id == 2 for t in runtime.trade_buffer)

    authority.close()


# --- 3. Quiet Market and Retention Horizon Tests ---

def test_quiet_interval_does_not_fail_oldest_trade_history_check():
    engine, store, authority = setup_engine()
    # First trade on market at 20,000
    engine.on_trade(make_trade(price=100.0, time_ms=20_000, seq=1), safe_cov())

    # Sweep at event_time_ms=0 with detection at 20,000 (within 180,000 retention window)
    obs = make_obs(event_time_ms=0, detection_time_ms=20_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    result = engine.on_sweep(obs, provider)

    # Must NOT fail with INSUFFICIENT_REPLAY_HISTORY because retention horizon is 20,000 - 180,000 <= 0
    # Event is admitted and active
    assert store.active_count("BTCUSDT") == 1
    assert result is not None

    authority.close()


def test_quiet_symbol_with_no_max_seen_can_have_valid_retained_history():
    engine, store, authority = setup_engine()
    # No trades seen yet on ETHUSDT (max_seen is None)
    obs = make_obs(event_id="eth-1", symbol="ETHUSDT", event_time_ms=1_000, detection_time_ms=2_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Admitted cleanly into volatile store
    assert store.active_count("ETHUSDT") == 1

    authority.close()


# --- 4. Interval Coverage and Reason Mapping Tests ---

def test_historical_gap_survives_later_safe_coverage():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # 1. Safe coverage 0..20k
    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=10_000, seq=2), safe_cov())

    # 2. Known gap at 20k..25k
    gap_cov = replace(safe_cov(), known_gap=True)
    engine.on_trade(make_trade(price=100.02, time_ms=25_000, seq=3), gap_cov)

    # 3. Later safe coverage 25k..62k
    engine.on_trade(make_trade(price=100.02, time_ms=30_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 62_000, safe_cov())

    # The historical gap at 20k..25k MUST invalidate the event despite later safe coverage
    assert store.active_count("BTCUSDT") == 0
    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"

    authority.close()


def test_gap_before_event_does_not_invalidate_event():
    engine, store, authority = setup_engine()
    # Gap at 0..5,000
    gap_cov = replace(safe_cov(), known_gap=True)
    engine.on_trade(make_trade(price=100.0, time_ms=5_000, seq=1), gap_cov)

    # Event starts at 10,000
    obs = make_obs(event_id="ev-10k", event_time_ms=10_000, detection_time_ms=11_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Safe trades during event window
    engine.on_trade(make_trade(price=99.98, time_ms=10_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=11_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=12_000, seq=4), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=14_000, seq=5), safe_cov())
    engine.advance_time("BTCUSDT", 16_000, safe_cov())

    # Event should resolve cleanly to FAILED_BREAKDOWN without being affected by the gap before 10,000
    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.FAILED_BREAKDOWN
    assert recent[0].market_resolution_time_ms == 14_000

    authority.close()


def test_coverage_ledger_is_symbol_local():
    engine, store, authority = setup_engine()
    # BTCUSDT has gap
    gap_cov = replace(safe_cov(), known_gap=True)
    engine.on_trade(make_trade(price=100.0, time_ms=5_000, seq=1, symbol="BTCUSDT"), gap_cov)

    # ETHUSDT is safe
    obs = make_obs(event_id="eth-clean", symbol="ETHUSDT", event_time_ms=5_000, detection_time_ms=6_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=5_000, seq=1, symbol="ETHUSDT"), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=6_000, seq=2, symbol="ETHUSDT"), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=7_000, seq=3, symbol="ETHUSDT"), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=9_000, seq=4, symbol="ETHUSDT"), safe_cov())
    engine.advance_time("ETHUSDT", 11_000, safe_cov())

    # ETHUSDT resolves normally without interference from BTCUSDT gap
    assert len(store.recent("ETHUSDT")) == 1
    assert store.recent("ETHUSDT")[0].classification is EventClassification.FAILED_BREAKDOWN

    authority.close()


def test_post_expiry_gap_does_not_invalidate_resolved_event():
    engine, store, authority = setup_engine()
    # Event 0..60,000 resolved at 4,000
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    assert len(store.recent("BTCUSDT")) == 1
    assert store.recent("BTCUSDT")[0].classification is EventClassification.FAILED_BREAKDOWN

    # Gap arrives at 75,000 (after expiry and resolution)
    gap_cov = replace(safe_cov(), known_gap=True)
    engine.on_trade(make_trade(price=100.0, time_ms=75_000, seq=10), gap_cov)

    # Finalized result remains immutable
    assert len(store.recent("BTCUSDT")) == 1
    assert store.recent("BTCUSDT")[0].classification is EventClassification.FAILED_BREAKDOWN

    authority.close()


def test_detection_clock_does_not_advance_across_unproven_coverage():
    engine, store, authority = setup_engine()
    provider = DynamicTradeCoverageProvider(safe_cov())
    # Event interval is safe
    provider.add_override("BTCUSDT", 0, 60_000, safe_cov())
    # Clock advancement interval 0..90k has unproven/unsafe coverage
    unsafe_cov = replace(safe_cov(), feed_safe=False)
    provider.add_override("BTCUSDT", 0, 90_000, unsafe_cov)

    obs = make_obs(event_time_ms=0, detection_time_ms=90_000)
    engine.on_sweep(obs, provider)

    # Clock should NOT advance to 90,000 because clock coverage was unsafe
    assert (engine.symbol_runtime("BTCUSDT").max_seen_exchange_time_ms or 0) < 90_000

    authority.close()


def test_feed_unsafe_maps_market_data_unsafe():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    unsafe_cov = replace(safe_cov(), feed_safe=False)
    engine.on_trade(make_trade(price=100.0, time_ms=2_000, seq=1), unsafe_cov)

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "MARKET_DATA_UNSAFE"

    authority.close()


def test_known_gap_maps_market_data_integrity_compromised():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    gap_cov = replace(safe_cov(), known_gap=True)
    engine.on_trade(make_trade(price=100.0, time_ms=2_000, seq=1), gap_cov)

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"

    authority.close()


def test_buffer_overflow_maps_market_data_integrity_compromised():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    overflow_cov = replace(safe_cov(), buffer_overflow=True)
    engine.on_trade(make_trade(price=100.0, time_ms=2_000, seq=1), overflow_cov)

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"

    authority.close()


def test_unresolved_sequence_maps_market_data_integrity_compromised():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    seq_cov = replace(safe_cov(), unresolved_sequence=True)
    engine.on_trade(make_trade(price=100.0, time_ms=2_000, seq=1), seq_cov)

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"

    authority.close()


# --- 5. Collision Orchestration Tests ---

def test_same_side_ambiguous_collision_invalidates_all_members():
    engine, store, authority = setup_engine()
    provider = StaticTradeCoverageProvider(safe_cov())

    # First event at 100.00
    obs1 = make_obs(event_id="sweep-col-1", level=100.00, event_time_ms=0, detection_time_ms=1_000)
    engine.on_sweep(obs1, provider)
    assert store.active_count("BTCUSDT") == 1

    # Second event on same side and close level within collision tolerance (e.g. 100.002, diff 0.2 bps <= 0.5 bps)
    obs2 = make_obs(event_id="sweep-col-2", level=100.002, event_time_ms=500, detection_time_ms=1_500)
    engine.on_sweep(obs2, provider)

    # Both colliding events must be finalized deterministically as INVALID / AMBIGUOUS_EVENT_COLLISION
    assert store.active_count("BTCUSDT") == 0
    recent = store.recent("BTCUSDT")
    assert len(recent) == 2
    assert {r.event_id for r in recent} == {"sweep-col-1", "sweep-col-2"}
    for r in recent:
        assert r.classification is EventClassification.INVALID
        assert r.reason_code == "AMBIGUOUS_EVENT_COLLISION"

    authority.close()


def test_opposite_side_overlap_remains_separate():
    engine, store, authority = setup_engine()
    provider = StaticTradeCoverageProvider(safe_cov())

    obs_sell = make_obs(event_id="sweep-sell", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=0)
    obs_buy = make_obs(event_id="sweep-buy", side=LiquiditySide.BUY_SIDE, level=100.0, event_time_ms=500)

    engine.on_sweep(obs_sell, provider)
    engine.on_sweep(obs_buy, provider)

    # Both active independently, no collision
    assert store.active_count("BTCUSDT") == 2
    authority.close()


# --- 6. Point-in-Time Evidence Finalization Tests ---

def test_final_evidence_uses_market_resolution_not_detection_time():
    engine, store, authority = setup_engine()
    # Feed trade history: penetration at 0, reclaim confirmation at 4,000
    engine.on_trade(make_trade(price=99.98, quantity=1.0, time_ms=0, seq=1, aggressor_side=AggressorSide.SELL), safe_cov())
    engine.on_trade(make_trade(price=100.02, quantity=2.0, time_ms=1_000, seq=2, aggressor_side=AggressorSide.BUY), safe_cov())
    engine.on_trade(make_trade(price=100.02, quantity=3.0, time_ms=2_000, seq=3, aggressor_side=AggressorSide.BUY), safe_cov())
    engine.on_trade(make_trade(price=100.02, quantity=4.0, time_ms=4_000, seq=4, aggressor_side=AggressorSide.BUY), safe_cov())

    # Trade at 8,000 after market resolution
    engine.on_trade(make_trade(price=100.50, quantity=100.0, time_ms=8_000, seq=5, aggressor_side=AggressorSide.BUY), safe_cov())

    # Late sweep detected at 9,000
    obs = make_obs(event_time_ms=0, detection_time_ms=9_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    res = engine.on_sweep(obs, provider)

    assert isinstance(res, LiquidityEventResult)
    assert res.market_resolution_time_ms == 4_000
    assert res.classification_time_ms == 9_000
    assert res.evidence.as_of_ms == 4_000

    # CVD must reflect volume strictly up to 4,000 (not including trade at 8,000)
    delta_val = res.evidence.values["post_sweep_cvd"].value
    # 0ms: sell 1.0 (-100), 1000ms: buy 2.0 (+200.04), 2000ms: buy 3.0 (+300.06), 4000ms: buy 4.0 (+400.08) -> sum ~ 800.18
    assert delta_val is not None
    assert float(delta_val) < 5000.0  # Trade at 8000 (100 * 100.50 = 10050) was NOT included!

    authority.close()


def test_post_resolution_trade_does_not_change_final_cvd():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    initial_result = store.recent("BTCUSDT")[0]

    # Additional trades arriving after finalization
    engine.on_trade(make_trade(price=105.0, quantity=50.0, time_ms=10_000, seq=10), safe_cov())

    # Result in store and authority is completely unchanged
    persisted = authority.lookup(obs.event_id)
    assert persisted is not None
    assert store.recent("BTCUSDT")[0] == initial_result

    authority.close()


# --- 7. Authority Failure and Callback Ordering Tests ---

def test_authority_transition_failure_emits_no_callback():
    engine, store, authority = setup_engine()
    transitions_emitted: list[LifecycleTransition] = []
    engine.on_transition = lambda t: transitions_emitted.append(t)

    # Close authority to force transition claim failure
    authority.close()

    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Because authority was closed/failed, no transition callback should be emitted
    assert len(transitions_emitted) == 0


def test_authority_result_failure_emits_no_result_callback():
    engine, store, authority = setup_engine()
    results_emitted: list[LiquidityEventResult] = []
    engine.on_result = lambda r: results_emitted.append(r)

    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Close authority before finalization
    authority.close()

    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    assert len(results_emitted) == 0


# --- 8. Restart Recovery Tests ---

def test_restart_recovery_does_not_call_normal_observation_claim():
    policy = LiquidityClassificationPolicy()
    authority = SQLiteIdentityAuthority(":memory:")
    obs = make_obs(event_id="rec-1", level="100.0")

    # 1. First run admits event into SQLite authority
    claim = authority.claim_observation(obs)
    assert claim.outcome is IdentityClaimOutcome.NEW

    # 2. Simulate crash and restart: create fresh store and engine with same SQLite db
    store2 = LiquidityEventStore(policy)
    engine2 = LiquidityEventEngine(
        policy=policy,
        store=store2,
        authority=authority,
        evidence_builder=LiquidityEvidenceBuilder(),
    )

    provider = StaticTradeCoverageProvider(safe_cov())
    # 3. Recover unresolved
    recovered = engine2.recover_claimed_unresolved(provider)

    # Event is recovered into volatile store
    assert store2.active_count("BTCUSDT") == 1
    authority.close()


def test_restart_recovery_restores_claimed_unresolved_into_volatile_store():
    policy = LiquidityClassificationPolicy()
    authority = SQLiteIdentityAuthority(":memory:")
    obs = make_obs(event_id="rec-2", level="100.0", event_time_ms=0)
    authority.claim_observation(obs)

    # Pre-populate some trades in new engine
    store = LiquidityEventStore(policy)
    engine = LiquidityEventEngine(
        policy=policy,
        store=store,
        authority=authority,
        evidence_builder=LiquidityEvidenceBuilder(),
    )

    # Feed history
    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())

    provider = StaticTradeCoverageProvider(safe_cov())
    recovered = engine.recover_claimed_unresolved(provider)

    # With history present, event settles immediately to FAILED_BREAKDOWN
    engine.advance_time("BTCUSDT", 6_000, safe_cov())
    assert len(store.recent("BTCUSDT")) == 1
    assert store.recent("BTCUSDT")[0].classification is EventClassification.FAILED_BREAKDOWN

    authority.close()


def test_restart_recovery_skips_persisted_transition_sequences():
    policy = LiquidityClassificationPolicy()
    authority = SQLiteIdentityAuthority(":memory:")
    obs = make_obs(event_id="rec-seq", level="100.0")
    authority.claim_observation(obs)

    # Persist transition sequence 0
    t0 = LifecycleTransition(
        event_id="rec-seq",
        previous_state=EventState.OBSERVED,
        next_state=EventState.PENETRATION_VALIDATING,
        transition_time_ms=500,
        transition_sequence=0,
        reason_code="START_VALIDATION",
    )
    authority.claim_transition(t0)

    store = LiquidityEventStore(policy)
    engine = LiquidityEventEngine(
        policy=policy,
        store=store,
        authority=authority,
        evidence_builder=LiquidityEvidenceBuilder(),
    )

    provider = StaticTradeCoverageProvider(safe_cov())
    engine.recover_claimed_unresolved(provider)

    tracker_ctx = engine.symbol_runtime("BTCUSDT").active_trackers.get("rec-seq")
    assert tracker_ctx is not None
    assert tracker_ctx.last_transition_sequence >= 0

    authority.close()


# --- 9. Replay, Late Callback & Eviction Tests ---

def test_late_callback_reconstructs_when_full_history_is_retained():
    engine, store, authority = setup_engine()
    # Feed trade history before sweep arrives
    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())

    # Late callback at detection 9,000
    obs = make_obs(event_time_ms=0, detection_time_ms=9_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    res = engine.on_sweep(obs, provider)

    assert isinstance(res, LiquidityEventResult)
    assert res.market_resolution_time_ms == 4_000
    assert res.classification_time_ms == 9_000
    assert res.classification is EventClassification.FAILED_BREAKDOWN

    authority.close()


def test_late_callback_is_invalid_when_history_was_evicted():
    policy = LiquidityClassificationPolicy(market_buffer_retention_ms=180_000)
    engine, store, authority = setup_engine(policy)

    # Market clock at 200,000 (retention horizon is 200,000 - 180,000 = 20,000)
    engine.on_trade(make_trade(price=100.0, time_ms=200_000, seq=1), safe_cov())

    # Late callback at event_time_ms = 0 (< 20,000 retention horizon)
    obs = make_obs(event_time_ms=0, detection_time_ms=200_000)
    # Coverage provider indicates interval not retained
    provider = StaticTradeCoverageProvider(replace(safe_cov(), interval_retained=False))
    res = engine.on_sweep(obs, provider)

    assert isinstance(res, LiquidityEventResult)
    assert res.classification is EventClassification.INVALID
    assert res.reason_code == "INSUFFICIENT_REPLAY_HISTORY"

    authority.close()


def test_premature_terminal_input_invalidates_unresolved_event():
    engine, store, authority = setup_engine()
    obs = make_obs(event_time_ms=0, detection_time_ms=1_000)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=500, seq=1), safe_cov())

    # Replay terminates prematurely at 30,000 with terminal_input=True
    results = engine.advance_time("BTCUSDT", 30_000, safe_cov(), terminal_input=True)
    assert len(results) == 1
    assert results[0].classification is EventClassification.INVALID
    assert results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"

    authority.close()


# --- 10. End-to-End Truth Table Outcome Tests ---

def test_e2e_failed_breakdown_outcome():
    engine, store, authority = setup_engine()
    obs = make_obs(side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=0)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.FAILED_BREAKDOWN
    assert recent[0].reason_code == "RECLAIM_CONFIRMED"
    authority.close()


def test_e2e_failed_breakout_outcome():
    engine, store, authority = setup_engine()
    obs = make_obs(side=LiquiditySide.BUY_SIDE, level=100.0, event_time_ms=0)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=100.02, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.FAILED_BREAKOUT
    assert recent[0].reason_code == "RECLAIM_CONFIRMED"
    authority.close()


def test_e2e_bearish_continuation_outcome():
    engine, store, authority = setup_engine()
    obs = make_obs(side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=0)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=99.98, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=99.98, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.BEARISH_CONTINUATION
    assert recent[0].reason_code == "ACCEPTANCE_CONFIRMED"
    authority.close()


def test_e2e_bullish_continuation_outcome():
    engine, store, authority = setup_engine()
    obs = make_obs(side=LiquiditySide.BUY_SIDE, level=100.0, event_time_ms=0)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    engine.on_trade(make_trade(price=100.02, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=1_000, seq=2), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=2_000, seq=3), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=4_000, seq=4), safe_cov())
    engine.advance_time("BTCUSDT", 6_000, safe_cov())

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.BULLISH_CONTINUATION
    assert recent[0].reason_code == "ACCEPTANCE_CONFIRMED"
    authority.close()


def test_e2e_unpenetrated_sweep_expires_to_invalid_penetration_not_confirmed():
    engine, store, authority = setup_engine()
    obs = make_obs(side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=0)
    provider = StaticTradeCoverageProvider(safe_cov())
    engine.on_sweep(obs, provider)

    # Trades at 100.02 (above level 100.0 -> no penetration for SELL_SIDE)
    engine.on_trade(make_trade(price=100.02, time_ms=0, seq=1), safe_cov())
    engine.on_trade(make_trade(price=100.02, time_ms=10_000, seq=2), safe_cov())

    # Advance beyond expiry + reorder tolerance
    engine.advance_time("BTCUSDT", 62_000, safe_cov())

    recent = store.recent("BTCUSDT")
    assert len(recent) == 1
    assert recent[0].classification is EventClassification.INVALID
    assert recent[0].reason_code == "PENETRATION_NOT_CONFIRMED"
    assert recent[0].market_resolution_time_ms == 60_000
    authority.close()
