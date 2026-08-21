from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import importlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, get_args, get_type_hints
from unittest.mock import MagicMock, patch

import pytest

import config
from config import (
    EXECUTION_DISABLED,
    LIQUIDITY_EVENT_ENGINE_ENABLED,
    LIQUIDITY_EVENT_ENFORCEMENT_ENABLED,
    REGIME_ENFORCEMENT_ENABLED,
)
from flow_metrics import FlowMetrics
from liquidity_event import (
    AggressorSide,
    ArtifactIntegrity,
    ConfidenceType,
    DepthObservation,
    EvidenceAvailability,
    EvidenceValue,
    EventClassification,
    EventFlowAccumulator,
    EventFlowSnapshot,
    EventState,
    IdentityClaimOutcome,
    LifecycleTransition,
    LegacyLiquidityContextAdapter,
    LiquidityClassificationPolicy,
    LiquidityEvent,
    LiquidityEventEngine,
    LiquidityEventResult,
    LiquidityEventStore,
    LiquidityEvidence,
    LiquidityEvidenceBuilder,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    PersistedIdentity,
    PriceOutcomeTracker,
    RejectedSweepInput,
    SQLiteIdentityAuthority,
    SweepSource,
    SweepsMonitorAdapter,
    TradeCoverage,
    canonical_hash,
)
from liquidity_event.classifier import CandidateProgress
import liquidity_event.recorder as rec_mod
from liquidity_event.recorder import (
    CanonicalRecord,
    EVENTS_FILENAME,
    LiquidityEventRecorder,
    REJECTED_FILENAME,
    TRANSITIONS_FILENAME,
)
from liquidity_event.replay import (
    LiquidityReplayRunner,
    ReplayInput,
    ReplayTradeCoverageProvider,
    main as replay_main,
)
from main import LiveTradeCoverageProvider, OrderFlowEngine
from scoring import OrderFlowScorer


# Session-level isolation for OrderFlowEngine to prevent default artifact leaks across test suites
_orig_engine_init = OrderFlowEngine.__init__
_session_artifact_dir = tempfile.mkdtemp(prefix="orderflow_session_artifacts_")
_default_artifact_dir = os.path.join(config.BASE_DIR, "liquidity_event_artifacts")

def _isolated_engine_init(self, *args, **kwargs):
    curr_dir = str(config.LIQUIDITY_EVENT_OUTPUT_DIR)
    if curr_dir in (_default_artifact_dir, "liquidity_event_artifacts", os.path.abspath("liquidity_event_artifacts")) or (
        curr_dir.endswith("liquidity_event_artifacts")
        and not "pytest" in curr_dir
        and not "orderflow_session_artifacts" in curr_dir
        and not "Temp" in curr_dir
        and not "temp" in curr_dir
    ):
        with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", _session_artifact_dir):
            _orig_engine_init(self, *args, **kwargs)
    else:
        _orig_engine_init(self, *args, **kwargs)

OrderFlowEngine.__init__ = _isolated_engine_init


# -----------------------------------------------------------------------------
# Test Fixtures & Helpers
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def auto_cleanup_test_resources(tmp_path):
    engines = []
    orig_init = OrderFlowEngine.__init__

    def tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        engines.append(self)

    OrderFlowEngine.__init__ = tracking_init

    with patch(
        "config.LIQUIDITY_EVENT_OUTPUT_DIR",
        str(tmp_path / "liquidity_event_artifacts"),
    ):
        try:
            yield
        finally:
            OrderFlowEngine.__init__ = orig_init

            for eng in engines:
                if (
                    hasattr(eng, "liquidity_identity_authority")
                    and eng.liquidity_identity_authority
                ):
                    try:
                        eng.liquidity_identity_authority.close()
                    except Exception:
                        pass

                if (
                    hasattr(eng, "regime_feed")
                    and eng.regime_feed
                    and hasattr(eng.regime_feed, "bar_store")
                ):
                    try:
                        eng.regime_feed.bar_store.close()
                    except Exception:
                        pass

                if (
                    hasattr(eng, "regime_engine")
                    and eng.regime_engine
                    and hasattr(eng.regime_engine, "bar_store")
                ):
                    try:
                        eng.regime_engine.bar_store.close()
                    except Exception:
                        pass


@pytest.fixture(scope="session", autouse=True)
def assert_no_default_liquidity_artifact_leak():
    artifact_dir = Path("liquidity_event_artifacts")

    assert not artifact_dir.exists(), (
        "pre-existing liquidity_event_artifacts directory must be "
        "removed before acceptance certification"
    )

    yield

    assert not artifact_dir.exists(), (
        "acceptance suite leaked liquidity_event_artifacts into worktree"
    )



def make_safe_coverage() -> TradeCoverage:
    return TradeCoverage(
        feed_safe=True,
        known_gap=False,
        buffer_overflow=False,
        unresolved_sequence=False,
        interval_retained=True,
    )


class StaticCoverageProvider:
    def __init__(self, coverage: TradeCoverage):
        self._coverage = coverage

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> TradeCoverage:
        return self._coverage


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
        price=Decimal(str(price)) if isinstance(price, (int, float)) else price,
        quantity=Decimal(str(quantity)) if isinstance(quantity, (int, float)) else quantity,
        aggressor_side=aggressor_side,
        exchange_time_ms=time_ms,
        sequence_id=seq,
    )


def make_obs(
    event_id: str = "sweep-1",
    symbol: str = "BTCUSDT",
    side: LiquiditySide = LiquiditySide.SELL_SIDE,
    level: float | Decimal = 100.0,
    event_time_ms: int = 10_000,
    detection_time_ms: int = 10_000,
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
        swept_level=Decimal(str(level)) if isinstance(level, (int, float)) else level,
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


def make_sweep_callback(
    symbol: str = "BTCUSDT",
    event_time_ms: int = 10_000,
    detection_time_ms: int = 10_000,
    side: str = "SELL_SIDE",
    level: float = 100.0,
    sweep_id: str = "swp-1",
) -> dict[str, Any]:
    legacy_type = "BULLISH" if side == "SELL_SIDE" else "BEARISH"
    dt = datetime.fromtimestamp(event_time_ms / 1000.0, tz=timezone.utc)
    ts_str = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{event_time_ms % 1000:03d}Z" if event_time_ms % 1000 else dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "timestamp": ts_str,
        "symbol": symbol,
        "type": legacy_type,
        "sweep_level": level,
        "sweep_id": sweep_id,
        "detection_time_ms": detection_time_ms,
        "source_file_id": "SWEEPS_MONITOR_CSV",
        "source_row_hash": "a" * 64,
        "source_sweep_price": level,
        "source_penetration_bps": 5.0,
    }


def make_test_engine(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = LiquidityClassificationPolicy()
    db_path = str(tmp_path / "test_identity.sqlite3")
    authority = SQLiteIdentityAuthority(db_path)
    store = LiquidityEventStore(policy)
    builder = LiquidityEvidenceBuilder()
    engine = LiquidityEventEngine(
        policy=policy,
        store=store,
        authority=authority,
        evidence_builder=builder,
    )
    return engine, store, authority, policy


def drive_live_engine(
    inputs: Iterable[ReplayInput],
    policy: LiquidityClassificationPolicy | None = None,
) -> tuple[tuple[LiquidityEventResult, ...], tuple[LifecycleTransition, ...]]:
    sorted_inputs = sorted(inputs, key=lambda x: x.processing_key)
    pol = policy or LiquidityClassificationPolicy()

    symbol_bounds: dict[str, tuple[int, int]] = {}
    for inp in sorted_inputs:
        if inp.source_type == "TRADE" and inp.trade is not None:
            sym = inp.trade.symbol
            t_time = inp.trade.exchange_time_ms
            if sym not in symbol_bounds:
                symbol_bounds[sym] = (t_time, t_time)
            else:
                first_t, last_t = symbol_bounds[sym]
                symbol_bounds[sym] = (min(first_t, t_time), max(last_t, t_time))

    cov_provider = ReplayTradeCoverageProvider(symbol_bounds)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "live_authority.sqlite3"
        auth = SQLiteIdentityAuthority(db_path)
        try:
            store = LiquidityEventStore(pol)
            adapter = SweepsMonitorAdapter(pol)
            results: list[LiquidityEventResult] = []
            transitions: list[LifecycleTransition] = []

            engine = LiquidityEventEngine(
                policy=pol,
                store=store,
                authority=auth,
                on_transition=lambda t: transitions.append(t),
                on_result=lambda r: results.append(r),
            )

            safe_trade_cov = TradeCoverage(
                feed_safe=True,
                known_gap=False,
                buffer_overflow=False,
                unresolved_sequence=False,
                interval_retained=True,
            )

            for inp in sorted_inputs:
                if inp.source_type == "TRADE" and inp.trade is not None:
                    engine.on_trade(inp.trade, safe_trade_cov)
                elif inp.source_type == "DEPTH" and inp.depth is not None:
                    engine.on_depth(inp.depth)
                elif inp.source_type == "SWEEP":
                    obs = None
                    if inp.sweep is not None:
                        obs = inp.sweep
                    elif inp.sweep_callback is not None:
                        adapted = adapter.adapt(inp.sweep_callback, int(inp.sweep_callback["detection_time_ms"]))
                        obs = adapted.observation
                    if obs is not None:
                        engine.on_sweep(obs, cov_provider)

            for symbol in list(engine._runtimes.keys()):
                runtime = engine.symbol_runtime(symbol)
                if runtime.active_trackers:
                    proven_horizon = runtime.proven_trade_coverage_time_ms
                    if proven_horizon is not None:
                        engine.advance_time(
                            symbol=symbol,
                            as_of_exchange_time_ms=proven_horizon,
                            coverage=TradeCoverage.safe_zero_activity(),
                            terminal_input=True,
                        )

            return tuple(results), tuple(transitions)
        finally:
            auth.close()


# -----------------------------------------------------------------------------
# GATE 01: SELL-side liquidity taken + sustained reclaim => FAILED_BREAKDOWN
# -----------------------------------------------------------------------------
def test_gate_01_failed_breakdown(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-01", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Penetration trade below 100.00
        engine.on_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        # Reclaim trades above 100.00 (>= 100.01) for hold duration
        engine.on_trade(make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Settle watermark: 20_000 - 2_000 = 18_000 >= 14_000
        engine.on_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.FAILED_BREAKDOWN
        assert results[0].reason_code == "RECLAIM_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 02: SELL-side liquidity taken + sustained acceptance below => BEARISH_CONTINUATION
# -----------------------------------------------------------------------------
def test_gate_02_bearish_continuation(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-02", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Penetration trade below 100.00
        engine.on_trade(make_trade(price=99.80, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        # Acceptance trades staying below threshold (<= 99.99) for hold duration
        engine.on_trade(make_trade(price=99.70, time_ms=11_000, seq=2, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=99.70, time_ms=12_000, seq=3, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=99.70, time_ms=14_000, seq=4, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        # Settle watermark
        engine.on_trade(make_trade(price=99.70, time_ms=20_000, seq=5, aggressor_side=AggressorSide.SELL), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.BEARISH_CONTINUATION
        assert results[0].reason_code == "ACCEPTANCE_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 03: BUY-side liquidity taken + sustained rejection/reclaim below => FAILED_BREAKOUT
# -----------------------------------------------------------------------------
def test_gate_03_failed_breakout(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-03", side=LiquiditySide.BUY_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Penetration trade above 100.00
        engine.on_trade(make_trade(price=100.10, time_ms=10_500, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Reclaim trades below 100.00 (<= 99.99) for hold duration
        engine.on_trade(make_trade(price=99.80, time_ms=11_000, seq=2, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=99.80, time_ms=12_000, seq=3, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=99.80, time_ms=14_000, seq=4, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        # Settle watermark
        engine.on_trade(make_trade(price=99.80, time_ms=20_000, seq=5, aggressor_side=AggressorSide.SELL), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.FAILED_BREAKOUT
        assert results[0].reason_code == "RECLAIM_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 04: BUY-side liquidity taken + sustained acceptance above => BULLISH_CONTINUATION
# -----------------------------------------------------------------------------
def test_gate_04_bullish_continuation(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-04", side=LiquiditySide.BUY_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Penetration trade above 100.00
        engine.on_trade(make_trade(price=100.20, time_ms=10_500, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Acceptance trades staying above threshold (>= 100.01) for hold duration
        engine.on_trade(make_trade(price=100.30, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.30, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.30, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Settle watermark
        engine.on_trade(make_trade(price=100.30, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.BULLISH_CONTINUATION
        assert results[0].reason_code == "ACCEPTANCE_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 05: Verified penetration but insufficient confirmation before deadline => INDETERMINATE
# -----------------------------------------------------------------------------
def test_gate_05_indeterminate_timeout_after_penetration(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-05", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Penetration trade below 100.00
        engine.on_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        # Neutral price trades (100.00) that never meet hold threshold duration
        engine.on_trade(make_trade(price=100.00, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Advance clock to expiry (10_000 + 60_000 = 70_000) with settlement watermark (72_000 - 2_000 = 70_000)
        engine.on_trade(make_trade(price=100.00, time_ms=72_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.INDETERMINATE
        assert results[0].reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 06: No independently verified penetration => INVALID / PENETRATION_NOT_CONFIRMED
# -----------------------------------------------------------------------------
def test_gate_06_unpenetrated_expires_to_invalid(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-06", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Trades only above 100.00 (no penetration on SELL_SIDE sweep)
        engine.on_trade(make_trade(price=100.50, time_ms=11_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        # Advance past confirmation window expiry
        engine.on_trade(make_trade(price=100.50, time_ms=72_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.INVALID
        assert results[0].reason_code == "PENETRATION_NOT_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 07: Duplicate source observation / source identity is idempotent
# -----------------------------------------------------------------------------
def test_gate_07_duplicate_observation_is_idempotent(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-07", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))
        assert len(store.active("BTCUSDT")) == 1

        # Duplicate sweep arrival
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))
        assert len(store.active("BTCUSDT")) == 1
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 08: Out-of-order evidence within reorder tolerance produces canonical outcome
# -----------------------------------------------------------------------------
def test_gate_08_out_of_order_within_reorder_tolerance(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-08", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Feed trades out-of-order within reorder tolerance (2_000 ms)
        t_penetration = make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)
        t_reclaim_1 = make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY)
        t_reclaim_2 = make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY)
        t_reclaim_3 = make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY)

        engine.on_trade(t_reclaim_2, make_safe_coverage())
        engine.on_trade(t_reclaim_1, make_safe_coverage())
        engine.on_trade(t_penetration, make_safe_coverage())
        engine.on_trade(t_reclaim_3, make_safe_coverage())

        # Watermark settlement
        engine.on_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.FAILED_BREAKDOWN
        assert results[0].reason_code == "RECLAIM_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 09: Evidence arriving behind settled watermark is quarantined
# -----------------------------------------------------------------------------
def test_gate_09_quarantined_late_trade_behind_watermark(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-09", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Settle watermark up to 20_000 - 2_000 = 18_000
        engine.on_trade(make_trade(price=100.00, time_ms=20_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        # Late trade arrives at 15_000 (behind settled watermark 18_000)
        late_trade = make_trade(price=90.00, time_ms=15_000, seq=2, aggressor_side=AggressorSide.SELL)
        engine.on_trade(late_trade, make_safe_coverage())

        # Advance to expiry past confirmation window
        engine.on_trade(make_trade(price=100.00, time_ms=72_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        # Late trade at 15_000 was quarantined and did not trigger penetration
        assert results[0].classification is EventClassification.INVALID
        assert results[0].reason_code == "PENETRATION_NOT_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 10: Confirmation-window expiry produces deterministic timeout result
# -----------------------------------------------------------------------------
def test_gate_10_deterministic_timeout_expiry(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-10", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Advance watermark strictly at expiry threshold (10_000 + 60_000 = 70_000)
        # Watermark is max_seen - 2000, so max_seen 72_000 produces watermark 70_000
        engine.on_trade(make_trade(price=100.00, time_ms=72_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].market_resolution_time_ms == 70_000
        assert results[0].reason_code == "PENETRATION_NOT_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 11: Compromised mandatory trade coverage => fail closed / INVALID
# -----------------------------------------------------------------------------
def test_gate_11_compromised_trade_coverage_fails_closed(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-11", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Trade with known gap in coverage (feed_safe=True with known_gap=True)
        gapped_coverage = TradeCoverage(
            feed_safe=True,
            known_gap=True,
            buffer_overflow=False,
            unresolved_sequence=False,
            interval_retained=True,
        )
        engine.on_trade(make_trade(price=99.80, time_ms=11_000, seq=1, aggressor_side=AggressorSide.SELL), gapped_coverage)
        engine.on_trade(make_trade(price=99.80, time_ms=20_000, seq=2, aggressor_side=AggressorSide.SELL), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.INVALID
        assert results[0].reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 12: Unavailable/unsafe optional depth context must NOT invalidate price classification
# -----------------------------------------------------------------------------
def test_gate_12_optional_depth_does_not_invalidate_price_classification(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-12", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Drive valid failed breakdown trades without any depth observations
        engine.on_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.FAILED_BREAKDOWN
        # Context shows unavailable appropriately without breaking classification
        assert results[0].evidence.values["depth_weighted_imbalance"].availability is EvidenceAvailability.UNAVAILABLE
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 13: Finalized events are immutable
# -----------------------------------------------------------------------------
def test_gate_13_finalized_events_are_immutable(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-13", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        # Settle to invalid at expiry
        engine.on_trade(make_trade(price=100.00, time_ms=72_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        results1 = store.recent("BTCUSDT")
        assert len(results1) == 1
        res = results1[0]

        # Post-finalization trades cannot change finalized state
        engine.on_trade(make_trade(price=90.00, time_ms=80_000, seq=2, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        results2 = store.recent("BTCUSDT")
        assert len(results2) == 1
        assert results2[0] == res

        # Attempt to mutate dataclass raises FrozenInstanceError / AttributeError
        with pytest.raises((AttributeError, TypeError)):
            res.classification = EventClassification.BEARISH_CONTINUATION  # type: ignore
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 14: Equivalent canonical timelines produce byte-equivalent deterministic artifacts
# -----------------------------------------------------------------------------
def test_gate_14_byte_equivalent_deterministic_artifacts(tmp_path: Path):
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0, sweep_id="swp-gate-14")
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY)),
    ]

    runner1 = LiquidityReplayRunner(inputs=inputs, output_dir=dir1)
    runner2 = LiquidityReplayRunner(inputs=inputs, output_dir=dir2)

    res1 = runner1.run()
    res2 = runner2.run()

    assert res1.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert res2.artifact_integrity is ArtifactIntegrity.COMPLETE

    bytes_events_1 = (dir1 / EVENTS_FILENAME).read_bytes()
    bytes_events_2 = (dir2 / EVENTS_FILENAME).read_bytes()
    assert bytes_events_1 == bytes_events_2

    bytes_transitions_1 = (dir1 / TRANSITIONS_FILENAME).read_bytes()
    bytes_transitions_2 = (dir2 / TRANSITIONS_FILENAME).read_bytes()
    assert bytes_transitions_1 == bytes_transitions_2


# -----------------------------------------------------------------------------
# GATE 15: Liquidity Event Engine enabled vs disabled does not alter scorer output
# -----------------------------------------------------------------------------
def test_gate_15_engine_toggle_scorer_invariance(tmp_path: Path):
    with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", str(tmp_path)):
        with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", False):
            engine_off = OrderFlowEngine()
        with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", True):
            engine_on = OrderFlowEngine()

        trade_raw = {
            "p": "50000.0",
            "q": "1.5",
            "m": False,
            "T": 10_000,
            "a": 100,
            "timestamp": 10.0,
            "price": 50000.0,
            "quantity": 1.5,
            "side": "BUY",
        }
        engine_off.metrics.add_trade("btcusdt", trade_raw)
        engine_on.metrics.add_trade("btcusdt", trade_raw)

        alerts = [{"symbol": "btcusdt", "type": "LARGE_TRADE", "timestamp": 10.0}]
        res_off = engine_off.scorer.evaluate_sweep("btcusdt", "BULLISH", 49950.0, alerts, 10.0)
        res_on = engine_on.scorer.evaluate_sweep("btcusdt", "BULLISH", 49950.0, alerts, 10.0)

        assert res_off == res_on


# -----------------------------------------------------------------------------
# GATE 16: Execution remains disabled and PaperTrader behavior unchanged
# -----------------------------------------------------------------------------
def test_gate_16_execution_disabled_and_paper_trader_isolated():
    assert EXECUTION_DISABLED is True
    engine = OrderFlowEngine()
    assert engine.paper_trader is not None
    assert engine.paper_trader.initial_balance == 10000.0
    for attr, val in vars(engine.paper_trader).items():
        assert not isinstance(val, LiquidityEventEngine)


# -----------------------------------------------------------------------------
# GATE 17: REGIME_ENFORCEMENT_ENABLED remains False and no regime authority
# -----------------------------------------------------------------------------
def test_gate_17_regime_enforcement_disabled():
    assert REGIME_ENFORCEMENT_ENABLED is False
    assert config.LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False
    engine = OrderFlowEngine()
    if hasattr(engine, "regime_engine") and engine.regime_engine:
        for attr, val in vars(engine.regime_engine).items():
            assert not isinstance(val, LiquidityEventEngine)


# -----------------------------------------------------------------------------
# GATE 18: Legacy SweepsMonitor direction semantics map correctly
# -----------------------------------------------------------------------------
def test_gate_18_legacy_direction_mapping():
    policy = LiquidityClassificationPolicy()
    adapter = SweepsMonitorAdapter(policy)

    raw_bullish = {
        "sweep_id": "swp-bullish",
        "symbol": "BTCUSDT",
        "type": "BULLISH",
        "sweep_level": 100.0,
        "timestamp": "1970-01-01T00:00:10Z",
        "source_file_id": "SWEEPS_MONITOR_CSV",
        "source_row_hash": "a" * 64,
    }
    adapted_bullish = adapter.adapt(raw_bullish, 10_000)
    assert adapted_bullish.observation is not None
    assert adapted_bullish.observation.liquidity_side is LiquiditySide.SELL_SIDE

    raw_bearish = {
        "sweep_id": "swp-bearish",
        "symbol": "BTCUSDT",
        "type": "BEARISH",
        "sweep_level": 100.0,
        "timestamp": "1970-01-01T00:00:10Z",
        "source_file_id": "SWEEPS_MONITOR_CSV",
        "source_row_hash": "b" * 64,
    }
    adapted_bearish = adapter.adapt(raw_bearish, 10_000)
    assert adapted_bearish.observation is not None
    assert adapted_bearish.observation.liquidity_side is LiquiditySide.BUY_SIDE


# -----------------------------------------------------------------------------
# GATE 19: Neutral price evidence pauses confirmation accumulation
# -----------------------------------------------------------------------------
def test_gate_19_neutral_zone_pauses_accumulation():
    policy = LiquidityClassificationPolicy()
    tracker = PriceOutcomeTracker(level=Decimal("100.00"), side=LiquiditySide.SELL_SIDE, policy=policy)

    # Penetration trade at 10_500
    t1 = make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)
    # Qualify reclaim at 11_000
    t2 = make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY)
    # Neutral trade (100.00) between 11_000 and 13_000
    t3 = make_trade(price=100.00, time_ms=13_000, seq=3, aggressor_side=AggressorSide.BUY)
    # Reclaim trade at 13_500 (duration only 13_500 - 13_000 + 0 = 500ms since neutral paused)
    t4 = make_trade(price=100.10, time_ms=13_500, seq=4, aggressor_side=AggressorSide.BUY)

    for t in [t1, t2, t3, t4]:
        tracker.on_trade(t)

    decision = tracker.decision_at(watermark_ms=13_500, expiry_ms=70_000)
    assert decision.classification is EventClassification.PENDING_SWEEP
    assert decision.has_penetration is True


# -----------------------------------------------------------------------------
# GATE 20: Opposite qualifying evidence resets competing branch
# -----------------------------------------------------------------------------
def test_gate_20_opposite_evidence_resets_unsatisfied_branch():
    policy = LiquidityClassificationPolicy()
    tracker = PriceOutcomeTracker(level=Decimal("100.00"), side=LiquiditySide.SELL_SIDE, policy=policy)

    # Penetration at 10_500
    t1 = make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)
    # Start acceptance branch at 11_000
    t2 = make_trade(price=99.70, time_ms=11_000, seq=2, aggressor_side=AggressorSide.SELL)
    # Opposite threshold (reclaim) at 12_000 resets acceptance candidate
    t3 = make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY)
    # Acceptance trade at 12_500 starts new candidate from scratch
    t4 = make_trade(price=99.70, time_ms=12_500, seq=4, aggressor_side=AggressorSide.SELL)

    for t in [t1, t2, t3, t4]:
        tracker.on_trade(t)

    decision = tracker.decision_at(watermark_ms=12_500, expiry_ms=70_000)
    assert decision.classification is EventClassification.PENDING_SWEEP
    assert decision.has_penetration is True


# -----------------------------------------------------------------------------
# GATE 21: Equal canonical sufficient-time conflict => INDETERMINATE / CONFLICTING_CONFIRMATION
# -----------------------------------------------------------------------------
def test_gate_21_equal_sufficient_time_conflict():
    policy = LiquidityClassificationPolicy()
    assert policy.confirmation_hold_ms == 3000
    assert policy.minimum_confirming_trades == 3

    tracker = PriceOutcomeTracker(
        level=Decimal("100.00"),
        side=LiquiditySide.SELL_SIDE,
        policy=policy,
    )

    # Drive genuine penetration at 10_000
    t_pen = make_trade(
        price=Decimal("99.90"),
        time_ms=10_000,
        seq=1,
        aggressor_side=AggressorSide.SELL,
    )
    tracker.on_trade(t_pen)
    assert tracker.has_penetration is True

    # Construct both immutable candidate witnesses with equal sufficient time
    witness_time = 15_000
    tracker._reclaim_progress = CandidateProgress.sufficient_at(witness_time)
    tracker._acceptance_progress = CandidateProgress.sufficient_at(witness_time)

    decision = tracker.decision_at(watermark_ms=20_000, expiry_ms=70_000)

    assert decision.classification is EventClassification.INDETERMINATE
    assert decision.reason_code == "CONFLICTING_CONFIRMATION"
    assert decision.reclaim_sufficient_time_ms == witness_time
    assert decision.acceptance_sufficient_time_ms == witness_time
    assert decision.reclaim_sufficient_time_ms == decision.acceptance_sufficient_time_ms
    assert decision.market_resolution_time_ms == witness_time


# -----------------------------------------------------------------------------
# GATE 22: Late callback within retained history reconstructs live resolution
# -----------------------------------------------------------------------------
def test_gate_22_late_callback_within_retained_history(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        # Pre-feed trades from 10_000 to 20_000
        engine.on_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY), make_safe_coverage())
        engine.on_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        # Late sweep detection arriving at 20_000 for event at 10_000
        obs = make_obs("gate-22", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000, detection_time_ms=20_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        assert results[0].classification is EventClassification.FAILED_BREAKDOWN
        assert results[0].reason_code == "RECLAIM_CONFIRMED"
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 23: Evicted / unavailable required history => fail closed (with retention boundary test)
# -----------------------------------------------------------------------------
def test_gate_23_evicted_and_boundary_history_fails_closed():
    policy = LiquidityClassificationPolicy()
    provider = LiveTradeCoverageProvider(retention_ms=policy.market_buffer_retention_ms)

    # Initial trades around 10_000
    provider.record_trade(
        "BTCUSDT",
        10_000,
        feed_safe=True,
        known_gap=False,
        sequence_id=1,
    )
    provider.record_trade(
        "BTCUSDT",
        20_000,
        feed_safe=True,
        known_gap=False,
        sequence_id=2,
    )

    # Advance time beyond retention horizon (180_000 ms)
    later_time = 20_000 + policy.market_buffer_retention_ms + 5_000  # 205_000 ms
    provider.record_trade(
        "BTCUSDT",
        later_time,
        feed_safe=True,
        known_gap=False,
        sequence_id=3,
    )

    # Wholly evicted interval (10_000 - 20_000)
    cov_evicted = provider.coverage("BTCUSDT", 10_000, 20_000)
    assert cov_evicted.interval_retained is False
    assert cov_evicted.valid is False

    # Boundary crossing intervals (starts before retention cutoff 25_000)
    cov_24 = provider.coverage("BTCUSDT", 24_000, later_time)
    assert cov_24.interval_retained is False
    assert cov_24.valid is False

    cov_24999 = provider.coverage("BTCUSDT", 24_999, later_time)
    assert cov_24999.interval_retained is False
    assert cov_24999.valid is False

    # Retained boundary interval (25_000 is exactly at cutoff 25_000)
    cov_25 = provider.coverage("BTCUSDT", 25_000, later_time)
    assert cov_25.interval_retained is True
    assert cov_25.valid is True

    # Inside retained interval (25_001 is above cutoff 25_000)
    cov_25001 = provider.coverage("BTCUSDT", 25_001, later_time)
    assert cov_25001.interval_retained is True
    assert cov_25001.valid is True


# -----------------------------------------------------------------------------
# GATE 24: Overlapping independent events remain separate
# -----------------------------------------------------------------------------
def test_gate_24_overlapping_independent_events_remain_separate(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs_sell = make_obs("gate-24-sell", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        obs_buy = make_obs("gate-24-buy", side=LiquiditySide.BUY_SIDE, level=105.0, event_time_ms=12_000)

        engine.on_sweep(obs_sell, StaticCoverageProvider(make_safe_coverage()))
        engine.on_sweep(obs_buy, StaticCoverageProvider(make_safe_coverage()))

        active = store.active("BTCUSDT")
        assert len(active) == 2
        assert {e.event_id for e in active} == {obs_sell.event_id, obs_buy.event_id}
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 25: Same-side collision fails closed according to identity policy
# -----------------------------------------------------------------------------
def test_gate_25_same_side_collision_fails_closed(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        # Two sell-side observations within collision tolerance (e.g. same level/time window)
        obs1 = make_obs("gate-25-a", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        obs2 = make_obs("gate-25-b", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_500)

        engine.on_sweep(obs1, StaticCoverageProvider(make_safe_coverage()))
        engine.on_sweep(obs2, StaticCoverageProvider(make_safe_coverage()))

        # Collision invalidates colliding events
        results = store.recent("BTCUSDT")
        assert any(r.classification is EventClassification.INVALID for r in results)
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 26: Every result carries deterministic frozen policy hash / model version
# -----------------------------------------------------------------------------
def test_gate_26_policy_hash_and_model_version(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs = make_obs("gate-26", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        engine.on_sweep(obs, StaticCoverageProvider(make_safe_coverage()))
        engine.on_trade(make_trade(price=100.00, time_ms=72_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        results = store.recent("BTCUSDT")
        assert len(results) == 1
        res = results[0]
        assert res.policy_hash == policy.policy_hash
        assert res.model_version == policy.model_version
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 27: Reasons, contradictions, and ordering are deterministic
# -----------------------------------------------------------------------------
def test_gate_27_reasons_contradictions_determinism():
    builder = LiquidityEvidenceBuilder()
    obs = make_obs("gate-27")
    event = LiquidityEvent(observation=obs, state=EventState.OBSERVED)
    evidence = builder.build(
        event=event,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=20_000,
    )
    assert isinstance(evidence.reasons, tuple)
    assert isinstance(evidence.contradictions, tuple)
    assert list(evidence.reasons) == sorted(evidence.reasons)
    assert list(evidence.contradictions) == sorted(evidence.contradictions)


# -----------------------------------------------------------------------------
# GATE 28: Architecture forbidden-import gate
# -----------------------------------------------------------------------------
def test_gate_28_forbidden_imports():
    root_dir = Path(__file__).parent
    liquidity_event_dir = root_dir / "liquidity_event"
    forbidden_modules = [
        "scoring",
        "data.paper_trader",
        "paper_trader",
        "regime.permissions",
        "execution",
        "position_sizing",
    ]

    # A. liquidity_event production package must not import authority modules
    for py_file in liquidity_event_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert forbidden not in alias.name, f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in forbidden_modules:
                        assert forbidden not in node.module, f"Forbidden import from {node.module} in {py_file}"

    # B. Reverse authority scan: authority files must not import liquidity_event as trading/permission authority dependency
    authority_files = [
        root_dir / "scoring.py",
        root_dir / "data" / "paper_trader.py",
        root_dir / "regime" / "permissions.py",
    ]
    for p in root_dir.glob("execution*"):
        if p.is_file() and p.suffix == ".py" and not p.name.startswith("test_"):
            authority_files.append(p)
        elif p.is_dir():
            for sub in p.rglob("*.py"):
                if not sub.name.startswith("test_"):
                    authority_files.append(sub)
    for p in root_dir.glob("position_sizing*"):
        if p.is_file() and p.suffix == ".py" and not p.name.startswith("test_"):
            authority_files.append(p)
        elif p.is_dir():
            for sub in p.rglob("*.py"):
                if not sub.name.startswith("test_"):
                    authority_files.append(sub)

    for py_file in authority_files:
        if py_file.exists():
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "liquidity_event" not in alias.name, f"Reverse authority import {alias.name} in {py_file}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert "liquidity_event" not in node.module, f"Reverse authority import from {node.module} in {py_file}"

    # C. No wall-clock classification authority in core classification modules
    core_modules = [
        liquidity_event_dir / "classifier.py",
        liquidity_event_dir / "evidence.py",
        liquidity_event_dir / "engine.py",
    ]
    for py_file in core_modules:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ("time", "now", "utcnow", "today"):
                        if isinstance(func.value, ast.Name) and func.value.id in ("time", "datetime"):
                            raise AssertionError(f"Wall-clock call {func.value.id}.{func.attr}() found in {py_file}")

    # D. Frozen authority flags
    assert LIQUIDITY_EVENT_ENGINE_ENABLED is True
    assert LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False
    assert REGIME_ENFORCEMENT_ENABLED is False
    assert EXECUTION_DISABLED is True


# -----------------------------------------------------------------------------
# GATE 29: Recorder failure is bounded and cannot change classification
# -----------------------------------------------------------------------------
def test_gate_29_recorder_failure_bounded_and_isolated(tmp_path: Path):
    dir_b = tmp_path / "baseline"
    dir_f = tmp_path / "failing"

    with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", str(dir_b)):
        engine_b = OrderFlowEngine()
    with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", str(dir_f)):
        engine_f = OrderFlowEngine()

    if engine_f.liquidity_recorder:
        engine_f.liquidity_recorder.telemetry.status = "DEGRADED"
        engine_f.liquidity_recorder.enqueue = MagicMock(side_effect=RuntimeError("Disk failure"))

    cb = make_sweep_callback(symbol="BTCUSDT", event_time_ms=10_000, detection_time_ms=10_000, level=100.0, sweep_id="gate-29")
    adapted_b = engine_b.liquidity_adapter.adapt(cb, 10_000)
    adapted_f = engine_f.liquidity_adapter.adapt(cb, 10_000)

    engine_b.liquidity_engine.on_sweep(adapted_b.observation, StaticCoverageProvider(make_safe_coverage()))
    engine_f.liquidity_engine.on_sweep(adapted_f.observation, StaticCoverageProvider(make_safe_coverage()))

    trades = [
        make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL),
        make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY),
        make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY),
        make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY),
        make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY),
    ]
    for t in trades:
        engine_b.liquidity_engine.on_trade(t, make_safe_coverage())
        engine_f.liquidity_engine.on_trade(t, make_safe_coverage())

    assert len(engine_b.recent_liquidity_results) == 1
    assert len(engine_f.recent_liquidity_results) == 1
    assert engine_b.recent_liquidity_results[0]["classification"] == engine_f.recent_liquidity_results[0]["classification"]
    assert engine_b.recent_liquidity_results[0]["reason_code"] == engine_f.recent_liquidity_results[0]["reason_code"]


# -----------------------------------------------------------------------------
# GATE 30: Equivalent live and replay event sequences produce identical semantics
# -----------------------------------------------------------------------------
def test_gate_30_live_and_replay_parity(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0, sweep_id="gate-30")
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2, aggressor_side=AggressorSide.SELL)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=13_000, seq=3, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=16_000, seq=4, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=19_000, seq=5, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=6, aggressor_side=AggressorSide.BUY)),
    ]

    # Live drive
    live_results, live_transitions = drive_live_engine(inputs)

    # Replay drive
    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "replay_live_parity")
    replay_res = runner.run()

    assert replay_res.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert len(live_results) == 1
    assert len(replay_res.results) == 1
    assert len(live_transitions) > 0
    assert len(replay_res.transitions) > 0

    # Full canonical dict equivalence
    assert [r.to_canonical_dict() for r in live_results] == [r.to_canonical_dict() for r in replay_res.results]
    assert [t.to_canonical_dict() for t in live_transitions] == [t.to_canonical_dict() for t in replay_res.transitions]

    # Explicit field equivalence
    assert live_results[0].classification == replay_res.results[0].classification
    assert live_results[0].reason_code == replay_res.results[0].reason_code
    assert live_results[0].market_resolution_time_ms == replay_res.results[0].market_resolution_time_ms
    assert live_results[0].classification_time_ms == replay_res.results[0].classification_time_ms
    assert live_results[0].policy_hash == replay_res.results[0].policy_hash
    assert live_results[0].model_version == replay_res.results[0].model_version


# -----------------------------------------------------------------------------
# GATE 31: Watermarks, active events, evidence remain isolated per symbol
# -----------------------------------------------------------------------------
def test_gate_31_symbol_isolation(tmp_path: Path):
    engine, store, authority, policy = make_test_engine(tmp_path)
    try:
        obs_btc = make_obs("gate-31-btc", symbol="BTCUSDT", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
        obs_eth = make_obs("gate-31-eth", symbol="ETHUSDT", side=LiquiditySide.SELL_SIDE, level=2000.0, event_time_ms=10_000)

        engine.on_sweep(obs_btc, StaticCoverageProvider(make_safe_coverage()))
        engine.on_sweep(obs_eth, StaticCoverageProvider(make_safe_coverage()))

        # ETH trades advance ETH watermark only
        engine.on_trade(make_trade(symbol="ETHUSDT", price=2000.00, time_ms=72_000, seq=1, aggressor_side=AggressorSide.BUY), make_safe_coverage())

        assert len(store.recent("ETHUSDT")) == 1
        assert len(store.recent("BTCUSDT")) == 0
        assert len(store.active("BTCUSDT")) == 1
    finally:
        authority.close()


# -----------------------------------------------------------------------------
# GATE 32: Replay with insufficient future coverage => fail closed
# -----------------------------------------------------------------------------
def test_gate_32_replay_insufficient_future_coverage(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0, sweep_id="gate-32")
    runner = LiquidityReplayRunner(
        inputs=[
            ReplayInput.from_sweep_callback(cb),
            ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)),
        ],
        output_dir=tmp_path / "replay_incomplete",
    )
    res = runner.run()
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code in ("INSUFFICIENT_FUTURE_COVERAGE", "INSUFFICIENT_REPLAY_HISTORY")


# -----------------------------------------------------------------------------
# GATE 33: Frozen bounds are enforced
# -----------------------------------------------------------------------------
def test_gate_33_frozen_bounds_enforced(tmp_path: Path):
    policy = LiquidityClassificationPolicy()
    assert policy.max_active_events_per_symbol == 128
    assert policy.recent_final_events_per_symbol == 1_000
    assert policy.recorder_queue_max_items == 10_000
    assert policy.market_buffer_retention_ms == 180_000

    # A. ACTIVE EVENT LIMIT (128)
    store = LiquidityEventStore(policy)
    auth = SQLiteIdentityAuthority(str(tmp_path / "bounds_auth.sqlite3"))
    try:
        for i in range(128):
            obs = make_obs(
                event_id=f"active-lim-{i}",
                symbol="BTCUSDT",
                side=LiquiditySide.SELL_SIDE,
                level=100.0 + i,
                source_level_id=f"LVL_{i}",
            )
            res = store.admit_observation(obs, auth)
            assert res.created is True, f"Failed to admit event {i}"

        assert store.active_count("BTCUSDT") == 128

        # Event 129 must be rejected with EVENT_CAPACITY_REACHED
        obs_129 = make_obs(
            event_id="active-lim-128",
            symbol="BTCUSDT",
            side=LiquiditySide.SELL_SIDE,
            level=300.0,
            source_level_id="LVL_128",
        )
        res_129 = store.admit_observation(obs_129, auth)
        assert res_129.created is False
        assert res_129.rejection is not None
        assert res_129.rejection.reason_code == "EVENT_CAPACITY_REACHED"
        assert store.active_count("BTCUSDT") == 128
    finally:
        auth.close()

    # B. RECENT FINAL LIMIT (1000)
    store_recent = LiquidityEventStore(policy)
    auth_recent = SQLiteIdentityAuthority(str(tmp_path / "recent_auth.sqlite3"))
    builder = LiquidityEvidenceBuilder()
    try:
        for i in range(1001):
            obs = make_obs(
                event_id=f"recent-lim-{i}",
                symbol="BTCUSDT",
                side=LiquiditySide.SELL_SIDE,
                level=100.0 + (i % 50),
                source_level_id=f"LVL_REC_{i}",
            )
            admit_res = store_recent.admit_observation(obs, auth_recent)
            assert admit_res.created is True
            event = admit_res.event
            evidence = builder.build(
                event=event,
                outcome_direction=EventClassification.FAILED_BREAKDOWN,
                as_of_ms=20_000 + i,
            )
            final_res = LiquidityEventResult(
                event_id=obs.event_id,
                symbol=obs.symbol,
                liquidity_side=obs.liquidity_side,
                classification=EventClassification.FAILED_BREAKDOWN,
                reason_code="RECLAIM_CONFIRMED",
                event_time_ms=obs.event_time_ms,
                detection_time_ms=obs.detection_time_ms,
                market_resolution_time_ms=20_000 + i,
                classification_time_ms=20_000 + i,
                source_observation_hash=obs.source_observation_hash,
                evidence=evidence,
                policy_hash=policy.policy_hash,
                model_version=policy.model_version,
            )
            auth_recent.claim_result(final_res)
            store_recent.finalize(final_res)

        recent_events = store_recent.recent("BTCUSDT")
        assert len(recent_events) == 1000
        # Oldest event (recent-lim-0) was evicted, newest (recent-lim-1000) is present
        assert any(r.event_id == "recent-lim-1000" for r in recent_events)
        assert all(r.event_id != "recent-lim-0" for r in recent_events)
    finally:
        auth_recent.close()

    # C. RECORDER QUEUE LIMIT (10000)
    recorder = LiquidityEventRecorder(
        mode="live",
        output_dir=tmp_path / "rec_queue_bound",
        queue_max_items=policy.recorder_queue_max_items,
    )
    try:
        t_sample = LifecycleTransition(
            event_id="q-ev",
            transition_sequence=1,
            transition_time_ms=10_000,
            previous_state=EventState.OBSERVED,
            next_state=EventState.SWEEP_DETECTED,
            reason_code="OBSERVATION_ADMITTED",
        )
        rec_sample = CanonicalRecord(record_type="TRANSITION", payload=t_sample)
        for _ in range(10_000):
            enq_ok = recorder.enqueue(rec_sample)
            assert enq_ok is True

        # Item 10,001 overflows
        enq_overflow = recorder.enqueue(rec_sample)
        assert enq_overflow is False
        assert recorder.artifact_integrity is ArtifactIntegrity.QUEUE_OVERFLOW
        assert recorder.telemetry.status == "DEGRADED"
        assert recorder.telemetry.failure_count >= 1
    finally:
        recorder.close()

    # D. MARKET RETENTION (180000)
    provider = LiveTradeCoverageProvider(retention_ms=policy.market_buffer_retention_ms)
    provider.record_trade("BTCUSDT", 10_000, feed_safe=True, known_gap=False, sequence_id=1)
    provider.record_trade("BTCUSDT", 20_000, feed_safe=True, known_gap=False, sequence_id=2)
    provider.record_trade("BTCUSDT", 205_000, feed_safe=True, known_gap=False, sequence_id=3)

    cutoff = 205_000 - policy.market_buffer_retention_ms
    assert cutoff == 25_000

    cov_below = provider.coverage("BTCUSDT", 24_999, 205_000)
    assert cov_below.interval_retained is False
    assert cov_below.valid is False

    cov_exact = provider.coverage("BTCUSDT", 25_000, 205_000)
    assert cov_exact.interval_retained is True
    assert cov_exact.valid is True


# -----------------------------------------------------------------------------
# GATE 34: Event-local order flow accumulator does not read global CVD
# -----------------------------------------------------------------------------
def test_gate_34_event_local_flow_no_global_cvd():
    # Behavioral arithmetic
    acc = EventFlowAccumulator(event_time_ms=10_000)
    acc.on_trade(make_trade(price=100.00, quantity=2.0, time_ms=10_000, seq=1, aggressor_side=AggressorSide.BUY))
    acc.on_trade(make_trade(price=100.00, quantity=1.0, time_ms=11_000, seq=2, aggressor_side=AggressorSide.SELL))

    snap = acc.snapshot(as_of_ms=12_000)
    assert snap.buy_volume_usdt == 200.0
    assert snap.sell_volume_usdt == 100.0
    assert snap.signed_delta_usdt == 100.0
    assert snap.post_sweep_cvd_usdt == 100.0

    # Static AST certification: liquidity_event core must not reference global FlowMetrics or session CVD fields
    core_files = [
        Path(__file__).parent / "liquidity_event" / "classifier.py",
        Path(__file__).parent / "liquidity_event" / "evidence.py",
        Path(__file__).parent / "liquidity_event" / "engine.py",
    ]
    forbidden_names = {"FlowMetrics", "flow_metrics"}
    forbidden_attrs = {"session_cvd", "session_cvd_usdt", "running_cvd", "running_cvd_usdt"}

    for py_file in core_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_names, f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module not in forbidden_names, f"Forbidden import from {node.module} in {py_file}"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, f"Forbidden name {node.id} in {py_file}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, f"Forbidden attribute {node.attr} in {py_file}"

    # Behavioral isolation: EventFlowAccumulator ignores external trades before event_time_ms
    acc_isolated = EventFlowAccumulator(event_time_ms=10_000)
    acc_isolated.on_trade(make_trade(price=500.00, quantity=100.0, time_ms=5_000, seq=1, aggressor_side=AggressorSide.BUY))
    snap_iso = acc_isolated.snapshot(as_of_ms=12_000)
    assert snap_iso.buy_volume_usdt == 0.0
    assert snap_iso.post_sweep_cvd_usdt == 0.0


# -----------------------------------------------------------------------------
# GATE 35: Context enters only through typed context adapters
# -----------------------------------------------------------------------------
def test_gate_35_typed_context_adapter():
    adapter = LegacyLiquidityContextAdapter()

    # 1. Pass representative raw legacy mappings and verify deterministic typed EvidenceValue outputs
    abs_val = adapter.absorption({"side": "BID", "event_type": "BULLISH_ABSORPTION"}, as_of_ms=20_000)
    assert isinstance(abs_val, EvidenceValue)
    assert abs_val.availability is EvidenceAvailability.AVAILABLE
    assert abs_val.value == 1.0
    assert abs_val.as_of_ms == 20_000

    rep_val = adapter.replenishment({"side": "BID", "event_type": "BID_REPLENISHMENT"}, as_of_ms=20_000)
    assert isinstance(rep_val, EvidenceValue)
    assert rep_val.availability is EvidenceAvailability.AVAILABLE
    assert rep_val.value == 1.0
    assert rep_val.as_of_ms == 20_000

    sp_val = adapter.stacking_pulling({"event_type": "BID_STACKING"}, as_of_ms=20_000)
    assert isinstance(sp_val, EvidenceValue)
    assert sp_val.availability is EvidenceAvailability.AVAILABLE
    assert sp_val.value == 1.0
    assert sp_val.as_of_ms == 20_000

    depth_val = EvidenceValue(EvidenceAvailability.AVAILABLE, 0.75, 20_000)

    # 2. Pass typed EvidenceValue into LiquidityEvidenceBuilder.build()
    builder = LiquidityEvidenceBuilder()
    obs = make_obs("gate-35")
    event = LiquidityEvent(observation=obs, state=EventState.OBSERVED)

    evidence = builder.build(
        event=event,
        outcome_direction=EventClassification.FAILED_BREAKDOWN,
        as_of_ms=20_000,
        absorption=abs_val,
        depth_weighted_imbalance=depth_val,
        replenishment=rep_val,
        stacking_pulling=sp_val,
    )

    assert evidence.values["bid_ask_absorption"] == abs_val
    assert evidence.values["depth_weighted_imbalance"] == depth_val
    assert evidence.values["wall_replenishment"] == rep_val
    assert evidence.values["stacking_pulling"] == sp_val

    # 3. Certify actual parameter type annotations using get_type_hints
    type_hints = get_type_hints(LiquidityEvidenceBuilder.build)
    for param_name in (
        "absorption",
        "depth_weighted_imbalance",
        "replenishment",
        "stacking_pulling",
    ):
        assert param_name in type_hints
        annotation_args = set(get_args(type_hints[param_name]))
        assert EvidenceValue in annotation_args
        assert type(None) in annotation_args

    # Retain inspect.signature as an additional check
    sig = inspect.signature(builder.build)
    for param_name in ("absorption", "depth_weighted_imbalance", "replenishment", "stacking_pulling"):
        assert param_name in sig.parameters

    # 4. Prove arbitrary raw legacy dict is NOT accepted directly by build()
    with pytest.raises((AttributeError, TypeError)):
        builder.build(
            event=event,
            outcome_direction=EventClassification.FAILED_BREAKDOWN,
            as_of_ms=20_000,
            absorption={"side": "BID", "event_type": "BULLISH_ABSORPTION"},  # type: ignore
        )


# -----------------------------------------------------------------------------
# GATE 36: source_level_id preserved when supplied, deterministically derived when absent
# -----------------------------------------------------------------------------
def test_gate_36_source_level_id_preservation():
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)

    obs_with_id = make_obs("swp-with-id", source_level_id="LEVEL_123")
    obs_without_id = make_obs("swp-without-id", source_level_id=None)

    assert store.level_identity(obs_with_id) == "LEVEL_123"
    assert store.level_identity(obs_without_id) is not None
    assert len(store.level_identity(obs_without_id)) > 0


# -----------------------------------------------------------------------------
# GATE 37: Recorder failure / incomplete artifact integrity produces non-complete
# -----------------------------------------------------------------------------
def test_gate_37_incomplete_artifact_integrity(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0, sweep_id="gate-37")
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY)),
    ]

    out_dir = tmp_path / "replay_fail"
    runner = LiquidityReplayRunner(inputs=inputs, output_dir=out_dir)

    # Patch open in recorder to simulate a genuine disk / I/O failure during flush_replay
    with patch("builtins.open", side_effect=OSError("Disk write failed")):
        result = runner.run()

    assert result.artifact_integrity is ArtifactIntegrity.RECORDER_FAILURE
    assert result.recorder_failure_count >= 1
    assert result.exit_code == 1

    # CLI test under genuine failure
    cli_out_dir = tmp_path / "cli_fail"
    with patch("builtins.open", side_effect=OSError("Disk write failed")):
        cli_exit = replay_main(["--output-dir", str(cli_out_dir)])
    assert cli_exit == 1


# -----------------------------------------------------------------------------
# GATE 38: Canonical recorder flush ordering exactly follows rank tuple
# -----------------------------------------------------------------------------
def test_gate_38_recorder_flush_canonical_ordering(tmp_path: Path):
    out_dir = tmp_path / "recorder_order"
    recorder = LiquidityEventRecorder(mode="replay", output_dir=out_dir)

    # 1. Construct deliberately scrambled set of records with diverse times, event IDs, sequences, ranks
    t1 = LifecycleTransition(
        event_id="evA",
        transition_sequence=1,
        transition_time_ms=10_000,
        previous_state=EventState.OBSERVED,
        next_state=EventState.SWEEP_DETECTED,
        reason_code="OBSERVATION_ADMITTED",
    )
    t2 = LifecycleTransition(
        event_id="evA",
        transition_sequence=2,
        transition_time_ms=12_000,
        previous_state=EventState.SWEEP_DETECTED,
        next_state=EventState.FINALIZED,
        reason_code="RECLAIM_CONFIRMED",
    )
    obs = make_obs("evA", event_time_ms=10_000, detection_time_ms=10_000)
    builder = LiquidityEvidenceBuilder()
    ev_obj = LiquidityEvent(observation=obs, state=EventState.FINALIZED)
    evidence = builder.build(event=ev_obj, outcome_direction=EventClassification.FAILED_BREAKDOWN, as_of_ms=12_000)
    res1 = LiquidityEventResult(
        event_id="evA",
        symbol="BTCUSDT",
        liquidity_side=LiquiditySide.SELL_SIDE,
        classification=EventClassification.FAILED_BREAKDOWN,
        reason_code="RECLAIM_CONFIRMED",
        event_time_ms=10_000,
        detection_time_ms=10_000,
        market_resolution_time_ms=12_000,
        classification_time_ms=12_000,
        source_observation_hash=obs.source_observation_hash,
        evidence=evidence,
        policy_hash="policy_hash_test",
        model_version="1.0.0",
    )
    rej1 = RejectedSweepInput(
        source=SweepSource.SWEEPS_MONITOR_CSV,
        detection_time_ms=11_000,
        reason_code="INVALID_SWEEP_OBSERVATION",
        reason_detail="INVALID_TIMESTAMP",
    )

    rec_t1 = CanonicalRecord(record_type="TRANSITION", payload=t1)
    rec_t2 = CanonicalRecord(record_type="TRANSITION", payload=t2)
    rec_res = CanonicalRecord(record_type="EVENT", payload=res1)
    rec_rej = CanonicalRecord(record_type="REJECTED_INPUT", payload=rej1)

    expected_records = [rec_t1, rec_rej, rec_t2, rec_res]
    expected_canonical_keys = [r.canonical_key for r in expected_records]

    # Explicitly assert canonical_key tuple structure
    for r in expected_records:
        assert r.canonical_key == (
            r.canonical_record_time_ms,
            r.event_id,
            r.transition_sequence,
            r.record_type_rank,
            r.content_hash,
        )

    # Enqueue in scrambled non-canonical order
    scrambled = [rec_res, rec_t2, rec_rej, rec_t1]
    for r in scrambled:
        recorder.enqueue(r)

    # 2. Spy on serialization calls during flush_replay
    observed_flush_keys: list[tuple[Any, ...]] = []
    orig_ser_trans = rec_mod._serialize_transition_row
    orig_ser_event = rec_mod._serialize_event_row
    orig_ser_rej = rec_mod._serialize_rejected_row

    def spy_ser_trans(payload):
        observed_flush_keys.append(CanonicalRecord(record_type="TRANSITION", payload=payload).canonical_key)
        return orig_ser_trans(payload)

    def spy_ser_event(payload):
        observed_flush_keys.append(CanonicalRecord(record_type="EVENT", payload=payload).canonical_key)
        return orig_ser_event(payload)

    def spy_ser_rej(payload):
        observed_flush_keys.append(CanonicalRecord(record_type="REJECTED_INPUT", payload=payload).canonical_key)
        return orig_ser_rej(payload)

    with patch("liquidity_event.recorder._serialize_transition_row", side_effect=spy_ser_trans), \
         patch("liquidity_event.recorder._serialize_event_row", side_effect=spy_ser_event), \
         patch("liquidity_event.recorder._serialize_rejected_row", side_effect=spy_ser_rej):
        recorder.flush_replay()

    assert observed_flush_keys == sorted(expected_canonical_keys)

    # Verify CSV files exist
    assert (out_dir / EVENTS_FILENAME).exists()
    assert (out_dir / TRANSITIONS_FILENAME).exists()
    assert (out_dir / REJECTED_FILENAME).exists()


# -----------------------------------------------------------------------------
# GATE 39: Guardian distinguishes quiet activity from outages
# -----------------------------------------------------------------------------
def test_gate_39_guardian_distinguishes_quiet_from_gapped():
    provider = LiveTradeCoverageProvider(retention_ms=180_000)
    provider.record_trade("BTCUSDT", 10_000, feed_safe=True, sequence_id=1)
    provider.record_trade("BTCUSDT", 20_000, feed_safe=True, sequence_id=2)

    # Quiet interval between contiguous sequences is healthy
    cov_quiet = provider.coverage("BTCUSDT", 12_000, 18_000)
    assert cov_quiet.feed_safe is True
    assert cov_quiet.known_gap is False

    # Gap between sequences
    provider.record_trade("BTCUSDT", 30_000, feed_safe=True, sequence_id=5)  # Missing 3, 4
    cov_gap = provider.coverage("BTCUSDT", 20_000, 30_000)
    assert cov_gap.known_gap is True
    assert cov_gap.valid is False


# -----------------------------------------------------------------------------
# GATE 40: Finalized identity survives restart
# -----------------------------------------------------------------------------
def test_gate_40_identity_restart_durability(tmp_path: Path):
    policy = LiquidityClassificationPolicy()
    builder = LiquidityEvidenceBuilder()

    # -------------------------------------------------------------------------
    # PART A — FINALIZED DUPLICATE
    # -------------------------------------------------------------------------
    db_path_a = str(tmp_path / "part_a.sqlite3")
    auth_a1 = SQLiteIdentityAuthority(db_path_a)
    obs_a = make_obs("gate-40-a", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
    claim_a1 = auth_a1.claim_observation(obs_a)
    assert claim_a1.outcome is IdentityClaimOutcome.NEW

    event_a = LiquidityEvent(observation=obs_a, state=EventState.FINALIZED, classification=EventClassification.FAILED_BREAKDOWN)
    ev_a = builder.build(event=event_a, outcome_direction=EventClassification.FAILED_BREAKDOWN, as_of_ms=20_000)
    res_a = LiquidityEventResult(
        event_id=obs_a.event_id,
        symbol=obs_a.symbol,
        liquidity_side=obs_a.liquidity_side,
        classification=EventClassification.FAILED_BREAKDOWN,
        reason_code="RECLAIM_CONFIRMED",
        event_time_ms=obs_a.event_time_ms,
        detection_time_ms=obs_a.detection_time_ms,
        market_resolution_time_ms=20_000,
        classification_time_ms=20_000,
        source_observation_hash=obs_a.source_observation_hash,
        evidence=ev_a,
        policy_hash=policy.policy_hash,
        model_version=policy.model_version,
    )
    auth_a1.claim_result(res_a)
    auth_a1.close()

    auth_a2 = SQLiteIdentityAuthority(db_path_a)
    try:
        claim_a2 = auth_a2.claim_observation(obs_a)
        assert claim_a2.outcome is IdentityClaimOutcome.DUPLICATE_EXISTING
        persisted = auth_a2.lookup(obs_a.event_id)
        assert persisted is not None
        assert persisted.status == "FINALIZED"
    finally:
        auth_a2.close()

    # -------------------------------------------------------------------------
    # PART B — IMMUTABLE IDENTITY CONFLICT
    # -------------------------------------------------------------------------
    auth_b = SQLiteIdentityAuthority(db_path_a)
    store_b = LiquidityEventStore(policy)
    try:
        altered_obs = make_obs(
            "gate-40-a",
            side=LiquiditySide.SELL_SIDE,
            level=999.0,
            event_time_ms=10_000,
        )
        claim_conflict = auth_b.claim_observation(altered_obs)
        assert claim_conflict.outcome is IdentityClaimOutcome.IDENTITY_CONFLICT

        admit_res = store_b.admit_observation(altered_obs, auth_b)
        assert admit_res.created is False
        assert admit_res.rejection is not None
        assert admit_res.rejection.reason_code == "IDENTITY_CONFLICT"
        assert store_b.active_count("BTCUSDT") == 0
    finally:
        auth_b.close()

    # -------------------------------------------------------------------------
    # PART C — CLAIMED_UNRESOLVED RECOVERY WITH RETAINED HISTORY
    # -------------------------------------------------------------------------
    db_path_c = str(tmp_path / "part_c.sqlite3")
    auth_c1 = SQLiteIdentityAuthority(db_path_c)
    store_c1 = LiquidityEventStore(policy)
    engine_c1 = LiquidityEventEngine(policy=policy, store=store_c1, authority=auth_c1)

    obs_c = make_obs("gate-40-c", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
    engine_c1.on_sweep(obs_c, StaticCoverageProvider(make_safe_coverage()))
    persisted_c = auth_c1.lookup(obs_c.event_id)
    assert persisted_c is not None
    assert persisted_c.status == "CLAIMED_UNRESOLVED"
    auth_c1.close()

    # Session 2: Reopen same DB, preload retained historical trades FIRST, then recover
    auth_c2 = SQLiteIdentityAuthority(db_path_c)
    store_c2 = LiquidityEventStore(policy)
    recovered_results: list[LiquidityEventResult] = []
    recovered_transitions: list[LifecycleTransition] = []
    engine_c2 = LiquidityEventEngine(
        policy=policy,
        store=store_c2,
        authority=auth_c2,
        on_result=lambda r: recovered_results.append(r),
        on_transition=lambda t: recovered_transitions.append(t),
    )

    try:
        # Preload retained historical trades into fresh engine before recovery
        trades_c = [
            make_trade(price=99.90, time_ms=10_500, seq=1, aggressor_side=AggressorSide.SELL),
            make_trade(price=100.10, time_ms=11_000, seq=2, aggressor_side=AggressorSide.BUY),
            make_trade(price=100.10, time_ms=12_000, seq=3, aggressor_side=AggressorSide.BUY),
            make_trade(price=100.10, time_ms=14_000, seq=4, aggressor_side=AggressorSide.BUY),
            make_trade(price=100.10, time_ms=20_000, seq=5, aggressor_side=AggressorSide.BUY),
        ]
        for t in trades_c:
            engine_c2.on_trade(t, make_safe_coverage())

        # The recovery call itself must resolve the event using the preloaded history
        rec_res_c = engine_c2.recover_claimed_unresolved(StaticCoverageProvider(make_safe_coverage()))
        assert engine_c2.telemetry.recovered_events_count == 1
        assert len(rec_res_c) == 1
        assert len(recovered_results) == 1

        assert rec_res_c[0].classification is EventClassification.FAILED_BREAKDOWN
        assert rec_res_c[0].reason_code == "RECLAIM_CONFIRMED"
        assert rec_res_c[0].to_canonical_dict() == recovered_results[0].to_canonical_dict()

        persisted_final = auth_c2.lookup(obs_c.event_id)
        assert persisted_final is not None
        assert persisted_final.status == "FINALIZED"

        # Check durable transition sequence continuity and uniqueness
        persisted_sequences = auth_c2.persisted_transition_sequences(obs_c.event_id)
        assert len(persisted_sequences) == len(set(persisted_sequences))
        assert persisted_sequences == set(range(min(persisted_sequences), max(persisted_sequences) + 1))

        # A second recovery call emits no duplicate result
        rec_res_c2 = engine_c2.recover_claimed_unresolved(StaticCoverageProvider(make_safe_coverage()))
        assert len(rec_res_c2) == 0
        assert len(recovered_results) == 1
    finally:
        auth_c2.close()

    # -------------------------------------------------------------------------
    # PART D — CLAIMED_UNRESOLVED WITHOUT RETAINED HISTORY
    # -------------------------------------------------------------------------
    db_path_d = str(tmp_path / "part_d.sqlite3")
    auth_d1 = SQLiteIdentityAuthority(db_path_d)
    store_d1 = LiquidityEventStore(policy)
    engine_d1 = LiquidityEventEngine(policy=policy, store=store_d1, authority=auth_d1)

    obs_d = make_obs("gate-40-d", side=LiquiditySide.SELL_SIDE, level=100.0, event_time_ms=10_000)
    engine_d1.on_sweep(obs_d, StaticCoverageProvider(make_safe_coverage()))
    auth_d1.close()

    auth_d2 = SQLiteIdentityAuthority(db_path_d)
    store_d2 = LiquidityEventStore(policy)
    results_d: list[LiquidityEventResult] = []
    engine_d2 = LiquidityEventEngine(
        policy=policy,
        store=store_d2,
        authority=auth_d2,
        on_result=lambda r: results_d.append(r),
    )

    try:
        missing_cov = TradeCoverage(
            feed_safe=True,
            known_gap=False,
            buffer_overflow=False,
            unresolved_sequence=False,
            interval_retained=False,
        )
        rec_res_d = engine_d2.recover_claimed_unresolved(StaticCoverageProvider(missing_cov))
        assert len(rec_res_d) == 1
        assert rec_res_d[0].classification is EventClassification.INVALID
        assert rec_res_d[0].reason_code == "INSUFFICIENT_REPLAY_HISTORY"
        persisted_d_final = auth_d2.lookup(obs_d.event_id)
        assert persisted_d_final is not None
        assert persisted_d_final.status == "FINALIZED"

        # Second recovery call produces no duplicate
        rec_res_d2 = engine_d2.recover_claimed_unresolved(StaticCoverageProvider(missing_cov))
        assert len(rec_res_d2) == 0
        assert len(results_d) == 1
    finally:
        auth_d2.close()


# -----------------------------------------------------------------------------
# Acceptance Gate Manifest & Meta-Test
# -----------------------------------------------------------------------------

ACCEPTANCE_GATE_MANIFEST: dict[int, str] = {
    1: "test_liquidity_event_acceptance.py::test_gate_01_failed_breakdown",
    2: "test_liquidity_event_acceptance.py::test_gate_02_bearish_continuation",
    3: "test_liquidity_event_acceptance.py::test_gate_03_failed_breakout",
    4: "test_liquidity_event_acceptance.py::test_gate_04_bullish_continuation",
    5: "test_liquidity_event_acceptance.py::test_gate_05_indeterminate_timeout_after_penetration",
    6: "test_liquidity_event_acceptance.py::test_gate_06_unpenetrated_expires_to_invalid",
    7: "test_liquidity_event_acceptance.py::test_gate_07_duplicate_observation_is_idempotent",
    8: "test_liquidity_event_acceptance.py::test_gate_08_out_of_order_within_reorder_tolerance",
    9: "test_liquidity_event_acceptance.py::test_gate_09_quarantined_late_trade_behind_watermark",
    10: "test_liquidity_event_acceptance.py::test_gate_10_deterministic_timeout_expiry",
    11: "test_liquidity_event_acceptance.py::test_gate_11_compromised_trade_coverage_fails_closed",
    12: "test_liquidity_event_acceptance.py::test_gate_12_optional_depth_does_not_invalidate_price_classification",
    13: "test_liquidity_event_acceptance.py::test_gate_13_finalized_events_are_immutable",
    14: "test_liquidity_event_acceptance.py::test_gate_14_byte_equivalent_deterministic_artifacts",
    15: "test_liquidity_event_acceptance.py::test_gate_15_engine_toggle_scorer_invariance",
    16: "test_liquidity_event_acceptance.py::test_gate_16_execution_disabled_and_paper_trader_isolated",
    17: "test_liquidity_event_acceptance.py::test_gate_17_regime_enforcement_disabled",
    18: "test_liquidity_event_acceptance.py::test_gate_18_legacy_direction_mapping",
    19: "test_liquidity_event_acceptance.py::test_gate_19_neutral_zone_pauses_accumulation",
    20: "test_liquidity_event_acceptance.py::test_gate_20_opposite_evidence_resets_unsatisfied_branch",
    21: "test_liquidity_event_acceptance.py::test_gate_21_equal_sufficient_time_conflict",
    22: "test_liquidity_event_acceptance.py::test_gate_22_late_callback_within_retained_history",
    23: "test_liquidity_event_acceptance.py::test_gate_23_evicted_and_boundary_history_fails_closed",
    24: "test_liquidity_event_acceptance.py::test_gate_24_overlapping_independent_events_remain_separate",
    25: "test_liquidity_event_acceptance.py::test_gate_25_same_side_collision_fails_closed",
    26: "test_liquidity_event_acceptance.py::test_gate_26_policy_hash_and_model_version",
    27: "test_liquidity_event_acceptance.py::test_gate_27_reasons_contradictions_determinism",
    28: "test_liquidity_event_acceptance.py::test_gate_28_forbidden_imports",
    29: "test_liquidity_event_acceptance.py::test_gate_29_recorder_failure_bounded_and_isolated",
    30: "test_liquidity_event_acceptance.py::test_gate_30_live_and_replay_parity",
    31: "test_liquidity_event_acceptance.py::test_gate_31_symbol_isolation",
    32: "test_liquidity_event_acceptance.py::test_gate_32_replay_insufficient_future_coverage",
    33: "test_liquidity_event_acceptance.py::test_gate_33_frozen_bounds_enforced",
    34: "test_liquidity_event_acceptance.py::test_gate_34_event_local_flow_no_global_cvd",
    35: "test_liquidity_event_acceptance.py::test_gate_35_typed_context_adapter",
    36: "test_liquidity_event_acceptance.py::test_gate_36_source_level_id_preservation",
    37: "test_liquidity_event_acceptance.py::test_gate_37_incomplete_artifact_integrity",
    38: "test_liquidity_event_acceptance.py::test_gate_38_recorder_flush_canonical_ordering",
    39: "test_liquidity_event_acceptance.py::test_gate_39_guardian_distinguishes_quiet_from_gapped",
    40: "test_liquidity_event_acceptance.py::test_gate_40_identity_restart_durability",
}


def test_acceptance_gate_manifest_completeness():
    assert set(ACCEPTANCE_GATE_MANIFEST.keys()) == set(range(1, 41))
    current_module = importlib.import_module(__name__)
    for gate_num, node_id in ACCEPTANCE_GATE_MANIFEST.items():
        _, _, fn_name = node_id.partition("::")
        assert hasattr(current_module, fn_name), f"Gate {gate_num} maps to nonexistent function {fn_name}"
        fn = getattr(current_module, fn_name)
        assert callable(fn), f"Gate {gate_num} target {fn_name} is not callable"
