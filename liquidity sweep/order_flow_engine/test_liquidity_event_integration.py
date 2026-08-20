from __future__ import annotations

import asyncio
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from config import (
    EXECUTION_DISABLED,
    LIQUIDITY_EVENT_ENGINE_ENABLED,
    LIQUIDITY_EVENT_ENFORCEMENT_ENABLED,
    LIQUIDITY_EVENT_OUTPUT_DIR,
    REGIME_ENFORCEMENT_ENABLED,
)
from dashboard import OrderFlowDashboard
from flow_metrics import FlowMetrics
from liquidity_event import (
    AggressorSide,
    DepthObservation,
    EventClassification,
    LiquidityClassificationPolicy,
    LiquidityEventEngine,
    LiquidityEventStore,
    LiquidityEvidenceBuilder,
    MarketTrade,
    SQLiteIdentityAuthority,
    SweepsMonitorAdapter,
    TradeCoverage,
)
from liquidity_event.recorder import LiquidityEventRecorder
from main import LiveTradeCoverageProvider, OrderFlowEngine
from scoring import OrderFlowScorer
from trade_stream import TradeStream

@pytest.fixture(autouse=True)
def auto_cleanup_engines():
    engines = []
    orig_init = OrderFlowEngine.__init__

    def tracking_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        engines.append(self)

    OrderFlowEngine.__init__ = tracking_init
    try:
        yield
    finally:
        OrderFlowEngine.__init__ = orig_init
        for eng in engines:
            if hasattr(eng, "liquidity_identity_authority") and eng.liquidity_identity_authority:
                try:
                    eng.liquidity_identity_authority.close()
                except Exception:
                    pass
            if hasattr(eng, "regime_feed") and eng.regime_feed and hasattr(eng.regime_feed, "bar_store"):
                try:
                    eng.regime_feed.bar_store.close()
                except Exception:
                    pass
            if hasattr(eng, "regime_engine") and eng.regime_engine and hasattr(eng.regime_engine, "bar_store"):
                try:
                    eng.regime_engine.bar_store.close()
                except Exception:
                    pass
        import gc
        gc.collect()


def test_frozen_authority_flags():
    assert LIQUIDITY_EVENT_ENGINE_ENABLED is True
    assert LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False
    assert REGIME_ENFORCEMENT_ENABLED is False
    assert EXECUTION_DISABLED is True
    assert "liquidity_event_artifacts" in LIQUIDITY_EVENT_OUTPUT_DIR


def test_liquidity_event_enforcement_remains_disabled():
    assert config.LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False


def test_regime_enforcement_remains_disabled():
    assert config.REGIME_ENFORCEMENT_ENABLED is False


def test_execution_disabled_and_paper_behavior_unchanged():
    assert config.EXECUTION_DISABLED is True
    engine = OrderFlowEngine()
    assert engine.paper_trader is not None
    assert engine.paper_trader.initial_balance == 10000.0


def test_liquidity_engine_not_passed_to_scorer():
    engine = OrderFlowEngine()
    assert not hasattr(engine.scorer, "liquidity_engine")
    assert not hasattr(engine.scorer, "liquidity_event_engine")
    for key, val in vars(engine.scorer).items():
        assert not isinstance(val, LiquidityEventEngine)


def test_liquidity_engine_not_passed_to_paper_trader():
    engine = OrderFlowEngine()
    for key, val in vars(engine.paper_trader).items():
        assert not isinstance(val, LiquidityEventEngine)


def test_engine_enabled_or_disabled_never_changes_scorer_output():
    metrics1 = FlowMetrics()
    metrics2 = FlowMetrics()
    scorer1 = OrderFlowScorer(metrics1)
    scorer2 = OrderFlowScorer(metrics2)

    raw_trade = {
        "p": "100.50",
        "q": "2.0",
        "m": False,
        "T": 10_000,
        "a": 12345,
        "timestamp": 10.0,
        "price": 100.50,
        "quantity": 2.0,
        "side": "BUY",
    }
    metrics1.add_trade("btcusdt", raw_trade)
    metrics2.add_trade("btcusdt", raw_trade)

    alerts_history = [{"symbol": "btcusdt", "type": "LARGE_TRADE", "timestamp": 10.0}]
    res1 = scorer1.evaluate_sweep("btcusdt", "BULLISH", 100.0, alerts_history, 10.0)
    res2 = scorer2.evaluate_sweep("btcusdt", "BULLISH", 100.0, alerts_history, 10.0)

    assert res1 == res2


def test_trade_feed_creates_canonical_market_trade():
    async def _run():
        engine = OrderFlowEngine()
        if not engine.liquidity_engine:
            pytest.fail("Liquidity engine not instantiated")

        raw_trade = {
            "p": "100.50",
            "q": "2.0",
            "m": False,
            "T": 10_000,
            "a": 12345,
            "timestamp": 10.0,
            "price": 100.50,
            "quantity": 2.0,
            "side": "BUY",
        }
        await engine.trade_queue.put(("btcusdt", raw_trade))

        with patch.object(engine.liquidity_engine, "on_trade", wraps=engine.liquidity_engine.on_trade) as mock_on_trade:
            task = asyncio.create_task(engine._process_trades_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert mock_on_trade.called
            market_trade, coverage = mock_on_trade.call_args[0]
            assert isinstance(market_trade, MarketTrade)
            assert market_trade.symbol == "BTCUSDT"
            assert market_trade.price == Decimal("100.50")
            assert market_trade.quantity == Decimal("2.0")
            assert market_trade.aggressor_side == AggressorSide.BUY
            assert market_trade.sequence_id == 12345
            assert market_trade.exchange_time_ms == 10_000

    asyncio.run(_run())


def test_tradestream_end_to_end_mapping():
    async def _run():
        engine = OrderFlowEngine()
        stream = TradeStream(engine._queue_trade)

        raw_binance_msg = {
            "e": "aggTrade",
            "E": 123456789,
            "s": "BTCUSDT",
            "a": 555666,
            "p": "68500.25",
            "q": "1.750",
            "f": 100,
            "l": 105,
            "T": 123456700,
            "m": True,
        }

        with patch.object(engine.liquidity_engine, "on_trade") as mock_on_trade:
            await stream._handle_message("btcusdt", raw_binance_msg)

            task = asyncio.create_task(engine._process_trades_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert mock_on_trade.called
            m_trade, cov = mock_on_trade.call_args[0]
            assert isinstance(m_trade, MarketTrade)
            assert m_trade.symbol == "BTCUSDT"
            assert m_trade.price == Decimal("68500.25")
            assert m_trade.quantity == Decimal("1.750")
            assert m_trade.aggressor_side == AggressorSide.SELL
            assert m_trade.exchange_time_ms == 123456700
            assert m_trade.sequence_id == 555666

    asyncio.run(_run())


def test_trade_guardian_coverage_is_explicit():
    async def _run():
        engine = OrderFlowEngine()
        raw_trade = {
            "p": "100.50",
            "q": "2.0",
            "m": False,
            "T": 10_000,
            "a": 12345,
            "timestamp": 10.0,
            "price": 100.50,
            "quantity": 2.0,
            "side": "BUY",
        }
        await engine.trade_queue.put(("btcusdt", raw_trade))

        with patch.object(engine.liquidity_engine, "on_trade") as mock_on_trade:
            task = asyncio.create_task(engine._process_trades_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert mock_on_trade.called
            _, cov = mock_on_trade.call_args[0]
            assert hasattr(cov, "feed_safe")
            assert hasattr(cov, "known_gap")
            assert hasattr(cov, "buffer_overflow")

    asyncio.run(_run())


def test_optional_depth_failure_does_not_invalidate_trade_coverage():
    async def _run():
        engine = OrderFlowEngine()
        # Mark local order book invalid while trade stream is healthy
        state = engine.metrics.get_state("btcusdt")
        state.local_book._is_valid = False

        now = time.time()
        now_ms = int(now * 1000)

        raw_trade = {
            "p": "100.50",
            "q": "2.0",
            "m": False,
            "T": now_ms,
            "a": 12345,
            "timestamp": now,
            "price": 100.50,
            "quantity": 2.0,
            "side": "BUY",
        }
        await engine.trade_queue.put(("btcusdt", raw_trade))

        with patch.object(engine.liquidity_engine, "on_trade") as mock_on_trade:
            task = asyncio.create_task(engine._process_trades_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert mock_on_trade.called
            _, cov = mock_on_trade.call_args[0]
            assert cov.feed_safe is True
            assert cov.known_gap is False
            assert cov.valid is True

    asyncio.run(_run())


def test_depth_feed_creates_canonical_depth_observation():
    async def _run():
        engine = OrderFlowEngine()
        raw_depth = {
            "b": [["99.90", "1.5"]],
            "a": [["100.10", "2.5"]],
            "bids": [["99.90", "1.5"]],
            "asks": [["100.10", "2.5"]],
            "E": 10_000,
            "T": 10_000,
            "U": 1,
            "u": 2,
            "pu": 0,
            "exchange_time_ms": 10_000,
            "last_update_id": 999,
        }

        with patch.object(engine.liquidity_engine, "on_depth") as mock_on_depth:
            await engine._handle_depth("btcusdt", raw_depth)
            assert mock_on_depth.called
            depth_obs = mock_on_depth.call_args[0][0]
            assert depth_obs.symbol == "BTCUSDT"
            assert depth_obs.bids[0] == (Decimal("99.90"), Decimal("1.5"))
            assert depth_obs.asks[0] == (Decimal("100.10"), Decimal("2.5"))

    asyncio.run(_run())


def test_live_coverage_provider_does_not_certify_unproven_history():
    provider = LiveTradeCoverageProvider(retention_ms=180_000)
    provider.record_trade("BTCUSDT", exchange_time_ms=10_000, feed_safe=True, known_gap=False)
    provider.record_trade("BTCUSDT", exchange_time_ms=20_000, feed_safe=True, known_gap=False)

    # Before proven min (5000)
    cov_before = provider.coverage("BTCUSDT", 5_000, 8_000)
    assert cov_before.interval_retained is False
    assert cov_before.feed_safe is False

    # After proven max (25000)
    cov_after = provider.coverage("BTCUSDT", 15_000, 25_000)
    assert cov_after.interval_retained is False
    assert cov_after.feed_safe is False

    # Within bounds (12000 - 18000)
    cov_within = provider.coverage("BTCUSDT", 12_000, 18_000)
    assert cov_within.interval_retained is True
    assert cov_within.feed_safe is True


def test_live_coverage_provider_preserves_historical_gap():
    provider = LiveTradeCoverageProvider(retention_ms=180_000)
    provider.record_trade("BTCUSDT", exchange_time_ms=10_000, feed_safe=True, known_gap=False)
    provider.record_trade("BTCUSDT", exchange_time_ms=15_000, feed_safe=False, known_gap=True)
    provider.record_trade("BTCUSDT", exchange_time_ms=20_000, feed_safe=True, known_gap=False)

    # Historical interval overlapping gap
    cov = provider.coverage("BTCUSDT", 12_000, 16_000)
    assert cov.interval_retained is True
    assert cov.feed_safe is False
    assert cov.known_gap is True


def test_end_to_end_trade_sequence_discontinuity_cannot_be_certified_safe():
    async def _run():
        engine = OrderFlowEngine()
        trade1 = {
            "p": "100.0",
            "q": "1.0",
            "m": False,
            "T": 10_000,
            "a": 100,
            "side": "BUY",
            "timestamp": 10.0,
            "price": 100.0,
            "quantity": 1.0,
        }
        trade2 = {
            "p": "100.5",
            "q": "1.0",
            "m": False,
            "T": 15_000,
            "a": 105,  # Missing 101, 102, 103, 104 -> Sequence Gap!
            "side": "BUY",
            "timestamp": 15.0,
            "price": 100.5,
            "quantity": 1.0,
        }
        await engine.trade_queue.put(("btcusdt", trade1))
        await engine.trade_queue.put(("btcusdt", trade2))

        task = asyncio.create_task(engine._process_trades_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        cov = engine.liquidity_coverage_provider.coverage("BTCUSDT", 10_000, 15_000)
        assert cov.known_gap is True
        assert cov.valid is False

    asyncio.run(_run())


def test_evicted_live_coverage_interval_fails_closed():
    policy = LiquidityClassificationPolicy()
    retention_ms = policy.market_buffer_retention_ms  # 180_000 ms
    provider = LiveTradeCoverageProvider(retention_ms=retention_ms)

    provider.record_trade("BTCUSDT", exchange_time_ms=10_000, feed_safe=True, sequence_id=1)
    provider.record_trade("BTCUSDT", exchange_time_ms=20_000, feed_safe=True, sequence_id=2)

    # Evict early interval by advancing beyond retention_ms
    later_time_ms = 20_000 + retention_ms + 10_000  # 210_000 ms
    provider.record_trade("BTCUSDT", exchange_time_ms=later_time_ms, feed_safe=True, sequence_id=3)

    # Query early evicted interval [10_000, 20_000]
    cov_evicted = provider.coverage("BTCUSDT", 10_000, 20_000)
    assert cov_evicted.interval_retained is False
    assert cov_evicted.feed_safe is False
    assert cov_evicted.valid is False


def test_partially_covered_interval_fails_closed():
    provider = LiveTradeCoverageProvider(retention_ms=180_000)
    provider.record_trade("BTCUSDT", exchange_time_ms=100_000, feed_safe=True, sequence_id=1)
    provider.record_trade("BTCUSDT", exchange_time_ms=120_000, feed_safe=True, sequence_id=2)

    # Requested interval starts before retained coverage
    cov_underflow = provider.coverage("BTCUSDT", 90_000, 110_000)
    assert cov_underflow.interval_retained is False
    assert cov_underflow.valid is False

    # Requested interval ends after retained coverage
    cov_overflow = provider.coverage("BTCUSDT", 110_000, 130_000)
    assert cov_overflow.interval_retained is False
    assert cov_overflow.valid is False


def test_late_sweep_cannot_reconstruct_across_unproven_interval():
    provider = LiveTradeCoverageProvider(retention_ms=180_000)
    provider.record_trade("BTCUSDT", exchange_time_ms=20_000, feed_safe=True, known_gap=False)

    # Sweep from timestamp 5000 before recorded trade horizon
    cov = provider.coverage("BTCUSDT", 5_000, 15_000)
    assert cov.interval_retained is False
    assert cov.feed_safe is False


def test_malformed_trade_without_timestamp_is_rejected_and_not_fed():
    async def _run():
        engine = OrderFlowEngine()
        malformed = {
            "p": "100.0",
            "q": "1.0",
            "a": 1,
            # Missing trade_time_ms / T
        }
        await engine.trade_queue.put(("btcusdt", malformed))

        with patch.object(engine.liquidity_engine, "on_trade") as mock_on_trade:
            task = asyncio.create_task(engine._process_trades_loop())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            assert not mock_on_trade.called

    asyncio.run(_run())


def test_malformed_depth_without_timestamp_is_rejected_and_not_fed():
    async def _run():
        engine = OrderFlowEngine()
        malformed = {
            "U": 1,
            "u": 2,
            "pu": 0,
            "b": [["100.0", "1.0"]],
            "a": [["101.0", "1.0"]],
            "bids": [["100.0", "1.0"]],
            "asks": [["101.0", "1.0"]],
            # Missing exchange_time_ms / E / T
            "last_update_id": 10,
        }
        with patch.object(engine.liquidity_engine, "on_depth") as mock_on_depth:
            await engine._handle_depth("btcusdt", malformed)
            assert not mock_on_depth.called

    asyncio.run(_run())


def test_sweep_handler_preserves_existing_scorer_logic():
    async def _run():
        engine = OrderFlowEngine()
        sweep = {
            "sweep_id": "swp-legacy-1",
            "symbol": "BTCUSDT",
            "type": "BULLISH",
            "sweep_level": 100.0,
            "timestamp": "1970-01-01T00:00:10Z",
        }
        with patch.object(engine.scorer, "evaluate_sweep", wraps=engine.scorer.evaluate_sweep) as mock_eval:
            await engine._handle_sweep(sweep)
            assert mock_eval.called

    asyncio.run(_run())


def test_sweep_handler_also_opens_shadow_liquidity_event():
    async def _run():
        engine = OrderFlowEngine()
        sweep = {
            "sweep_id": "swp-shadow-1",
            "symbol": "BTCUSDT",
            "type": "BULLISH",
            "sweep_level": 100.0,
            "timestamp": "1970-01-01T00:00:10Z",
            "source_file_id": "SWEEPS_MONITOR_CSV",
            "source_row_hash": "a" * 64,
            "detection_time_ms": 10_000,
        }
        with patch.object(engine.liquidity_engine, "on_sweep") as mock_on_sweep:
            await engine._handle_sweep(sweep)
            assert mock_on_sweep.called
            obs, _ = mock_on_sweep.call_args[0]
            assert obs.symbol == "BTCUSDT"
            assert obs.swept_level == Decimal("100.0")

    asyncio.run(_run())


def test_event_local_flow_does_not_read_global_cvd():
    engine = OrderFlowEngine()
    state = engine.metrics.get_state("btcusdt")
    state.session_cvd_usdt = 999_999.0
    state.running_cvd_usdt = 888_888.0

    runtime = engine.liquidity_engine.symbol_runtime("BTCUSDT")
    assert runtime is not None


def test_context_enters_only_through_typed_adapter():
    engine = OrderFlowEngine()
    assert hasattr(engine, "context_adapter") or hasattr(engine, "liquidity_context_adapter")


def test_read_only_liquidity_api_payload_has_no_mutation_controls():
    engine = OrderFlowEngine()
    payload = engine._build_liquidity_events_payload()
    assert "active" in payload
    assert "recent" in payload
    assert "engine" in payload
    eng_info = payload["engine"]
    assert eng_info["enabled"] is True
    assert eng_info["enforcement_enabled"] is False

    forbidden_keys = ["mutate", "classify", "set_policy", "trade", "execute", "size", "allocation"]
    for k in forbidden_keys:
        assert k not in payload
        assert k not in eng_info


def test_liquidity_api_get_returns_active_recent_and_engine_telemetry():
    async def _run():
        engine = OrderFlowEngine()
        reader = asyncio.StreamReader()
        reader.feed_data(b"GET /api/liquidity-events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        reader.feed_eof()

        writer = MagicMock()
        written_chunks = []
        writer.write = lambda data: written_chunks.append(data)
        writer.drain = AsyncMock()

        await engine._handle_http_client(reader, writer)
        response_bytes = b"".join(written_chunks)
        assert b"200 OK" in response_bytes
        header, _, body = response_bytes.partition(b"\r\n\r\n")
        data = json.loads(body.decode("utf-8"))
        assert "engine" in data
        assert data["engine"]["enabled"] is True
        assert data["engine"]["enforcement_enabled"] is False

    asyncio.run(_run())


def test_liquidity_api_rejects_post_or_mutation():
    async def _run():
        engine = OrderFlowEngine()
        reader = asyncio.StreamReader()
        reader.feed_data(b"POST /api/liquidity-events HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
        reader.feed_eof()

        writer = MagicMock()
        written_chunks = []
        writer.write = lambda data: written_chunks.append(data)
        writer.drain = AsyncMock()

        await engine._handle_http_client(reader, writer)
        response_bytes = b"".join(written_chunks)
        assert b"404 Not Found" in response_bytes or b"405 Method Not Allowed" in response_bytes

    asyncio.run(_run())


def test_dashboard_labels_confidence_as_uncalibrated():
    dash = OrderFlowDashboard(["btcusdt"])
    html = dash.render_html()
    assert "Uncalibrated score" in html or "uncalibrated" in html.lower()

    forbidden = ["win probability", "confidence probability", "chance of"]
    for term in forbidden:
        assert term not in html.lower()


def test_dashboard_contains_liquidity_events_shadow_section():
    dash = OrderFlowDashboard(["btcusdt"])
    html = dash.render_html()
    assert "Liquidity Events (Shadow)" in html or "liquidity-events" in html.lower()


def test_dashboard_js_fetches_and_renders_liquidity_events():
    dash = OrderFlowDashboard(["btcusdt"])
    html = dash.render_html()
    assert "fetchLiquidityEvents" in html
    assert "updateLiquidityEventsUI" in html
    assert "/api/liquidity-events" in html
    assert "liquidity-active-events" in html
    assert "liquidity-recent-events" in html
    assert "Uncalibrated score:" in html


def test_disabled_mode_starts_without_liquidity_engine():
    with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", False):
        engine = OrderFlowEngine()
        assert engine.liquidity_engine is None


def test_disabled_mode_market_ingestion_remains_operational():
    async def _run():
        with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", False):
            engine = OrderFlowEngine()
            raw_trade = {
                "p": "100.50",
                "q": "2.0",
                "m": False,
                "T": 10_000,
                "a": 1,
                "timestamp": 10.0,
                "price": 100.50,
                "quantity": 2.0,
                "side": "BUY",
            }
            await engine._queue_trade("btcusdt", raw_trade)
            assert not engine.trade_queue.empty()

    asyncio.run(_run())


def test_disabled_mode_api_is_safe():
    with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", False):
        engine = OrderFlowEngine()
        payload = engine._build_liquidity_events_payload()
        assert payload["engine"]["enabled"] is False
        assert payload["active"] == []
        assert payload["recent"] == []


def test_real_two_engine_scorer_invariance():
    async def _run():
        with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", False):
            engine_disabled = OrderFlowEngine()
        with patch("config.LIQUIDITY_EVENT_ENGINE_ENABLED", True):
            engine_enabled = OrderFlowEngine()

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
        engine_disabled.metrics.add_trade("btcusdt", trade_raw)
        engine_enabled.metrics.add_trade("btcusdt", trade_raw)

        alerts = [{"symbol": "btcusdt", "type": "LARGE_TRADE", "timestamp": 10.0}]
        score_dis, det_dis, stat_dis, ob_dis, depth_age_dis = engine_disabled.scorer.evaluate_sweep(
            "btcusdt", "BULLISH", 49950.0, alerts, 10.0
        )
        score_en, det_en, stat_en, ob_en, depth_age_en = engine_enabled.scorer.evaluate_sweep(
            "btcusdt", "BULLISH", 49950.0, alerts, 10.0
        )

        assert (score_dis, det_dis, stat_dis, ob_dis, depth_age_dis) == (score_en, det_en, stat_en, ob_en, depth_age_en)

    asyncio.run(_run())


def test_real_recorder_failure_isolation():
    engine_baseline = OrderFlowEngine()
    engine_failure = OrderFlowEngine()

    # Degrade recorder on failure engine
    if engine_failure.liquidity_recorder:
        engine_failure.liquidity_recorder.telemetry.status = "DEGRADED"
        engine_failure.liquidity_recorder.telemetry.failure_count += 5

    alerts_history = [{"symbol": "btcusdt", "type": "LARGE_TRADE", "timestamp": 10.0}]
    base_res = engine_baseline.scorer.evaluate_sweep("btcusdt", "BULLISH", 100.0, alerts_history, 10.0)
    fail_res = engine_failure.scorer.evaluate_sweep("btcusdt", "BULLISH", 100.0, alerts_history, 10.0)

    assert base_res == fail_res


def test_recorder_failure_classification_exact_isolation():
    async def _run():
        with tempfile.TemporaryDirectory() as dir_b, tempfile.TemporaryDirectory() as dir_f:
            with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", dir_b):
                engine_baseline = OrderFlowEngine()
            with patch("config.LIQUIDITY_EVENT_OUTPUT_DIR", dir_f):
                engine_failure = OrderFlowEngine()

            # Force failure on engine_failure recorder
            if engine_failure.liquidity_recorder:
                engine_failure.liquidity_recorder.telemetry.status = "DEGRADED"
                engine_failure.liquidity_recorder.enqueue = MagicMock(side_effect=RuntimeError("I/O write failure"))

            sweep = {
                "sweep_id": "swp-isolation-1",
                "symbol": "BTCUSDT",
                "type": "BULLISH",
                "sweep_level": 100.0,
                "timestamp": "1970-01-01T00:00:00Z",
                "source_file_id": "SWEEPS_MONITOR_CSV",
                "source_row_hash": "a" * 64,
                "detection_time_ms": 0,
            }
            with patch("time.time", return_value=0.0):
                initial_trade = {"p": "99.98", "q": "1.0", "m": True, "T": 0, "a": 1, "side": "SELL", "timestamp": 0.0, "price": 99.98, "quantity": 1.0}
                await engine_baseline.trade_queue.put(("btcusdt", initial_trade))
                await engine_failure.trade_queue.put(("btcusdt", initial_trade))

                task_b0 = asyncio.create_task(engine_baseline._process_trades_loop())
                task_f0 = asyncio.create_task(engine_failure._process_trades_loop())
                await asyncio.sleep(0.05)
                task_b0.cancel()
                task_f0.cancel()
                try:
                    await task_b0
                except asyncio.CancelledError:
                    pass
                try:
                    await task_f0
                except asyncio.CancelledError:
                    pass

                await engine_baseline._handle_sweep(sweep)
                await engine_failure._handle_sweep(sweep)

                # Drive subsequent trades to resolve both events identically
                trades = [
                    {"p": "100.02", "q": "2.0", "m": False, "T": 1_000, "a": 2, "side": "BUY", "timestamp": 1.0, "price": 100.02, "quantity": 2.0},  # Reclaim
                    {"p": "100.02", "q": "2.0", "m": False, "T": 2_000, "a": 3, "side": "BUY", "timestamp": 2.0, "price": 100.02, "quantity": 2.0},
                    {"p": "100.02", "q": "2.0", "m": False, "T": 4_000, "a": 4, "side": "BUY", "timestamp": 4.0, "price": 100.02, "quantity": 2.0},  # Hold qualified
                    {"p": "100.02", "q": "2.0", "m": False, "T": 6_000, "a": 5, "side": "BUY", "timestamp": 6.0, "price": 100.02, "quantity": 2.0},  # Watermark settlement
                ]
                for t in trades:
                    await engine_baseline.trade_queue.put(("btcusdt", t))
                    await engine_failure.trade_queue.put(("btcusdt", t))

                task_b = asyncio.create_task(engine_baseline._process_trades_loop())
                task_f = asyncio.create_task(engine_failure._process_trades_loop())
                await asyncio.sleep(0.05)
                task_b.cancel()
                task_f.cancel()
                try:
                    await task_b
                except asyncio.CancelledError:
                    pass
                try:
                    await task_f
                except asyncio.CancelledError:
                    pass

            assert len(engine_baseline.recent_liquidity_results) == 1
            assert len(engine_failure.recent_liquidity_results) == 1

            res_b = engine_baseline.recent_liquidity_results[0]
            res_f = engine_failure.recent_liquidity_results[0]

            assert res_b["classification"] == res_f["classification"]
            assert res_b["reason_code"] == res_f["reason_code"]
            assert res_b["market_resolution_time_ms"] == res_f["market_resolution_time_ms"]
            assert res_b["classification_time_ms"] == res_f["classification_time_ms"]
            assert res_b["confidence"] == res_f["confidence"]
            assert res_b["confidence_type"] == res_f["confidence_type"]
            assert res_b["reasons"] == res_f["reasons"]
            assert res_b["contradictions"] == res_f["contradictions"]
            assert res_b["context_coverage"] == res_f["context_coverage"]

            engine_baseline.liquidity_identity_authority.close()
            engine_failure.liquidity_identity_authority.close()

    asyncio.run(_run())


def test_adapter_and_engine_rejections_update_telemetry_without_double_counting():
    async def _run():
        engine = OrderFlowEngine()
        invalid_sweep = {
            "sweep_id": "bad-swp-1",
            "symbol": "BTCUSDT",
            "type": "UNKNOWN_INVALID_SIDE",
            "sweep_level": 100.0,
            "timestamp": "1970-01-01T00:00:10Z",
            "source_file_id": "SWEEPS_MONITOR_CSV",
            "source_row_hash": "a" * 64,
            "detection_time_ms": 10_000,
        }
        await engine._handle_sweep(invalid_sweep)
        payload = engine._build_liquidity_events_payload()
        assert payload["engine"]["rejected_input_count"] == 1

    asyncio.run(_run())


def test_partial_init_recorder_failure_leaves_engine_operational():
    with patch("liquidity_event.recorder.LiquidityEventRecorder.__init__", side_effect=RuntimeError("Disk full")):
        engine = OrderFlowEngine()
        assert engine.liquidity_engine is None
        assert engine.metrics is not None
        assert engine.scorer is not None
        assert engine.paper_trader is not None
        payload = engine._build_liquidity_events_payload()
        assert payload["engine"]["enabled"] is False


def test_partial_init_authority_cleanup_on_later_failure():
    closed_authorities = []
    orig_close = SQLiteIdentityAuthority.close

    def spy_close(self):
        closed_authorities.append(self)
        orig_close(self)

    with patch("liquidity_event.recorder.LiquidityEventRecorder.__init__", side_effect=RuntimeError("Recorder fail")), \
         patch.object(SQLiteIdentityAuthority, "close", spy_close):
        engine = OrderFlowEngine()
        assert len(closed_authorities) >= 1
        assert engine.liquidity_engine is None
        assert engine.liquidity_identity_authority is None


def test_partial_init_recorder_and_authority_cleanup_on_engine_failure():
    flushed_recorders = []
    closed_authorities = []
    orig_flush = LiquidityEventRecorder.flush
    orig_close = SQLiteIdentityAuthority.close

    def spy_flush(self):
        flushed_recorders.append(self)
        orig_flush(self)

    def spy_close(self):
        closed_authorities.append(self)
        orig_close(self)

    with patch("liquidity_event.LiquidityEventEngine.__init__", side_effect=RuntimeError("Engine fail")), \
         patch.object(LiquidityEventRecorder, "flush", spy_flush), \
         patch.object(SQLiteIdentityAuthority, "close", spy_close):
        engine = OrderFlowEngine()
        assert len(flushed_recorders) >= 1
        assert len(closed_authorities) >= 1
        assert engine.liquidity_engine is None


def test_shutdown_flushes_recorder_and_closes_identity_authority():
    async def _run():
        engine = OrderFlowEngine()
        if hasattr(engine, "regime_feed") and engine.regime_feed and hasattr(engine.regime_feed, "bar_store"):
            engine.regime_feed.bar_store.close()
        if hasattr(engine, "regime_engine") and engine.regime_engine and hasattr(engine.regime_engine, "bar_store"):
            engine.regime_engine.bar_store.close()
        recorder_flush_spy = MagicMock(wraps=engine.liquidity_recorder.flush)
        engine.liquidity_recorder.flush = recorder_flush_spy
        auth_close_spy = MagicMock(wraps=engine.liquidity_identity_authority.close)
        engine.liquidity_identity_authority.close = auth_close_spy

        await engine._shutdown_liquidity_event_engine()
        assert recorder_flush_spy.called
        assert auth_close_spy.called

    asyncio.run(_run())


def test_degraded_shutdown_remains_safe():
    async def _run():
        engine = OrderFlowEngine()
        if hasattr(engine, "regime_feed") and engine.regime_feed and hasattr(engine.regime_feed, "bar_store"):
            engine.regime_feed.bar_store.close()
        if hasattr(engine, "regime_engine") and engine.regime_engine and hasattr(engine.regime_engine, "bar_store"):
            engine.regime_engine.bar_store.close()
        if engine.liquidity_recorder:
            engine.liquidity_recorder.telemetry.status = "DEGRADED"

        await engine._shutdown_liquidity_event_engine()

    asyncio.run(_run())


def test_retention_crossing_segment_is_clamped_to_cutoff():
    policy = LiquidityClassificationPolicy()
    retention_ms = policy.market_buffer_retention_ms

    provider = LiveTradeCoverageProvider(retention_ms=retention_ms)
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
    provider.record_trade(
        "BTCUSDT",
        205_000,
        feed_safe=True,
        known_gap=False,
        sequence_id=3,
    )

    cutoff = 205_000 - retention_ms
    assert cutoff == 25_000

    cov_below = provider.coverage("BTCUSDT", cutoff - 1, 205_000)
    assert cov_below.interval_retained is False
    assert cov_below.valid is False

    cov_exact = provider.coverage("BTCUSDT", cutoff, 205_000)
    assert cov_exact.interval_retained is True
    assert cov_exact.feed_safe is True
    assert cov_exact.known_gap is False
    assert cov_exact.unresolved_sequence is False
    assert cov_exact.valid is True

    retained_segments = provider._segments.get("BTCUSDT", [])
    assert len(retained_segments) > 0
    for seg in retained_segments:
        assert seg.start_ms >= cutoff
