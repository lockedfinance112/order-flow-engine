from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable
from unittest.mock import patch

import pytest

from liquidity_event import (
    EventClassification,
    EventState,
    LifecycleTransition,
    LiquidityClassificationPolicy,
    LiquidityEventEngine,
    LiquidityEventResult,
    LiquidityEventStore,
    LiquidityEvidence,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    SQLiteIdentityAuthority,
    SweepsMonitorAdapter,
    TradeCoverage,
)
from liquidity_event.models import (
    AggressorSide,
    ArtifactIntegrity,
    ConfidenceType,
    DepthObservation,
    RejectedSweepInput,
    SweepSource,
)
from liquidity_event.recorder import (
    EVENTS_FILENAME,
    LiquidityEventRecorder,
    REJECTED_FILENAME,
    TRANSITIONS_FILENAME,
)
from liquidity_event.replay import (
    LiquidityReplayResult,
    LiquidityReplayRunner,
    ReplayInput,
    ReplayTradeCoverageProvider,
    main,
)


def make_trade(
    symbol: str = "BTCUSDT",
    price: float | Decimal = 100.0,
    quantity: float | Decimal = 1.0,
    side: AggressorSide = AggressorSide.BUY,
    time_ms: int = 1_000,
    seq: int = 1,
) -> MarketTrade:
    return MarketTrade(
        symbol=symbol,
        price=Decimal(str(price)) if isinstance(price, float) else price,
        quantity=Decimal(str(quantity)) if isinstance(quantity, float) else quantity,
        aggressor_side=side,
        exchange_time_ms=time_ms,
        sequence_id=seq,
    )


def make_depth(
    symbol: str = "BTCUSDT",
    time_ms: int = 1_000,
    seq: int = 1,
) -> DepthObservation:
    return DepthObservation(
        symbol=symbol,
        bids=((Decimal("99.98"), Decimal("1.0")),),
        asks=((Decimal("100.02"), Decimal("1.0")),),
        exchange_time_ms=time_ms,
        sequence_id=seq,
    )


def make_sweep_callback(
    symbol: str = "BTCUSDT",
    event_time_ms: int = 1_000,
    detection_time_ms: int = 2_000,
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


def drive_live_engine(
    inputs: Iterable[ReplayInput],
    policy: LiquidityClassificationPolicy | None = None,
) -> tuple[tuple[LiquidityEventResult, ...], tuple[LifecycleTransition, ...]]:
    """Helper driving live engine path with production components."""
    sorted_inputs = sorted(inputs, key=lambda x: x.processing_key)
    pol = policy or LiquidityClassificationPolicy()

    # Calculate symbol bounds
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


def test_live_and_replay_produce_identical_event_sequence_and_flow(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        # penetrating trade below level for sell side
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
        # reclaim trades above level
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=13_000, seq=3)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=16_000, seq=4)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=19_000, seq=5)),
        # advance past watermark to settle
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=6)),
    ]

    # Live drive
    live_results, live_transitions = drive_live_engine(inputs)

    # Replay drive
    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "replay_live_parity")
    res = runner.run()

    assert res.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert len(res.results) == 1
    assert len(live_results) == 1

    # Exact comparison
    assert res.results == live_results
    assert res.transitions == live_transitions
    assert res.results[0].classification == live_results[0].classification
    assert res.results[0].evidence == live_results[0].evidence
    assert res.results[0].market_resolution_time_ms == live_results[0].market_resolution_time_ms
    assert res.results[0].classification_time_ms == live_results[0].classification_time_ms


def test_replay_twice_produces_byte_identical_artifacts(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=13_000, seq=3)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=16_000, seq=4)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=19_000, seq=5)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=6)),
    ]

    one = LiquidityReplayRunner(inputs, output_dir=tmp_path / "run_one").run()
    two = LiquidityReplayRunner(inputs, output_dir=tmp_path / "run_two").run()

    assert one.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert two.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert one.artifact_hashes == two.artifact_hashes


def test_replay_twice_produces_identical_artifact_hashes(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=3)),
    ]

    res1 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "hash_one").run()
    res2 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "hash_two").run()

    assert res1.artifact_hashes == res2.artifact_hashes
    assert "liquidity_events.csv" in res1.artifact_hashes
    assert "liquidity_event_transitions.csv" in res1.artifact_hashes
    assert "liquidity_rejected_inputs.csv" in res1.artifact_hashes


def test_sweep_is_presented_at_detection_time_not_event_time():
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=15_000)
    inp = ReplayInput.from_sweep_callback(cb)
    assert inp.processing_time_ms == 15_000
    assert inp.source_rank == 2


def test_missing_detection_time_never_falls_back_to_event_time():
    cb = {
        "timestamp": "1970-01-01T00:00:10Z",
        "symbol": "BTCUSDT",
        "type": "BULLISH",
        "sweep_level": 100.0,
        "sweep_id": "swp-1",
    }
    inp = ReplayInput.from_sweep_callback(cb)
    with pytest.raises(ValueError, match="missing mandatory detection_time_ms"):
        _ = inp.processing_time_ms


def test_trade_depth_sweep_same_timestamp_follow_source_rank():
    t = ReplayInput.from_trade(make_trade(time_ms=10_000, seq=0))
    d = ReplayInput.from_depth(make_depth(time_ms=10_000, seq=0))
    s = ReplayInput.from_sweep_callback(make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000))

    sorted_inputs = sorted([s, d, t], key=lambda x: x.processing_key)
    assert sorted_inputs == [t, d, s]
    assert [x.source_rank for x in sorted_inputs] == [0, 1, 2]


def test_out_of_order_input_iterable_is_stably_canonicalized(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    t1 = ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1))
    s1 = ReplayInput.from_sweep_callback(cb)
    t2 = ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2))
    t3 = ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=3))

    inputs_forward = [t1, s1, t2, t3]
    inputs_reverse = [t3, t2, s1, t1]

    res_f = LiquidityReplayRunner(inputs_forward, output_dir=tmp_path / "f").run()
    res_r = LiquidityReplayRunner(inputs_reverse, output_dir=tmp_path / "r").run()

    assert res_f.artifact_hashes == res_r.artifact_hashes
    assert res_f.results == res_r.results


def test_each_replay_run_uses_fresh_identity_authority(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=3)),
    ]

    r1 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "auth1")
    r2 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "auth2")

    res1 = r1.run()
    res2 = r2.run()

    assert res1.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert res2.artifact_integrity is ArtifactIntegrity.COMPLETE


def test_replay_identity_state_does_not_leak_between_runs(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=3)),
    ]

    res1 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "leak1").run()
    res2 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "leak2").run()

    assert res1.event_count == 1
    assert res2.event_count == 1


def test_incomplete_future_coverage_is_not_certified(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_005, seq=2)),
    ]

    res = LiquidityReplayRunner(inputs, output_dir=tmp_path / "incomplete_cov").run()
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_terminal_advance_is_symbol_local(tmp_path: Path):
    cb_btc = make_sweep_callback(symbol="BTCUSDT", event_time_ms=10_000, detection_time_ms=10_000, level=100.0, sweep_id="btc-1")
    cb_eth = make_sweep_callback(symbol="ETHUSDT", event_time_ms=10_000, detection_time_ms=10_000, level=2000.0, sweep_id="eth-1")

    inputs = [
        ReplayInput.from_sweep_callback(cb_btc),
        ReplayInput.from_sweep_callback(cb_eth),
        # BTC trades advance and reclaim
        ReplayInput.from_trade(make_trade(symbol="BTCUSDT", price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(symbol="BTCUSDT", price=100.10, time_ms=15_000, seq=2)),
        ReplayInput.from_trade(make_trade(symbol="BTCUSDT", price=100.10, time_ms=20_000, seq=3)),
        ReplayInput.from_trade(make_trade(symbol="BTCUSDT", price=100.10, time_ms=25_000, seq=4)),
        ReplayInput.from_trade(make_trade(symbol="BTCUSDT", price=100.10, time_ms=80_000, seq=5)),
        # ETH has only 1 trade at 10,000
        ReplayInput.from_trade(make_trade(symbol="ETHUSDT", price=1999.0, time_ms=10_000, seq=1)),
    ]

    res = LiquidityReplayRunner(inputs, output_dir=tmp_path / "sym_local").run()
    assert len(res.results) == 2
    btc_res = next(r for r in res.results if r.symbol == "BTCUSDT")
    eth_res = next(r for r in res.results if r.symbol == "ETHUSDT")

    # BTC should be resolved
    assert btc_res.classification is EventClassification.FAILED_BREAKDOWN
    # ETH should be invalid due to insufficient future coverage on ETH
    assert eth_res.classification is EventClassification.INVALID
    assert eth_res.reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_replay_does_not_invent_post_recording_trade_coverage(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=2)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "no_invent")
    res = runner.run()

    # Active tracker must not be advanced beyond recording end
    assert res.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_terminal_depth_after_last_trade_does_not_extend_trade_coverage(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        # Trade only at 10,000
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        # Depth arrives far into future at 80,000
        ReplayInput.from_depth(make_depth(time_ms=80_000, seq=1)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "depth_future")
    res = runner.run()

    # Must settle as INVALID due to insufficient future trade coverage, not timeout at 70,000!
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_terminal_depth_only_future_interval_yields_insufficient_future_coverage(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_depth(make_depth(time_ms=20_000, seq=1)),
        ReplayInput.from_depth(make_depth(time_ms=30_000, seq=2)),
        ReplayInput.from_depth(make_depth(time_ms=75_000, seq=3)),
    ]

    res = LiquidityReplayRunner(inputs, output_dir=tmp_path / "depth_only_fut").run()
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_late_sweep_detection_beyond_recorded_trade_horizon_is_not_given_safe_coverage(tmp_path: Path):
    # Sweep detected at 90,000, but trades end at 20,000
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=90_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=20_000, seq=2)),
        ReplayInput.from_sweep_callback(cb),
    ]

    res = LiquidityReplayRunner(inputs, output_dir=tmp_path / "late_sweep").run()
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code in ("MARKET_DATA_UNSAFE", "INSUFFICIENT_FUTURE_COVERAGE", "INSUFFICIENT_REPLAY_HISTORY")


def test_successful_replay_flushes_recorder_exactly_once(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "single_flush")
    orig_flush = LiquidityEventRecorder.flush_replay
    flush_calls: list[Any] = []

    def counting_flush(self):
        flush_calls.append(self)
        return orig_flush(self)

    with patch.object(LiquidityEventRecorder, "flush_replay", counting_flush):
        res = runner.run()
        assert res.artifact_integrity is ArtifactIntegrity.COMPLETE
        assert len(flush_calls) == 1


def test_engine_rejected_inputs_are_recorded_in_replay_artifact(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    # Bad callback with missing fields to trigger adapter/engine rejection
    bad_cb = {
        "timestamp": "1970-01-01T00:00:10Z",
        "symbol": "BTCUSDT",
        "type": "UNKNOWN_DIRECTION",
        "detection_time_ms": 10_000,
    }
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_sweep_callback(bad_cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    out_dir = tmp_path / "rej_art"
    runner = LiquidityReplayRunner(inputs, output_dir=out_dir)
    res = runner.run()

    rej_csv = out_dir / REJECTED_FILENAME
    assert rej_csv.exists()
    content = rej_csv.read_text(encoding="utf-8")
    assert "INVALID_DIRECTION" in content


def test_from_recordings_preserves_exact_trade_decimal_values(tmp_path: Path):
    recording_file = tmp_path / "trades.jsonl.gz"
    trade_line = {
        "e": "aggTrade",
        "s": "BTCUSDT",
        "p": "100.123456789012345678",
        "q": "1.500000000000000001",
        "m": False,
        "T": 10_000,
        "a": 12345,
    }
    with gzip.open(recording_file, "wt", encoding="utf-8") as f:
        f.write(json.dumps(trade_line) + "\n")

    runner = LiquidityReplayRunner.from_recordings(
        market_paths=[recording_file],
        sweep_callbacks=[],
        output_dir=tmp_path / "out",
    )
    assert len(runner.inputs) == 1
    t = runner.inputs[0].trade
    assert isinstance(t.price, Decimal)
    assert isinstance(t.quantity, Decimal)
    assert t.price == Decimal("100.123456789012345678")
    assert t.quantity == Decimal("1.500000000000000001")


def test_from_recordings_trade_hash_matches_direct_canonical_market_trade(tmp_path: Path):
    recording_file = tmp_path / "trades.jsonl"
    trade_line = {
        "e": "aggTrade",
        "s": "BTCUSDT",
        "p": "100.50",
        "q": "2.00",
        "m": False,
        "T": 10_000,
        "a": 123,
    }
    recording_file.write_text(json.dumps(trade_line) + "\n", encoding="utf-8")

    runner = LiquidityReplayRunner.from_recordings(
        market_paths=[recording_file],
        sweep_callbacks=[],
        output_dir=tmp_path / "out",
    )
    direct_trade = MarketTrade(
        symbol="BTCUSDT",
        price=Decimal("100.50"),
        quantity=Decimal("2.00"),
        aggressor_side=AggressorSide.BUY,
        exchange_time_ms=10_000,
        sequence_id=123,
    )
    assert runner.inputs[0].trade.content_hash == direct_trade.content_hash


def test_recorder_failure_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
    ]

    bad_dir = tmp_path / "file_exists"
    bad_dir.write_text("blocking file")
    runner = LiquidityReplayRunner(inputs, output_dir=bad_dir / "sub")
    res = runner.run()

    assert res.artifact_integrity is ArtifactIntegrity.RECORDER_FAILURE
    assert res.exit_code == 1


def test_queue_overflow_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    policy = LiquidityClassificationPolicy(recorder_queue_max_items=1)
    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "q_overflow", policy=policy)
    res = runner.run()

    assert res.artifact_integrity is ArtifactIntegrity.QUEUE_OVERFLOW
    assert res.exit_code == 1


def test_missing_event_row_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "missing_evt")
    # Fault-inject recorder to drop event enqueue
    orig_enqueue = LiquidityEventRecorder.enqueue

    def fault_enqueue(self, record):
        if isinstance(record, LiquidityEventResult):
            return True  # drop without adding to _records
        return orig_enqueue(self, record)

    with patch.object(LiquidityEventRecorder, "enqueue", fault_enqueue):
        res = runner.run()
        assert res.artifact_integrity is ArtifactIntegrity.MISSING_EVENT_ROW
        assert res.exit_code == 1
        assert res.missing_event_row_count > 0


def test_missing_transition_row_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "missing_trans")
    orig_enqueue = LiquidityEventRecorder.enqueue

    def fault_enqueue(self, record):
        if isinstance(record, LifecycleTransition):
            return True  # drop transition
        return orig_enqueue(self, record)

    with patch.object(LiquidityEventRecorder, "enqueue", fault_enqueue):
        res = runner.run()
        assert res.artifact_integrity is ArtifactIntegrity.MISSING_TRANSITION_ROW
        assert res.exit_code == 1
        assert res.missing_transition_row_count > 0


def test_cli_complete_run_returns_zero(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    sweeps_file = tmp_path / "sweeps.json"
    sweeps_file.write_text(json.dumps([cb]), encoding="utf-8")

    trades_file = tmp_path / "trades.jsonl"
    trade1 = {"e": "aggTrade", "s": "BTCUSDT", "p": "99.90", "q": "1.0", "m": False, "T": 10_000, "a": 1}
    trade2 = {"e": "aggTrade", "s": "BTCUSDT", "p": "100.10", "q": "1.0", "m": False, "T": 25_000, "a": 2}
    trades_file.write_text(f"{json.dumps(trade1)}\n{json.dumps(trade2)}\n", encoding="utf-8")

    out_dir = tmp_path / "cli_out_0"
    exit_code = main(["--output-dir", str(out_dir), "--market-recordings", str(trades_file), "--sweeps", str(sweeps_file)])
    assert exit_code == 0


def test_cli_noncomplete_run_returns_one(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    sweeps_file = tmp_path / "sweeps.json"
    sweeps_file.write_text(json.dumps([cb]), encoding="utf-8")

    # Bad output directory (cannot write into file)
    bad_dir = tmp_path / "blocked_file"
    bad_dir.write_text("blocker")

    exit_code = main(["--output-dir", str(bad_dir / "sub"), "--sweeps", str(sweeps_file)])
    assert exit_code == 1


def test_replay_uses_production_engine_and_classifier_path(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=13_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=16_000, seq=3)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=19_000, seq=4)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=5)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "prod_path")
    res = runner.run()

    # Verifies production PriceOutcomeTracker and LiquidityEvidenceBuilder outcomes
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.FAILED_BREAKDOWN
    assert res.results[0].evidence.confidence > 0.0


def test_forbidden_trading_authority_imports_absent():
    with open("liquidity_event/replay.py", "r", encoding="utf-8") as f:
        code = f.read()

    forbidden = [
        "OrderFlowScorer",
        "PaperTrader",
        "scoring",
        "regime",
        "allocation",
        "position",
    ]
    for term in forbidden:
        assert term not in code, f"Forbidden term '{term}' found in liquidity_event/replay.py"
