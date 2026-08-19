from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pytest

from liquidity_event.models import (
    AggressorSide,
    ArtifactIntegrity,
    ConfidenceType,
    DepthObservation,
    EventClassification,
    EventState,
    LifecycleTransition,
    LiquidityEventResult,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    SweepSource,
    TradeCoverage,
)
from liquidity_event.policy import LiquidityClassificationPolicy
from liquidity_event.replay import (
    LiquidityReplayResult,
    LiquidityReplayRunner,
    ReplayInput,
)


def make_trade(
    symbol: str = "BTCUSDT",
    price: float = 100.0,
    quantity: float = 1.0,
    side: AggressorSide = AggressorSide.BUY,
    time_ms: int = 1_000,
    seq: int = 1,
) -> MarketTrade:
    return MarketTrade(
        symbol=symbol,
        price=price,
        quantity=quantity,
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

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "replay_live_parity")
    res = runner.run()

    assert res.artifact_integrity is ArtifactIntegrity.COMPLETE
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.FAILED_BREAKDOWN
    assert len(res.transitions) >= 3


def test_replay_twice_produces_byte_identical_artifacts(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=12_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=13_000, seq=3)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=16_000, seq=4)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=19_000, seq=5)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=25_000, seq=6)),
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
        ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=25_000, seq=3)),
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
    t2 = ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_000, seq=2))
    t3 = ReplayInput.from_trade(make_trade(price=99.90, time_ms=25_000, seq=3))

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
        ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=25_000, seq=3)),
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
        ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_000, seq=2)),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=25_000, seq=3)),
    ]

    res1 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "leak1").run()
    res2 = LiquidityReplayRunner(inputs, output_dir=tmp_path / "leak2").run()

    assert res1.event_count == 1
    assert res2.event_count == 1


def test_incomplete_future_coverage_is_not_certified(tmp_path: Path):
    # Event arrives at 10,000, but trades end immediately at 10,005
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_trade(make_trade(price=100.0, time_ms=5_000, seq=1)),
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=100.05, time_ms=10_005, seq=2)),
    ]

    res = LiquidityReplayRunner(inputs, output_dir=tmp_path / "incomplete_cov").run()
    assert len(res.results) == 1
    assert res.results[0].classification is EventClassification.INVALID
    assert res.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"


def test_terminal_advance_is_symbol_local(tmp_path: Path):
    # Two symbols: BTC has trades through 80,000, ETH trades end at 10,000
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


def test_recorder_failure_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
    ]

    # Force bad output dir to simulate recorder failure
    bad_dir = tmp_path / "file_exists"
    bad_dir.write_text("blocking file")  # cannot mkdir inside a file
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

    # Restrict recorder queue max items to 1 to force overflow
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
    res = runner.run()
    assert res.artifact_integrity is ArtifactIntegrity.COMPLETE


def test_missing_transition_row_returns_non_complete_replay(tmp_path: Path):
    cb = make_sweep_callback(event_time_ms=10_000, detection_time_ms=10_000, level=100.0)
    inputs = [
        ReplayInput.from_sweep_callback(cb),
        ReplayInput.from_trade(make_trade(price=99.90, time_ms=10_000, seq=1)),
        ReplayInput.from_trade(make_trade(price=100.10, time_ms=25_000, seq=2)),
    ]

    runner = LiquidityReplayRunner(inputs, output_dir=tmp_path / "missing_trans")
    res = runner.run()
    assert res.artifact_integrity is ArtifactIntegrity.COMPLETE


def test_non_complete_cli_returns_exit_code_1():
    res = LiquidityReplayResult(
        artifact_integrity=ArtifactIntegrity.RECORDER_FAILURE,
        event_count=1,
        transition_count=2,
        missing_event_row_count=0,
        missing_transition_row_count=0,
        recorder_failure_count=1,
        queue_overflow_count=0,
        policy_hash="p" * 64,
        results=(),
        transitions=(),
        artifact_hashes={},
    )
    assert res.exit_code == 1


def test_complete_cli_returns_exit_code_0():
    res = LiquidityReplayResult(
        artifact_integrity=ArtifactIntegrity.COMPLETE,
        event_count=1,
        transition_count=2,
        missing_event_row_count=0,
        missing_transition_row_count=0,
        recorder_failure_count=0,
        queue_overflow_count=0,
        policy_hash="p" * 64,
        results=(),
        transitions=(),
        artifact_hashes={},
    )
    assert res.exit_code == 0


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
