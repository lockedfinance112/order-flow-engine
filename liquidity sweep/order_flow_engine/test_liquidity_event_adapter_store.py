import asyncio
import csv
from datetime import datetime, timezone

import pytest

from liquidity_event import LiquidityClassificationPolicy, LiquiditySide, canonical_hash
from sweeps_monitor import SweepsMonitor


def valid_raw(**overrides):
    raw = {
        "timestamp": "1970-01-01T00:00:01Z",
        "symbol": "BTCUSDT",
        "type": "BULLISH",
        "sweep_level": 100.0,
        "sweep_id": "sweep-a",
        "source_file_id": "SWEEPS_MONITOR_CSV",
        "source_row_hash": "a" * 64,
    }
    raw.update(overrides)
    return raw


def adapter_api():
    try:
        from liquidity_event.sweep_adapter import SweepsMonitorAdapter
    except ImportError:
        pytest.fail("required Phase 1C.1 API is not implemented")
    return SweepsMonitorAdapter


def test_bullish_and_bearish_legacy_directions_map_only_in_adapter():
    SweepsMonitorAdapter = adapter_api()
    adapter = SweepsMonitorAdapter(LiquidityClassificationPolicy())

    bullish = adapter.adapt(valid_raw(type="BULLISH"), detection_time_ms=2_000)
    bearish = adapter.adapt(valid_raw(type="BEARISH", sweep_id="b"), detection_time_ms=2_000)

    assert bullish.observation.liquidity_side is LiquiditySide.SELL_SIDE
    assert bearish.observation.liquidity_side is LiquiditySide.BUY_SIDE


def test_adapter_is_deterministic_and_does_not_invent_sweep_price():
    SweepsMonitorAdapter = adapter_api()
    adapter = SweepsMonitorAdapter(LiquidityClassificationPolicy())

    first = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation
    second = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation

    assert first == second
    assert first.event_time_ms == 1_000
    assert first.source_sweep_price is None
    assert first.source_penetration_bps is None


@pytest.mark.parametrize(
    ("raw", "detection_time_ms"),
    [
        (valid_raw(type="UP"), 2_000),
        (valid_raw(type=None), 2_000),
        (valid_raw(sweep_level=0), 2_000),
        (valid_raw(timestamp="not-a-timestamp"), 2_000),
        (valid_raw(), -1),
        (valid_raw(sweep_id=None), 2_000),
        (valid_raw(symbol=None), 2_000),
        (valid_raw(sweep_id=""), 2_000),
        (valid_raw(symbol=""), 2_000),
        (valid_raw(unhashable={"value"}), 2_000),
    ],
)
def test_adapter_rejects_invalid_observations_without_guessing(raw, detection_time_ms):
    SweepsMonitorAdapter = adapter_api()
    result = SweepsMonitorAdapter(LiquidityClassificationPolicy()).adapt(raw, detection_time_ms)

    assert result.observation is None
    assert result.rejected.reason_code == "INVALID_SWEEP_OBSERVATION"


def test_adapter_preserves_supplied_source_level_id():
    SweepsMonitorAdapter = adapter_api()
    result = SweepsMonitorAdapter(LiquidityClassificationPolicy()).adapt(
        valid_raw(source_level_id="detector-level-42"), detection_time_ms=2_000
    )

    assert result.observation.source_level_id == "detector-level-42"


def test_monitor_callback_provenance_is_stable_for_replay_and_live_rows(tmp_path):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = [timestamp, "BTCUSDT", "BULLISH", "100.0", "sweep-a", "", "", "RAW_SWEEP"]
    csv_path = tmp_path / "sweeps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "symbol", "type", "sweep_level", "sweep_id", "", "", "state"])
        writer.writerow(row)

    callbacks = []

    async def callback(sweep):
        callbacks.append(sweep)

    async def exercise_monitor():
        monitor = SweepsMonitor(callback)
        monitor.csv_path = str(csv_path)
        await monitor._replay_recent_sweeps()
        monitor.last_position = 0
        monitor.is_running = True
        task = asyncio.create_task(monitor._monitor_loop())
        while len(callbacks) < 2:
            await asyncio.sleep(0.01)
        monitor.is_running = False
        await task

    asyncio.run(exercise_monitor())

    assert callbacks[0] == callbacks[1]
    if "source_file_id" not in callbacks[0]:
        pytest.fail("required Phase 1C.1 API is not implemented")
    assert callbacks[0]["source_file_id"] == "SWEEPS_MONITOR_CSV"
    assert callbacks[0]["source_row_hash"] == canonical_hash(row)
    assert callbacks[0]["source_level_id"] is None
    assert callbacks[0]["detector_version"] is None
    assert callbacks[0]["source_sweep_price"] is None
    assert callbacks[0]["source_penetration_bps"] is None
    assert {"timestamp", "symbol", "type", "sweep_level", "sweep_id"}.issubset(callbacks[0])
