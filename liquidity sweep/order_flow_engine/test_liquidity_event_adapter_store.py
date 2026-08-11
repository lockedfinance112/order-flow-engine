import asyncio
import csv
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from liquidity_event import (
    EventClassification,
    LiquidityClassificationPolicy,
    LiquidityEventResult,
    LiquidityEvidence,
    LiquiditySide,
    canonical_hash,
)
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


def store_api():
    try:
        from liquidity_event.event_store import LiquidityEventStore
    except ImportError:
        pytest.fail("required Phase 1C.1 API is not implemented")
    return LiquidityEventStore


def observation(**overrides):
    aliases = {
        "level": "swept_level",
        "side": "liquidity_side",
        "event_time_ms": "event_time_ms",
    }
    adapted = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(), detection_time_ms=2_000
    ).observation
    values = {aliases.get(key, key): value for key, value in overrides.items()}
    values.setdefault("event_id", "same")
    return replace(adapted, **values)


def final_result(event, market_resolution_time_ms=2_000):
    source = event.observation
    classification_time_ms = max(
        market_resolution_time_ms,
        source.detection_time_ms,
    )
    return LiquidityEventResult(
        event_id=event.event_id,
        symbol=source.symbol,
        liquidity_side=source.liquidity_side,
        classification=EventClassification.INVALID,
        reason_code="TEST_FINALIZATION",
        event_time_ms=source.event_time_ms,
        detection_time_ms=source.detection_time_ms,
        market_resolution_time_ms=market_resolution_time_ms,
        classification_time_ms=classification_time_ms,
        source_observation_hash=source.source_observation_hash,
        evidence=LiquidityEvidence(),
        policy_hash=LiquidityClassificationPolicy().policy_hash,
        model_version="1C.1-v1",
    )


def test_duplicate_callback_returns_same_event_without_duplicate():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)

    first = store.open_event(observation(event_id="same"))
    second = store.open_event(observation(event_id="same"))

    assert first.created is True
    assert second.duplicate is True
    assert second.event is first.event
    assert store.active() == (first.event,)


def test_distinct_levels_and_opposite_sides_remain_separate():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())

    first = store.open_event(observation(level="117250"))
    second = store.open_event(observation(event_id="two", level="117210"))
    third = store.open_event(
        observation(event_id="three", side=LiquiditySide.BUY_SIDE)
    )

    assert first.created and second.created and third.created
    assert tuple(event.event_id for event in store.active()) == ("same", "three", "two")


def test_close_same_side_levels_report_ambiguous_collision():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())

    store.open_event(observation(level="117250.00"))
    result = store.open_event(
        observation(event_id="two", level="117251.00", event_time_ms=1_500)
    )

    assert result.created is True
    assert result.collision_event_ids == ("same", "two")
    assert tuple(event.event_id for event in store.active()) == ("same", "two")


def test_collision_window_boundary_is_inclusive_and_deterministic():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)

    store.open_event(observation(event_id="later", level="100.004", event_time_ms=3_000))
    result = store.open_event(
        observation(event_id="earlier", level="100.000", event_time_ms=1_000)
    )

    assert result.collision_event_ids == ("earlier", "later")


def test_collision_bps_boundary_is_inclusive_and_just_over_is_separate():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()

    exact_store = LiquidityEventStore(policy)
    exact_store.open_event(observation(event_id="base", level="100.000"))
    exact = exact_store.open_event(observation(event_id="exact", level="100.005"))

    over_store = LiquidityEventStore(policy)
    over_store.open_event(observation(event_id="base", level="100.000"))
    just_over = over_store.open_event(
        observation(event_id="over", level="100.005001")
    )

    assert exact.collision_event_ids == ("base", "exact")
    assert just_over.collision_event_ids == ()


def test_level_identity_preserves_source_and_has_deterministic_fallback():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)

    first = observation(event_id="one", level="100.000000004")
    equivalent = observation(event_id="two", level="100.000000003")
    supplied = observation(event_id="three", source_level_id="level-3")
    supplied_other = observation(event_id="four", source_level_id="level-4")

    assert store.level_identity(first) == store.level_identity(equivalent)
    assert store.level_identity(supplied) == "level-3"
    assert store.level_identity(supplied_other) == "level-4"
    assert store.open_event(supplied).created is True
    assert store.open_event(supplied_other).collision_event_ids == ()
    assert store.open_event(observation(event_id="five")).collision_event_ids == ()


def test_active_capacity_rejects_without_unbounded_growth():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = LiquidityEventStore(policy)

    first = store.open_event(observation())
    rejected = store.open_event(observation(event_id="two", level="118000"))

    assert first.created is True
    assert rejected.rejection.reason_code == "EVENT_CAPACITY_REACHED"
    assert tuple(event.event_id for event in store.active()) == ("same",)


def test_active_capacity_is_independent_per_symbol():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = LiquidityEventStore(policy)

    btc = store.open_event(observation(event_id="btc"))
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="200")
    )

    assert btc.created is True
    assert eth.created is True
    assert tuple(event.event_id for event in store.active("BTCUSDT")) == ("btc",)
    assert tuple(event.event_id for event in store.active("ETHUSDT")) == ("eth",)


def test_finalize_moves_matching_result_once_and_bounds_recent_results():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    first = store.open_event(observation(event_id="one")).event
    second = store.open_event(observation(event_id="two", level="200")).event
    first_result = final_result(first, market_resolution_time_ms=3_000)
    second_result = final_result(second, market_resolution_time_ms=2_000)

    store.finalize(first_result)
    store.finalize(first_result)
    store.finalize(second_result)

    assert tuple(event.event_id for event in store.active()) == ()
    assert store.recent() == (second_result,)


def test_recent_retention_eviction_is_independent_per_symbol():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    btc_old = store.open_event(observation(event_id="btc-old")).event
    btc_new = store.open_event(observation(event_id="btc-new", level="200")).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="300")
    ).event
    btc_old_result = final_result(btc_old, market_resolution_time_ms=1_000)
    btc_new_result = final_result(btc_new, market_resolution_time_ms=2_000)
    eth_result = final_result(eth, market_resolution_time_ms=1_500)

    store.finalize(btc_old_result)
    store.finalize(eth_result)
    store.finalize(btc_new_result)

    assert store.recent("BTCUSDT") == (btc_new_result,)
    assert store.recent("ETHUSDT") == (eth_result,)


def test_recent_symbol_filtering_returns_multiple_results_in_canonical_order():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())
    later = store.open_event(
        observation(event_id="later", event_time_ms=2_000)
    ).event
    earlier = store.open_event(
        observation(event_id="earlier", event_time_ms=1_000, level="200")
    ).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="300")
    ).event
    later_result = final_result(later, market_resolution_time_ms=5_000)
    earlier_result = final_result(earlier, market_resolution_time_ms=3_000)
    eth_result = final_result(eth, market_resolution_time_ms=1_000)

    store.finalize(later_result)
    store.finalize(eth_result)
    store.finalize(earlier_result)

    assert store.recent("BTCUSDT") == (earlier_result, later_result)
    assert store.recent() == (earlier_result, later_result, eth_result)


def test_unknown_finalization_is_inert_and_symbol_reads_are_sorted():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())
    btc = store.open_event(observation(event_id="btc", event_time_ms=2_000)).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", event_time_ms=1_000)
    ).event
    unknown = final_result(replace(btc, observation=observation(event_id="unknown")))

    store.finalize(unknown)

    assert tuple(event.event_id for event in store.active()) == ("btc", "eth")
    assert store.active("BTCUSDT") == (btc,)
    assert store.recent("BTCUSDT") == ()
    assert eth.event_id == "eth"


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
