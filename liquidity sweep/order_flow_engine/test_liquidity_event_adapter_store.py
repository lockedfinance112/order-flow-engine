import asyncio
import csv
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from liquidity_event import (
    EventClassification,
    EventState,
    IdentityAlreadyExists,
    IdentityAuthorityUnavailable,
    IdentityClaimOutcome,
    IdentityClaimResult,
    LifecycleTransition,
    LiquidityClassificationPolicy,
    LiquidityEventResult,
    LiquidityEvidence,
    LiquiditySide,
    SQLiteIdentityAuthority,
    canonical_hash,
    canonical_json,
    recover_claimed_unresolved,
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


def broken_sqlite_execute(*args, **kwargs):
    raise RuntimeError("sqlite unavailable")


def new_claim(event_id="same"):
    return IdentityClaimResult(IdentityClaimOutcome.NEW, event_id, None, None)


def duplicate_claim(event_id="same"):
    return IdentityClaimResult(
        IdentityClaimOutcome.DUPLICATE_EXISTING, event_id, None, None
    )


def conflict_claim(event_id="same"):
    return IdentityClaimResult(
        IdentityClaimOutcome.IDENTITY_CONFLICT,
        event_id,
        None,
        "same event_id has different immutable identity or provenance hash",
    )


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


def lifecycle_transition(event_id="same", transition_sequence=0):
    previous = None if transition_sequence == 0 else EventState.OBSERVED
    next_state = EventState.OBSERVED if transition_sequence == 0 else EventState.UNRESOLVED
    return LifecycleTransition(
        event_id=event_id,
        previous_state=previous,
        next_state=next_state,
        transition_time_ms=2_000 + transition_sequence,
        transition_sequence=transition_sequence,
        reason_code=f"TEST_TRANSITION_{transition_sequence}",
    )


def finalized_result(event_id="same", policy=None):
    source = observation(event_id=event_id)
    policy = policy or LiquidityClassificationPolicy()
    return LiquidityEventResult(
        event_id=event_id,
        symbol=source.symbol,
        liquidity_side=source.liquidity_side,
        classification=EventClassification.INVALID,
        reason_code="TEST_FINALIZATION",
        event_time_ms=source.event_time_ms,
        detection_time_ms=source.detection_time_ms,
        market_resolution_time_ms=2_000,
        classification_time_ms=2_000,
        source_observation_hash=source.source_observation_hash,
        evidence=LiquidityEvidence(),
        policy_hash=policy.policy_hash,
        model_version=policy.model_version,
    )


def complete_history_for(event):
    return {
        "complete": True,
        "transitions": (
            lifecycle_transition(event_id=event.event_id, transition_sequence=0),
            lifecycle_transition(event_id=event.event_id, transition_sequence=1),
            lifecycle_transition(event_id=event.event_id, transition_sequence=2),
        ),
    }


def missing_history_for(event):
    policy = LiquidityClassificationPolicy()
    return {
        "complete": False,
        "result": LiquidityEventResult(
            event_id=event.event_id,
            symbol=event.symbol,
            liquidity_side=event.liquidity_side,
            classification=EventClassification.INVALID,
            reason_code="INSUFFICIENT_REPLAY_HISTORY",
            event_time_ms=event.event_time_ms,
            detection_time_ms=event.detection_time_ms,
            market_resolution_time_ms=event.detection_time_ms,
            classification_time_ms=event.detection_time_ms,
            source_observation_hash=event.source_observation_hash,
            evidence=LiquidityEvidence(),
            policy_hash=policy.policy_hash,
            model_version=policy.model_version,
        ),
    }


def final_result(event, market_resolution_time_ms=2_000, policy=None):
    source = event.observation
    policy = policy or LiquidityClassificationPolicy()
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
        policy_hash=policy.policy_hash,
        model_version=policy.model_version,
    )


def result_with_mismatch(event, field, value):
    result = final_result(event, market_resolution_time_ms=3_000)
    changes = {field: value}
    if field == "detection_time_ms":
        changes["classification_time_ms"] = max(
            result.market_resolution_time_ms,
            value,
        )
    return replace(result, **changes)


def result_with_invalid_timing(event):
    result = final_result(event, market_resolution_time_ms=3_000)
    object.__setattr__(result, "classification_time_ms", 2_999)
    return result


def test_sqlite_identity_claim_is_exact_and_survives_restart(tmp_path):
    db_path = tmp_path / "identity.sqlite3"
    first = SQLiteIdentityAuthority(db_path)
    assert first.claim_observation(observation(event_id="same")).outcome is IdentityClaimOutcome.NEW
    first.close()

    restarted = SQLiteIdentityAuthority(db_path)
    result = restarted.claim_observation(observation(event_id="same"))
    assert result.outcome is IdentityClaimOutcome.DUPLICATE_EXISTING
    assert result.event_id == "same"


def test_identity_conflict_fails_closed_for_same_event_id(tmp_path):
    authority = SQLiteIdentityAuthority(tmp_path / "identity.sqlite3")
    assert authority.claim_observation(observation(event_id="same", level="117250.00")).outcome is IdentityClaimOutcome.NEW
    conflict = authority.claim_observation(observation(event_id="same", level="117251.00"))
    assert conflict.outcome is IdentityClaimOutcome.IDENTITY_CONFLICT
    assert "immutable identity" in conflict.conflict_reason


def test_finalized_identity_survives_eviction_restart_and_conflict_fails_closed(tmp_path):
    db_path = tmp_path / "identity.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=0)
    store = store_api()(policy)
    event = observation(event_id="same", level="117250.00")
    claim = authority.claim_observation(event)
    opened = store.open_event(event, claim)
    final = finalized_result(event_id="same", policy=policy)
    authority.claim_result(final)
    store.finalize(final)
    assert claim.outcome is IdentityClaimOutcome.NEW
    assert opened.event.event_id not in {result.event_id for result in store.recent()}
    authority.close()

    restarted = SQLiteIdentityAuthority(db_path)
    duplicate = restarted.claim_observation(event)
    conflict = restarted.claim_observation(observation(event_id="same", level="117251.00"))
    assert duplicate.outcome is IdentityClaimOutcome.DUPLICATE_EXISTING
    assert conflict.outcome is IdentityClaimOutcome.IDENTITY_CONFLICT


def test_transition_and_result_uniqueness_are_hard_constraints(tmp_path):
    authority = SQLiteIdentityAuthority(tmp_path / "identity.sqlite3")
    event = observation(event_id="same")
    authority.claim_observation(event)
    transition = lifecycle_transition(event_id="same", transition_sequence=0)
    result = finalized_result(event_id="same")
    authority.claim_transition(transition)
    authority.claim_result(result)
    with pytest.raises(IdentityAlreadyExists):
        authority.claim_transition(transition)
    with pytest.raises(IdentityAlreadyExists):
        authority.claim_result(result)


def test_orphan_transition_and_result_claims_fail_closed(tmp_path):
    transition_authority = SQLiteIdentityAuthority(tmp_path / "transition.sqlite3")
    with pytest.raises(IdentityAuthorityUnavailable):
        transition_authority.claim_transition(
            lifecycle_transition(event_id="orphan", transition_sequence=0)
        )
    assert transition_authority.lookup("orphan") is None
    assert transition_authority.persisted_transition_sequences("orphan") == set()

    result_authority = SQLiteIdentityAuthority(tmp_path / "result.sqlite3")
    with pytest.raises(IdentityAuthorityUnavailable):
        result_authority.claim_result(finalized_result(event_id="orphan"))
    assert result_authority.lookup("orphan") is None


def test_claimed_unresolved_restart_recovery_emits_only_missing_transition_keys(tmp_path):
    db_path = tmp_path / "identity.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)
    event = observation(event_id="same")
    authority.claim_observation(event)
    authority.claim_transition(lifecycle_transition(event_id="same", transition_sequence=0))
    authority.close()

    restarted = SQLiteIdentityAuthority(db_path)
    pending = restarted.pending_unresolved()
    assert [item.event_id for item in pending] == ["same"]
    assert restarted.persisted_transition_sequences("same") == {0}
    recovery = recover_claimed_unresolved(
        authority=restarted,
        pending_identity=pending[0],
        retained_history=complete_history_for(event),
    )
    assert [transition.transition_sequence for transition in recovery.emitted_transitions] == [1, 2]


def test_claimed_unresolved_restart_without_history_finalizes_once_as_insufficient_replay_history(tmp_path):
    db_path = tmp_path / "identity.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)
    event = observation(event_id="same")
    authority.claim_observation(event)
    authority.close()

    restarted = SQLiteIdentityAuthority(db_path)
    recovery = recover_claimed_unresolved(
        authority=restarted,
        pending_identity=restarted.pending_unresolved()[0],
        retained_history=missing_history_for(event),
    )
    assert recovery.result.reason_code == "INSUFFICIENT_REPLAY_HISTORY"
    with pytest.raises(IdentityAlreadyExists):
        restarted.claim_result(recovery.result)


def test_authority_failure_prevents_new_event(monkeypatch, tmp_path):
    authority = SQLiteIdentityAuthority(tmp_path / "identity.sqlite3")
    monkeypatch.setattr(authority, "_execute", broken_sqlite_execute)
    with pytest.raises(IdentityAuthorityUnavailable):
        authority.claim_observation(observation(event_id="same"))


def test_duplicate_callback_uses_identity_authority_without_reopening_event():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)
    first_observation = observation(event_id="same")

    first = store.open_event(first_observation, new_claim("same"))
    second = store.open_event(first_observation, duplicate_claim("same"))

    assert first.created is True
    assert second.duplicate is True
    assert second.event is first.event
    assert tuple(event.event_id for event in store.active()) == ("same",)


def test_identity_conflict_claim_fails_closed_without_opening_event():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())

    result = store.open_event(
        observation(event_id="conflict"),
        conflict_claim("conflict"),
    )

    assert result.event is None
    assert result.created is False
    assert result.duplicate is False
    assert result.collision_event_ids == ()
    assert result.rejection.reason_code == "IDENTITY_CONFLICT"
    assert store.active() == ()


def test_distinct_levels_and_opposite_sides_remain_separate():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())

    first = store.open_event(observation(level="117250"), new_claim("same"))
    second = store.open_event(
        observation(event_id="two", level="117210"), new_claim("two")
    )
    third = store.open_event(
        observation(event_id="three", side=LiquiditySide.BUY_SIDE),
        new_claim("three"),
    )

    assert first.created and second.created and third.created
    assert tuple(event.event_id for event in store.active()) == ("same", "three", "two")


def test_close_same_side_levels_report_ambiguous_collision():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())

    store.open_event(observation(level="117250.00"), new_claim("same"))
    result = store.open_event(
        observation(event_id="two", level="117251.00", event_time_ms=1_500),
        new_claim("two"),
    )

    assert result.created is True
    assert result.collision_event_ids == ("same", "two")
    assert tuple(event.event_id for event in store.active()) == ("same", "two")


def test_collision_window_boundary_is_inclusive_and_deterministic():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)

    store.open_event(
        observation(event_id="later", level="100.004", event_time_ms=3_000),
        new_claim("later"),
    )
    result = store.open_event(
        observation(event_id="earlier", level="100.000", event_time_ms=1_000),
        new_claim("earlier"),
    )

    assert result.collision_event_ids == ("earlier", "later")


def test_collision_bps_boundary_is_inclusive_and_just_over_is_separate():
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()

    exact_store = LiquidityEventStore(policy)
    exact_store.open_event(
        observation(event_id="base", level="100.000"), new_claim("base")
    )
    exact = exact_store.open_event(
        observation(event_id="exact", level="100.005"), new_claim("exact")
    )

    over_store = LiquidityEventStore(policy)
    over_store.open_event(
        observation(event_id="base", level="100.000"), new_claim("base")
    )
    just_over = over_store.open_event(
        observation(event_id="over", level="100.005001"),
        new_claim("over"),
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
    assert store.open_event(supplied, new_claim("three")).created is True
    assert store.open_event(supplied_other, new_claim("four")).collision_event_ids == ()
    assert store.open_event(
        observation(event_id="five"), new_claim("five")
    ).collision_event_ids == ()


def test_active_capacity_rejects_without_unbounded_growth():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = LiquidityEventStore(policy)

    first = store.open_event(observation(), new_claim("same"))
    rejected = store.open_event(
        observation(event_id="two", level="118000"), new_claim("two")
    )

    assert first.created is True
    assert rejected.rejection.reason_code == "EVENT_CAPACITY_REACHED"
    assert tuple(event.event_id for event in store.active()) == ("same",)


def test_active_capacity_is_independent_per_symbol():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = LiquidityEventStore(policy)

    btc = store.open_event(observation(event_id="btc"), new_claim("btc"))
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="200"),
        new_claim("eth"),
    )

    assert btc.created is True
    assert eth.created is True
    assert tuple(event.event_id for event in store.active("BTCUSDT")) == ("btc",)
    assert tuple(event.event_id for event in store.active("ETHUSDT")) == ("eth",)


def test_active_returns_defensive_snapshots_for_events_and_transitions():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())
    store.open_event(observation(event_id="snapshot"), new_claim("snapshot"))

    exposed = store.active()[0]
    exposed.state = EventState.FINALIZED
    exposed.transitions.append(
        lifecycle_transition(event_id="snapshot", transition_sequence=0)
    )

    fresh = store.active()[0]
    assert fresh.state is EventState.OBSERVED
    assert fresh.transitions == []


def test_finalize_moves_matching_result_once_and_bounds_recent_results():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    first = store.open_event(observation(event_id="one"), new_claim("one")).event
    second = store.open_event(
        observation(event_id="two", level="200"), new_claim("two")
    ).event
    first_result = final_result(
        first,
        market_resolution_time_ms=3_000,
        policy=policy,
    )
    second_result = final_result(
        second,
        market_resolution_time_ms=2_000,
        policy=policy,
    )

    store.finalize(first_result)
    store.finalize(first_result)
    store.finalize(second_result)

    assert tuple(event.event_id for event in store.active()) == ()
    assert store.recent() == (second_result,)


def test_finalized_duplicate_after_eviction_uses_duplicate_claim_not_tombstone():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=0)
    store = LiquidityEventStore(policy)
    source = observation(event_id="finalized")
    event = store.open_event(source, new_claim("finalized")).event
    result = final_result(event, policy=policy)

    store.finalize(result)
    duplicate = store.open_event(source, duplicate_claim("finalized"))
    reopened = store.open_event(source, new_claim("finalized"))

    assert duplicate.event is None
    assert duplicate.created is False
    assert duplicate.duplicate is True
    assert duplicate.collision_event_ids == ()
    assert duplicate.rejection is None
    assert reopened.created is True
    assert tuple(event.event_id for event in store.active()) == ("finalized",)
    assert store.recent() == ()


def test_duplicate_claim_wins_over_capacity_without_opening():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    active = store.open_event(
        observation(event_id="active", level="200"), new_claim("active")
    )
    duplicate = store.open_event(
        observation(event_id="finalized"), duplicate_claim("finalized")
    )

    assert active.created is True
    assert duplicate.event is None
    assert duplicate.created is False
    assert duplicate.duplicate is True
    assert duplicate.rejection is None
    assert tuple(event.event_id for event in store.active()) == ("active",)


def test_volatile_store_does_not_own_finalized_id_tombstones():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    btc_evicted_observation = observation(event_id="btc-evicted")
    btc_evicted = store.open_event(
        btc_evicted_observation, new_claim("btc-evicted")
    ).event
    btc_retained = store.open_event(
        observation(event_id="btc-retained", level="200"),
        new_claim("btc-retained"),
    ).event
    eth_retained = store.open_event(
        observation(event_id="eth-retained", symbol="ETHUSDT", level="300"),
        new_claim("eth-retained"),
    ).event

    store.finalize(final_result(btc_evicted, policy=policy))
    store.finalize(final_result(eth_retained, policy=policy))
    store.finalize(final_result(btc_retained, policy=policy))

    evicted = store.open_event(
        btc_evicted_observation, duplicate_claim("btc-evicted")
    )
    retained_new_claim = store.open_event(
        btc_retained.observation, new_claim("btc-retained")
    )

    assert evicted.event is None and evicted.duplicate is True
    assert retained_new_claim.created is True
    assert tuple(result.event_id for result in store.recent("BTCUSDT")) == (
        "btc-retained",
    )
    assert tuple(result.event_id for result in store.recent("ETHUSDT")) == (
        "eth-retained",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", "other-event"),
        ("symbol", "ETHUSDT"),
        ("liquidity_side", LiquiditySide.BUY_SIDE),
        ("source_observation_hash", "b" * 64),
        ("event_time_ms", 1_001),
        ("detection_time_ms", 3_001),
        ("policy_hash", "c" * 64),
        ("model_version", "other-model"),
    ],
)
def test_finalize_rejects_each_nonmatching_result_field_without_mutation(field, value):
    LiquidityEventStore = store_api()
    policy = LiquidityClassificationPolicy()
    store = LiquidityEventStore(policy)
    event = store.open_event(observation(event_id="strict"), new_claim("strict")).event
    valid = final_result(event, market_resolution_time_ms=3_000)

    mismatched = result_with_mismatch(event, field, value)
    if field == "event_id":
        store.finalize(mismatched)
    else:
        with pytest.raises(ValueError, match="does not match active event"):
            store.finalize(mismatched)

    assert tuple(item.event_id for item in store.active()) == ("strict",)
    assert store.recent() == ()

    store.finalize(valid)
    store.finalize(valid)

    assert store.active() == ()
    assert store.recent() == (valid,)


def test_finalize_rejects_invalid_model_timing_without_mutation():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())
    event = store.open_event(
        observation(event_id="invalid-timing"), new_claim("invalid-timing")
    ).event

    with pytest.raises(ValueError, match="does not match active event"):
        store.finalize(result_with_invalid_timing(event))

    assert tuple(item.event_id for item in store.active()) == ("invalid-timing",)
    assert store.recent() == ()


def test_recent_retention_eviction_is_independent_per_symbol():
    LiquidityEventStore = store_api()
    policy = replace(LiquidityClassificationPolicy(), recent_final_events_per_symbol=1)
    store = LiquidityEventStore(policy)
    btc_old = store.open_event(
        observation(event_id="btc-old"), new_claim("btc-old")
    ).event
    btc_new = store.open_event(
        observation(event_id="btc-new", level="200"), new_claim("btc-new")
    ).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="300"),
        new_claim("eth"),
    ).event
    btc_old_result = final_result(
        btc_old,
        market_resolution_time_ms=1_000,
        policy=policy,
    )
    btc_new_result = final_result(
        btc_new,
        market_resolution_time_ms=2_000,
        policy=policy,
    )
    eth_result = final_result(
        eth,
        market_resolution_time_ms=1_500,
        policy=policy,
    )

    store.finalize(btc_old_result)
    store.finalize(eth_result)
    store.finalize(btc_new_result)

    assert store.recent("BTCUSDT") == (btc_new_result,)
    assert store.recent("ETHUSDT") == (eth_result,)


def test_recent_symbol_filtering_returns_multiple_results_in_canonical_order():
    LiquidityEventStore = store_api()
    store = LiquidityEventStore(LiquidityClassificationPolicy())
    later = store.open_event(
        observation(event_id="later", event_time_ms=2_000),
        new_claim("later"),
    ).event
    earlier = store.open_event(
        observation(event_id="earlier", event_time_ms=1_000, level="200"),
        new_claim("earlier"),
    ).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", level="300"),
        new_claim("eth"),
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
    btc = store.open_event(
        observation(event_id="btc", event_time_ms=2_000), new_claim("btc")
    ).event
    eth = store.open_event(
        observation(event_id="eth", symbol="ETHUSDT", event_time_ms=1_000),
        new_claim("eth"),
    ).event
    unknown = final_result(replace(btc, observation=observation(event_id="unknown")))

    store.finalize(unknown)

    assert tuple(event.event_id for event in store.active()) == ("btc", "eth")
    assert tuple(event.event_id for event in store.active("BTCUSDT")) == ("btc",)
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


def test_adapter_event_id_uses_frozen_canonical_identity_payload():
    observation = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(), detection_time_ms=2_000
    ).observation

    assert observation.event_id == (
        "d8aad5699389f81a034433669a9bf97dea0a1ff9c66ef40e05ace14dfaef270e"
    )
    assert observation.source_event_id == "sweep-a"
    assert observation.event_id != observation.source_event_id


def test_adapter_event_id_distinguishes_frozen_identity_components():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    base = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation
    same_legacy_id_variants = (
        valid_raw(symbol="ETHUSDT"),
        valid_raw(type="BEARISH"),
        valid_raw(timestamp="1970-01-01T00:00:02Z"),
        valid_raw(sweep_level="100.00000001"),
    )

    variant_ids = {
        adapter.adapt(raw, detection_time_ms=2_000).observation.event_id
        for raw in same_legacy_id_variants
    }
    different_source_id = adapter.adapt(
        valid_raw(sweep_id="sweep-b"), detection_time_ms=2_000
    ).observation

    assert base.event_id not in variant_ids
    assert len(variant_ids) == len(same_legacy_id_variants)
    assert different_source_id.event_id != base.event_id
    assert different_source_id.source_event_id == "sweep-b"


def test_adapter_event_id_matches_for_semantically_identical_callbacks():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    base = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation
    equivalent_raw = valid_raw(
        symbol="btcusdt",
        sweep_level="100.000000004",
        source_row_hash="b" * 64,
        detector_version="legacy-v2",
    )
    equivalent_raw.pop("source_file_id")
    equivalent = adapter.adapt(equivalent_raw, detection_time_ms=9_000).observation

    assert equivalent.event_id == base.event_id
    assert equivalent.source_event_id == base.source_event_id == "sweep-a"


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


def test_adapter_rejects_absolute_source_file_id_without_path_provenance():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(source_file_id=r"C:\\Users\\someone\\sweeps.csv"),
        detection_time_ms=2_000,
    )

    assert result.observation is None
    assert result.rejected.reason_detail == "INVALID_SOURCE_FILE_ID"
    assert result.rejected.source_file_id is None
    assert result.rejected.source_observation_hash is None
    assert "Users" not in canonical_json(result.rejected.to_canonical_dict())


def test_adapter_object_timestamp_rejection_is_byte_deterministic():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    first_value = object()
    second_value = object()

    first = adapter.adapt(
        valid_raw(timestamp=first_value), detection_time_ms=2_000
    ).rejected
    second = adapter.adapt(
        valid_raw(timestamp=second_value), detection_time_ms=2_000
    ).rejected

    first_bytes = canonical_json(first.to_canonical_dict()).encode("ascii")
    second_bytes = canonical_json(second.to_canonical_dict()).encode("ascii")
    assert first.reason_detail == second.reason_detail == "INVALID_TIMESTAMP"
    assert first_bytes == second_bytes
    assert b"0x" not in first_bytes


def test_adapter_extreme_finite_level_returns_deterministic_rejection():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = []

    for _ in range(2):
        try:
            result = adapter.adapt(
                valid_raw(sweep_level="1e100"), detection_time_ms=2_000
            )
        except Exception as exc:
            pytest.fail(f"adapter escaped instead of rejecting: {type(exc).__name__}")
        rejections.append(result.rejected)

    first_bytes = canonical_json(rejections[0].to_canonical_dict()).encode("ascii")
    second_bytes = canonical_json(rejections[1].to_canonical_dict()).encode("ascii")
    assert rejections[0].reason_detail == "INVALID_SWEPT_LEVEL"
    assert rejections[1].reason_detail == "INVALID_SWEPT_LEVEL"
    assert first_bytes == second_bytes


def test_adapter_level_that_normalizes_to_zero_returns_deterministic_rejection():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = [
        adapter.adapt(
            valid_raw(sweep_level="1e-100"), detection_time_ms=2_000
        ).rejected
        for _ in range(2)
    ]

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "INVALID_SWEPT_LEVEL"
        for rejection in rejections
    )
    assert rejection_bytes[0] == rejection_bytes[1]


def test_adapter_accepts_smallest_canonical_positive_level():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(sweep_level="0.00000001"), detection_time_ms=2_000
    )

    assert result.rejected is None
    assert result.observation.to_canonical_dict()["swept_level"] == "0.00000001"


def test_adapter_extreme_source_sweep_price_returns_deterministic_rejection():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = []

    for _ in range(2):
        try:
            result = adapter.adapt(
                valid_raw(source_sweep_price="1e100"), detection_time_ms=2_000
            )
            if result.observation is not None:
                canonical_json(result.observation.to_canonical_dict())
        except Exception as exc:
            pytest.fail(
                "adapter accepted a non-representable source sweep price: "
                f"{type(exc).__name__}"
            )
        rejections.append(result.rejected)

    assert all(rejection is not None for rejection in rejections)
    first_bytes = canonical_json(rejections[0].to_canonical_dict()).encode("ascii")
    second_bytes = canonical_json(rejections[1].to_canonical_dict()).encode("ascii")
    assert rejections[0].reason_detail == "INVALID_SOURCE_SWEEP_PRICE"
    assert rejections[1].reason_detail == "INVALID_SOURCE_SWEEP_PRICE"
    assert first_bytes == second_bytes


def test_adapter_source_price_that_normalizes_to_zero_is_rejected_deterministically():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = [
        adapter.adapt(
            valid_raw(source_sweep_price="1e-100"), detection_time_ms=2_000
        ).rejected
        for _ in range(2)
    ]

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "INVALID_SOURCE_SWEEP_PRICE"
        for rejection in rejections
    )
    assert rejection_bytes[0] == rejection_bytes[1]


def test_adapter_accepts_smallest_canonical_positive_source_price():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(source_sweep_price="0.00000001"), detection_time_ms=2_000
    )

    assert result.rejected is None
    assert (
        result.observation.to_canonical_dict()["source_sweep_price"]
        == "0.00000001"
    )


@pytest.mark.parametrize(
    "source_penetration_bps",
    ["1e-1000000", "9007199254740993"],
)
def test_adapter_rejects_lossy_source_penetration_conversion_deterministically(
    source_penetration_bps,
):
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = [
        adapter.adapt(
            valid_raw(source_penetration_bps=source_penetration_bps),
            detection_time_ms=2_000,
        ).rejected
        for _ in range(2)
    ]

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "INVALID_SOURCE_PENETRATION_BPS"
        for rejection in rejections
    )
    assert rejection_bytes[0] == rejection_bytes[1]


@pytest.mark.parametrize(
    ("source_penetration_bps", "expected"),
    [("0", 0.0), ("5e-324", 5e-324), ("0.1", 0.1)],
)
def test_adapter_preserves_representable_source_penetration_values(
    source_penetration_bps, expected
):
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(source_penetration_bps=source_penetration_bps),
        detection_time_ms=2_000,
    )

    assert result.rejected is None
    assert result.observation.source_penetration_bps == expected


@pytest.mark.parametrize(
    "source_row_hash",
    ["", "a" * 63, "A" * 64, "g" * 64, 123],
)
def test_adapter_rejects_malformed_supplied_source_row_hash_deterministically(
    source_row_hash,
):
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = [
        adapter.adapt(
            valid_raw(source_row_hash=source_row_hash), detection_time_ms=2_000
        ).rejected
        for _ in range(2)
    ]

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "INVALID_SOURCE_ROW_HASH"
        for rejection in rejections
    )
    assert all(rejection.source_row_hash is None for rejection in rejections)
    assert rejection_bytes[0] == rejection_bytes[1]


@pytest.mark.parametrize("source_row_hash", [None, "b" * 64])
def test_adapter_preserves_absent_or_valid_lowercase_source_row_hash(source_row_hash):
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(source_row_hash=source_row_hash), detection_time_ms=2_000
    )

    assert result.rejected is None
    assert result.observation.source_row_hash == source_row_hash


def test_adapter_accepts_missing_source_row_hash():
    raw = valid_raw()
    raw.pop("source_row_hash")

    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        raw, detection_time_ms=2_000
    )

    assert result.rejected is None
    assert result.observation.source_row_hash is None


def test_adapter_rejects_sub_millisecond_timestamp_without_truncating():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(timestamp="1970-01-01T00:00:01.000001Z"),
        detection_time_ms=2_000,
    )

    assert result.observation is None
    assert result.rejected.reason_detail == "INVALID_TIMESTAMP"


def test_adapter_preserves_exact_millisecond_timestamp():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(timestamp="1970-01-01T00:00:01.001000Z"),
        detection_time_ms=2_000,
    )

    assert result.rejected is None
    assert result.observation.event_time_ms == 1_001


@pytest.mark.parametrize(
    "timestamp",
    [
        "1970-01-01T00:00:01.0011Z",
        "1970-01-01T00:00:01.0010001Z",
        "1970-01-01T00:00:01.0010000001Z",
        "1970-01-01T00:00:01.0010001+00:00",
        "1970-01-01T00:00:01.0010001-00:00",
        "1970-01-01t00:00:01.0010001+00:00",
        "1970-01-01_00:00:01.0010001+00:00",
        "1970-01-01T00:00:01.001+00:00:00.0000001",
    ],
)
def test_adapter_rejects_nonzero_precision_beyond_milliseconds(timestamp):
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(timestamp=timestamp), detection_time_ms=2_000
    )

    assert result.observation is None
    assert result.rejected.reason_detail == "INVALID_TIMESTAMP"


@pytest.mark.parametrize(
    "timestamp",
    [
        "1970-01-01T00:00:01.0010Z",
        "1970-01-01T00:00:01.0010000Z",
        "1970-01-01T00:00:01.0010000000Z",
        "1970-01-01T00:00:01.0010000+00:00",
        "1970-01-01T00:00:01.0010000-00:00",
        "1970-01-01t00:00:01.0010000+00:00",
    ],
)
def test_adapter_accepts_zero_precision_beyond_milliseconds(timestamp):
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(timestamp=timestamp), detection_time_ms=2_000
    )

    assert result.rejected is None
    assert result.observation.event_time_ms == 1_001


@pytest.mark.parametrize(
    "source_sweep_price",
    [
        0,
        "0",
        "-0",
        Decimal("-0.00000000"),
        "-0.00000001",
        "-100",
        "0.000000004",
        "-0.000000004",
    ],
    ids=[
        "integer-zero",
        "string-zero",
        "signed-zero",
        "decimal-signed-zero",
        "smallest-negative",
        "negative",
        "positive-rounds-to-zero",
        "negative-rounds-to-zero",
    ],
)
def test_adapter_rejects_source_price_not_positive_after_normalization(
    source_sweep_price,
):
    adapter = adapter_api()(LiquidityClassificationPolicy())
    rejections = [
        adapter.adapt(
            valid_raw(source_sweep_price=source_sweep_price),
            detection_time_ms=2_000,
        ).rejected
        for _ in range(2)
    ]

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "INVALID_SOURCE_SWEEP_PRICE"
        for rejection in rejections
    )
    assert rejection_bytes[0] == rejection_bytes[1]


def _assert_noncanonical_callback_rejection_is_stable(raw, detection_time_ms=2_000):
    adapter = adapter_api()(LiquidityClassificationPolicy())
    try:
        rejections = [
            adapter.adapt(raw, detection_time_ms=detection_time_ms).rejected
            for _ in range(2)
        ]
    except RecursionError:
        pytest.fail("recursive callback content escaped instead of being rejected")

    assert all(rejection is not None for rejection in rejections)
    rejection_bytes = [
        canonical_json(rejection.to_canonical_dict()).encode("ascii")
        for rejection in rejections
    ]
    assert all(
        rejection.reason_detail == "NON_CANONICAL_CALLBACK"
        for rejection in rejections
    )
    assert all(rejection.source_observation_hash is None for rejection in rejections)
    assert rejection_bytes[0] == rejection_bytes[1]


def test_adapter_rejects_cyclic_callback_content_without_recursion_error():
    cycle = []
    cycle.append(cycle)

    _assert_noncanonical_callback_rejection_is_stable(
        valid_raw(callback_metadata=cycle)
    )
    _assert_noncanonical_callback_rejection_is_stable(
        valid_raw(type="UP", callback_metadata=cycle), detection_time_ms=-1
    )


def test_adapter_rejects_excessively_deep_callback_content_without_recursion_error():
    nested = []
    for _ in range(2_000):
        nested = [nested]

    _assert_noncanonical_callback_rejection_is_stable(
        valid_raw(callback_metadata=nested)
    )


@pytest.mark.parametrize(
    "detection_time_ms",
    [
        True,
        -1,
        253_402_300_800_000,
        10**5_000,
        -(10**5_000),
    ],
    ids=["bool", "negative", "past-utc-domain", "huge-positive", "huge-negative"],
)
def test_adapter_rejects_noncanonical_detection_time_with_safe_sentinel(
    detection_time_ms,
):
    adapter = adapter_api()(LiquidityClassificationPolicy())
    try:
        rejections = [
            adapter.adapt(valid_raw(), detection_time_ms=detection_time_ms).rejected
            for _ in range(2)
        ]
        assert all(rejection is not None for rejection in rejections)
        rejection_bytes = [
            canonical_json(rejection.to_canonical_dict()).encode("ascii")
            for rejection in rejections
        ]
    except (OverflowError, ValueError) as exc:
        pytest.fail(
            "invalid detection time escaped canonical rejection: "
            f"{type(exc).__name__}"
        )

    assert all(rejection.detection_time_ms == -1 for rejection in rejections)
    assert all(
        rejection.reason_detail == "INVALID_DETECTION_TIME"
        for rejection in rejections
    )
    assert rejection_bytes[0] == rejection_bytes[1]


def test_adapter_accepts_maximum_utc_domain_detection_time():
    result = adapter_api()(LiquidityClassificationPolicy()).adapt(
        valid_raw(), detection_time_ms=253_402_300_799_999
    )

    assert result.rejected is None
    assert result.observation.detection_time_ms == 253_402_300_799_999


def test_adapter_object_detection_time_uses_deterministic_integer_sentinel():
    adapter = adapter_api()(LiquidityClassificationPolicy())
    first = adapter.adapt(valid_raw(), detection_time_ms=object()).rejected
    second = adapter.adapt(valid_raw(), detection_time_ms=object()).rejected

    try:
        first_bytes = canonical_json(first.to_canonical_dict()).encode("ascii")
        second_bytes = canonical_json(second.to_canonical_dict()).encode("ascii")
    except (TypeError, ValueError) as exc:
        pytest.fail(
            f"rejection must remain canonical-serializable: {type(exc).__name__}"
        )

    assert first.detection_time_ms == second.detection_time_ms == -1
    assert first.reason_detail == second.reason_detail == "INVALID_DETECTION_TIME"
    assert first_bytes == second_bytes
    assert b"0x" not in first_bytes


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
        try:
            async def _wait_for_callbacks():
                while len(callbacks) < 2:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_wait_for_callbacks(), timeout=2.0)
        finally:
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


def test_sqlite_transaction_in_transaction_is_false_after_claims_and_rollback(tmp_path):
    authority = SQLiteIdentityAuthority(tmp_path / "trans.sqlite3")
    obs = observation(event_id="tx-test")

    # Initial state
    assert authority._connection.in_transaction is False

    # Claim observation
    claim = authority.claim_observation(obs)
    assert claim.outcome is IdentityClaimOutcome.NEW
    assert authority._connection.in_transaction is False

    # Claim transition
    trans = lifecycle_transition(event_id="tx-test", transition_sequence=0)
    authority.claim_transition(trans)
    assert authority._connection.in_transaction is False

    # Claim result
    res = finalized_result(event_id="tx-test")
    authority.claim_result(res)
    assert authority._connection.in_transaction is False

    # Forced rollback test
    try:
        authority._begin_immediate()
        assert authority._connection.in_transaction is True
        authority._execute("INSERT INTO event_identity VALUES ('bad', 'bad', 'bad', 'bad', 'bad', 'bad', 0)")
        # Force check constraint failure or duplicate
        authority._execute("INSERT INTO event_identity VALUES ('bad', 'bad', 'bad', 'bad', 'bad', 'bad', 0)")
    except Exception:
        authority._rollback()

    assert authority._connection.in_transaction is False
    # Verify no partial rows remained from failed transaction
    assert authority.lookup("bad") is None


def test_sqlite_foreign_key_constraints_enforced(tmp_path):
    import sqlite3
    db_path = tmp_path / "fk.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)

    # Try inserting orphan transition directly via SQL
    with pytest.raises(sqlite3.IntegrityError):
        authority._execute(
            """
            INSERT INTO event_transition (event_id, transition_sequence, transition_hash, transition_payload_json)
            VALUES ('non-existent', 0, 'hash', '{}')
            """
        )

    # Try inserting orphan result directly via SQL
    with pytest.raises(sqlite3.IntegrityError):
        authority._execute(
            """
            INSERT INTO event_result (event_id, result_hash, result_payload_json, classification_time_ms)
            VALUES ('non-existent', 'hash', '{}', 1000)
            """
        )
    authority.close()


def test_sqlite_pragmas_and_runtime_state_configured_for_maximum_durability(tmp_path):
    db_path = tmp_path / "pragmas.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)

    sync_row = authority._execute("PRAGMA synchronous;").fetchone()
    assert sync_row is not None and sync_row[0] == 2  # 2 == FULL

    fk_row = authority._execute("PRAGMA foreign_keys;").fetchone()
    assert fk_row is not None and fk_row[0] == 1  # 1 == ON

    timeout_row = authority._execute("PRAGMA busy_timeout;").fetchone()
    assert timeout_row is not None and timeout_row[0] == 5000

    journal_row = authority._execute("PRAGMA journal_mode;").fetchone()
    assert journal_row is not None and journal_row[0] in ("delete", "wal")

    authority.close()


def test_capacity_rejection_leaves_no_persistent_identity_and_recovers_cleanly(tmp_path):
    db_path = tmp_path / "capacity.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = store_api()(policy)

    # 1. Admit first event -> admitted and claimed
    obs1 = observation(event_id="event-1", symbol="BTCUSDT")
    res1 = store.admit_observation(obs1, authority)
    assert res1.created is True
    assert authority.lookup("event-1") is not None
    assert len(authority.pending_unresolved()) == 1

    # 2. Try admitting second event when capacity=1 is full
    obs2 = observation(event_id="event-2", symbol="BTCUSDT")
    res2 = store.admit_observation(obs2, authority)
    assert res2.created is False
    assert res2.rejection is not None
    assert res2.rejection.reason_code == "EVENT_CAPACITY_REACHED"

    # Critical invariant: event-2 was NEVER written to SQLite!
    assert authority.lookup("event-2") is None
    assert len(authority.pending_unresolved()) == 1
    assert authority.pending_unresolved()[0].event_id == "event-1"

    # Restart authority -> verify event-2 is still absent
    authority.close()
    restarted = SQLiteIdentityAuthority(db_path)
    assert restarted.lookup("event-2") is None
    assert [p.event_id for p in restarted.pending_unresolved()] == ["event-1"]

    # 3. Finalize event-1 -> capacity is now freed
    final1 = final_result(res1.event, policy=policy)
    restarted.claim_result(final1)
    store.finalize(final1)
    assert store.active_count("BTCUSDT") == 0
    assert store.has_capacity("BTCUSDT") is True

    # 4. Now event-2 can legitimately claim and open!
    res2_retry = store.admit_observation(obs2, restarted)
    assert res2_retry.created is True
    assert restarted.lookup("event-2") is not None
    assert restarted.lookup("event-2").status == "CLAIMED_UNRESOLVED"
    restarted.close()


def test_capacity_concurrency_race_free_admission(tmp_path):
    import concurrent.futures
    db_path = tmp_path / "concurrent_cap.sqlite3"
    authority = SQLiteIdentityAuthority(db_path)
    policy = replace(LiquidityClassificationPolicy(), max_active_events_per_symbol=1)
    store = store_api()(policy)

    obs_a = observation(event_id="event-a", symbol="BTCUSDT")
    obs_b = observation(event_id="event-b", symbol="BTCUSDT")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_a = executor.submit(store.admit_observation, obs_a, authority)
        f_b = executor.submit(store.admit_observation, obs_b, authority)
        results = [f_a.result(), f_b.result()]

    created = [r for r in results if r.created]
    rejected = [r for r in results if r.rejection is not None and r.rejection.reason_code == "EVENT_CAPACITY_REACHED"]

    assert len(created) == 1
    assert len(rejected) == 1

    # Exactly one admitted event in SQLite
    admitted_id = created[0].event.event_id
    rejected_id = "event-b" if admitted_id == "event-a" else "event-a"

    assert authority.lookup(admitted_id) is not None
    assert authority.lookup(rejected_id) is None
    assert len(authority.pending_unresolved()) == 1
    assert authority.pending_unresolved()[0].event_id == admitted_id
    authority.close()
