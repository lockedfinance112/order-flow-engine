# Phase 1C.1 Liquidity Event Engine Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic, shadow-only Phase 1C.1 Liquidity Event Engine defined by frozen specification version `1.0.1`, including exact live/replay parity and all 40 acceptance gates.

**Architecture:** Normalize existing `SweepsMonitor` callbacks into immutable sweep observations, claim exact event identity through a persistent standard-library SQLite authority, retain symbol-local point-in-time market buffers, validate penetration, and classify a 60-second post-sweep response with a watermark-settled deterministic state machine. Event-local trade accumulators and typed optional-context adapters build evidence; immutable results flow to canonical recorders, exact replay, a read-only API, and a compact shadow display without touching trading authority.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `decimal`, `hashlib`, `bisect`, `asyncio`, `csv`, `json`, `sqlite3`), existing `pytest`/`unittest` suite, existing Rich dashboard and local asyncio HTTP server. No new runtime dependency.

## Global Constraints

- Frozen source of truth: `docs/superpowers/specs/2026-08-11-liquidity-event-engine-foundation-design.md`, version `1.0.1`, including Spec Patch 001 persistent identity authority.
- Do not edit or reinterpret the frozen specification beyond the approved Spec Patch 001 contract. Stop and raise a spec issue if an invariant is contradictory or impossible.
- `LIQUIDITY_EVENT_ENGINE_ENABLED = True`.
- `LIQUIDITY_EVENT_ENFORCEMENT_ENABLED = False`.
- `REGIME_ENFORCEMENT_ENABLED = False`.
- `EXECUTION_DISABLED = True`.
- `liquidity_event.*` must not import scoring, paper trading, execution, position sizing, or `regime.permissions`.
- Price behavior alone selects the outcome; contextual evidence cannot change classification.
- Classification logic must not call `time.time()` or sample mutable global `FlowMetrics` delta/CVD state.
- Missing compatible optional context is `UNAVAILABLE`, never numeric zero.
- `identity_authority.py` is the exact persistent identity authority; bounded volatile stores and recorders are not identity authority.
- Event identity retention is indefinite for Phase 1C.1; bounded tombstone caches are rejected.
- Persistent identity uses Python standard-library SQLite (`sqlite3`) only.
- All production changes follow red-green-refactor: run each named test and observe the expected failure before implementing.
- Missing modules or symbols must be imported inside the test and converted to an explicit `pytest.fail("required Phase 1C.1 API is not implemented")`; a collection error does not count as RED.
- Review-fix commits are temporary. After a task review is clean, squash that task's implementation and fix commits into one scoped commit, verify the tree hash is unchanged, and record only the final commit in the ledger.
- Run Python and pytest commands from `C:\Users\karan\Desktop\orderflow\liquidity sweep\order_flow_engine`; run repository-level Git commands from `C:\Users\karan\Desktop\orderflow`.
- Keep the unrelated modifications in `data/local_order_book.py` and `test_data_integrity.py` unstaged; every commit command below names files explicitly.

## File Map

```text
liquidity sweep/order_flow_engine/
    liquidity_event/
        __init__.py       public exports only
        models.py         enums, immutable inputs/results, mutable event state
        policy.py         frozen v1 policy, validation, canonical hash
        sweep_adapter.py  legacy callback normalization and rejection
        identity_authority.py exact SQLite event identity, transition, and result uniqueness
        event_store.py    bounded volatile working state, active collisions, recent snapshots
        evidence.py       event-local flow and typed optional context
        classifier.py     branch timers and pure outcome truth table
        engine.py         buffers, coverage, watermark, lifecycle orchestration
        recorder.py       canonical rows, bounded live queue, replay certification
        replay.py         canonical input merge and exact replay result
    test_liquidity_event_models.py
    test_liquidity_event_adapter_store.py
    test_liquidity_event_evidence.py
    test_liquidity_event_classifier.py
    test_liquidity_event_engine.py
    test_liquidity_event_recorder.py
    test_liquidity_event_replay.py
    test_liquidity_event_integration.py
    config.py
    sweeps_monitor.py
    main.py
    dashboard.py
    walkthrough.md
```

---

### Task 1: Canonical Models and Frozen Policy

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/__init__.py`
- Create: `liquidity sweep/order_flow_engine/liquidity_event/models.py`
- Create: `liquidity sweep/order_flow_engine/liquidity_event/policy.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_models.py`

**Interfaces:**
- Produces: `LiquiditySide`, `EventState`, `EventClassification`, `EvidenceAvailability`, `ConfidenceType`, `ArtifactIntegrity`, `SweepSource`, `AggressorSide`, `EvidenceValue`, `MarketTrade`, `DepthObservation`, `TradeCoverage`, `TradeCoverageProvider`, `LiquiditySweepObservation`, `RejectedSweepInput`, `LifecycleTransition`, `LiquidityEvidence`, `LiquidityEvent`, `LiquidityEventResult`, and `LiquidityClassificationPolicy`.
- Produces: `canonical_json(value) -> str`, `canonical_hash(value) -> str`, and `normalize_price(price, decimal_places) -> str`.

- [ ] **Step 1: Write failing model and policy tests**

```python
def test_v1_policy_hash_is_stable_and_validates_retention():
    first = LiquidityClassificationPolicy()
    second = LiquidityClassificationPolicy()
    assert first.policy_hash == second.policy_hash
    assert len(first.policy_hash) == 64
    assert first.market_buffer_retention_ms == 180_000
    with pytest.raises(ValueError, match="retention"):
        replace(first, market_buffer_retention_ms=179_999).validate()

def test_price_normalization_is_fixed_half_even_eight_places():
    assert normalize_price(Decimal("117250.123456785"), 8) == "117250.12345678"
    assert normalize_price(Decimal("117250.123456795"), 8) == "117250.12345680"

def test_models_keep_missing_optional_evidence_distinct_from_zero():
    missing = EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
    zero = EvidenceValue(EvidenceAvailability.AVAILABLE, 0.0, 1000)
    assert missing != zero
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest test_liquidity_event_models.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement the canonical contracts**

Use string enums and dataclasses. The policy must contain every frozen behavior-affecting value:

```python
@dataclass(frozen=True)
class LiquidityClassificationPolicy:
    confirmation_window_ms: int = 60_000
    confirmation_hold_ms: int = 3_000
    minimum_confirming_trades: int = 3
    reclaim_buffer_bps: float = 1.0
    acceptance_buffer_bps: float = 1.0
    reorder_tolerance_ms: int = 2_000
    collision_window_ms: int = 2_000
    collision_level_tolerance_bps: float = 0.5
    canonical_price_decimal_places: int = 8
    maximum_supported_detection_lag_ms: int = 90_000
    buffer_safety_margin_ms: int = 28_000
    market_buffer_retention_ms: int = 180_000
    max_active_events_per_symbol: int = 128
    recent_final_events_per_symbol: int = 1_000
    recorder_queue_max_items: int = 10_000
    model_version: str = "1C.1-v1"

    def validate(self) -> "LiquidityClassificationPolicy":
        required = (
            self.confirmation_window_ms
            + self.reorder_tolerance_ms
            + self.maximum_supported_detection_lag_ms
            + self.buffer_safety_margin_ms
        )
        if self.market_buffer_retention_ms < required:
            raise ValueError("market buffer retention is below the frozen minimum")
        if self.minimum_confirming_trades < 1 or self.confirmation_hold_ms < 0:
            raise ValueError("confirmation constraints must be non-negative")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_hash(asdict(self))
```

Define `MarketTrade.canonical_key` as `(exchange_time_ms, sequence_id, 0, content_hash)` and `DepthObservation.canonical_key` as `(exchange_time_ms, sequence_id, 1, content_hash)`. Define observation/result `to_canonical_dict()` methods so recorders never serialize object representations.

Define mandatory coverage and optional evidence exactly as:

```python
@dataclass(frozen=True)
class TradeCoverage:
    feed_safe: bool
    known_gap: bool
    buffer_overflow: bool
    unresolved_sequence: bool
    interval_retained: bool

    @property
    def valid(self) -> bool:
        return (
            self.feed_safe
            and not self.known_gap
            and not self.buffer_overflow
            and not self.unresolved_sequence
            and self.interval_retained
        )

    @classmethod
    def safe_zero_activity(cls) -> "TradeCoverage":
        return cls(True, False, False, False, True)

class TradeCoverageProvider(Protocol):
    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> TradeCoverage:
        raise NotImplementedError

@dataclass(frozen=True)
class EvidenceValue:
    availability: EvidenceAvailability
    value: float | None
    as_of_ms: int | None
```

- [ ] **Step 4: Run model tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_models.py -q`

Expected: all Task 1 tests pass with no warnings.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/__init__.py" "liquidity sweep/order_flow_engine/liquidity_event/models.py" "liquidity sweep/order_flow_engine/liquidity_event/policy.py" "liquidity sweep/order_flow_engine/test_liquidity_event_models.py"
git commit -m "feat: add liquidity event contracts and policy"
```

### Task 2: SweepsMonitor Adapter and Source Provenance

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/sweep_adapter.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py`
- Modify: `liquidity sweep/order_flow_engine/sweeps_monitor.py:54-155`

**Interfaces:**
- Consumes: `LiquidityClassificationPolicy`, `LiquiditySweepObservation`, `RejectedSweepInput`.
- Produces: `SweepAdapterResult(observation, rejected)` and `SweepsMonitorAdapter.adapt(raw, detection_time_ms) -> SweepAdapterResult`.
- Extends callback dictionaries with optional `source_file_id`, `source_row_hash`, `source_level_id`, `detector_version`, `source_sweep_price`, and `source_penetration_bps`; existing keys remain unchanged.

The adapter test module defines this complete raw fixture before its tests:

```python
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
```

- [ ] **Step 1: Write failing adapter tests**

```python
def test_bullish_and_bearish_legacy_directions_map_only_in_adapter():
    adapter = SweepsMonitorAdapter(LiquidityClassificationPolicy())
    bullish = adapter.adapt(valid_raw(type="BULLISH"), detection_time_ms=2_000)
    bearish = adapter.adapt(valid_raw(type="BEARISH", sweep_id="b"), detection_time_ms=2_000)
    assert bullish.observation.liquidity_side is LiquiditySide.SELL_SIDE
    assert bearish.observation.liquidity_side is LiquiditySide.BUY_SIDE

def test_adapter_is_deterministic_and_does_not_invent_sweep_price():
    first = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation
    second = adapter.adapt(valid_raw(), detection_time_ms=2_000).observation
    assert first == second
    assert first.source_sweep_price is None
    assert first.source_penetration_bps is None

def test_malformed_direction_is_rejected_without_guessing():
    result = adapter.adapt(valid_raw(type="UP"), detection_time_ms=2_000)
    assert result.observation is None
    assert result.rejected.reason_code == "INVALID_SWEEP_OBSERVATION"
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest test_liquidity_event_adapter_store.py -k adapter -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement strict normalization and provenance**

```python
class SweepsMonitorAdapter:
    SIDE_MAP = {"BULLISH": LiquiditySide.SELL_SIDE, "BEARISH": LiquiditySide.BUY_SIDE}

    def adapt(self, raw: Mapping[str, Any], detection_time_ms: int) -> SweepAdapterResult:
        try:
            side = self.SIDE_MAP[str(raw["type"]).upper()]
            event_time_ms = parse_utc_ms(str(raw["timestamp"]))
            swept_level = Decimal(str(raw["sweep_level"]))
            if swept_level <= 0 or detection_time_ms < 0:
                raise ValueError("invalid level or detection time")
        except (KeyError, TypeError, ValueError) as exc:
            return SweepAdapterResult(None, rejected_from(raw, detection_time_ms, str(exc)))
        return SweepAdapterResult(build_observation(raw, side, swept_level, event_time_ms, detection_time_ms), None)
```

In `SweepsMonitor`, calculate `source_row_hash` from canonical JSON of the parsed row, use logical `source_file_id="SWEEPS_MONITOR_CSV"`, and pass metadata in both startup replay and live polling callbacks. Do not change CSV contents or existing scorer-facing keys.

- [ ] **Step 4: Run adapter tests and existing sweep tests**

Run: `python -m pytest test_liquidity_event_adapter_store.py -k adapter test_liquidity.py -q`

Expected: adapter and existing liquidity tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/sweep_adapter.py" "liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py" "liquidity sweep/order_flow_engine/sweeps_monitor.py"
git commit -m "feat: normalize sweep observations"
```

### Task 3A: Persistent Identity Authority

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/identity_authority.py`
- Modify: `liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py`

**Interfaces:**
- Consumes: `LiquiditySweepObservation`, `LifecycleTransition`, `LiquidityEventResult`, `LiquidityClassificationPolicy`.
- Produces: `IdentityClaimOutcome.NEW`, `IdentityClaimOutcome.DUPLICATE_EXISTING`, and `IdentityClaimOutcome.IDENTITY_CONFLICT`.
- Produces: `IdentityClaimResult(outcome, event_id, existing_identity, conflict_reason)`.
- Produces: `SQLiteIdentityAuthority(path)`, `claim_observation(observation) -> IdentityClaimResult`, `claim_transition(transition)`, `claim_result(result)`, `lookup(event_id)`, `pending_unresolved()`, and `persisted_transition_sequences(event_id)`.

The store test module defines `observation(**overrides)` by adapting `valid_raw()` and applying `dataclasses.replace`; `level`, `side`, and `event_time_ms` overrides map to `swept_level`, `liquidity_side`, and `event_time_ms` respectively. Its default event ID is `"same"`.

- [ ] **Step 1: Add failing persistent identity tests**

```python
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
    store = LiquidityEventStore(replace(policy, recent_final_events_per_symbol=0))
    event = observation(event_id="same", level="117250.00")
    claim = authority.claim_observation(event)
    opened = store.open_event(event, claim)
    final = finalized_result(event_id="same")
    authority.claim_result(final)
    store.finalize(final)
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
```

- [ ] **Step 2: Run identity authority tests and verify RED**

Run: `python -m pytest test_liquidity_event_adapter_store.py -k "sqlite_identity or identity_conflict or finalized_identity_survives or transition_and_result_uniqueness or claimed_unresolved_restart or authority_failure" -q`

Expected: import or attribute failure for `SQLiteIdentityAuthority`.

- [ ] **Step 3: Implement exact SQLite identity authority**

Use Python standard-library `sqlite3`, explicit transactions, and schema initialization on construction. Store canonical immutable identity payloads, source observation hash, claim status, and timestamps without absolute paths, hostnames, process IDs, or wall-clock-only values. Enforce these indexes and table constraints:

```sql
CREATE TABLE IF NOT EXISTS event_identity (
    event_id TEXT PRIMARY KEY,
    identity_hash TEXT NOT NULL,
    identity_payload_json TEXT NOT NULL,
    observation_payload_json TEXT NOT NULL,
    source_observation_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('CLAIMED_UNRESOLVED', 'FINALIZED')),
    first_detection_time_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_transition (
    event_id TEXT NOT NULL,
    transition_sequence INTEGER NOT NULL,
    transition_hash TEXT NOT NULL,
    transition_payload_json TEXT NOT NULL,
    PRIMARY KEY (event_id, transition_sequence)
);

CREATE TABLE IF NOT EXISTS event_result (
    event_id TEXT PRIMARY KEY,
    result_hash TEXT NOT NULL,
    result_payload_json TEXT NOT NULL,
    classification_time_ms INTEGER NOT NULL
);
```

`claim_observation()` must start an immediate transaction. If `event_id` is absent, insert a `CLAIMED_UNRESOLVED` row and return `NEW`. If `event_id` exists with the same immutable identity hash, return `DUPLICATE_EXISTING`. If `event_id` exists with a different immutable identity hash or source observation hash, return `IDENTITY_CONFLICT`. Do not fall back to a bounded tombstone cache.

`claim_result()` inserts into `event_result` and updates `event_identity.status` to `FINALIZED` in the same transaction. This commit occurs before recorder notification. `claim_transition()` relies on the `(event_id, transition_sequence)` primary key and raises `IdentityAlreadyExists` on duplicate sequence.

`pending_unresolved()` returns claimed identities whose status is `CLAIMED_UNRESOLVED`, including the persisted observation payload needed to reconstruct the event after process restart. `persisted_transition_sequences(event_id)` returns the committed transition sequence numbers for recovery. On restart, recovery replays each claimed-unresolved identity only when complete required market history is available; it emits only missing transition keys, and otherwise finalizes exactly once as `INVALID / INSUFFICIENT_REPLAY_HISTORY`.

- [ ] **Step 4: Run identity authority tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_adapter_store.py -k "sqlite_identity or identity_conflict or finalized_identity_survives or transition_and_result_uniqueness or claimed_unresolved_restart or authority_failure" -q`

Expected: persistent duplicate, conflict, uniqueness, and failure tests pass.

- [ ] **Step 5: Commit Task 3A**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/identity_authority.py" "liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py"
git commit -m "feat: add persistent liquidity identity authority"
```

### Task 3B: Bounded Volatile Event Store and Collision Bounds

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/event_store.py`
- Modify: `liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py`

**Interfaces:**
- Consumes: `LiquiditySweepObservation`, `LiquidityClassificationPolicy`, `LiquidityEvent`, `LiquidityEventResult`, and `IdentityClaimResult`.
- Produces: `OpenEventResult(event, created, duplicate, collision_event_ids, rejection)`.
- Produces: `LiquidityEventStore.open_event(observation, identity_claim)`, `finalize(result)`, `active(symbol=None)`, and `recent(symbol=None)`.

The store is bounded volatile working state only. It does not own lifetime identity, finalized duplicate suppression, restart semantics, or tombstones.

- [ ] **Step 1: Add failing collision and capacity tests**

```python
def test_duplicate_callback_uses_identity_authority_without_reopening_event():
    store = LiquidityEventStore(policy)
    first_claim = IdentityClaimResult(IdentityClaimOutcome.NEW, "same", None, None)
    duplicate_claim = IdentityClaimResult(IdentityClaimOutcome.DUPLICATE_EXISTING, "same", existing_identity(), None)
    first = store.open_event(observation(event_id="same"), first_claim)
    second = store.open_event(observation(event_id="same"), duplicate_claim)
    assert first.created is True
    assert second.duplicate is True
    assert second.event is first.event

def test_distinct_levels_and_opposite_sides_remain_separate():
    assert store.open_event(observation(level="117250"), new_claim("same")).created
    assert store.open_event(observation(event_id="two", level="117210"), new_claim("two")).created
    assert store.open_event(observation(event_id="three", side=LiquiditySide.BUY_SIDE), new_claim("three")).created

def test_close_same_side_levels_report_ambiguous_collision():
    store.open_event(observation(level="117250.00"), new_claim("same"))
    result = store.open_event(observation(event_id="two", level="117251.00", event_time_ms=1_500), new_claim("two"))
    assert result.collision_event_ids == ("same", "two")

def test_active_capacity_rejects_without_unbounded_growth():
    tiny = replace(policy, max_active_events_per_symbol=1)
    store = LiquidityEventStore(tiny)
    store.open_event(observation(), new_claim("same"))
    assert store.open_event(observation(event_id="two", level="118000"), new_claim("two")).rejection.reason_code == "EVENT_CAPACITY_REACHED"
```

- [ ] **Step 2: Run store tests and verify RED**

Run: `python -m pytest test_liquidity_event_adapter_store.py -k "store or duplicate or collision or capacity" -q`

Expected: import or attribute failure for `LiquidityEventStore` or its identity-claim-aware `open_event` signature.

- [ ] **Step 3: Implement bounded volatile storage**

Use `dict[event_id, LiquidityEvent]` for active events and `dict[symbol, deque(maxlen=recent_final_events_per_symbol)]` for immutable recent results. Do not maintain finalized-ID tombstones. When `identity_claim.outcome` is `DUPLICATE_EXISTING`, return a duplicate result if the event remains active or a duplicate-without-event result if the volatile event has been evicted. When the claim is `IDENTITY_CONFLICT`, return a rejection that the engine treats as fail-closed without opening/classifying a new event.

Compute fallback level identity with the Task 1 Decimal normalizer. Compare same-side levels in bps only inside `collision_window_ms`; return collisions to the engine instead of silently merging different IDs.

```python
def fallback_level_identity(obs, policy):
    payload = {
        "symbol": obs.symbol,
        "side": obs.liquidity_side.value,
        "level": normalize_price(obs.swept_level, policy.canonical_price_decimal_places),
    }
    return canonical_hash(payload)
```

- [ ] **Step 4: Run adapter/store tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_adapter_store.py -q`

Expected: all adapter/store tests pass.

- [ ] **Step 5: Commit Task 3B**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/event_store.py" "liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py"
git commit -m "feat: add bounded volatile liquidity event store"
```

### Task 4: Event-Local Flow and Typed Context Evidence

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/evidence.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_evidence.py`

**Interfaces:**
- Consumes: `MarketTrade`, `DepthObservation`, `LiquiditySweepObservation`, `EvidenceValue`.
- Produces: `EventFlowAccumulator.on_trade(trade)`, `snapshot(as_of_ms) -> EventFlowSnapshot`.
- Produces: `LegacyLiquidityContextAdapter.absorption(raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue`, `.replenishment(raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue`, and `.stacking_pulling(raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue`.
- Produces: `LiquidityEvidenceBuilder.build(event, outcome_direction, as_of_ms) -> LiquidityEvidence`.

The evidence test module defines `trade(ts, side, notional, sequence=1)` as a `MarketTrade` at price `100.0` with quantity `notional / 100.0`, and `accumulate(trades, canonical_sort=False)` as a fresh accumulator fed either in supplied or canonical-key order.

- [ ] **Step 1: Write failing event-local flow tests**

```python
def test_event_flow_accumulates_only_point_in_time_trades():
    flow = EventFlowAccumulator(event_time_ms=1_000, end_time_ms=61_000)
    flow.on_trade(trade(900, "BUY", 100))
    flow.on_trade(trade(1_100, "SELL", 40))
    flow.on_trade(trade(1_200, "BUY", 70))
    snap = flow.snapshot(1_200)
    assert snap.buy_volume_usdt == 70
    assert snap.sell_volume_usdt == 40
    assert snap.signed_delta_usdt == 30
    assert snap.cvd_min_usdt == -40
    assert snap.cvd_max_usdt == 30

def test_live_and_replay_trade_order_produce_identical_flow_snapshot():
    chronological = [trade(1_000, "SELL", 50, 1), trade(1_100, "BUY", 80, 2)]
    assert accumulate(chronological) == accumulate(reversed(chronological), canonical_sort=True)

def test_optional_context_is_typed_and_never_reimplemented():
    adapter = LegacyLiquidityContextAdapter()
    assert adapter.absorption(None, as_of_ms=2_000).availability is EvidenceAvailability.UNAVAILABLE
    value = adapter.absorption({"event_type": "BULLISH_ABSORPTION"}, as_of_ms=2_000)
    assert value.value == 1.0
```

- [ ] **Step 2: Run evidence tests and verify RED**

Run: `python -m pytest test_liquidity_event_evidence.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement event-local accumulation and evidence scoring inputs**

`EventFlowAccumulator` must sort by `MarketTrade.canonical_key`, deduplicate by `(symbol, sequence_id)`, start CVD at zero, and reject trades outside `[event_time_ms, end_time_ms]`. It must not accept a `FlowMetrics` instance.

`LiquidityEvidenceBuilder` assigns contextual direction values only:

```python
def directional_value(value, bullish):
    if value.availability is not EvidenceAvailability.AVAILABLE or value.value == 0:
        return 0
    return 1 if (value.value > 0) == bullish else -1

evidence_strength = 0.60 + 0.10 * support_count
contradiction_strength = 0.10 * contradiction_count
confidence = max(0.0, min(1.0, evidence_strength - contradiction_strength))
```

Delta, event CVD, absorption, and depth-weighted imbalance are the only v1 contextual confidence groups. Replenishment, stacking/pulling, displacement, and volume expansion produce ordered reasons/contradictions but no confidence weight.

- [ ] **Step 4: Run evidence tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_evidence.py -q`

Expected: all evidence tests pass and no `FlowMetrics` import exists in `liquidity_event`.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/evidence.py" "liquidity sweep/order_flow_engine/test_liquidity_event_evidence.py"
git commit -m "feat: build event-local liquidity evidence"
```

### Task 5: Deterministic Price Outcome Classifier

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/classifier.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_classifier.py`

**Interfaces:**
- Consumes: `LiquiditySide`, `MarketTrade`, `LiquidityClassificationPolicy`.
- Produces: `PriceOutcomeTracker(level, side, policy)`, `on_trade(trade)`, `reclaim_sufficient_time_ms`, `acceptance_sufficient_time_ms`, `penetration`, and `decision_at(watermark_ms, expiry_ms) -> PriceDecision`.

The classifier test module defines `feed_prices(side, prices, times)` by constructing a tracker at level `100.0` and feeding one-unit BUY trades with matching timestamps and increasing sequence IDs. `tracker_with_sufficient_times(reclaim, acceptance)` creates a tracker and sets its two candidate progress objects through their public test factory `CandidateProgress.sufficient_at(timestamp_ms)`; production code never exposes mutable setters.

- [ ] **Step 1: Write the four truth-table tests and timer edge tests**

```python
def test_sell_side_reclaim_is_failed_breakdown():
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_000])
    assert tracker.decision_at(6_000, 60_000).classification is EventClassification.FAILED_BREAKDOWN

def test_sell_side_acceptance_is_bearish_continuation():
    tracker = feed_prices(LiquiditySide.SELL_SIDE, [99.98, 99.98, 99.98, 99.98], [0, 1_000, 2_000, 4_000])
    assert tracker.decision_at(6_000, 60_000).classification is EventClassification.BEARISH_CONTINUATION

def test_buy_side_reclaim_is_failed_breakout():
    tracker = feed_prices(LiquiditySide.BUY_SIDE, [100.02, 99.98, 99.98, 99.98], [0, 1_000, 2_000, 4_000])
    assert tracker.decision_at(6_000, 60_000).classification is EventClassification.FAILED_BREAKOUT

def test_buy_side_acceptance_is_bullish_continuation():
    tracker = feed_prices(LiquiditySide.BUY_SIDE, [100.02, 100.02, 100.02, 100.02], [0, 1_000, 2_000, 4_000])
    assert tracker.decision_at(6_000, 60_000).classification is EventClassification.BULLISH_CONTINUATION

def test_neutral_zone_pauses_qualifying_duration():
    tracker = feed_prices(SELL_SIDE, [99.98, 100.02, 100.02, 100.00, 100.02, 100.02], [0, 100, 1_100, 1_200, 9_100, 10_100])
    assert tracker.reclaim_qualifying_duration_ms == 2_000

def test_opposite_threshold_resets_candidate():
    tracker = feed_prices(SELL_SIDE, [99.98, 100.02, 100.02, 99.98], [0, 100, 2_000, 2_100])
    assert tracker.reclaim_qualifying_duration_ms == 0
    assert tracker.acceptance_trade_count == 1

def test_equal_sufficient_times_are_indeterminate():
    tracker = tracker_with_sufficient_times(reclaim=5_000, acceptance=5_000)
    assert tracker.decision_at(7_000, 60_000).reason_code == "CONFLICTING_CONFIRMATION"
```

- [ ] **Step 2: Run classifier tests and verify RED**

Run: `python -m pytest test_liquidity_event_classifier.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement exact thresholds and paused segment timing**

Represent each competing branch with `CandidateProgress(accumulated_ms, segment_start_ms, last_qualifying_ms, qualifying_trade_count, sufficient_time_ms)`. On each consecutive qualifying trade, add `trade.exchange_time_ms - last_qualifying_ms`; when resuming after neutral, set `last_qualifying_ms` without adding the neutral interval. A neutral trade pauses at the last qualifying timestamp. An opposite-threshold trade resets the competing branch and starts the new branch. Do not finalize until sufficient time is behind the supplied watermark.

Use Decimal-derived thresholds from the policy and strict `>`/`<` comparisons. A trade exactly on a threshold is neutral.

- [ ] **Step 4: Run classifier tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_classifier.py -q`

Expected: all four outcomes, timeout, tie, neutral pause, and reset tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/classifier.py" "liquidity sweep/order_flow_engine/test_liquidity_event_classifier.py"
git commit -m "feat: classify deterministic liquidity outcomes"
```

### Task 6: Symbol-Local Engine, Buffers, Coverage, and Watermark

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/engine.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_engine.py`

**Interfaces:**
- Consumes: policy, adapter observations, store, evidence builder, classifier, `TradeCoverage`.
- Produces: `LiquidityEventEngine.on_trade(trade, coverage)`, `on_depth(depth)`, `on_sweep(observation, coverage_provider: TradeCoverageProvider)`, `advance_time(symbol, as_of_exchange_time_ms, coverage)`, `snapshot(symbol=None)`, and callbacks `on_transition`, `on_result`, `on_rejected`.

The engine test module uses `safe_coverage = TradeCoverage(feed_safe=True, known_gap=False, buffer_overflow=False, unresolved_sequence=False, interval_retained=True)` and `coverage_provider = StaticTradeCoverageProvider(safe_coverage)`, whose `coverage()` method returns its constructor value. Its helper factories create a new policy/store/engine for each test, feed explicitly listed trades, and return the engine; they must not monkeypatch internal timestamps.

- [ ] **Step 1: Write failing engine lifecycle and safety tests**

Cover these behaviors with explicit fixtures:

```python
def test_late_callback_reconstructs_when_full_history_is_retained():
    engine = engine_with_trades(sell_side_failed_breakdown_trades())
    result = engine.on_sweep(observation(event_time_ms=0, detection_time_ms=9_000), coverage_provider)
    assert result.market_resolution_time_ms == 5_000
    assert result.classification_time_ms == 9_000

def test_late_callback_is_invalid_when_history_was_evicted():
    engine = engine_with_oldest_retained_ms(5_000)
    result = engine.on_sweep(observation(event_time_ms=0, detection_time_ms=200_000), coverage_provider)
    assert result.reason_code == "INSUFFICIENT_REPLAY_HISTORY"

def test_late_trade_behind_watermark_is_quarantined_without_mutation():
    finalized = finalized_engine()
    before = finalized.results()[0]
    finalized.on_trade(trade(1_000), safe_coverage)
    assert finalized.results()[0] == before
    assert finalized.telemetry.late_data_count == 1

def test_quiet_healthy_interval_is_valid_but_gap_is_invalid():
    assert engine.advance_time("btcusdt", 60_000, TradeCoverage.safe_zero_activity()).reason_code == "CONFIRMATION_WINDOW_EXPIRED"
    gap = replace(safe_coverage, known_gap=True)
    assert engine_with_open_event().advance_time("btcusdt", 60_000, gap).reason_code == "MARKET_DATA_INTEGRITY_COMPROMISED"

def test_symbol_watermarks_are_isolated():
    engine.on_trade(trade(100_000, symbol="btcusdt"), safe_coverage)
    assert engine.watermark("ethusdt") is None
```

Also test no penetration, incomplete future coverage, unsafe mandatory trade data, optional unsafe depth, duplicate observations, collisions, immutable final events, and a 60-second indeterminate timeout.

- [ ] **Step 2: Run engine tests and verify RED**

Run: `python -m pytest test_liquidity_event_engine.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement symbol-local orchestration**

Use a `SymbolRuntime` containing sorted trade/depth buffers, `max_seen_exchange_time_ms`, watermark, and open event trackers. Prune only data older than `max_seen - market_buffer_retention_ms`. Insert tolerated out-of-order records with `bisect`; quarantine records older than the settled watermark.

On engine startup, query `SQLiteIdentityAuthority.pending_unresolved()`. For each claimed-unresolved identity, recover from the persisted observation payload only when complete required market history is retained; skip already persisted transition keys returned by `persisted_transition_sequences(event_id)`. If the complete required history is unavailable, finalize that identity exactly once as `INVALID / INSUFFICIENT_REPLAY_HISTORY`.

`on_sweep` must:

```python
1. atomically claim identity in `SQLiteIdentityAuthority`
2. open/deduplicate bounded working state in `LiquidityEventStore`
3. reject capacity, duplicate existing identities, identity conflicts, or ambiguous collisions
4. prove retained interval and TradeCoverage validity
5. replay buffered trades from event_time_ms in canonical order
6. validate strict penetration
7. settle candidates only through the symbol watermark
8. claim each transition sequence in the authority
9. commit final result identity in the authority before recorder notification
10. emit ordered lifecycle transitions and at most one immutable result
```

`advance_time` must require explicit coverage and never inspect wall clock. Mark unresolved replay events `INSUFFICIENT_FUTURE_COVERAGE` when the caller declares terminal input before the full interval.

- [ ] **Step 4: Run engine and lower-layer tests**

Run: `python -m pytest test_liquidity_event_models.py test_liquidity_event_adapter_store.py test_liquidity_event_evidence.py test_liquidity_event_classifier.py test_liquidity_event_engine.py -q`

Expected: all Phase 1C.1 core tests pass without warnings.

- [ ] **Step 5: Commit Task 6**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/engine.py" "liquidity sweep/order_flow_engine/test_liquidity_event_engine.py"
git commit -m "feat: orchestrate liquidity event lifecycle"
```

### Task 7: Canonical Recorder and Artifact Integrity

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/recorder.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_recorder.py`

**Interfaces:**
- Consumes: transitions, final results, rejected inputs, policy.
- Produces: `CanonicalRecord`, `RecorderTelemetry`, `LiquidityEventRecorder(mode, output_dir, queue_max_items)`, and `certify(expected_event_ids, expected_transition_keys) -> ArtifactIntegrity`.

The recorder test module's `render_records(output_dir, ordered_records)` creates a replay-mode recorder, enqueues every supplied canonical record, calls `flush_replay()`, and returns an object containing bytes read from all three fixed artifact filenames.

The recorder is not the identity authority. Recorder rows and certification can report artifact integrity, but duplicate suppression and one-result uniqueness are enforced by `SQLiteIdentityAuthority` before recorder notification.

- [ ] **Step 1: Write failing deterministic recorder tests**

```python
def test_replay_flush_is_byte_equivalent_across_enqueue_order(tmp_path):
    first = render_records(tmp_path / "a", records)
    second = render_records(tmp_path / "b", reversed(records))
    assert first.events_bytes == second.events_bytes
    assert first.transitions_bytes == second.transitions_bytes

def test_canonical_artifacts_exclude_machine_runtime_values(tmp_path):
    output = render_records(tmp_path, records).all_bytes.decode("utf-8")
    assert str(tmp_path) not in output
    assert "object at 0x" not in output

def test_queue_overflow_preserves_classification_and_fails_integrity(tmp_path):
    recorder = LiquidityEventRecorder("live", tmp_path, queue_max_items=1)
    recorder.enqueue(record_one)
    recorder.enqueue(record_two)
    assert recorder.telemetry.failure_count == 1
    assert recorder.artifact_integrity is ArtifactIntegrity.QUEUE_OVERFLOW
    assert immutable_result.classification is EventClassification.FAILED_BREAKDOWN
```

- [ ] **Step 2: Run recorder tests and verify RED**

Run: `python -m pytest test_liquidity_event_recorder.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement the three append-only schemas and two write modes**

Define fixed columns in module constants. Serialize floats with fixed decimal formats, enums by value, missing values as empty strings, and reasons/contradictions as compact sorted JSON arrays. Define canonical write keys exactly as the spec. Live mode uses a bounded deque/async flush; replay mode accumulates, stable-sorts, then synchronously writes UTF-8 with `newline=""` and `lineterminator="\n"`.

`certify` compares emitted event IDs and transition keys to expected sets and returns `MISSING_EVENT_ROW` or `MISSING_TRANSITION_ROW` before allowing `COMPLETE`.

- [ ] **Step 4: Run recorder tests and verify GREEN**

Run: `python -m pytest test_liquidity_event_recorder.py -q`

Expected: byte-equivalence, bounded queue, failure isolation, and integrity tests pass.

- [ ] **Step 5: Commit Task 7**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/recorder.py" "liquidity sweep/order_flow_engine/test_liquidity_event_recorder.py"
git commit -m "feat: record canonical liquidity artifacts"
```

### Task 8: Exact Liquidity Replay

**Files:**
- Create: `liquidity sweep/order_flow_engine/liquidity_event/replay.py`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_replay.py`

**Interfaces:**
- Consumes: existing gzip JSONL trade/depth recordings, sweep callback dictionaries, the production adapter/engine/identity authority/recorder/policy.
- Produces: `LiquidityReplayRunner.run() -> LiquidityReplayResult` with `artifact_integrity`, counts, failures, overflows, and policy hash.

`LiquidityReplayRunner` accepts `inputs: Iterable[ReplayInput]`, `output_dir: Path`, and optional policy. `ReplayInput` is a frozen union wrapper with constructors `from_trade`, `from_depth`, and `from_sweep_callback`. File parsing is provided by `LiquidityReplayRunner.from_recordings(market_paths, sweep_callbacks, output_dir, policy=None)`. Replay tests build `canonical_inputs` exclusively through these constructors; `drive_engine(inputs)` feeds the same normalized values to a production engine in processing-key order.

- [ ] **Step 1: Write failing live/replay parity and certification tests**

```python
def test_live_and_replay_produce_identical_event_sequence_and_flow(tmp_path):
    live = drive_engine(canonical_inputs)
    replay = LiquidityReplayRunner(canonical_inputs, tmp_path).run()
    assert replay.results == live.results
    assert replay.transitions == live.transitions
    assert replay.results[0].evidence.event_flow == live.results[0].evidence.event_flow
    assert replay.artifact_integrity is ArtifactIntegrity.COMPLETE

def test_replay_twice_produces_byte_identical_artifacts(tmp_path):
    one = run_replay(tmp_path / "one")
    two = run_replay(tmp_path / "two")
    assert one.artifact_hashes == two.artifact_hashes

def test_incomplete_future_coverage_is_not_certified(tmp_path):
    result = LiquidityReplayRunner(inputs_ending_before_event_window, tmp_path).run()
    assert result.results[0].reason_code == "INSUFFICIENT_FUTURE_COVERAGE"
```

- [ ] **Step 2: Run replay tests and verify RED**

Run: `python -m pytest test_liquidity_event_replay.py -q`

Expected: tests fail explicitly with `required Phase 1C.1 API is not implemented`; no collection error is accepted as RED.

- [ ] **Step 3: Implement canonical input merge and replay certification**

Normalize recording messages into `MarketTrade` and `DepthObservation`. Present sweeps at `detection_time_ms`, not event time. Stable-sort by `(processing_time_ms, exchange_sequence, source_rank, content_hash)`, where trade rank is `0`, depth `1`, and sweep `2`. After the last input, call per-symbol terminal advance only through proven coverage; do not invent coverage beyond the recording.

Use production `SweepsMonitorAdapter`, `LiquidityEventEngine`, `LiquidityEvidenceBuilder`, classifier, store, a fresh per-run `SQLiteIdentityAuthority`, and replay-mode recorder. Do not reuse live identity authority or any identity authority from a prior independent replay run. Return a non-complete result and CLI exit code `1` on artifact-integrity failure.

- [ ] **Step 4: Run replay and existing replay tests**

Run: `python -m pytest test_liquidity_event_replay.py test_replay.py -q`

Expected: exact liquidity replay and legacy replay tests pass.

- [ ] **Step 5: Commit Task 8**

```powershell
git add -- "liquidity sweep/order_flow_engine/liquidity_event/replay.py" "liquidity sweep/order_flow_engine/test_liquidity_event_replay.py"
git commit -m "feat: add exact liquidity event replay"
```

### Task 9: Shadow-Only Application Integration

**Files:**
- Modify: `liquidity sweep/order_flow_engine/config.py:94-135`
- Modify: `liquidity sweep/order_flow_engine/main.py:43-120,208-263,457-540,765-805,962-1135`
- Modify: `liquidity sweep/order_flow_engine/dashboard.py:19-90`
- Modify: `liquidity sweep/order_flow_engine/walkthrough.md`
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_integration.py`

**Interfaces:**
- Consumes: production Phase 1C.1 package.
- Produces: `/api/liquidity-events`, engine telemetry in dashboard payload, compact recent-event display, and application lifecycle startup/shutdown.

- [ ] **Step 1: Write failing authority, API, and integration tests**

```python
def test_frozen_authority_flags():
    assert config.LIQUIDITY_EVENT_ENGINE_ENABLED is True
    assert config.LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False
    assert config.REGIME_ENFORCEMENT_ENABLED is False
    assert config.EXECUTION_DISABLED is True

def test_engine_enabled_or_disabled_never_changes_scorer_output():
    baseline = evaluate_existing_scorer(engine_enabled=False)
    shadow = evaluate_existing_scorer(engine_enabled=True, liquidity_inputs=resolved_event_inputs)
    assert shadow == baseline

def test_read_only_liquidity_api_payload_has_no_mutation_controls():
    payload = app._liquidity_events_payload()
    assert set(payload) >= {"active", "recent", "engine"}
    assert payload["engine"]["enforcement_enabled"] is False
    assert not ({"classify", "open_event", "set_policy", "execute"} & set(payload))

def test_dashboard_labels_confidence_as_uncalibrated():
    html = OrderFlowDashboard(["BTCUSDT"]).render_html()
    assert "Liquidity Events (Shadow)" in html
    assert "Uncalibrated score" in html
    assert "chance of" not in html.lower()
```

- [ ] **Step 2: Run integration tests and verify RED**

Run: `python -m pytest test_liquidity_event_integration.py -q`

Expected: missing flags, payload method, and dashboard label failures.

- [ ] **Step 3: Add flags and instantiate the shadow engine**

In `config.py` add exactly:

```python
LIQUIDITY_EVENT_ENGINE_ENABLED = True
LIQUIDITY_EVENT_ENFORCEMENT_ENABLED = False
LIQUIDITY_EVENT_OUTPUT_DIR = os.path.join(BASE_DIR, "liquidity_event_artifacts")
```

In `OrderFlowEngine.__init__`, instantiate policy/identity authority/store/evidence/recorder/engine only when enabled. Wire callbacks to the recorder and dashboard. Do not pass the event engine to `OrderFlowScorer`, `PaperTrader`, or regime permissions.

- [ ] **Step 4: Feed canonical market and sweep observations**

After `FlowMetrics.add_trade`, create `MarketTrade` directly from the raw trade and pass explicit Guardian-derived `TradeCoverage`. After `FlowMetrics.update_depth`, pass `DepthObservation` with exchange event time and depth-weighted imbalance availability. In `_handle_sweep`, preserve all existing scorer logic and additionally adapt/open the shadow event.

Do not source event-local delta/CVD from `state.running_cvd_usdt`, `state.session_cvd_usdt`, or window metrics. Existing absorption/replenishment/stacking-pulling data may be passed only through `LegacyLiquidityContextAdapter`; incompatible observations remain unavailable.

- [ ] **Step 5: Add the read-only endpoint and compact display**

Add `elif path == "/api/liquidity-events"` to the existing GET route and return:

```python
{
    "active": engine.snapshot()["active"],
    "recent": engine.snapshot()["recent"],
    "engine": {
        "enabled": True,
        "enforcement_enabled": False,
        "active_event_count": len(snapshot["active"]),
        "rejected_input_count": telemetry.rejected_input_count,
        "late_data_count": telemetry.late_data_count,
        "recorder_status": recorder.telemetry.status,
        "recorder_failure_count": recorder.telemetry.failure_count,
        "pending_write_count": recorder.telemetry.pending_write_count,
        "policy_version": policy.model_version,
        "policy_hash": policy.policy_hash,
    },
}
```

Add a small shadow section to the existing web payload and recent-event stream. Do not add any POST route, controls, probability wording, scorer gates, or trade actions. Document flags, endpoint, and artifacts in `walkthrough.md`.

- [ ] **Step 6: Run integration and legacy safety tests**

Run: `python -m pytest test_liquidity_event_integration.py test_liquidity.py test_data_integrity.py test_regime.py -q`

Expected: shadow integration and legacy safety/regime tests pass without warnings.

- [ ] **Step 7: Commit Task 9**

```powershell
git add -- "liquidity sweep/order_flow_engine/config.py" "liquidity sweep/order_flow_engine/main.py" "liquidity sweep/order_flow_engine/dashboard.py" "liquidity sweep/order_flow_engine/walkthrough.md" "liquidity sweep/order_flow_engine/test_liquidity_event_integration.py"
git commit -m "feat: integrate liquidity engine in shadow mode"
```

### Task 10: Architecture and 40-Gate Acceptance Audit

**Files:**
- Modify: all Phase 1C.1 test files created in Tasks 1-9 only where a frozen gate lacks direct coverage.
- Create: `liquidity sweep/order_flow_engine/test_liquidity_event_acceptance.py`

**Interfaces:**
- Consumes: complete Phase 1C.1 implementation.
- Produces: one explicit audit test or referenced lower-level test for every frozen gate.

- [ ] **Step 1: Write failing architecture-boundary tests**

```python
FORBIDDEN = {"scoring", "data.paper_trader", "regime.permissions", "execution", "position_sizing"}

def test_liquidity_event_package_has_no_authority_imports():
    imports = collect_imports(Path("liquidity_event"))
    assert not any(name == item or name.startswith(item + ".") for name in imports for item in FORBIDDEN)

def test_classifier_is_independent_of_wall_clock(monkeypatch):
    def forbidden_wall_clock():
        raise AssertionError("classification consulted wall clock")
    monkeypatch.setattr(time, "time", forbidden_wall_clock)
    tracker = feed_completed_sell_side_reclaim()
    assert tracker.decision_at(6_000, 60_000).classification is EventClassification.FAILED_BREAKDOWN

def test_core_event_modules_do_not_import_flow_metrics():
    imports = collect_imports(Path("liquidity_event"), only={"classifier.py", "engine.py", "evidence.py"})
    assert "flow_metrics" not in imports

def test_phase1c_does_not_mutate_paper_or_regime_authority():
    assert config.EXECUTION_DISABLED is True
    assert config.REGIME_ENFORCEMENT_ENABLED is False
    assert config.LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False
```

- [ ] **Step 2: Add a gate-to-test manifest and fail on missing coverage**

Define the manifest with these exact pytest node IDs and assert that its keys equal `set(range(1, 41))`. Use the frozen spec's numbered requirement text as comments beside the corresponding entries. This is traceability, not a substitute for behavioral assertions.

```python
ACCEPTANCE_GATE_TESTS = {
    1: "test_liquidity_event_classifier.py::test_sell_side_reclaim_is_failed_breakdown",
    2: "test_liquidity_event_classifier.py::test_sell_side_acceptance_is_bearish_continuation",
    3: "test_liquidity_event_classifier.py::test_buy_side_reclaim_is_failed_breakout",
    4: "test_liquidity_event_classifier.py::test_buy_side_acceptance_is_bullish_continuation",
    5: "test_liquidity_event_engine.py::test_insufficient_follow_through_is_indeterminate",
    6: "test_liquidity_event_engine.py::test_no_actual_penetration_is_invalid",
    7: "test_liquidity_event_adapter_store.py::test_duplicate_callback_uses_identity_authority_without_reopening_event",
    8: "test_liquidity_event_engine.py::test_out_of_order_inside_tolerance_settles_deterministically",
    9: "test_liquidity_event_engine.py::test_late_trade_behind_watermark_is_quarantined_without_mutation",
    10: "test_liquidity_event_engine.py::test_confirmation_window_timeout_is_indeterminate",
    11: "test_liquidity_event_engine.py::test_unsafe_trade_coverage_invalidates_unresolved_event",
    12: "test_liquidity_event_engine.py::test_unsafe_depth_keeps_price_classification_and_marks_context_unavailable",
    13: "test_liquidity_event_engine.py::test_finalized_event_is_immutable",
    14: "test_liquidity_event_replay.py::test_replay_twice_produces_byte_identical_artifacts",
    15: "test_liquidity_event_integration.py::test_engine_enabled_or_disabled_never_changes_scorer_output",
    16: "test_liquidity_event_integration.py::test_execution_disabled_and_paper_behavior_unchanged",
    17: "test_liquidity_event_integration.py::test_regime_enforcement_remains_disabled",
    18: "test_liquidity_event_adapter_store.py::test_bullish_and_bearish_legacy_directions_map_only_in_adapter",
    19: "test_liquidity_event_classifier.py::test_neutral_zone_pauses_qualifying_duration",
    20: "test_liquidity_event_classifier.py::test_opposite_threshold_resets_candidate",
    21: "test_liquidity_event_classifier.py::test_equal_sufficient_times_are_indeterminate",
    22: "test_liquidity_event_engine.py::test_late_callback_reconstructs_when_full_history_is_retained",
    23: "test_liquidity_event_engine.py::test_late_callback_is_invalid_when_history_was_evicted",
    24: "test_liquidity_event_adapter_store.py::test_distinct_levels_and_opposite_sides_remain_separate",
    25: "test_liquidity_event_adapter_store.py::test_close_same_side_levels_report_ambiguous_collision",
    26: "test_liquidity_event_models.py::test_v1_policy_hash_is_stable_and_validates_retention",
    27: "test_liquidity_event_recorder.py::test_reasons_and_rows_serialize_deterministically",
    28: "test_liquidity_event_acceptance.py::test_liquidity_event_package_has_no_authority_imports",
    29: "test_liquidity_event_recorder.py::test_queue_overflow_preserves_classification_and_fails_integrity",
    30: "test_liquidity_event_replay.py::test_live_and_replay_produce_identical_event_sequence_and_flow",
    31: "test_liquidity_event_engine.py::test_symbol_watermarks_are_isolated",
    32: "test_liquidity_event_replay.py::test_incomplete_future_coverage_is_not_certified",
    33: "test_liquidity_event_recorder.py::test_store_and_recorder_bounds_are_enforced",
    34: "test_liquidity_event_evidence.py::test_live_and_replay_trade_order_produce_identical_flow_snapshot",
    35: "test_liquidity_event_evidence.py::test_optional_context_is_typed_and_never_reimplemented",
    36: "test_liquidity_event_adapter_store.py::test_level_identity_preserves_source_and_has_deterministic_fallback",
    37: "test_liquidity_event_recorder.py::test_recorder_failure_preserves_result_but_invalidates_research",
    38: "test_liquidity_event_recorder.py::test_replay_flush_is_byte_equivalent_across_enqueue_order",
    39: "test_liquidity_event_engine.py::test_quiet_healthy_interval_is_valid_but_gap_is_invalid",
    40: "test_liquidity_event_adapter_store.py::test_finalized_identity_survives_eviction_restart_and_conflict_fails_closed",
}

def test_all_frozen_acceptance_gates_have_named_behavioral_tests():
    assert set(ACCEPTANCE_GATE_TESTS) == set(range(1, 41))
```

- [ ] **Step 3: Run acceptance audit and verify RED/GREEN honestly**

Run: `python -m pytest test_liquidity_event_acceptance.py -q`

Expected before filling any uncovered gate: failure naming the missing gate. Add only the missing behavioral test to the owning lower-level file, rerun its focused node to observe RED, implement/fix production behavior, then rerun until all 40 mappings pass.

- [ ] **Step 4: Run the complete Phase 1C.1 suite**

Run:

```powershell
python -m pytest test_liquidity_event_models.py test_liquidity_event_adapter_store.py test_liquidity_event_evidence.py test_liquidity_event_classifier.py test_liquidity_event_engine.py test_liquidity_event_recorder.py test_liquidity_event_replay.py test_liquidity_event_integration.py test_liquidity_event_acceptance.py -q
```

Expected: all Phase 1C.1 tests pass with no warnings.

- [ ] **Step 5: Commit the acceptance audit**

```powershell
git add -- "liquidity sweep/order_flow_engine/test_liquidity_event_models.py" "liquidity sweep/order_flow_engine/test_liquidity_event_adapter_store.py" "liquidity sweep/order_flow_engine/test_liquidity_event_evidence.py" "liquidity sweep/order_flow_engine/test_liquidity_event_classifier.py" "liquidity sweep/order_flow_engine/test_liquidity_event_engine.py" "liquidity sweep/order_flow_engine/test_liquidity_event_recorder.py" "liquidity sweep/order_flow_engine/test_liquidity_event_replay.py" "liquidity sweep/order_flow_engine/test_liquidity_event_integration.py" "liquidity sweep/order_flow_engine/test_liquidity_event_acceptance.py"
git commit -m "test: certify Phase 1C.1 acceptance gates"
```

### Task 11: Full Regression and Final Integrity Verification

**Files:**
- No production edits unless a failing test exposes a real implementation defect.
- Verify: `docs/superpowers/specs/2026-08-11-liquidity-event-engine-foundation-design.md` is version `1.0.1` and includes Spec Patch 001 persistent identity authority.

**Interfaces:**
- Consumes: complete repository and clean Phase 1C.1 commits.
- Produces: final verification evidence suitable for Phase 1C.1 sign-off.

- [ ] **Step 1: Prove the patched frozen specification is present**

Run:

```powershell
Select-String -Path "docs/superpowers/specs/2026-08-11-liquidity-event-engine-foundation-design.md" -Pattern "Spec Version:\\*\\* 1.0.1","PERSISTENT_EXACT_IDENTITY_AUTHORITY = APPROVED","IDENTITY_RETENTION = INDEFINITE"
```

Expected: all three patterns are found.

- [ ] **Step 2: Run syntax/import verification**

Run:

```powershell
python -m compileall -q liquidity_event
python -c "from liquidity_event.engine import LiquidityEventEngine; from liquidity_event.identity_authority import SQLiteIdentityAuthority; from liquidity_event.replay import LiquidityReplayRunner"
```

Expected: exit code `0`, no output or warnings.

- [ ] **Step 3: Run the complete repository test suite**

Run: `python -m pytest -q`

Expected: all legacy and Phase 1C.1 tests pass with no warning summary.

- [ ] **Step 4: Verify authority flags and forbidden imports independently**

Run:

```powershell
python -c "import config; assert config.LIQUIDITY_EVENT_ENGINE_ENABLED is True; assert config.LIQUIDITY_EVENT_ENFORCEMENT_ENABLED is False; assert config.REGIME_ENFORCEMENT_ENABLED is False; assert config.EXECUTION_DISABLED is True"
python -m pytest test_liquidity_event_acceptance.py::test_liquidity_event_package_has_no_authority_imports -q
```

Expected: both commands pass.

- [ ] **Step 5: Inspect final scope without staging unrelated work**

Run:

```powershell
git status --short
git diff --stat
git log --oneline --decorate -n 20
```

Expected: implementation commits contain only Phase 1C.1 files; the pre-existing order-book modifications may still appear as unstaged working-tree changes and must remain outside every Phase 1C.1 commit.

- [ ] **Step 6: Record final verification in the implementation handoff**

Report exact test counts, artifact-integrity result, policy hash, patched spec version, authority flag values, and any residual operational limitation. Do not claim completion if artifact integrity is not `COMPLETE` or if any warning appears.

## Acceptance Gate Traceability

| Frozen gates | Owning task |
|---|---|
| 1-5 outcome truth table and unresolved behavior | Task 5, Task 6 |
| 6 penetration validation | Task 6 |
| 7 duplicate suppression | Task 2, Task 3A, Task 3B |
| 8-9 reorder settlement and late-data immutability | Task 6 |
| 10 timeout | Task 5, Task 6 |
| 11-12 Guardian trade/depth semantics | Task 6 |
| 13 finalized immutability | Task 6 |
| 14 deterministic bytes | Task 7, Task 8 |
| 15-17 scorer/paper/regime authority isolation | Task 9, Task 10 |
| 18 legacy side mapping | Task 2 |
| 19-21 neutral pause, opposite reset, tie | Task 5 |
| 22-23 late callback reconstruction/eviction | Task 6 |
| 24-25 overlap and collision semantics | Task 3A, Task 3B, Task 6 |
| 26-27 policy hash and serialization | Task 1, Task 7 |
| 28 forbidden imports | Task 10 |
| 29 recorder failure isolation/bounds | Task 7 |
| 30 live/replay parity | Task 8 |
| 31 symbol-local watermarks | Task 6 |
| 32 incomplete future coverage | Task 6, Task 8 |
| 33 bounded stores/queues | Task 3B, Task 7 |
| 34 event-local flow parity | Task 4, Task 8 |
| 35 typed legacy detector boundary | Task 4, Task 9 |
| 36 source-level identity fallback | Task 2, Task 3A, Task 3B |
| 37 artifact-integrity invalidation | Task 7, Task 8 |
| 38 canonical flush ordering | Task 7 |
| 39 healthy quiet interval versus known gap | Task 6 |
| 40 finalized identity survives volatile eviction/restart and conflicts fail closed | Task 3A, Task 3B, Task 6 |
