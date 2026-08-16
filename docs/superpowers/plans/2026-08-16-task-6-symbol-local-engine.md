# TICS Phase 1C.1 — Task 6 Implementation Plan: Symbol-Local Engine, Buffers, Coverage, and Watermark (Final)

## Executive Summary

Task 6 implements **`LiquidityEventEngine`** in `liquidity_event/engine.py`, adds the narrow **`restore_claimed_unresolved`** recovery API to `LiquidityEventStore` (`liquidity_event/event_store.py`), and implements the complete Task 6 test suite in `test_liquidity_event_engine.py` (with adapter/store recovery tests in `test_liquidity_event_adapter_store.py`).

Task 6 is the central deterministic orchestrator of Phase 1C.1 that joins all lower layers (models, policy, SQLite identity authority, bounded volatile store, sweep adapters, typed evidence builder, deterministic price outcome classifier) into an integrated point-in-time liquidity lifecycle engine.

---

## User Review Required

> [!IMPORTANT]
> **Key Architectural Invariants & Final Plan Hardening:**
> 1. **Real Recovery Types (`PersistedIdentity`)**: `store.restore_claimed_unresolved(observation, persisted_identity)` takes the actual `PersistedIdentity` type from `identity_authority.py` and checks `if persisted_identity.status != "CLAIMED_UNRESOLVED":`. No custom status enums are introduced.
> 2. **Bounded Interval-Aware Coverage Ledger (`CoverageSegment`)**:
>    - Each `SymbolRuntime` maintains a `coverage_ledger: list[CoverageSegment]` where `CoverageSegment(start_ms, end_ms, coverage: TradeCoverage)`.
>    - Event integrity is evaluated strictly across overlapping segments in $[E, \text{expiry\_ms}]$. A later safe segment does NOT erase an earlier historical gap. A gap before $E$ does NOT invalidate an event starting at $E$.
>    - Ledger segments are pruned beyond the retention horizon (`end_ms < max_seen - market_buffer_retention_ms`).
> 3. **Split Event-Integrity vs Clock-Advancement Coverage**:
>    - `event_integrity_coverage` is evaluated over $[E, \min(\text{watermark}, \text{expiry\_ms})]$ or $[E, \text{expiry\_ms}]$.
>    - `clock_advancement_coverage` is evaluated over $[\text{current\_proven\_time}, \text{detection\_time\_ms}]$.
>    - A post-expiry gap (e.g. $[75k, 80k]$) does NOT invalidate an already-resolved event $[0, 60k]$, but prevents advancing the symbol clock across unproven intervals to detection time ($90k$).
> 4. **`max_seen_exchange_time_ms is None` Handling**: If no trades have been observed on a symbol (`max_seen is None`), missing history is not inferred from absence of trades; the engine relies on explicit `TradeCoverage.interval_retained` and provider state. Quiet healthy markets remain valid zero-activity evidence.
> 5. **Settled-Only Trade Delivery**: Provisional trades newer than the settled watermark are held in sorted buffers and deduplicated. Only newly settled canonical records ($\le \text{watermark}$) are fed to `PriceOutcomeTracker` and `EventFlowAccumulator` in strict `canonical_key` order.
> 6. **Watermark Calculation & Timeout**: $\text{watermark} = \text{max\_seen\_exchange\_time\_ms} - \text{reorder\_tolerance\_ms}$ (with 2,000 ms tolerance). Timeout at $\text{expiry\_ms} = E + 60,000$ resolves only when $\text{watermark} \ge \text{expiry\_ms}$ (i.e. $\text{max\_seen} \ge E + 62,000$) with $\text{market\_resolution\_time\_ms} = \text{expiry\_ms}$.
> 7. **Previous Watermark Late Boundary**: Late data is evaluated against the *previous settled watermark* using $\text{exchange\_time\_ms} \le \text{previous\_watermark}$. Accepted records update `max_seen`, calculate the new watermark, and process newly settled trades.
> 8. **Ambiguous Collision Orchestration**: When `admit_observation` returns `collision_event_ids`, all members in the collision set are finalized deterministically in sorted event ID order as `INVALID / AMBIGUOUS_EVENT_COLLISION`.
> 9. **Strict Point-in-Time Evidence Finalization**: Flow snapshot is taken strictly at `as_of_ms = market_resolution_time_ms` (or `expiry_ms` for timeout). Final evidence values are enriched with price-state witness fields (`confirmed_penetration_price`, `confirmed_penetration_bps`, `penetration_time_ms`, `reclaim_sufficient_time_ms`, `reclaim_duration_ms`, `reclaim_trade_count`, `acceptance_sufficient_time_ms`, `acceptance_duration_ms`, `acceptance_trade_count`).
> 10. **Authority-Before-Memory/Callback Ordering**: Hard transactional ordering where failure of an authority operation halts mutation, emits no external callbacks, and flags failure/degraded telemetry.

---

## Core Engine Flow Specifications

### 1. `on_trade(trade, coverage)`
```text
1. Identify/create symbol runtime.
2. Record coverage in runtime.coverage_ledger: CoverageSegment(trade.exchange_time_ms, trade.exchange_time_ms, coverage).
3. Read PREVIOUS settled watermark: prev_wm = runtime.watermark_ms.
4. If prev_wm is not None and trade.exchange_time_ms <= prev_wm:
      Quarantine trade; telemetry.late_data_count += 1; return.
5. Deduplicate provisional trade identity (symbol, sequence_id):
      - Exact duplicate -> ignore.
      - Conflicting duplicate while provisional -> keep canonical minimum by canonical_key.
6. Insert/update sorted provisional buffer (bisect by canonical_key).
7. Advance runtime max_seen_exchange_time_ms = max(max_seen or 0, trade.exchange_time_ms).
8. Calculate new watermark: new_wm = max_seen - policy.reorder_tolerance_ms.
9. Extract newly settled canonical records (where prev_wm is None or t > prev_wm, and t <= new_wm).
10. For each active event on this symbol:
      a. Query coverage_ledger for overlapping segments in [event_time_ms, min(new_wm, expiry)]:
         - If any segment has feed_safe is False -> finalize INVALID / MARKET_DATA_UNSAFE.
         - If any segment has known_gap / buffer_overflow / unresolved_sequence -> finalize INVALID / MARKET_DATA_INTEGRITY_COMPROMISED.
      b. Feed newly settled trades in canonical_key order where event_time <= t <= expiry.
      c. Evaluate tracker.decision_at(watermark_ms=new_wm, expiry_ms=expiry).
      d. If decision resolved:
         - Emit intermediate transitions (OBSERVED -> PENETRATION_VALIDATING -> SWEEP_DETECTED -> RECLAIMING/ACCEPTING/UNRESOLVED -> FINALIZED).
         - Persist transitions in SQLite authority BEFORE in-memory state mutation and on_transition callback.
         - Build point-in-time LiquidityEvidence at as_of_ms=decision.market_resolution_time_ms.
         - Persist final LiquidityEventResult in SQLite authority BEFORE store.finalize() and on_result callback.
11. Prune trade buffer, dedup index, and coverage_ledger for entries older than max_seen - market_buffer_retention_ms.
```

### 2. `on_sweep(observation, coverage_provider)`
```text
1. Obtain event-integrity coverage: cov_event = coverage_provider.coverage(observation.symbol, observation.event_time_ms, min(observation.detection_time_ms, observation.event_time_ms + policy.confirmation_window_ms)).
2. Record cov_event in runtime.coverage_ledger.
3. Call store.admit_observation(observation, authority).
4. If admission.rejection is not None:
      on_rejected(admission.rejection); return admission.rejection.
5. If admission.duplicate is True:
      return admission.event.
6. If admission.collision_event_ids:
      Deterministically finalize all colliding events in sorted event_id order as INVALID / AMBIGUOUS_EVENT_COLLISION.
7. If admission.created is False or admission.event is None:
      return None.
8. Verify retained history:
      If runtime.max_seen_exchange_time_ms is not None:
         retained_from_ms = runtime.max_seen_exchange_time_ms - policy.market_buffer_retention_ms
         if observation.event_time_ms < retained_from_ms or not cov_event.interval_retained:
            Finalize immediately as INVALID / INSUFFICIENT_REPLAY_HISTORY; return result.
      Else:
         if not cov_event.interval_retained:
            Finalize immediately as INVALID / INSUFFICIENT_REPLAY_HISTORY; return result.
9. Verify event coverage validity:
      If not cov_event.valid:
         Map reason (MARKET_DATA_UNSAFE or MARKET_DATA_INTEGRITY_COMPROMISED) and finalize immediately; return result.
10. Initialize EventTrackerContext (PriceOutcomeTracker, EventFlowAccumulator).
11. Replay SETTLED canonical trades from buffer where event_time <= t <= min(current_watermark or -1, expiry).
12. If observation.detection_time_ms > (runtime.max_seen_exchange_time_ms or 0):
       Query clock-advancement coverage: cov_clock = coverage_provider.coverage(observation.symbol, runtime.max_seen_exchange_time_ms or observation.event_time_ms, observation.detection_time_ms).
       If cov_clock.valid:
          Advance symbol max_seen to observation.detection_time_ms and process newly settled trades.
13. Evaluate tracker.decision_at(watermark_ms=runtime.watermark_ms, expiry_ms=expiry).
14. If resolved: finalize and persist result point-in-time at market_resolution_time_ms.
```

### 3. `advance_time(symbol, as_of_exchange_time_ms, coverage, terminal_input=False)`
```text
1. Record coverage in runtime.coverage_ledger: CoverageSegment(runtime.max_seen_exchange_time_ms or as_of_exchange_time_ms, as_of_exchange_time_ms, coverage).
2. Advance ONLY the requested symbol: runtime.max_seen_exchange_time_ms = max(runtime.max_seen_exchange_time_ms or 0, as_of_exchange_time_ms).
3. Calculate new watermark: new_wm = runtime.max_seen_exchange_time_ms - policy.reorder_tolerance_ms.
4. Extract and feed newly settled canonical records to active events.
5. For each active event on symbol:
      a. Check coverage_ledger overlapping [event_time_ms, min(new_wm, expiry)] -> finalize with exact reason mapping if compromised.
      b. Evaluate tracker.decision_at(watermark_ms=new_wm, expiry_ms=expiry).
      c. If resolved (including CONFIRMATION_WINDOW_EXPIRED when new_wm >= expiry):
         Finalize with market_resolution_time_ms = decision.market_resolution_time_ms.
      d. If terminal_input is True and event remains unresolved and new_wm < expiry:
         Finalize as INVALID / INSUFFICIENT_FUTURE_COVERAGE.
```

### 4. `recover_claimed_unresolved(coverage_provider)`
```text
1. Query authority.pending_unresolved().
2. For each persisted_identity in pending_unresolved():
      a. Check persisted_identity.status: must be "CLAIMED_UNRESOLVED" (skip if "FINALIZED").
      b. Reconstruct LiquiditySweepObservation from persisted_identity.observation payload.
      c. Call store.restore_claimed_unresolved(observation, persisted_identity).
      d. Query coverage for [event_time_ms, detection_time_ms]. If history evicted -> finalize as INVALID / INSUFFICIENT_REPLAY_HISTORY.
      e. Replay settled canonical trades from event_time_ms.
      f. Read existing persisted transition sequence set: persisted_seqs = authority.persisted_transition_sequences(event_id).
      g. Skip already persisted transition sequences; persist only new transitions.
      h. If settled, persist result and finalize.
```

---

## Proposed Code Changes

### Component 1: Store Recovery Extension (`liquidity_event/event_store.py`) [MODIFY]

#### [MODIFY] [event_store.py](file:///C:/Users/karan/Desktop/orderflow/.worktrees/phase-1c1-liquidity-event-engine/liquidity%20sweep/order_flow_engine/liquidity_event/event_store.py)
- Import `PersistedIdentity` from `.identity_authority`.
- Add `restore_claimed_unresolved(observation: LiquiditySweepObservation, persisted_identity: PersistedIdentity) -> OpenEventResult`:
  - Enforces `persisted_identity.status == "CLAIMED_UNRESOLVED"`.
  - Checks volatile capacity for `observation.symbol`.
  - If event is already active, returns duplicate safely.
  - Opens `LiquidityEvent(observation=observation)` in volatile `self._active`.
  - Does NOT call `claim_observation()` or modify SQLite authority.

### Component 2: Primary Engine Implementation (`liquidity_event/engine.py`) [NEW]

#### [NEW] [engine.py](file:///C:/Users/karan/Desktop/orderflow/.worktrees/phase-1c1-liquidity-event-engine/liquidity%20sweep/order_flow_engine/liquidity_event/engine.py)
- Implement `CoverageSegment` dataclass and interval-aware `coverage_ledger` in `SymbolRuntime`.
- Implement `SymbolRuntime`, `EventTrackerContext`, `EngineTelemetry`, `EngineSnapshot`, and `LiquidityEventEngine`.
- Implement settled-only trade delivery, previous-watermark late data quarantine, provisional dedup, `max_seen is None` quiet market support, split event-integrity vs clock-advancement coverage, exact coverage reason mapping, collision set resolution, point-in-time evidence builder, and atomic authority persistence.

### Component 3: Package Exports (`liquidity_event/__init__.py`) [MODIFY]

#### [MODIFY] [__init__.py](file:///C:/Users/karan/Desktop/orderflow/.worktrees/phase-1c1-liquidity-event-engine/liquidity%20sweep/order_flow_engine/liquidity_event/__init__.py)
- Export `LiquidityEventEngine`, `EngineTelemetry`, `EngineSnapshot`, `SymbolRuntime`, `CoverageSegment`.

### Component 4: Store & Adapter Unit Tests (`test_liquidity_event_adapter_store.py`) [MODIFY]

#### [MODIFY] [test_liquidity_event_adapter_store.py](file:///C:/Users/karan/Desktop/orderflow/.worktrees/phase-1c1-liquidity-event-engine/liquidity%20sweep/order_flow_engine/test_liquidity_event_adapter_store.py)
- Add unit tests for `store.restore_claimed_unresolved`:
  - `test_restore_claimed_unresolved_uses_persisted_identity_without_new_claim`
  - Rejection of `FINALIZED` record restore (`status != "CLAIMED_UNRESOLVED"`).
  - Idempotent duplicate handling if restored twice.
  - Capacity check enforcement on restore.

### Component 5: Engine Unit & Integration Test Suite (`test_liquidity_event_engine.py`) [NEW]

#### [NEW] [test_liquidity_event_engine.py](file:///C:/Users/karan/Desktop/orderflow/.worktrees/phase-1c1-liquidity-event-engine/liquidity%20sweep/order_flow_engine/test_liquidity_event_engine.py)
- Comprehensive test catalog including:
  1. `test_advance_time_at_expiry_does_not_timeout_before_reorder_settlement`
  2. `test_timeout_occurs_only_after_expiry_plus_reorder_tolerance`
  3. `test_duplicate_market_trade_does_not_double_confirm`
  4. `test_provisional_conflicting_trade_duplicate_resolves_canonically`
  5. `test_record_at_previous_watermark_is_quarantined`
  6. `test_quiet_interval_does_not_fail_oldest_trade_history_check`
  7. `test_quiet_symbol_with_no_max_seen_can_have_valid_retained_history`
  8. `test_late_callback_advances_from_detection_only_with_valid_coverage`
  9. `test_detection_clock_does_not_advance_across_unproven_coverage`
  10. `test_post_expiry_gap_does_not_invalidate_resolved_event`
  11. `test_historical_gap_survives_later_safe_coverage`
  12. `test_gap_before_event_does_not_invalidate_event`
  13. `test_coverage_ledger_is_symbol_local`
  14. `test_coverage_ledger_prunes_with_retention`
  15. `test_depth_timestamp_alone_cannot_finalize_without_trade_coverage`
  16. `test_feed_unsafe_maps_market_data_unsafe`
  17. `test_known_gap_maps_market_data_integrity_compromised`
  18. `test_buffer_overflow_maps_market_data_integrity_compromised`
  19. `test_unresolved_sequence_maps_market_data_integrity_compromised`
  20. `test_same_side_ambiguous_collision_invalidates_all_members`
  21. `test_opposite_side_overlap_remains_separate`
  22. `test_final_evidence_uses_market_resolution_not_detection_time`
  23. `test_post_resolution_trade_does_not_change_final_cvd`
  24. `test_authority_transition_failure_emits_no_callback`
  25. `test_authority_result_failure_emits_no_result_callback`
  26. `test_restart_recovery_does_not_call_normal_observation_claim`
  27. `test_restart_recovery_restores_claimed_unresolved_into_volatile_store`
  28. `test_restart_recovery_skips_persisted_transition_sequences`
  29. End-to-end truth table tests (`FAILED_BREAKDOWN`, `FAILED_BREAKOUT`, `BEARISH_CONTINUATION`, `BULLISH_CONTINUATION`).

---

## Verification Plan

### Automated Tests
1. **RED Verification**: `python -m pytest test_liquidity_event_engine.py -q`
2. **Focused Store Extension Tests**: `python -m pytest test_liquidity_event_adapter_store.py -q`
3. **Focused Engine Tests**: `python -m pytest test_liquidity_event_engine.py -q`
4. **All Phase 1C.1 Tests (Tasks 1–6)**: `python -m pytest test_liquidity_event_models.py test_liquidity_event_adapter_store.py test_liquidity_event_evidence.py test_liquidity_event_classifier.py test_liquidity_event_engine.py -q`
5. **Compilation Verification**: `python -m compileall -q liquidity_event`
6. **Full Test Suite**: `python -m pytest -ra -W default`
7. **Git Diff Hygiene**: `git diff --check`
8. **Specialist Subagent Reviews**:
   - Timing & Boundary Reviewer
   - Determinism Reviewer
   - Architecture & Authority Boundary Reviewer
