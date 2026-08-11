# Phase 1C.1 Liquidity Event Engine Foundation Design

**Status:** Approved design, ready for implementation planning

**Date:** 2026-08-11

## Objective

Phase 1C.1 will detect liquidity being taken, reconstruct the point-in-time market response, and deterministically classify the break as rejected, accepted, unresolved, or invalid. It will produce canonical evidence and research artifacts without receiving any trading authority.

The milestone is an intelligence and research layer. It must not alter scanner scores, suppress or open trades, change paper trading, modify sizing or cooldowns, or consume regime permissions.

## Scope

Phase 1C.1 includes:

- `SweepsMonitor` CSV callback normalization behind a typed adapter.
- Independent traded-price penetration validation.
- Canonical event identity and semantic deduplication.
- Buffered point-in-time trade and depth evidence.
- A deterministic lifecycle and outcome classifier.
- Timestamped supporting and contradictory evidence.
- Watermark-based handling of tolerated out-of-order market data.
- Deterministic CSV recording, a read-only API, a compact shadow display, and exact replay.

Phase 1C.1 excludes:

- Native sweep detection or a rewrite of `SweepsMonitor`.
- Regime-conditioned liquidity classification.
- Machine learning or calibrated probability claims.
- Any scoring, execution, allocation, sizing, or trade-governance behavior.
- Phase 1C.2 absorption/replenishment expansion and Phase 1C.3 liquidity mapping.

## Authority Boundary

The engine may observe, classify, record, replay, expose, and display liquidity events. It may not alter scores, block trades, open trades, size positions, or execute orders.

The following settings remain frozen for this milestone:

```python
LIQUIDITY_EVENT_ENGINE_ENABLED = True
LIQUIDITY_EVENT_ENFORCEMENT_ENABLED = False
REGIME_ENFORCEMENT_ENABLED = False
EXECUTION_DISABLED = True
```

Modules under `liquidity_event` must not import `scoring`, paper-trading modules, execution modules, position-sizing modules, or `regime.permissions`. They may consume neutral market-data models and normalized observations.

## Architecture

```text
SweepsMonitor callback
        |
        v
SweepsMonitorAdapter ----> rejected input recorder
        |
        v
immutable LiquiditySweepObservation
        |
        v
LiquidityEventEngine <---- bounded trade/depth buffers
        |
        +---- penetration validation
        +---- semantic deduplication
        +---- lifecycle management
        +---- canonical ordering and watermark settlement
        +---- timeout handling
        |
        v
LiquidityEvidenceBuilder
        |
        v
DeterministicLiquidityClassifier
        |
        v
immutable LiquidityEventResult
        |
        +---- recorder
        +---- read-only API/display
        +---- exact replay
```

The production package is decomposed as follows:

```text
liquidity_event/
    __init__.py
    models.py
    policy.py
    sweep_adapter.py
    event_store.py
    evidence.py
    classifier.py
    engine.py
    recorder.py
    replay.py
```

`models.py` owns typed immutable inputs/results, mutable internal event state, enums, and availability wrappers. `policy.py` owns all behavior-affecting constants and the policy hash. `sweep_adapter.py` is the only module that knows legacy sweep terminology. `event_store.py` owns identity, collisions, retention, and lookup. `evidence.py` derives timestamped evidence without choosing an outcome. `classifier.py` applies the frozen truth table. `engine.py` owns buffers, ordering, watermarking, and lifecycle transitions. `recorder.py` owns deterministic artifacts and telemetry. `replay.py` merges recorded sources and drives the same live components.

## Canonical Sweep Input

For Phase 1C.1, `SweepsMonitor` CSV callbacks are the authoritative candidate-sweep source. The engine never reads CSV columns directly.

The existing contract uses reaction-direction terminology. This mapping is frozen in adapter tests:

```text
BULLISH -> SELL_SIDE
BEARISH -> BUY_SIDE
```

`BULLISH` currently describes a low/sell-side sweep with an expected bullish response. `BEARISH` describes a high/buy-side sweep with an expected bearish response. No code outside the adapter may depend on these legacy strings.

The adapter produces a frozen observation containing:

```text
event_id
source_event_id
symbol
liquidity_side
swept_level
event_time_ms
detection_time_ms
source_sweep_price
source_penetration_bps
source
source_file_id
source_row_hash
source_observation_hash
detector_version
```

`source_file_id` is a stable logical identifier, never an absolute path. `source_row_hash` is populated when the monitor can provide the original row; otherwise it is `None`. `source_observation_hash`, computed from canonical callback content, is always present.

Missing source sweep price and penetration remain `None`. The event engine derives `confirmed_sweep_price` and `confirmed_penetration_bps` only from retained canonical trades.

Malformed observations are not guessed or partially accepted. They are written to `liquidity_rejected_inputs.csv` with `INVALID_SWEEP_OBSERVATION` and a deterministic reason code.

## Event Identity and Collision Rules

The semantic event ID is SHA-256 over canonical serialized values:

```text
source
stable source_event_id when available
normalized symbol
liquidity side
event_time_ms
normalized swept level
```

CSV path, row number, temporary directory, process identity, and machine identity are not part of canonical identity.

Repeated callbacks with the same event ID return the existing event and create no duplicate transition or result. A semantic collision candidate is defined by the same symbol, side, source level identity, and event time within the policy collision window. Identical candidates merge provenance. Materially different levels and opposite-side sweeps remain separate events. Only same-side events whose levels are too close to attribute market evidence safely are finalized `INVALID` with `AMBIGUOUS_EVENT_COLLISION`.

## Time, Ordering, and Buffers

Market evidence uses exchange event time. Receipt time is diagnostic only. Classification logic must not call `time.time()`.

Canonical market ordering is:

1. Exchange event timestamp.
2. Exchange sequence or aggregate trade ID when available.
3. Source precedence: trade, depth, sweep observation.
4. Stable content hash as the final tie breaker.

In replay, a sweep observation is presented to the engine at `detection_time_ms`, while retaining its earlier `event_time_ms` for evidence reconstruction. This models when TICS could know the source claim.

Each symbol owns an independent market-data watermark:

```text
max_seen_exchange_time_ms - reorder_tolerance_ms
```

Evidence newer than the watermark is provisional. A sufficient branch cannot finalize until its sufficient timestamp is behind the watermark.

The engine exposes `advance_time(symbol, as_of_exchange_time_ms)` and never advances one symbol from another symbol's activity. Live trade and depth observations advance it from exchange timestamps. A sweep callback may advance it to detection time only after retained market data proves continuous mandatory trade coverage through that point. Replay performs an explicit deterministic final advance for each symbol. If replay input ends before an unresolved event's required interval is covered, the event becomes `INVALID / INSUFFICIENT_FUTURE_COVERAGE`, not an assumed timeout.

The v1 buffer policy is:

```text
confirmation_window_ms = 60_000
reorder_tolerance_ms = 2_000
maximum_supported_detection_lag_ms = 90_000
buffer_safety_margin_ms = 28_000
market_buffer_retention_ms = 180_000
max_active_events_per_symbol = 128
recent_final_events_per_symbol = 1_000
recorder_queue_max_items = 10_000
```

Policy validation requires retention to be at least the sum of the first four values.

A late trade or depth update older than the settled watermark is quarantined and increments late-data telemetry. It cannot mutate settled evidence. An overlapping unresolved event whose mandatory trade interval is compromised becomes `INVALID`; finalized events remain immutable.

A late sweep callback is allowed when the complete event interval remains buffered. The engine reconstructs evidence from `event_time_ms`. If required history has been evicted, it finalizes `INVALID` with `INSUFFICIENT_REPLAY_HISTORY`.

## Lifecycle and Classification

Lifecycle state and market classification are separate concepts.

Lifecycle transitions are:

```text
OBSERVED
    -> PENETRATION_VALIDATING
        -> INVALID -> FINALIZED
        -> SWEEP_DETECTED
            -> RECLAIMING -> FINALIZED
            -> ACCEPTING -> FINALIZED
            -> UNRESOLVED -> FINALIZED
```

Classifications are:

```text
PENDING_SWEEP
FAILED_BREAKDOWN
FAILED_BREAKOUT
BEARISH_CONTINUATION
BULLISH_CONTINUATION
INDETERMINATE
INVALID
```

`SweepsMonitor` supplies a candidate claim. A sweep becomes confirmed only when an actual canonical trade strictly penetrates the claimed level in the correct direction. Otherwise it becomes `INVALID / PENETRATION_NOT_CONFIRMED` when the confirmation window closes.

## Frozen Classification Policy

The v1 policy is:

```text
confirmation_window_ms = 60_000
confirmation_hold_ms = 3_000
minimum_confirming_trades = 3
reclaim_buffer_bps = 1.0
acceptance_buffer_bps = 1.0
reorder_tolerance_ms = 2_000
collision_window_ms = 2_000
collision_level_tolerance_bps = 0.5
model_version = 1C.1-v1
```

Every field is serialized in canonical key order and hashed with SHA-256. Every final event records `policy_hash` and `model_version`.

For a sell-side sweep:

```text
reclaim_threshold = swept_level * (1 + reclaim_buffer_bps / 10_000)
acceptance_threshold = swept_level * (1 - acceptance_buffer_bps / 10_000)

sustained price > reclaim_threshold -> FAILED_BREAKDOWN
sustained price < acceptance_threshold -> BEARISH_CONTINUATION
```

For a buy-side sweep:

```text
reclaim_threshold = swept_level * (1 - reclaim_buffer_bps / 10_000)
acceptance_threshold = swept_level * (1 + acceptance_buffer_bps / 10_000)

sustained price < reclaim_threshold -> FAILED_BREAKOUT
sustained price > acceptance_threshold -> BULLISH_CONTINUATION
```

A branch candidate starts on its first qualifying trade. It tracks qualifying duration and qualifying trade count. Trades in the neutral zone pause both the candidate and its timer. Neutral elapsed time does not count toward the hold. A trade across the opposite threshold resets the current candidate and may start the competing candidate.

A branch becomes sufficient after at least 3,000 milliseconds of accumulated qualifying duration and at least three qualifying trades. The earlier canonical sufficient time wins after watermark settlement. Equal sufficient times produce `INDETERMINATE / CONFLICTING_CONFIRMATION`.

If no branch resolves by `event_time_ms + 60_000`, the result is `INDETERMINATE / CONFIRMATION_WINDOW_EXPIRED`. A late callback reconstructs this outcome when history exists.

The result stores both:

```text
market_resolution_time_ms
classification_time_ms = max(market_resolution_time_ms, detection_time_ms)
```

Final results are immutable.

## Evidence Model

Price behavior determines the outcome. Context may explain, support, or contradict that outcome but cannot change it.

Evidence fields include their as-of timestamps and availability state. The model covers:

- Confirmed extreme price and penetration in bps.
- Reclaim and acceptance thresholds, distance, duration, and trade count.
- Post-sweep signed delta and delta recovery.
- CVD movement and recovery from the post-sweep extreme.
- Normalized bid/ask absorption.
- Spread, depth quality, depth-weighted imbalance, and book availability.
- Wall replenishment and stacking/pulling observations.
- Post-sweep displacement and volume expansion.
- Deterministically ordered reasons and contradictions.

Availability is tri-state: `AVAILABLE`, `UNAVAILABLE`, or `UNSAFE`. An unavailable value is `None`, never numeric zero.

The v1 confidence index is deliberately simple and uncalibrated:

```text
price outcome confirmation                         +0.60 support
post-sweep delta aligned/opposed                    +/-0.10
CVD movement aligned/opposed                        +/-0.10
absorption aligned/opposed                          +/-0.10
depth-weighted imbalance aligned/opposed            +/-0.10
```

Unavailable or neutral contextual groups contribute zero and reduce `context_coverage`; they are not treated as contradictions. `evidence_strength` is the sum of positive contributions, `contradiction_strength` is the absolute sum of negative contributions, and:

```text
confidence = clamp(evidence_strength - contradiction_strength, 0, 1)
confidence_type = UNCALIBRATED_DETERMINISTIC_SCORE
```

Indeterminate and invalid events have confidence `0.0`. Replenishment, stacking/pulling, displacement, and volume expansion are recorded as reasons or contradictions in v1 but do not affect confidence, avoiding double counting before validation.

## Guardian Semantics

Trusted trade timing and price provenance are mandatory. An unsafe trade feed during an unresolved event's required interval finalizes it `INVALID / MARKET_DATA_UNSAFE`.

Depth context is optional for the price outcome. Unsafe or missing depth marks depth, absorption, replenishment, and stacking/pulling evidence `UNAVAILABLE` or `UNSAFE`; a trustworthy price classification may still finalize.

Guardian failures after finalization do not mutate an event.

## Integration

The application creates one event engine, store, and recorder. Trade and depth handlers feed normalized observations from startup so delayed sweep callbacks can use retained history. Existing `_handle_sweep` scoring behavior remains unchanged and additionally sends the callback through the adapter when the engine is enabled.

The existing read-only API exposes `/api/liquidity-events` with:

```text
active events
recent final events
engine enabled
enforcement enabled
active event count
rejected input count
late data count
recorder status
recorder failure count
pending write count
policy version
policy hash
```

There are no mutation, manual-classification, policy-change, enforcement, or trading endpoints.

The dashboard receives a compact shadow status and recent-event readout. It must not imply that confidence is a probability or add trading controls.

## Recording

Phase 1C.1 writes three append-only artifacts:

```text
liquidity_events.csv
    one immutable final row per canonical event

liquidity_event_transitions.csv
    one row per lifecycle transition

liquidity_rejected_inputs.csv
    malformed or unusable source observations
```

Canonical output uses fixed column order, fixed float formatting, UTF-8, LF newlines, canonical enum values, and deterministically sorted JSON arrays. It excludes absolute paths, hostnames, process IDs, object representations, runtime wall-clock timestamps, and other machine-dependent values.

Every final event includes `event_id`, `source_observation_hash`, `policy_hash`, and `model_version`.

Recorder I/O is independent of classification. Writes use a bounded queue. Recorder telemetry exposes status, failure count, last error, and pending writes. Queue overflow or disk failure is logged and counted without changing an event outcome or allowing unbounded memory growth.

The event store retains at most 128 active events and 1,000 recent final events per symbol. A source observation that would exceed active capacity is rejected with `EVENT_CAPACITY_REACHED`. The recorder queue retains at most 10,000 pending rows and rejects additional writes with an operational failure counter rather than growing without bound.

## Replay

Exact replay merges sweep observations, trades, and depth observations in canonical order. Sweep observations enter at detection time and reconstruct from event time. Replay uses the same adapter, policy, event store, engine, evidence builder, classifier, and recorder as live mode.

There is no research-only classifier, alternate replay threshold, or bar approximation in Phase 1C.1. Identical canonical inputs and policy hashes must produce byte-equivalent event and transition artifacts.

## Testing and Acceptance

The implementation must prove:

1. A sell-side sweep with sustained reclaim becomes `FAILED_BREAKDOWN`.
2. A sell-side sweep with sustained acceptance becomes `BEARISH_CONTINUATION`.
3. A buy-side sweep with sustained reclaim becomes `FAILED_BREAKOUT`.
4. A buy-side sweep with sustained acceptance becomes `BULLISH_CONTINUATION`.
5. Insufficient follow-through becomes `INDETERMINATE`.
6. No actual penetration becomes `INVALID`, not a sweep outcome.
7. Repeated CSV ingestion and duplicate callbacks create one event.
8. Out-of-order market observations inside tolerance settle deterministically.
9. Market observations behind the watermark are quarantined and cannot mutate a final event.
10. Timeout produces `INDETERMINATE / CONFIRMATION_WINDOW_EXPIRED`.
11. Unsafe mandatory trade data invalidates an unresolved event.
12. Missing or unsafe depth still permits a price outcome with contextual evidence unavailable.
13. Later market data cannot change an earlier final classification or timestamp.
14. Identical inputs produce byte-equivalent timelines.
15. Enabling or disabling the engine leaves scorer output unchanged.
16. `EXECUTION_DISABLED` remains true and paper behavior is unchanged.
17. `REGIME_ENFORCEMENT_ENABLED` remains false and regime permissions are not consumed.
18. Legacy `BULLISH` and `BEARISH` fixtures map to `SELL_SIDE` and `BUY_SIDE` respectively.
19. Neutral-zone time pauses qualifying duration.
20. Crossing the opposite threshold resets the candidate.
21. Equal canonical sufficient times become `INDETERMINATE / CONFLICTING_CONFIRMATION`.
22. A late sweep callback reconstructs successfully while complete history is retained.
23. A callback whose history was evicted becomes `INVALID / INSUFFICIENT_REPLAY_HISTORY`.
24. Semantically distinct levels and opposite-side overlaps remain separate events.
25. Ambiguous same-side collisions become explicit invalid events.
26. Policy serialization and hashing are stable.
27. Reasons, contradictions, and artifact rows serialize deterministically.
28. The `liquidity_event` package respects its forbidden-import boundary.
29. Recorder failures do not alter classification and cannot grow memory without bound.
30. Live and replay event sequences are identical for the same canonical inputs.
31. One symbol's activity cannot advance another symbol's watermark.
32. Replay without complete future coverage produces `INVALID / INSUFFICIENT_FUTURE_COVERAGE`.
33. Active-event, recent-event, and recorder queues enforce their frozen bounds.

Phase 1C.1 is complete only when these engineering gates pass and the full existing test suite remains green without new warnings.

## Follow-on Sequence

After Phase 1C.1 is frozen:

```text
Phase 1C.2  Absorption and Replenishment Intelligence
Phase 1C.3  Liquidity Map
Phase 1C-V  Historical Liquidity Validation
Phase 1D    Derivatives Pressure Engine
```

Profitability is not an acceptance criterion for Phase 1C.1. The milestone establishes canonical semantics, point-in-time correctness, explainability, replay parity, and a safe research boundary.
