from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from typing import Any, Callable

from .classifier import PriceDecision, PriceOutcomeTracker
from .evidence import EventFlowAccumulator, LiquidityEvidenceBuilder
from .event_store import LiquidityEventStore, OpenEventResult
from .identity_authority import (
    IdentityAuthorityUnavailable,
    PersistedIdentity,
    SQLiteIdentityAuthority,
)
from .models import (
    AggressorSide,
    DepthObservation,
    EventClassification,
    EventState,
    EvidenceAvailability,
    EvidenceValue,
    LifecycleTransition,
    LiquidityEvent,
    LiquidityEventResult,
    LiquidityEvidence,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    RejectedSweepInput,
    TradeCoverage,
    TradeCoverageProvider,
    canonical_hash,
)
from .policy import LiquidityClassificationPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageSegment:
    start_ms: int
    end_ms: int
    coverage: TradeCoverage


@dataclass
class EngineTelemetry:
    late_data_count: int = 0
    quarantined_trade_count: int = 0
    quarantined_depth_count: int = 0
    recovered_events_count: int = 0
    authority_failures_count: int = 0


@dataclass
class EventTrackerContext:
    event: LiquidityEvent
    classifier: PriceOutcomeTracker
    flow_accumulator: EventFlowAccumulator
    expiry_ms: int
    last_transition_sequence: int = -1
    last_state: EventState = EventState.OBSERVED
    fed_trade_keys: set[Any] = field(default_factory=set)


class SymbolRuntime:
    def __init__(self, symbol: str, policy: LiquidityClassificationPolicy):
        self.symbol = symbol
        self.policy = policy
        self.trade_buffer: list[MarketTrade] = []
        self._trade_index: dict[tuple[str, int | str], MarketTrade] = {}
        self.depth_buffer: list[DepthObservation] = []
        self._depth_index: dict[tuple[str, int | str], DepthObservation] = {}
        self.coverage_ledger: list[CoverageSegment] = []
        self.max_seen_exchange_time_ms: int | None = None
        self.active_trackers: dict[str, EventTrackerContext] = {}

    @property
    def watermark_ms(self) -> int | None:
        if self.max_seen_exchange_time_ms is not None:
            return self.max_seen_exchange_time_ms - self.policy.reorder_tolerance_ms
        return None

    def record_coverage(self, start_ms: int, end_ms: int, coverage: TradeCoverage) -> None:
        self.coverage_ledger.append(CoverageSegment(start_ms=start_ms, end_ms=end_ms, coverage=coverage))

    def check_interval_coverage(self, start_ms: int, end_ms: int) -> tuple[bool, str | None]:
        """Evaluates all coverage ledger segments overlapping [start_ms, end_ms] monotonically without last-write-wins."""
        for seg in self.coverage_ledger:
            # Overlap check: max(start_ms, seg.start_ms) <= min(end_ms, seg.end_ms)
            if max(start_ms, seg.start_ms) <= min(end_ms, seg.end_ms):
                cov = seg.coverage
                if not cov.feed_safe:
                    return False, "MARKET_DATA_UNSAFE"
                if cov.known_gap or cov.buffer_overflow or cov.unresolved_sequence:
                    return False, "MARKET_DATA_INTEGRITY_COMPROMISED"
        return True, None

    def insert_trade(self, trade: MarketTrade) -> bool:
        """Inserts trade into provisional sorted buffer with canonical duplicate tie-breaking."""
        key = (trade.symbol, trade.sequence_id)
        if key in self._trade_index:
            existing = self._trade_index[key]
            if existing == trade or existing.content_hash == trade.content_hash:
                return False
            # Conflicting duplicate while provisional: keep canonical minimum
            winner = min(existing, trade, key=lambda t: t.canonical_key)
            if winner == existing:
                return False
            # Replace existing in buffer
            self._trade_index[key] = winner
            self.trade_buffer = [t if (t.symbol, t.sequence_id) != key else winner for t in self.trade_buffer]
            self.trade_buffer.sort(key=lambda t: t.canonical_key)
            return True

        self._trade_index[key] = trade
        # Bisect insertion maintaining canonical order
        bisect.insort(self.trade_buffer, trade, key=lambda t: t.canonical_key)
        return True

    def insert_depth(self, depth: DepthObservation) -> bool:
        key = (depth.symbol, depth.sequence_id)
        if key in self._depth_index:
            return False
        self._depth_index[key] = depth
        bisect.insort(self.depth_buffer, depth, key=lambda d: d.canonical_key)
        return True

    def prune(self) -> None:
        if self.max_seen_exchange_time_ms is None:
            return
        cutoff = self.max_seen_exchange_time_ms - self.policy.market_buffer_retention_ms
        if cutoff <= 0:
            return

        # Prune trade buffer and index
        retained_trades = []
        for t in self.trade_buffer:
            if t.exchange_time_ms >= cutoff:
                retained_trades.append(t)
            else:
                self._trade_index.pop((t.symbol, t.sequence_id), None)
        self.trade_buffer = retained_trades

        # Prune depth buffer and index
        retained_depth = []
        for d in self.depth_buffer:
            if d.exchange_time_ms >= cutoff:
                retained_depth.append(d)
            else:
                self._depth_index.pop((d.symbol, d.sequence_id), None)
        self.depth_buffer = retained_depth

        # Prune coverage ledger
        self.coverage_ledger = [seg for seg in self.coverage_ledger if seg.end_ms >= cutoff]


@dataclass(frozen=True)
class EngineSnapshot:
    symbol_watermarks: dict[str, int | None]
    active_event_count: int
    recent_event_count: int
    telemetry: EngineTelemetry


class LiquidityEventEngine:
    def __init__(
        self,
        policy: LiquidityClassificationPolicy,
        store: LiquidityEventStore,
        authority: SQLiteIdentityAuthority,
        evidence_builder: LiquidityEvidenceBuilder | None = None,
        on_transition: Callable[[LifecycleTransition], None] | None = None,
        on_result: Callable[[LiquidityEventResult], None] | None = None,
        on_rejected: Callable[[RejectedSweepInput], None] | None = None,
    ):
        self.policy = policy
        self.store = store
        self.authority = authority
        self.evidence_builder = evidence_builder or LiquidityEvidenceBuilder()
        self.on_transition = on_transition
        self.on_result = on_result
        self.on_rejected = on_rejected
        self.telemetry = EngineTelemetry()
        self._runtimes: dict[str, SymbolRuntime] = {}

    def symbol_runtime(self, symbol: str) -> SymbolRuntime:
        if symbol not in self._runtimes:
            self._runtimes[symbol] = SymbolRuntime(symbol, self.policy)
        return self._runtimes[symbol]

    def watermark(self, symbol: str) -> int | None:
        if symbol not in self._runtimes:
            return None
        return self._runtimes[symbol].watermark_ms

    def on_trade(self, trade: MarketTrade, coverage: TradeCoverage) -> None:
        runtime = self.symbol_runtime(trade.symbol)
        runtime.record_coverage(start_ms=trade.exchange_time_ms, end_ms=trade.exchange_time_ms, coverage=coverage)

        prev_wm = runtime.watermark_ms
        # Check against previous settled watermark
        if prev_wm is not None and trade.exchange_time_ms <= prev_wm:
            self.telemetry.late_data_count += 1
            self.telemetry.quarantined_trade_count += 1
            return

        inserted = runtime.insert_trade(trade)
        if not inserted:
            return

        # Advance symbol max seen
        runtime.max_seen_exchange_time_ms = max(
            runtime.max_seen_exchange_time_ms or 0,
            trade.exchange_time_ms,
        )
        new_wm = runtime.watermark_ms

        # Process newly settled trades across active events
        self._process_settled_records(runtime, prev_wm, new_wm)
        runtime.prune()

    def on_depth(self, depth: DepthObservation) -> None:
        runtime = self.symbol_runtime(depth.symbol)
        prev_wm = runtime.watermark_ms
        if prev_wm is not None and depth.exchange_time_ms <= prev_wm:
            self.telemetry.late_data_count += 1
            self.telemetry.quarantined_depth_count += 1
            return

        runtime.insert_depth(depth)
        runtime.max_seen_exchange_time_ms = max(
            runtime.max_seen_exchange_time_ms or 0,
            depth.exchange_time_ms,
        )
        new_wm = runtime.watermark_ms
        # Process newly settled trades without independent coverage advancement
        self._process_settled_records(runtime, prev_wm, new_wm)
        runtime.prune()

    def on_sweep(
        self,
        observation: LiquiditySweepObservation,
        coverage_provider: TradeCoverageProvider,
    ) -> LiquidityEventResult | RejectedSweepInput | LiquidityEvent | None:
        symbol = observation.symbol
        runtime = self.symbol_runtime(symbol)
        expiry_ms = observation.event_time_ms + self.policy.confirmation_window_ms

        # 1. Obtain event-integrity coverage
        cov_event = coverage_provider.coverage(
            symbol,
            observation.event_time_ms,
            min(observation.detection_time_ms, expiry_ms),
        )
        runtime.record_coverage(
            start_ms=observation.event_time_ms,
            end_ms=min(observation.detection_time_ms, expiry_ms),
            coverage=cov_event,
        )

        # 2. Admit observation
        try:
            admission = self.store.admit_observation(observation, self.authority)
        except Exception as e:
            self.telemetry.authority_failures_count += 1
            logger.error("authority failure during admission: %s", e)
            return None

        if admission.rejection is not None:
            if self.on_rejected:
                self.on_rejected(admission.rejection)
            return admission.rejection

        if admission.duplicate:
            return admission.event

        # 3. Collision handling
        if admission.collision_event_ids:
            return self._handle_collisions(runtime, admission.collision_event_ids, observation.detection_time_ms)

        if not admission.created or admission.event is None:
            return None

        event = admission.event
        # 4. Check retained past history
        if runtime.max_seen_exchange_time_ms is not None:
            retained_from_ms = runtime.max_seen_exchange_time_ms - self.policy.market_buffer_retention_ms
            if observation.event_time_ms < retained_from_ms or not cov_event.interval_retained:
                return self._finalize_direct(
                    runtime,
                    event,
                    EventClassification.INVALID,
                    "INSUFFICIENT_REPLAY_HISTORY",
                    market_resolution_time_ms=observation.event_time_ms,
                )
        else:
            if not cov_event.interval_retained:
                return self._finalize_direct(
                    runtime,
                    event,
                    EventClassification.INVALID,
                    "INSUFFICIENT_REPLAY_HISTORY",
                    market_resolution_time_ms=observation.event_time_ms,
                )

        # 5. Check coverage validity for event interval
        if not cov_event.valid:
            reason = "MARKET_DATA_UNSAFE" if not cov_event.feed_safe else "MARKET_DATA_INTEGRITY_COMPROMISED"
            return self._finalize_direct(
                runtime,
                event,
                EventClassification.INVALID,
                reason,
                market_resolution_time_ms=observation.event_time_ms,
            )

        # 6. Initialize context
        ctx = EventTrackerContext(
            event=event,
            classifier=PriceOutcomeTracker(
                level=observation.swept_level,
                side=observation.liquidity_side,
                policy=self.policy,
                symbol=symbol,
            ),
            flow_accumulator=EventFlowAccumulator(event_time_ms=observation.event_time_ms),
            expiry_ms=expiry_ms,
        )
        runtime.active_trackers[event.event_id] = ctx

        # Emit initial lifecycle start transition
        self._transition_event(
            ctx,
            EventState.PENETRATION_VALIDATING,
            "START_VALIDATION",
            transition_time_ms=observation.event_time_ms,
        )

        # 7. Check if clock can advance toward detection_time_ms
        if observation.detection_time_ms > (runtime.max_seen_exchange_time_ms or 0):
            cov_clock = coverage_provider.coverage(
                symbol,
                runtime.max_seen_exchange_time_ms or observation.event_time_ms,
                observation.detection_time_ms,
            )
            runtime.record_coverage(
                start_ms=runtime.max_seen_exchange_time_ms or observation.event_time_ms,
                end_ms=observation.detection_time_ms,
                coverage=cov_clock,
            )
            if cov_clock.valid:
                runtime.max_seen_exchange_time_ms = max(
                    runtime.max_seen_exchange_time_ms or 0,
                    observation.detection_time_ms,
                )

        # 8. Replay SETTLED canonical trades
        wm = runtime.watermark_ms
        settled_cutoff = min(wm if wm is not None else -1, expiry_ms)
        for t in runtime.trade_buffer:
            if observation.event_time_ms <= t.exchange_time_ms <= settled_cutoff:
                if t.canonical_key not in ctx.fed_trade_keys:
                    ctx.fed_trade_keys.add(t.canonical_key)
                    ctx.classifier.on_trade(t)
                    ctx.flow_accumulator.on_trade(t)

        # 9. Evaluate decision
        if wm is not None:
            decision = ctx.classifier.decision_at(watermark_ms=wm, expiry_ms=expiry_ms)
            if decision.classification is not EventClassification.PENDING_SWEEP:
                return self._finalize_from_decision(runtime, ctx, decision)

        return event

    def advance_time(
        self,
        symbol: str,
        as_of_exchange_time_ms: int,
        coverage: TradeCoverage,
        terminal_input: bool = False,
    ) -> tuple[LiquidityEventResult, ...]:
        runtime = self.symbol_runtime(symbol)
        start_ms = runtime.max_seen_exchange_time_ms or as_of_exchange_time_ms
        runtime.record_coverage(start_ms=start_ms, end_ms=as_of_exchange_time_ms, coverage=coverage)

        prev_wm = runtime.watermark_ms
        runtime.max_seen_exchange_time_ms = max(
            runtime.max_seen_exchange_time_ms or 0,
            as_of_exchange_time_ms,
        )
        new_wm = runtime.watermark_ms

        results = self._process_settled_records(runtime, prev_wm, new_wm)

        # If terminal input is declared and events remain unresolved
        if terminal_input and new_wm is not None:
            unresolved = list(runtime.active_trackers.values())
            for ctx in unresolved:
                if ctx.event.event_id in runtime.active_trackers:
                    if new_wm < ctx.expiry_ms:
                        res = self._finalize_direct(
                            runtime,
                            ctx.event,
                            EventClassification.INVALID,
                            "INSUFFICIENT_FUTURE_COVERAGE",
                            market_resolution_time_ms=as_of_exchange_time_ms,
                            ctx=ctx,
                        )
                        if res is not None:
                            results.append(res)

        runtime.prune()
        return tuple(results)

    def recover_claimed_unresolved(
        self,
        coverage_provider: TradeCoverageProvider,
    ) -> tuple[LiquidityEventResult, ...]:
        try:
            pending = self.authority.pending_unresolved()
        except Exception as e:
            self.telemetry.authority_failures_count += 1
            logger.error("authority lookup pending failed: %s", e)
            return ()

        results: list[LiquidityEventResult] = []
        for p in pending:
            if p.status != "CLAIMED_UNRESOLVED":
                continue

            obs_data = dict(p.observation_payload)
            # Reconstruct LiquiditySweepObservation
            obs = LiquiditySweepObservation(
                event_id=obs_data["event_id"],
                source_event_id=obs_data.get("source_event_id"),
                source_level_id=obs_data.get("source_level_id"),
                symbol=obs_data["symbol"],
                liquidity_side=LiquiditySide(obs_data["liquidity_side"]),
                swept_level=Decimal(str(obs_data["swept_level"])),
                event_time_ms=obs_data["event_time_ms"],
                detection_time_ms=obs_data["detection_time_ms"],
                source_sweep_price=obs_data.get("source_sweep_price"),
                source_penetration_bps=obs_data.get("source_penetration_bps"),
                source=obs_data.get("source", "SWEEPS_MONITOR_CSV"),
                source_file_id=obs_data.get("source_file_id", "sweeps.csv"),
                source_row_hash=obs_data.get("source_row_hash"),
                source_observation_hash=obs_data["source_observation_hash"],
                detector_version=obs_data.get("detector_version", "1.0.0"),
            )

            runtime = self.symbol_runtime(obs.symbol)
            try:
                open_res = self.store.restore_claimed_unresolved(obs, p)
            except Exception as e:
                logger.error("error restoring unresolved event %s: %s", obs.event_id, e)
                continue

            if not open_res.created or open_res.event is None:
                continue

            event = open_res.event
            expiry_ms = obs.event_time_ms + self.policy.confirmation_window_ms
            cov_event = coverage_provider.coverage(obs.symbol, obs.event_time_ms, min(obs.detection_time_ms, expiry_ms))

            if not cov_event.interval_retained:
                res = self._finalize_direct(
                    runtime,
                    event,
                    EventClassification.INVALID,
                    "INSUFFICIENT_REPLAY_HISTORY",
                    market_resolution_time_ms=obs.event_time_ms,
                )
                if res:
                    results.append(res)
                continue

            # Check already persisted transitions
            persisted_seqs = self.authority.persisted_transition_sequences(obs.event_id)
            max_seq = max(persisted_seqs) if persisted_seqs else -1

            ctx = EventTrackerContext(
                event=event,
                classifier=PriceOutcomeTracker(
                    level=obs.swept_level,
                    side=obs.liquidity_side,
                    policy=self.policy,
                    symbol=obs.symbol,
                ),
                flow_accumulator=EventFlowAccumulator(event_time_ms=obs.event_time_ms),
                expiry_ms=expiry_ms,
                last_transition_sequence=max_seq,
            )
            runtime.active_trackers[obs.event_id] = ctx
            self.telemetry.recovered_events_count += 1

            # Replay settled trades
            wm = runtime.watermark_ms
            settled_cutoff = min(wm if wm is not None else -1, expiry_ms)
            for t in runtime.trade_buffer:
                if obs.event_time_ms <= t.exchange_time_ms <= settled_cutoff:
                    if t.canonical_key not in ctx.fed_trade_keys:
                        ctx.fed_trade_keys.add(t.canonical_key)
                        ctx.classifier.on_trade(t)
                        ctx.flow_accumulator.on_trade(t)

            if wm is not None:
                decision = ctx.classifier.decision_at(watermark_ms=wm, expiry_ms=expiry_ms)
                if decision.classification is not EventClassification.PENDING_SWEEP:
                    res = self._finalize_from_decision(runtime, ctx, decision)
                    if res:
                        results.append(res)

        return tuple(results)

    def snapshot(self, symbol: str | None = None) -> EngineSnapshot:
        watermarks = {s: r.watermark_ms for s, r in self._runtimes.items()}
        if symbol is not None:
            watermarks = {symbol: self._runtimes[symbol].watermark_ms} if symbol in self._runtimes else {symbol: None}
        return EngineSnapshot(
            symbol_watermarks=watermarks,
            active_event_count=self.store.active_count(symbol),
            recent_event_count=len(self.store.recent(symbol)),
            telemetry=self.telemetry,
        )

    def _process_settled_records(
        self,
        runtime: SymbolRuntime,
        prev_wm: int | None,
        new_wm: int | None,
    ) -> list[LiquidityEventResult]:
        if new_wm is None:
            return []

        results: list[LiquidityEventResult] = []
        # Find active events
        active_list = list(runtime.active_trackers.values())
        for ctx in active_list:
            if ctx.event.event_id not in runtime.active_trackers:
                continue

            event_id = ctx.event.event_id
            obs = ctx.event.observation
            expiry = ctx.expiry_ms

            # 1. Coverage check for event-required interval seen so far
            cov_check_end = min(runtime.max_seen_exchange_time_ms if runtime.max_seen_exchange_time_ms is not None else new_wm, expiry)
            cov_ok, cov_reason = runtime.check_interval_coverage(
                obs.event_time_ms,
                cov_check_end,
            )
            if not cov_ok:
                res = self._finalize_direct(
                    runtime,
                    ctx.event,
                    EventClassification.INVALID,
                    cov_reason or "MARKET_DATA_INTEGRITY_COMPROMISED",
                    market_resolution_time_ms=cov_check_end,
                    ctx=ctx,
                )
                if res is not None:
                    results.append(res)
                continue

            # 2. Feed newly settled trades in canonical_key order
            for t in runtime.trade_buffer:
                if obs.event_time_ms <= t.exchange_time_ms <= min(new_wm, expiry):
                    if t.canonical_key not in ctx.fed_trade_keys:
                        ctx.fed_trade_keys.add(t.canonical_key)
                        ctx.classifier.on_trade(t)
                        ctx.flow_accumulator.on_trade(t)

            # 3. Lifecycle progression based on classifier state
            if ctx.classifier.has_penetration and ctx.last_state == EventState.PENETRATION_VALIDATING:
                pen_t = ctx.classifier.penetration_time_ms or obs.event_time_ms
                self._transition_event(ctx, EventState.SWEEP_DETECTED, "PENETRATION_CONFIRMED", pen_t)

            if ctx.classifier.reclaim_trade_count > 0 and ctx.last_state in (EventState.PENETRATION_VALIDATING, EventState.SWEEP_DETECTED):
                self._transition_event(ctx, EventState.RECLAIMING, "RECLAIM_PROGRESS", new_wm)
            elif ctx.classifier.acceptance_trade_count > 0 and ctx.last_state in (EventState.PENETRATION_VALIDATING, EventState.SWEEP_DETECTED):
                self._transition_event(ctx, EventState.ACCEPTING, "ACCEPTANCE_PROGRESS", new_wm)

            # 4. Evaluate decision at watermark
            decision = ctx.classifier.decision_at(watermark_ms=new_wm, expiry_ms=expiry)
            if decision.classification is not EventClassification.PENDING_SWEEP:
                res = self._finalize_from_decision(runtime, ctx, decision)
                if res is not None:
                    results.append(res)

        return results

    def _transition_event(
        self,
        ctx: EventTrackerContext,
        to_state: EventState,
        reason_code: str,
        transition_time_ms: int,
    ) -> bool:
        next_seq = ctx.last_transition_sequence + 1
        transition = LifecycleTransition(
            event_id=ctx.event.event_id,
            previous_state=ctx.last_state,
            next_state=to_state,
            transition_time_ms=transition_time_ms,
            transition_sequence=next_seq,
            reason_code=reason_code,
        )

        try:
            persisted_seqs = self.authority.persisted_transition_sequences(ctx.event.event_id)
            if next_seq not in persisted_seqs:
                self.authority.claim_transition(transition)
        except Exception as e:
            self.telemetry.authority_failures_count += 1
            logger.error("authority claim transition failed: %s", e)
            return False

        ctx.event.state = to_state
        ctx.event.transitions.append(transition)
        ctx.last_transition_sequence = next_seq
        ctx.last_state = to_state

        if self.on_transition:
            try:
                self.on_transition(transition)
            except Exception as e:
                logger.error("on_transition callback error: %s", e)

        return True

    def _finalize_from_decision(
        self,
        runtime: SymbolRuntime,
        ctx: EventTrackerContext,
        decision: PriceDecision,
    ) -> LiquidityEventResult | None:
        target_state = EventState.FINALIZED
        if decision.classification is EventClassification.INVALID:
            self._transition_event(ctx, EventState.INVALID, decision.reason_code or "INVALID_OUTCOME", decision.market_resolution_time_ms)
        elif decision.classification is EventClassification.INDETERMINATE:
            self._transition_event(ctx, EventState.UNRESOLVED, decision.reason_code or "UNRESOLVED_OUTCOME", decision.market_resolution_time_ms)

        self._transition_event(ctx, target_state, decision.reason_code or "FINALIZED", decision.market_resolution_time_ms)

        # Snapshot point-in-time flow evidence
        flow_snap = ctx.flow_accumulator.snapshot(as_of_ms=decision.market_resolution_time_ms)
        evidence = self.evidence_builder.build(
            event=ctx.event,
            outcome_direction=decision.classification,
            as_of_ms=decision.market_resolution_time_ms,
            flow_snapshot=flow_snap,
        )

        # Enrich evidence with price witness fields
        evidence_dict = dict(evidence.values)
        if decision.penetration_price is not None:
            evidence_dict["price.penetration_price"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(decision.penetration_price),
                as_of_ms=decision.penetration_time_ms,
            )
        if decision.penetration_bps is not None:
            evidence_dict["price.penetration_bps"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(decision.penetration_bps),
                as_of_ms=decision.penetration_time_ms,
            )
        if decision.reclaim_sufficient_time_ms is not None:
            evidence_dict["price.reclaim_sufficient_time_ms"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(decision.reclaim_sufficient_time_ms),
                as_of_ms=decision.reclaim_sufficient_time_ms,
            )
            evidence_dict["price.reclaim_duration_ms"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(ctx.classifier.reclaim_qualifying_duration_ms),
                as_of_ms=decision.reclaim_sufficient_time_ms,
            )
            evidence_dict["price.reclaim_trade_count"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(ctx.classifier.reclaim_trade_count),
                as_of_ms=decision.reclaim_sufficient_time_ms,
            )
        if decision.acceptance_sufficient_time_ms is not None:
            evidence_dict["price.acceptance_sufficient_time_ms"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(decision.acceptance_sufficient_time_ms),
                as_of_ms=decision.acceptance_sufficient_time_ms,
            )
            evidence_dict["price.acceptance_duration_ms"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(ctx.classifier.acceptance_qualifying_duration_ms),
                as_of_ms=decision.acceptance_sufficient_time_ms,
            )
            evidence_dict["price.acceptance_trade_count"] = EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=float(ctx.classifier.acceptance_trade_count),
                as_of_ms=decision.acceptance_sufficient_time_ms,
            )

        final_evidence = LiquidityEvidence(
            values=evidence_dict,
            reasons=evidence.reasons,
            contradictions=evidence.contradictions,
            context_coverage=evidence.context_coverage,
            evidence_strength=evidence.evidence_strength,
            contradiction_strength=evidence.contradiction_strength,
            confidence=evidence.confidence,
            confidence_type=evidence.confidence_type,
            as_of_ms=decision.market_resolution_time_ms,
        )

        obs = ctx.event.observation
        classification_time = max(decision.market_resolution_time_ms, obs.detection_time_ms)
        result = LiquidityEventResult(
            event_id=obs.event_id,
            symbol=obs.symbol,
            liquidity_side=obs.liquidity_side,
            classification=decision.classification,
            reason_code=decision.reason_code,
            event_time_ms=obs.event_time_ms,
            detection_time_ms=obs.detection_time_ms,
            market_resolution_time_ms=decision.market_resolution_time_ms,
            classification_time_ms=classification_time,
            source_observation_hash=obs.source_observation_hash,
            evidence=final_evidence,
            policy_hash=self.policy.policy_hash,
            model_version=self.policy.model_version,
        )

        # Atomic authority persistence BEFORE volatile store finalization and external callbacks
        try:
            self.authority.claim_result(result)
        except Exception as e:
            self.telemetry.authority_failures_count += 1
            logger.error("authority claim result failed: %s", e)
            return None

        self.store.finalize(result)
        runtime.active_trackers.pop(obs.event_id, None)

        if self.on_result:
            try:
                self.on_result(result)
            except Exception as e:
                logger.error("on_result callback error: %s", e)

        return result

    def _finalize_direct(
        self,
        runtime: SymbolRuntime,
        event: LiquidityEvent,
        classification: EventClassification,
        reason_code: str,
        market_resolution_time_ms: int,
        ctx: EventTrackerContext | None = None,
    ) -> LiquidityEventResult | None:
        obs = event.observation
        if ctx is None:
            ctx = EventTrackerContext(
                event=event,
                classifier=PriceOutcomeTracker(level=obs.swept_level, side=obs.liquidity_side, policy=self.policy, symbol=obs.symbol),
                flow_accumulator=EventFlowAccumulator(event_time_ms=obs.event_time_ms),
                expiry_ms=obs.event_time_ms + self.policy.confirmation_window_ms,
            )

        self._transition_event(ctx, EventState.INVALID, reason_code, market_resolution_time_ms)
        self._transition_event(ctx, EventState.FINALIZED, reason_code, market_resolution_time_ms)

        flow_snap = ctx.flow_accumulator.snapshot(as_of_ms=market_resolution_time_ms)
        evidence = self.evidence_builder.build(
            event=ctx.event,
            outcome_direction=classification,
            as_of_ms=market_resolution_time_ms,
            flow_snapshot=flow_snap,
        )

        classification_time = max(market_resolution_time_ms, obs.detection_time_ms)
        result = LiquidityEventResult(
            event_id=obs.event_id,
            symbol=obs.symbol,
            liquidity_side=obs.liquidity_side,
            classification=classification,
            reason_code=reason_code,
            event_time_ms=obs.event_time_ms,
            detection_time_ms=obs.detection_time_ms,
            market_resolution_time_ms=market_resolution_time_ms,
            classification_time_ms=classification_time,
            source_observation_hash=obs.source_observation_hash,
            evidence=evidence,
            policy_hash=self.policy.policy_hash,
            model_version=self.policy.model_version,
        )

        try:
            self.authority.claim_result(result)
        except Exception as e:
            self.telemetry.authority_failures_count += 1
            logger.error("authority claim result failed: %s", e)
            return None

        self.store.finalize(result)
        runtime.active_trackers.pop(obs.event_id, None)

        if self.on_result:
            try:
                self.on_result(result)
            except Exception as e:
                logger.error("on_result callback error: %s", e)

        return result

    def _handle_collisions(
        self,
        runtime: SymbolRuntime,
        collision_event_ids: tuple[str, ...],
        detection_time_ms: int,
    ) -> LiquidityEventResult | None:
        last_res: LiquidityEventResult | None = None
        for eid in sorted(collision_event_ids):
            active_events = {ev.event_id: ev for ev in self.store.active(runtime.symbol)}
            if eid in active_events:
                ev = active_events[eid]
                ctx = runtime.active_trackers.get(eid)
                res = self._finalize_direct(
                    runtime,
                    ev,
                    EventClassification.INVALID,
                    "AMBIGUOUS_EVENT_COLLISION",
                    market_resolution_time_ms=detection_time_ms,
                    ctx=ctx,
                )
                if res is not None:
                    last_res = res
        return last_res
