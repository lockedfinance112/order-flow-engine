from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal

from .identity_authority import IdentityClaimOutcome, IdentityClaimResult
from .models import (
    LiquidityEvent,
    LiquidityEventResult,
    LiquiditySweepObservation,
    RejectedSweepInput,
    canonical_hash,
    normalize_price,
)
from .policy import LiquidityClassificationPolicy


@dataclass(frozen=True)
class OpenEventResult:
    event: LiquidityEvent | None
    created: bool
    duplicate: bool
    collision_event_ids: tuple[str, ...]
    rejection: RejectedSweepInput | None


class LiquidityEventStore:
    def __init__(self, policy: LiquidityClassificationPolicy):
        self._policy = policy
        self._active: dict[str, LiquidityEvent] = {}
        self._recent: dict[str, deque[LiquidityEventResult]] = {}

    def level_identity(self, observation: LiquiditySweepObservation) -> str:
        if observation.source_level_id is not None:
            return observation.source_level_id
        return canonical_hash(
            {
                "level": normalize_price(
                    observation.swept_level,
                    self._policy.canonical_price_decimal_places,
                ),
                "side": observation.liquidity_side.value,
                "symbol": observation.symbol,
            }
        )

    def open_event(
        self,
        observation: LiquiditySweepObservation,
        identity_claim: IdentityClaimResult,
    ) -> OpenEventResult:
        if identity_claim.outcome is IdentityClaimOutcome.DUPLICATE_EXISTING:
            return OpenEventResult(
                self._active.get(identity_claim.event_id),
                False,
                True,
                (),
                None,
            )

        if identity_claim.outcome is IdentityClaimOutcome.IDENTITY_CONFLICT:
            return OpenEventResult(
                None,
                False,
                False,
                (),
                RejectedSweepInput(
                    source=observation.source,
                    detection_time_ms=observation.detection_time_ms,
                    reason_code="IDENTITY_CONFLICT",
                    reason_detail=identity_claim.conflict_reason
                    or "identity authority rejected conflicting event identity",
                    source_file_id=observation.source_file_id,
                    source_row_hash=observation.source_row_hash,
                    source_observation_hash=observation.source_observation_hash,
                ),
            )

        active_for_symbol = sum(
            event.observation.symbol == observation.symbol
            for event in self._active.values()
        )
        if active_for_symbol >= self._policy.max_active_events_per_symbol:
            return OpenEventResult(
                None,
                False,
                False,
                (),
                RejectedSweepInput(
                    source=observation.source,
                    detection_time_ms=observation.detection_time_ms,
                    reason_code="EVENT_CAPACITY_REACHED",
                    reason_detail="maximum active events reached for symbol",
                    source_file_id=observation.source_file_id,
                    source_row_hash=observation.source_row_hash,
                    source_observation_hash=observation.source_observation_hash,
                ),
            )

        event = LiquidityEvent(observation=observation)
        self._active[event.event_id] = event
        collisions = self._collision_event_ids(event)
        return OpenEventResult(event, True, False, collisions, None)

    def finalize(self, result: LiquidityEventResult) -> None:
        active = self._active.get(result.event_id)
        if active is None or not self._matches_result(active, result):
            return

        del self._active[result.event_id]
        self._recent.setdefault(
            result.symbol,
            deque(maxlen=self._policy.recent_final_events_per_symbol),
        ).append(result)

    def active(self, symbol: str | None = None) -> tuple[LiquidityEvent, ...]:
        events = self._active.values()
        if symbol is not None:
            events = (
                event for event in events if event.observation.symbol == symbol
            )
        return tuple(deepcopy(event) for event in sorted(events, key=self._event_order))

    def recent(self, symbol: str | None = None) -> tuple[LiquidityEventResult, ...]:
        if symbol is not None:
            return tuple(sorted(self._recent.get(symbol, ()), key=self._result_order))
        return tuple(
            sorted(
                (result for results in self._recent.values() for result in results),
                key=self._result_order,
            )
        )

    def _collision_event_ids(self, event: LiquidityEvent) -> tuple[str, ...]:
        candidates = [event.event_id]
        for other in self._active.values():
            if other.event_id == event.event_id:
                continue
            if self._is_collision_candidate(event.observation, other.observation):
                candidates.append(other.event_id)
        return tuple(sorted(candidates)) if len(candidates) > 1 else ()

    def _is_collision_candidate(
        self,
        first: LiquiditySweepObservation,
        second: LiquiditySweepObservation,
    ) -> bool:
        if (
            first.symbol != second.symbol
            or first.liquidity_side is not second.liquidity_side
            or abs(first.event_time_ms - second.event_time_ms)
            > self._policy.collision_window_ms
        ):
            return False

        if (
            first.source_level_id is not None or second.source_level_id is not None
        ) and self.level_identity(first) != self.level_identity(second):
            return False

        first_level = Decimal(str(first.swept_level))
        second_level = Decimal(str(second.swept_level))
        difference_bps = (
            abs(first_level - second_level) / min(first_level, second_level) * 10_000
        )
        return difference_bps <= Decimal(str(self._policy.collision_level_tolerance_bps))

    def _matches_result(
        self,
        event: LiquidityEvent,
        result: LiquidityEventResult,
    ) -> bool:
        observation = event.observation
        return (
            event.event_id == result.event_id
            and observation.symbol == result.symbol
            and observation.liquidity_side is result.liquidity_side
            and observation.source_observation_hash == result.source_observation_hash
            and observation.event_time_ms == result.event_time_ms
            and observation.detection_time_ms == result.detection_time_ms
            and result.policy_hash == self._policy.policy_hash
            and result.model_version == self._policy.model_version
            and result.classification_time_ms
            == max(result.market_resolution_time_ms, result.detection_time_ms)
        )

    @staticmethod
    def _event_order(event: LiquidityEvent) -> tuple[str, int, int, str]:
        observation = event.observation
        return (
            observation.symbol,
            observation.event_time_ms,
            observation.detection_time_ms,
            event.event_id,
        )

    @staticmethod
    def _result_order(result: LiquidityEventResult) -> tuple[str, int, int, str]:
        return (
            result.symbol,
            result.classification_time_ms,
            result.event_time_ms,
            result.event_id,
        )
