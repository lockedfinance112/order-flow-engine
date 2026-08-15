"""Event-local flow accumulation and typed liquidity context evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from typing import Any, Mapping

from .models import (
    AggressorSide,
    ConfidenceType,
    EventClassification,
    EvidenceAvailability,
    EvidenceValue,
    LiquidityEvent,
    LiquidityEvidence,
    MarketTrade,
)


@dataclass(frozen=True)
class EventFlowSnapshot:
    buy_volume_usdt: float
    sell_volume_usdt: float
    signed_delta_usdt: float
    cumulative_delta_usdt: float
    post_sweep_cvd_usdt: float
    cvd_min_usdt: float
    cvd_max_usdt: float
    cvd_recovery_usdt: float
    as_of_ms: int | None = None


class EventFlowAccumulator:
    """Accumulates point-in-time trades for a single liquidity event.

    Strictly point-in-time from event_time_ms to end_time_ms.
    Derived solely from canonical point-in-time trades.
    """

    def __init__(self, event_time_ms: int, end_time_ms: int | None = None) -> None:
        self.event_time_ms = int(event_time_ms)
        self.end_time_ms = int(end_time_ms) if end_time_ms is not None else None
        self._trades: dict[tuple[str, int | str], MarketTrade] = {}

    def on_trade(self, trade: MarketTrade) -> None:
        if trade.exchange_time_ms < self.event_time_ms:
            return
        if self.end_time_ms is not None and trade.exchange_time_ms > self.end_time_ms:
            return
        key = (trade.symbol, trade.sequence_id)
        if key not in self._trades:
            self._trades[key] = trade

    def trades(self, as_of_ms: int | None = None) -> tuple[MarketTrade, ...]:
        sorted_trades = sorted(self._trades.values(), key=lambda t: t.canonical_key)
        if as_of_ms is not None:
            sorted_trades = [t for t in sorted_trades if t.exchange_time_ms <= as_of_ms]
        return tuple(sorted_trades)

    def snapshot(self, as_of_ms: int | None = None) -> EventFlowSnapshot:
        active_trades = self.trades(as_of_ms)
        buy_vol = 0.0
        sell_vol = 0.0
        cvd = 0.0
        cvd_min = 0.0
        cvd_max = 0.0

        for t in active_trades:
            notional = float(t.price) * float(t.quantity)
            if t.aggressor_side in (AggressorSide.BUY, "BUY"):
                buy_vol += notional
                cvd += notional
            elif t.aggressor_side in (AggressorSide.SELL, "SELL"):
                sell_vol += notional
                cvd -= notional
            if cvd < cvd_min:
                cvd_min = cvd
            if cvd > cvd_max:
                cvd_max = cvd

        signed_delta = buy_vol - sell_vol
        cvd_recovery = cvd - cvd_min if cvd >= 0 else cvd_max - cvd

        return EventFlowSnapshot(
            buy_volume_usdt=round(buy_vol, 8),
            sell_volume_usdt=round(sell_vol, 8),
            signed_delta_usdt=round(signed_delta, 8),
            cumulative_delta_usdt=round(signed_delta, 8),
            post_sweep_cvd_usdt=round(cvd, 8),
            cvd_min_usdt=round(cvd_min, 8),
            cvd_max_usdt=round(cvd_max, 8),
            cvd_recovery_usdt=round(cvd_recovery, 8),
            as_of_ms=as_of_ms,
        )


class LegacyLiquidityContextAdapter:
    """Adapts existing contextual signals without re-implementing detectors."""

    def absorption(self, raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue:
        if not raw:
            return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
        if raw.get("unsafe") or raw.get("feed_unsafe"):
            return EvidenceValue(EvidenceAvailability.UNSAFE, None, as_of_ms)

        event_type = str(raw.get("event_type", "")).upper()
        if "BULLISH" in event_type or raw.get("side") == "BID":
            return EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, as_of_ms)
        if "BEARISH" in event_type or raw.get("side") == "ASK":
            return EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, as_of_ms)

        val = raw.get("value", raw.get("score"))
        if val is not None and isfinite(float(val)):
            return EvidenceValue(EvidenceAvailability.AVAILABLE, float(val), as_of_ms)

        return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)

    def replenishment(self, raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue:
        if not raw:
            return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
        if raw.get("unsafe") or raw.get("feed_unsafe"):
            return EvidenceValue(EvidenceAvailability.UNSAFE, None, as_of_ms)

        event_type = str(raw.get("event_type", "")).upper()
        if "BID" in event_type or raw.get("side") == "BID":
            return EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, as_of_ms)
        if "ASK" in event_type or raw.get("side") == "ASK":
            return EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, as_of_ms)

        val = raw.get("value", raw.get("score"))
        if val is not None and isfinite(float(val)):
            return EvidenceValue(EvidenceAvailability.AVAILABLE, float(val), as_of_ms)

        return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)

    def stacking_pulling(self, raw: Mapping[str, Any] | None, as_of_ms: int) -> EvidenceValue:
        if not raw:
            return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
        if raw.get("unsafe") or raw.get("feed_unsafe"):
            return EvidenceValue(EvidenceAvailability.UNSAFE, None, as_of_ms)

        event_type = str(raw.get("event_type", "")).upper()
        if "BID_STACKING" in event_type or "ASK_PULLING" in event_type:
            return EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, as_of_ms)
        if "ASK_STACKING" in event_type or "BID_PULLING" in event_type:
            return EvidenceValue(EvidenceAvailability.AVAILABLE, -1.0, as_of_ms)

        val = raw.get("value", raw.get("score"))
        if val is not None and isfinite(float(val)):
            return EvidenceValue(EvidenceAvailability.AVAILABLE, float(val), as_of_ms)

        return EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)


class LiquidityEvidenceBuilder:
    """Builds deterministic LiquidityEvidence based on Spec 1.0.1 scoring model."""

    def build(
        self,
        event: LiquidityEvent,
        outcome_direction: EventClassification,
        as_of_ms: int,
        flow_snapshot: EventFlowSnapshot | None = None,
        absorption: EvidenceValue | None = None,
        depth_weighted_imbalance: EvidenceValue | None = None,
        replenishment: EvidenceValue | None = None,
        stacking_pulling: EvidenceValue | None = None,
        displacement_bps: float | None = None,
        volume_expansion_ratio: float | None = None,
    ) -> LiquidityEvidence:
        values: dict[str, EvidenceValue] = {}
        reasons: list[str] = []
        contradictions: list[str] = []

        # Determine directional orientation of outcome
        is_bullish: bool | None = None
        is_confirmed = False

        if outcome_direction in (
            EventClassification.FAILED_BREAKDOWN,
            EventClassification.BULLISH_CONTINUATION,
        ):
            is_bullish = True
            is_confirmed = True
        elif outcome_direction in (
            EventClassification.FAILED_BREAKOUT,
            EventClassification.BEARISH_CONTINUATION,
        ):
            is_bullish = False
            is_confirmed = True

        if not is_confirmed or is_bullish is None:
            # Indeterminate or Invalid outcomes always have zero confidence
            return LiquidityEvidence(
                values={},
                reasons=("OUTCOME_UNCONFIRMED",),
                contradictions=(),
                context_coverage=0.0,
                evidence_strength=0.0,
                contradiction_strength=0.0,
                confidence=0.0,
                confidence_type=ConfidenceType.UNCALIBRATED_DETERMINISTIC_SCORE,
                as_of_ms=as_of_ms,
            )

        reasons.append("PRICE_OUTCOME_CONFIRMED")
        base_support = 0.60
        support_count = 0
        contradiction_count = 0
        available_groups = 0
        total_groups = 4

        # 1. Delta Group
        if flow_snapshot is not None:
            delta_val = flow_snapshot.signed_delta_usdt
            values["post_sweep_delta"] = EvidenceValue(EvidenceAvailability.AVAILABLE, delta_val, as_of_ms)
            available_groups += 1
            if delta_val != 0.0:
                if (delta_val > 0) == is_bullish:
                    support_count += 1
                    reasons.append("POST_SWEEP_DELTA_ALIGNED")
                else:
                    contradiction_count += 1
                    contradictions.append("POST_SWEEP_DELTA_OPPOSED")
        else:
            values["post_sweep_delta"] = EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)

        # 2. CVD Group
        if flow_snapshot is not None:
            cvd_val = flow_snapshot.post_sweep_cvd_usdt
            values["post_sweep_cvd"] = EvidenceValue(EvidenceAvailability.AVAILABLE, cvd_val, as_of_ms)
            available_groups += 1
            if cvd_val != 0.0:
                if (cvd_val > 0) == is_bullish:
                    support_count += 1
                    reasons.append("CVD_MOVEMENT_ALIGNED")
                else:
                    contradiction_count += 1
                    contradictions.append("CVD_MOVEMENT_OPPOSED")
        else:
            values["post_sweep_cvd"] = EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)

        # 3. Absorption Group
        abs_ev = absorption if absorption is not None else EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
        values["bid_ask_absorption"] = abs_ev
        if abs_ev.availability is EvidenceAvailability.AVAILABLE and abs_ev.value is not None:
            available_groups += 1
            if abs_ev.value != 0.0:
                if (abs_ev.value > 0) == is_bullish:
                    support_count += 1
                    reasons.append("ABSORPTION_ALIGNED")
                else:
                    contradiction_count += 1
                    contradictions.append("ABSORPTION_OPPOSED")

        # 4. Depth-weighted Imbalance Group
        depth_ev = depth_weighted_imbalance if depth_weighted_imbalance is not None else EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
        values["depth_weighted_imbalance"] = depth_ev
        if depth_ev.availability is EvidenceAvailability.AVAILABLE and depth_ev.value is not None:
            available_groups += 1
            if depth_ev.value != 0.0:
                if (depth_ev.value > 0) == is_bullish:
                    support_count += 1
                    reasons.append("DEPTH_IMBALANCE_ALIGNED")
                else:
                    contradiction_count += 1
                    contradictions.append("DEPTH_IMBALANCE_OPPOSED")

        # Contextual observation reasons (do not add confidence weight in v1)
        if replenishment and replenishment.availability is EvidenceAvailability.AVAILABLE and replenishment.value:
            values["wall_replenishment"] = replenishment
            if (replenishment.value > 0) == is_bullish:
                reasons.append("REPLENISHMENT_OBSERVED")
            else:
                contradictions.append("REPLENISHMENT_OPPOSED")

        if stacking_pulling and stacking_pulling.availability is EvidenceAvailability.AVAILABLE and stacking_pulling.value:
            values["stacking_pulling"] = stacking_pulling
            if (stacking_pulling.value > 0) == is_bullish:
                reasons.append("STACKING_PULLING_ALIGNED")
            else:
                contradictions.append("STACKING_PULLING_OPPOSED")

        if displacement_bps is not None and isfinite(displacement_bps):
            values["displacement_bps"] = EvidenceValue(EvidenceAvailability.AVAILABLE, float(displacement_bps), as_of_ms)
            reasons.append("DISPLACEMENT_RECORDED")

        if volume_expansion_ratio is not None and isfinite(volume_expansion_ratio):
            values["volume_expansion_ratio"] = EvidenceValue(EvidenceAvailability.AVAILABLE, float(volume_expansion_ratio), as_of_ms)
            reasons.append("VOLUME_EXPANSION_RECORDED")

        evidence_strength = base_support + (0.10 * support_count)
        contradiction_strength = 0.10 * contradiction_count
        confidence = max(0.0, min(1.0, round(evidence_strength - contradiction_strength, 4)))
        context_coverage = round(available_groups / float(total_groups), 4)

        return LiquidityEvidence(
            values=values,
            reasons=tuple(sorted(reasons)),
            contradictions=tuple(sorted(contradictions)),
            context_coverage=context_coverage,
            evidence_strength=round(evidence_strength, 4),
            contradiction_strength=round(contradiction_strength, 4),
            confidence=confidence,
            confidence_type=ConfidenceType.UNCALIBRATED_DETERMINISTIC_SCORE,
            as_of_ms=as_of_ms,
        )
