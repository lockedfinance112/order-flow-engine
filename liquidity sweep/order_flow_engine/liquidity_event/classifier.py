"""Deterministic price outcome classifier for Phase 1C.1 liquidity events."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .models import (
    EventClassification,
    LiquiditySide,
    MarketTrade,
)
from .policy import LiquidityClassificationPolicy


@dataclass(frozen=True)
class CandidateProgress:
    accumulated_ms: int = 0
    segment_start_ms: int | None = None
    last_qualifying_ms: int | None = None
    qualifying_trade_count: int = 0
    sufficient_time_ms: int | None = None

    @classmethod
    def sufficient_at(cls, timestamp_ms: int) -> CandidateProgress:
        return cls(
            accumulated_ms=3000,
            segment_start_ms=timestamp_ms - 3000,
            last_qualifying_ms=timestamp_ms,
            qualifying_trade_count=3,
            sufficient_time_ms=timestamp_ms,
        )


@dataclass(frozen=True)
class PriceDecision:
    classification: EventClassification
    reason_code: str
    market_resolution_time_ms: int | None
    reclaim_sufficient_time_ms: int | None = None
    acceptance_sufficient_time_ms: int | None = None
    has_penetration: bool = False
    penetration_price: Decimal | None = None
    penetration_bps: float | None = None
    penetration_time_ms: int | None = None


class PriceOutcomeTracker:
    """Tracks point-in-time canonical trade stream to classify post-sweep price outcome.

    Strictly price-driven, deterministic, and free of contextual evidence dependencies.
    """

    def __init__(
        self,
        level: float | Decimal,
        side: LiquiditySide,
        policy: LiquidityClassificationPolicy | None = None,
        symbol: str | None = None,
    ) -> None:
        self.level = level
        self.side = side
        self.policy = policy or LiquidityClassificationPolicy()
        self.symbol = symbol

        self._level_dec = level if isinstance(level, Decimal) else Decimal(str(level))
        bps_factor = Decimal("10000")
        reclaim_bps = Decimal(str(self.policy.reclaim_buffer_bps)) / bps_factor
        acceptance_bps = Decimal(str(self.policy.acceptance_buffer_bps)) / bps_factor

        if self.side is LiquiditySide.SELL_SIDE:
            self._reclaim_threshold = self._level_dec * (Decimal("1") + reclaim_bps)
            self._acceptance_threshold = self._level_dec * (Decimal("1") - acceptance_bps)
        else:
            self._reclaim_threshold = self._level_dec * (Decimal("1") - reclaim_bps)
            self._acceptance_threshold = self._level_dec * (Decimal("1") + acceptance_bps)

        self._has_penetration: bool = False
        self._penetration_price: Decimal | None = None
        self._penetration_bps: float | None = None
        self._penetration_time_ms: int | None = None

        self._reclaim_progress = CandidateProgress()
        self._acceptance_progress = CandidateProgress()

    @property
    def has_penetration(self) -> bool:
        return self._has_penetration

    @property
    def penetration_price(self) -> Decimal | None:
        return self._penetration_price

    @property
    def penetration_bps(self) -> float | None:
        return self._penetration_bps

    @property
    def penetration_time_ms(self) -> int | None:
        return self._penetration_time_ms

    @property
    def reclaim_sufficient_time_ms(self) -> int | None:
        return self._reclaim_progress.sufficient_time_ms

    @property
    def acceptance_sufficient_time_ms(self) -> int | None:
        return self._acceptance_progress.sufficient_time_ms

    @property
    def reclaim_qualifying_duration_ms(self) -> int:
        return self._reclaim_progress.accumulated_ms

    @property
    def acceptance_qualifying_duration_ms(self) -> int:
        return self._acceptance_progress.accumulated_ms

    @property
    def reclaim_trade_count(self) -> int:
        return self._reclaim_progress.qualifying_trade_count

    @property
    def acceptance_trade_count(self) -> int:
        return self._acceptance_progress.qualifying_trade_count

    def on_trade(self, trade: MarketTrade) -> None:
        if self.symbol is not None and trade.symbol != self.symbol:
            return

        p_dec = trade.price if isinstance(trade.price, Decimal) else Decimal(str(trade.price))

        # Check / update penetration
        if not self._has_penetration:
            if self.side is LiquiditySide.SELL_SIDE and p_dec < self._level_dec:
                self._has_penetration = True
                self._penetration_time_ms = trade.exchange_time_ms
                self._penetration_price = p_dec
                pen_diff = self._level_dec - p_dec
                self._penetration_bps = float(round((pen_diff / self._level_dec) * Decimal("10000"), 4))
            elif self.side is LiquiditySide.BUY_SIDE and p_dec > self._level_dec:
                self._has_penetration = True
                self._penetration_time_ms = trade.exchange_time_ms
                self._penetration_price = p_dec
                pen_diff = p_dec - self._level_dec
                self._penetration_bps = float(round((pen_diff / self._level_dec) * Decimal("10000"), 4))
        else:
            if self.side is LiquiditySide.SELL_SIDE:
                if self._penetration_price is not None and p_dec < self._penetration_price:
                    self._penetration_price = p_dec
                    pen_diff = self._level_dec - p_dec
                    self._penetration_bps = float(round((pen_diff / self._level_dec) * Decimal("10000"), 4))
            elif self.side is LiquiditySide.BUY_SIDE:
                if self._penetration_price is not None and p_dec > self._penetration_price:
                    self._penetration_price = p_dec
                    pen_diff = p_dec - self._level_dec
                    self._penetration_bps = float(round((pen_diff / self._level_dec) * Decimal("10000"), 4))

        if not self._has_penetration:
            # Pre-penetration trades cannot accumulate candidate progress
            return

        # Evaluate threshold qualification
        if self.side is LiquiditySide.SELL_SIDE:
            is_reclaim = p_dec > self._reclaim_threshold
            is_acceptance = p_dec < self._acceptance_threshold
        else:
            is_reclaim = p_dec < self._reclaim_threshold
            is_acceptance = p_dec > self._acceptance_threshold

        if is_reclaim:
            self._acceptance_progress = self._reset_candidate(self._acceptance_progress)
            self._reclaim_progress = self._advance_candidate(self._reclaim_progress, trade.exchange_time_ms)
        elif is_acceptance:
            self._reclaim_progress = self._reset_candidate(self._reclaim_progress)
            self._acceptance_progress = self._advance_candidate(self._acceptance_progress, trade.exchange_time_ms)
        else:
            # Neutral trade (between or exactly on thresholds) pauses current segments
            self._reclaim_progress = self._pause_candidate(self._reclaim_progress)
            self._acceptance_progress = self._pause_candidate(self._acceptance_progress)

    def _advance_candidate(self, progress: CandidateProgress, t_ms: int) -> CandidateProgress:
        if progress.sufficient_time_ms is not None:
            return progress

        trade_count = progress.qualifying_trade_count + 1
        if progress.last_qualifying_ms is None:
            seg_start = t_ms
            last_q = t_ms
            acc = progress.accumulated_ms
        else:
            delta = max(0, t_ms - progress.last_qualifying_ms)
            seg_start = progress.segment_start_ms
            last_q = t_ms
            acc = progress.accumulated_ms + delta

        suff_time = progress.sufficient_time_ms
        if (
            suff_time is None
            and acc >= self.policy.confirmation_hold_ms
            and trade_count >= self.policy.minimum_confirming_trades
        ):
            suff_time = t_ms

        return CandidateProgress(
            accumulated_ms=acc,
            segment_start_ms=seg_start,
            last_qualifying_ms=last_q,
            qualifying_trade_count=trade_count,
            sufficient_time_ms=suff_time,
        )

    def _reset_candidate(self, progress: CandidateProgress) -> CandidateProgress:
        """Resets unsatisfied candidate progress; preserves immutable sufficient witness."""
        if progress.sufficient_time_ms is not None:
            return progress
        return CandidateProgress()

    def _pause_candidate(self, progress: CandidateProgress) -> CandidateProgress:
        if progress.sufficient_time_ms is not None or progress.last_qualifying_ms is None:
            return progress
        return CandidateProgress(
            accumulated_ms=progress.accumulated_ms,
            segment_start_ms=None,
            last_qualifying_ms=None,
            qualifying_trade_count=progress.qualifying_trade_count,
            sufficient_time_ms=progress.sufficient_time_ms,
        )

    def decision_at(self, watermark_ms: int, expiry_ms: int) -> PriceDecision:
        penetration_in_window = (
            self._has_penetration
            and self._penetration_time_ms is not None
            and self._penetration_time_ms <= expiry_ms
        )

        reclaim_suff = self._reclaim_progress.sufficient_time_ms
        reclaim_settled = (
            penetration_in_window
            and (reclaim_suff is not None)
            and (reclaim_suff <= expiry_ms)
            and (reclaim_suff <= watermark_ms)
        )

        accept_suff = self._acceptance_progress.sufficient_time_ms
        accept_settled = (
            penetration_in_window
            and (accept_suff is not None)
            and (accept_suff <= expiry_ms)
            and (accept_suff <= watermark_ms)
        )

        if reclaim_settled and accept_settled:
            if reclaim_suff < accept_suff:
                classification = (
                    EventClassification.FAILED_BREAKDOWN
                    if self.side is LiquiditySide.SELL_SIDE
                    else EventClassification.FAILED_BREAKOUT
                )
                return PriceDecision(
                    classification=classification,
                    reason_code="RECLAIM_CONFIRMED",
                    market_resolution_time_ms=reclaim_suff,
                    reclaim_sufficient_time_ms=reclaim_suff,
                    acceptance_sufficient_time_ms=accept_suff,
                    has_penetration=self._has_penetration,
                    penetration_price=self._penetration_price,
                    penetration_bps=self._penetration_bps,
                    penetration_time_ms=self._penetration_time_ms,
                )
            elif accept_suff < reclaim_suff:
                classification = (
                    EventClassification.BEARISH_CONTINUATION
                    if self.side is LiquiditySide.SELL_SIDE
                    else EventClassification.BULLISH_CONTINUATION
                )
                return PriceDecision(
                    classification=classification,
                    reason_code="ACCEPTANCE_CONFIRMED",
                    market_resolution_time_ms=accept_suff,
                    reclaim_sufficient_time_ms=reclaim_suff,
                    acceptance_sufficient_time_ms=accept_suff,
                    has_penetration=self._has_penetration,
                    penetration_price=self._penetration_price,
                    penetration_bps=self._penetration_bps,
                    penetration_time_ms=self._penetration_time_ms,
                )
            else:
                return PriceDecision(
                    classification=EventClassification.INDETERMINATE,
                    reason_code="CONFLICTING_CONFIRMATION",
                    market_resolution_time_ms=reclaim_suff,
                    reclaim_sufficient_time_ms=reclaim_suff,
                    acceptance_sufficient_time_ms=accept_suff,
                    has_penetration=self._has_penetration,
                    penetration_price=self._penetration_price,
                    penetration_bps=self._penetration_bps,
                    penetration_time_ms=self._penetration_time_ms,
                )

        if reclaim_settled:
            classification = (
                EventClassification.FAILED_BREAKDOWN
                if self.side is LiquiditySide.SELL_SIDE
                else EventClassification.FAILED_BREAKOUT
            )
            return PriceDecision(
                classification=classification,
                reason_code="RECLAIM_CONFIRMED",
                market_resolution_time_ms=reclaim_suff,
                reclaim_sufficient_time_ms=reclaim_suff,
                acceptance_sufficient_time_ms=accept_suff,
                has_penetration=self._has_penetration,
                penetration_price=self._penetration_price,
                penetration_bps=self._penetration_bps,
                penetration_time_ms=self._penetration_time_ms,
            )

        if accept_settled:
            classification = (
                EventClassification.BEARISH_CONTINUATION
                if self.side is LiquiditySide.SELL_SIDE
                else EventClassification.BULLISH_CONTINUATION
            )
            return PriceDecision(
                classification=classification,
                reason_code="ACCEPTANCE_CONFIRMED",
                market_resolution_time_ms=accept_suff,
                reclaim_sufficient_time_ms=reclaim_suff,
                acceptance_sufficient_time_ms=accept_suff,
                has_penetration=self._has_penetration,
                penetration_price=self._penetration_price,
                penetration_bps=self._penetration_bps,
                penetration_time_ms=self._penetration_time_ms,
            )

        if watermark_ms >= expiry_ms:
            if not penetration_in_window:
                return PriceDecision(
                    classification=EventClassification.INVALID,
                    reason_code="PENETRATION_NOT_CONFIRMED",
                    market_resolution_time_ms=expiry_ms,
                    reclaim_sufficient_time_ms=reclaim_suff,
                    acceptance_sufficient_time_ms=accept_suff,
                    has_penetration=self._has_penetration,
                    penetration_price=self._penetration_price,
                    penetration_bps=self._penetration_bps,
                    penetration_time_ms=self._penetration_time_ms,
                )
            return PriceDecision(
                classification=EventClassification.INDETERMINATE,
                reason_code="CONFIRMATION_WINDOW_EXPIRED",
                market_resolution_time_ms=expiry_ms,
                reclaim_sufficient_time_ms=reclaim_suff,
                acceptance_sufficient_time_ms=accept_suff,
                has_penetration=True,
                penetration_price=self._penetration_price,
                penetration_bps=self._penetration_bps,
                penetration_time_ms=self._penetration_time_ms,
            )

        return PriceDecision(
            classification=EventClassification.PENDING_SWEEP,
            reason_code="PENDING_CONFIRMATION",
            market_resolution_time_ms=None,
            reclaim_sufficient_time_ms=reclaim_suff,
            acceptance_sufficient_time_ms=accept_suff,
            has_penetration=self._has_penetration,
            penetration_price=self._penetration_price,
            penetration_bps=self._penetration_bps,
            penetration_time_ms=self._penetration_time_ms,
        )
