from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol


class _StringEnum(str, Enum):
    pass


class LiquiditySide(_StringEnum):
    SELL_SIDE = "SELL_SIDE"
    BUY_SIDE = "BUY_SIDE"


class EventState(_StringEnum):
    OBSERVED = "OBSERVED"
    PENETRATION_VALIDATING = "PENETRATION_VALIDATING"
    SWEEP_DETECTED = "SWEEP_DETECTED"
    RECLAIMING = "RECLAIMING"
    ACCEPTING = "ACCEPTING"
    UNRESOLVED = "UNRESOLVED"
    INVALID = "INVALID"
    FINALIZED = "FINALIZED"


class EventClassification(_StringEnum):
    PENDING_SWEEP = "PENDING_SWEEP"
    FAILED_BREAKDOWN = "FAILED_BREAKDOWN"
    FAILED_BREAKOUT = "FAILED_BREAKOUT"
    BEARISH_CONTINUATION = "BEARISH_CONTINUATION"
    BULLISH_CONTINUATION = "BULLISH_CONTINUATION"
    INDETERMINATE = "INDETERMINATE"
    INVALID = "INVALID"


class EvidenceAvailability(_StringEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNSAFE = "UNSAFE"


class ConfidenceType(_StringEnum):
    UNCALIBRATED_DETERMINISTIC_SCORE = "UNCALIBRATED_DETERMINISTIC_SCORE"


class ArtifactIntegrity(_StringEnum):
    COMPLETE = "COMPLETE"
    RECORDER_FAILURE = "RECORDER_FAILURE"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    MISSING_EVENT_ROW = "MISSING_EVENT_ROW"
    MISSING_TRANSITION_ROW = "MISSING_TRANSITION_ROW"


class SweepSource(_StringEnum):
    SWEEPS_MONITOR_CSV = "SWEEPS_MONITOR_CSV"


class AggressorSide(_StringEnum):
    BUY = "BUY"
    SELL = "SELL"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return format(value, "f")
    if is_dataclass(value) and not isinstance(value, type):
        to_canonical_dict = getattr(value, "to_canonical_dict", None)
        if callable(to_canonical_dict):
            return _canonical_value(to_canonical_dict())
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_price(price: Decimal | float | int | str, decimal_places: int) -> str:
    if decimal_places < 0:
        raise ValueError("decimal places must be non-negative")
    normalized = Decimal(str(price)).quantize(
        Decimal(1).scaleb(-decimal_places), rounding=ROUND_HALF_EVEN
    )
    return format(normalized, f".{decimal_places}f")


def _canonical_sequence_component(sequence_id: int | str) -> tuple[int, int | str]:
    if isinstance(sequence_id, int):
        return (0, sequence_id)
    return (1, sequence_id)


@dataclass(frozen=True)
class EvidenceValue:
    availability: EvidenceAvailability
    value: float | None
    as_of_ms: int | None

    def __post_init__(self) -> None:
        timestamp_valid = (
            isinstance(self.as_of_ms, int)
            and not isinstance(self.as_of_ms, bool)
            and self.as_of_ms >= 0
        )
        if self.availability is EvidenceAvailability.UNAVAILABLE:
            if self.value is not None or self.as_of_ms is not None:
                raise ValueError("unavailable evidence must have no value or timestamp")
            return
        if self.availability is EvidenceAvailability.UNSAFE:
            if self.value is not None or not timestamp_valid:
                raise ValueError(
                    "unsafe evidence requires no value and a non-negative timestamp"
                )
            return
        if self.value is None or not timestamp_valid:
            raise ValueError(
                "available evidence requires a value and non-negative timestamp"
            )
        if not isfinite(self.value):
            raise ValueError("available evidence value must be finite")


@dataclass(frozen=True)
class MarketTrade:
    symbol: str
    price: Decimal | float
    quantity: Decimal | float
    aggressor_side: AggressorSide
    exchange_time_ms: int
    sequence_id: int | str
    source: str = "BINANCE_AGG_TRADE"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "aggressor_side": self.aggressor_side.value,
            "exchange_time_ms": self.exchange_time_ms,
            "price": format(Decimal(str(self.price)), "f"),
            "quantity": format(Decimal(str(self.quantity)), "f"),
            "sequence_id": self.sequence_id,
            "source": self.source,
            "symbol": self.symbol,
        }

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_canonical_dict())

    @property
    def canonical_key(self) -> tuple[int, tuple[int, int | str], int, str]:
        return (
            self.exchange_time_ms,
            _canonical_sequence_component(self.sequence_id),
            0,
            self.content_hash,
        )


@dataclass(frozen=True)
class DepthObservation:
    symbol: str
    bids: tuple[tuple[Decimal | float, Decimal | float], ...]
    asks: tuple[tuple[Decimal | float, Decimal | float], ...]
    exchange_time_ms: int
    sequence_id: int | str
    source: str = "BINANCE_DEPTH"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bids", tuple(tuple(level) for level in self.bids))
        object.__setattr__(self, "asks", tuple(tuple(level) for level in self.asks))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "asks": [[format(Decimal(str(price)), "f"), format(Decimal(str(size)), "f")] for price, size in self.asks],
            "bids": [[format(Decimal(str(price)), "f"), format(Decimal(str(size)), "f")] for price, size in self.bids],
            "exchange_time_ms": self.exchange_time_ms,
            "sequence_id": self.sequence_id,
            "source": self.source,
            "symbol": self.symbol,
        }

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.to_canonical_dict())

    @property
    def canonical_key(self) -> tuple[int, tuple[int, int | str], int, str]:
        return (
            self.exchange_time_ms,
            _canonical_sequence_component(self.sequence_id),
            1,
            self.content_hash,
        )


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
class LiquiditySweepObservation:
    event_id: str
    source_event_id: str | None
    source_level_id: str | None
    symbol: str
    liquidity_side: LiquiditySide
    swept_level: Decimal | float
    event_time_ms: int
    detection_time_ms: int
    source_sweep_price: Decimal | float | None
    source_penetration_bps: float | None
    source: SweepSource
    source_file_id: str
    source_row_hash: str | None
    source_observation_hash: str
    detector_version: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "detection_time_ms": self.detection_time_ms,
            "detector_version": self.detector_version,
            "event_id": self.event_id,
            "event_time_ms": self.event_time_ms,
            "liquidity_side": self.liquidity_side.value,
            "source": self.source.value,
            "source_event_id": self.source_event_id,
            "source_file_id": self.source_file_id,
            "source_level_id": self.source_level_id,
            "source_observation_hash": self.source_observation_hash,
            "source_penetration_bps": self.source_penetration_bps,
            "source_row_hash": self.source_row_hash,
            "source_sweep_price": None if self.source_sweep_price is None else normalize_price(self.source_sweep_price, 8),
            "swept_level": normalize_price(self.swept_level, 8),
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class RejectedSweepInput:
    source: SweepSource
    detection_time_ms: int
    reason_code: str
    reason_detail: str
    source_file_id: str | None = None
    source_row_hash: str | None = None
    source_observation_hash: str | None = None

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "detection_time_ms": self.detection_time_ms,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "source": self.source.value,
            "source_file_id": self.source_file_id,
            "source_observation_hash": self.source_observation_hash,
            "source_row_hash": self.source_row_hash,
        }


@dataclass(frozen=True)
class LifecycleTransition:
    event_id: str
    previous_state: EventState | None
    next_state: EventState
    transition_time_ms: int
    transition_sequence: int
    reason_code: str | None = None

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "next_state": self.next_state.value,
            "previous_state": None if self.previous_state is None else self.previous_state.value,
            "reason_code": self.reason_code,
            "transition_sequence": self.transition_sequence,
            "transition_time_ms": self.transition_time_ms,
        }


@dataclass(frozen=True)
class LiquidityEvidence:
    values: Mapping[str, EvidenceValue] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    context_coverage: float = 0.0
    evidence_strength: float = 0.0
    contradiction_strength: float = 0.0
    confidence: float = 0.0
    confidence_type: ConfidenceType = ConfidenceType.UNCALIBRATED_DETERMINISTIC_SCORE
    as_of_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "contradictions", tuple(self.contradictions))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "as_of_ms": self.as_of_ms,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type.value,
            "context_coverage": self.context_coverage,
            "contradiction_strength": self.contradiction_strength,
            "contradictions": sorted(self.contradictions),
            "evidence_strength": self.evidence_strength,
            "reasons": sorted(self.reasons),
            "values": {
                key: {
                    "as_of_ms": value.as_of_ms,
                    "availability": value.availability.value,
                    "value": value.value,
                }
                for key, value in sorted(self.values.items())
            },
        }


@dataclass
class LiquidityEvent:
    observation: LiquiditySweepObservation
    state: EventState = EventState.OBSERVED
    classification: EventClassification = EventClassification.PENDING_SWEEP
    reason_code: str | None = None
    transitions: list[LifecycleTransition] = field(default_factory=list)

    @property
    def event_id(self) -> str:
        return self.observation.event_id

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "event_id": self.event_id,
            "observation": self.observation.to_canonical_dict(),
            "reason_code": self.reason_code,
            "state": self.state.value,
            "transitions": [transition.to_canonical_dict() for transition in self.transitions],
        }


@dataclass(frozen=True)
class LiquidityEventResult:
    event_id: str
    symbol: str
    liquidity_side: LiquiditySide
    classification: EventClassification
    reason_code: str | None
    event_time_ms: int
    detection_time_ms: int
    market_resolution_time_ms: int
    classification_time_ms: int
    source_observation_hash: str
    evidence: LiquidityEvidence
    policy_hash: str
    model_version: str

    def __post_init__(self) -> None:
        expected = max(self.market_resolution_time_ms, self.detection_time_ms)
        if self.classification_time_ms != expected:
            raise ValueError(
                "classification time must equal the later of market resolution and detection"
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "classification_time_ms": self.classification_time_ms,
            "detection_time_ms": self.detection_time_ms,
            "event_id": self.event_id,
            "event_time_ms": self.event_time_ms,
            "evidence": self.evidence.to_canonical_dict(),
            "liquidity_side": self.liquidity_side.value,
            "market_resolution_time_ms": self.market_resolution_time_ms,
            "model_version": self.model_version,
            "policy_hash": self.policy_hash,
            "reason_code": self.reason_code,
            "source_observation_hash": self.source_observation_hash,
            "symbol": self.symbol,
        }
