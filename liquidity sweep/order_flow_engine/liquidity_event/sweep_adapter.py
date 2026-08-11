from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping

from .models import (
    LiquiditySide,
    LiquiditySweepObservation,
    RejectedSweepInput,
    SweepSource,
    canonical_hash,
    normalize_price,
)
from .policy import LiquidityClassificationPolicy


@dataclass(frozen=True)
class SweepAdapterResult:
    observation: LiquiditySweepObservation | None
    rejected: RejectedSweepInput | None


class _InvalidSweepObservation(ValueError):
    def __init__(self, detail_code: str):
        super().__init__(detail_code)
        self.detail_code = detail_code


def parse_utc_ms(value: str) -> int:
    if not value:
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    utc_value = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_value - epoch
    if delta.microseconds % 1_000:
        raise ValueError("timestamp must have millisecond precision")
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _required_text(
    raw: Mapping[str, Any], key: str, detail_code: str
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise _InvalidSweepObservation(detail_code)
    return value


def _optional_text(
    raw: Mapping[str, Any], key: str, detail_code: str
) -> str | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise _InvalidSweepObservation(detail_code)
    return value


def _optional_decimal(
    raw: Mapping[str, Any], key: str, detail_code: str
) -> Decimal | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (Decimal, float, int, str)):
        raise _InvalidSweepObservation(detail_code)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise _InvalidSweepObservation(detail_code) from None
    if not parsed.is_finite():
        raise _InvalidSweepObservation(detail_code)
    return parsed


def _optional_float(
    raw: Mapping[str, Any], key: str, detail_code: str
) -> float | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (Decimal, float, int, str)):
        raise _InvalidSweepObservation(detail_code)
    try:
        supplied = Decimal(str(value))
        parsed = float(supplied)
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        raise _InvalidSweepObservation(detail_code) from None
    if (
        not supplied.is_finite()
        or not math.isfinite(parsed)
        or Decimal(str(parsed)) != supplied
    ):
        raise _InvalidSweepObservation(detail_code)
    return parsed


def _source_file_id(raw: Mapping[str, Any]) -> str:
    value = raw.get("source_file_id")
    expected = SweepSource.SWEEPS_MONITOR_CSV.value
    if value is None:
        return expected
    if not isinstance(value, str) or value != expected:
        raise _InvalidSweepObservation("INVALID_SOURCE_FILE_ID")
    return value


def _sanitized_source_file_id(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("source_file_id")
    if value == SweepSource.SWEEPS_MONITOR_CSV.value:
        return SweepSource.SWEEPS_MONITOR_CSV.value
    return None


def _sanitized_source_row_hash(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("source_row_hash")
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def _source_row_hash(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("source_row_hash")
    if value is None:
        return None
    sanitized = _sanitized_source_row_hash(raw)
    if sanitized is None:
        raise _InvalidSweepObservation("INVALID_SOURCE_ROW_HASH")
    return sanitized


def _source_observation_hash(raw: Mapping[str, Any]) -> str | None:
    source_file_id = raw.get("source_file_id")
    if source_file_id not in (None, SweepSource.SWEEPS_MONITOR_CSV.value):
        return None
    try:
        return canonical_hash(dict(raw))
    except (TypeError, ValueError):
        return None


def rejected_from(
    raw: Mapping[str, Any], detection_time_ms: int, reason_detail: str
) -> RejectedSweepInput:
    return RejectedSweepInput(
        source=SweepSource.SWEEPS_MONITOR_CSV,
        detection_time_ms=detection_time_ms,
        reason_code="INVALID_SWEEP_OBSERVATION",
        reason_detail=reason_detail,
        source_file_id=_sanitized_source_file_id(raw),
        source_row_hash=_sanitized_source_row_hash(raw),
        source_observation_hash=_source_observation_hash(raw),
    )


class SweepsMonitorAdapter:
    SIDE_MAP = {
        "BULLISH": LiquiditySide.SELL_SIDE,
        "BEARISH": LiquiditySide.BUY_SIDE,
    }
    DEFAULT_DETECTOR_VERSION = "UNKNOWN"

    def __init__(self, policy: LiquidityClassificationPolicy):
        self.policy = policy

    def adapt(
        self, raw: Mapping[str, Any], detection_time_ms: int
    ) -> SweepAdapterResult:
        rejection_detection_time_ms = (
            detection_time_ms
            if isinstance(detection_time_ms, int)
            and not isinstance(detection_time_ms, bool)
            else -1
        )
        try:
            if not isinstance(detection_time_ms, int) or isinstance(detection_time_ms, bool):
                raise _InvalidSweepObservation("INVALID_DETECTION_TIME")
            if detection_time_ms < 0:
                raise _InvalidSweepObservation("INVALID_DETECTION_TIME")

            direction = _required_text(raw, "type", "INVALID_DIRECTION")
            try:
                side = self.SIDE_MAP[direction.upper()]
            except KeyError:
                raise _InvalidSweepObservation("INVALID_DIRECTION") from None

            timestamp = _required_text(raw, "timestamp", "INVALID_TIMESTAMP")
            try:
                event_time_ms = parse_utc_ms(timestamp)
            except (TypeError, ValueError, OverflowError):
                raise _InvalidSweepObservation("INVALID_TIMESTAMP") from None

            raw_swept_level = raw.get("sweep_level")
            if isinstance(raw_swept_level, bool) or not isinstance(
                raw_swept_level, (Decimal, float, int, str)
            ):
                raise _InvalidSweepObservation("INVALID_SWEPT_LEVEL")
            try:
                swept_level = Decimal(str(raw_swept_level))
            except (InvalidOperation, TypeError, ValueError):
                raise _InvalidSweepObservation("INVALID_SWEPT_LEVEL") from None
            if not swept_level.is_finite() or swept_level <= 0:
                raise _InvalidSweepObservation("INVALID_SWEPT_LEVEL")
            try:
                normalized_swept_level = normalize_price(
                    swept_level,
                    self.policy.canonical_price_decimal_places,
                )
            except (InvalidOperation, TypeError, ValueError):
                raise _InvalidSweepObservation("INVALID_SWEPT_LEVEL") from None
            if Decimal(normalized_swept_level) <= 0:
                raise _InvalidSweepObservation("INVALID_SWEPT_LEVEL")

            source_event_id = _required_text(
                raw, "sweep_id", "INVALID_SOURCE_EVENT_ID"
            )
            symbol = _required_text(raw, "symbol", "INVALID_SYMBOL").upper()
            source_file_id = _source_file_id(raw)
            source_row_hash = _source_row_hash(raw)

            source_sweep_price = _optional_decimal(
                raw, "source_sweep_price", "INVALID_SOURCE_SWEEP_PRICE"
            )
            if source_sweep_price is not None:
                try:
                    normalized_source_sweep_price = normalize_price(
                        source_sweep_price,
                        self.policy.canonical_price_decimal_places,
                    )
                except (InvalidOperation, TypeError, ValueError):
                    raise _InvalidSweepObservation(
                        "INVALID_SOURCE_SWEEP_PRICE"
                    ) from None
                if (
                    source_sweep_price != 0
                    and Decimal(normalized_source_sweep_price) == 0
                ):
                    raise _InvalidSweepObservation("INVALID_SOURCE_SWEEP_PRICE")
            source_penetration_bps = _optional_float(
                raw,
                "source_penetration_bps",
                "INVALID_SOURCE_PENETRATION_BPS",
            )
            source_level_id = _optional_text(
                raw, "source_level_id", "INVALID_SOURCE_LEVEL_ID"
            )
            detector_version = (
                _optional_text(
                    raw, "detector_version", "INVALID_DETECTOR_VERSION"
                )
                or self.DEFAULT_DETECTOR_VERSION
            )
            source_observation_hash = _source_observation_hash(raw)
            if source_observation_hash is None:
                raise _InvalidSweepObservation("NON_CANONICAL_CALLBACK")

            event_id = canonical_hash(
                {
                    "event_time_ms": event_time_ms,
                    "liquidity_side": side.value,
                    "source": SweepSource.SWEEPS_MONITOR_CSV.value,
                    "source_event_id": source_event_id,
                    "swept_level": normalized_swept_level,
                    "symbol": symbol,
                }
            )
        except _InvalidSweepObservation as exc:
            return SweepAdapterResult(
                observation=None,
                rejected=rejected_from(
                    raw,
                    rejection_detection_time_ms,
                    exc.detail_code,
                ),
            )

        observation = LiquiditySweepObservation(
            event_id=event_id,
            source_event_id=source_event_id,
            source_level_id=source_level_id,
            symbol=symbol,
            liquidity_side=side,
            swept_level=swept_level,
            event_time_ms=event_time_ms,
            detection_time_ms=detection_time_ms,
            source_sweep_price=source_sweep_price,
            source_penetration_bps=source_penetration_bps,
            source=SweepSource.SWEEPS_MONITOR_CSV,
            source_file_id=source_file_id,
            source_row_hash=source_row_hash,
            source_observation_hash=source_observation_hash,
            detector_version=detector_version,
        )
        return SweepAdapterResult(observation=observation, rejected=None)
