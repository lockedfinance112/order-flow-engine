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
)
from .policy import LiquidityClassificationPolicy


@dataclass(frozen=True)
class SweepAdapterResult:
    observation: LiquiditySweepObservation | None
    rejected: RejectedSweepInput | None


def parse_utc_ms(value: str) -> int:
    if not value:
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    utc_value = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_value - epoch
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _optional_text(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None or str(value) == "":
        return None
    return str(value)


def _optional_decimal(raw: Mapping[str, Any], key: str) -> Decimal | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError(f"{key} must be finite")
    return parsed


def _optional_float(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    return parsed


def _source_observation_hash(raw: Mapping[str, Any]) -> str | None:
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
        source_file_id=_optional_text(raw, "source_file_id"),
        source_row_hash=_optional_text(raw, "source_row_hash"),
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
        try:
            if not isinstance(detection_time_ms, int) or isinstance(detection_time_ms, bool):
                raise ValueError("detection time must be an integer")
            if detection_time_ms < 0:
                raise ValueError("detection time must be non-negative")

            direction = raw["type"]
            if direction is None:
                raise ValueError("direction is missing")
            side = self.SIDE_MAP[str(direction).upper()]
            event_time_ms = parse_utc_ms(str(raw["timestamp"]))
            swept_level = Decimal(str(raw["sweep_level"]))
            if not swept_level.is_finite() or swept_level <= 0:
                raise ValueError("swept level must be positive")

            raw_event_id = raw["sweep_id"]
            if raw_event_id is None or raw_event_id == "":
                raise ValueError("sweep id is missing")
            event_id = str(raw_event_id)
            raw_symbol = raw["symbol"]
            if raw_symbol is None or raw_symbol == "":
                raise ValueError("symbol is missing")
            symbol = str(raw_symbol).upper()

            source_sweep_price = _optional_decimal(raw, "source_sweep_price")
            source_penetration_bps = _optional_float(raw, "source_penetration_bps")
            source_observation_hash = _source_observation_hash(raw)
            if source_observation_hash is None:
                raise ValueError("callback content is not canonicalizable")
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            return SweepAdapterResult(
                observation=None,
                rejected=rejected_from(raw, detection_time_ms, str(exc)),
            )

        observation = LiquiditySweepObservation(
            event_id=event_id,
            source_event_id=_optional_text(raw, "source_event_id") or event_id,
            source_level_id=_optional_text(raw, "source_level_id"),
            symbol=symbol,
            liquidity_side=side,
            swept_level=swept_level,
            event_time_ms=event_time_ms,
            detection_time_ms=detection_time_ms,
            source_sweep_price=source_sweep_price,
            source_penetration_bps=source_penetration_bps,
            source=SweepSource.SWEEPS_MONITOR_CSV,
            source_file_id=(
                _optional_text(raw, "source_file_id")
                or SweepSource.SWEEPS_MONITOR_CSV.value
            ),
            source_row_hash=_optional_text(raw, "source_row_hash"),
            source_observation_hash=source_observation_hash,
            detector_version=(
                _optional_text(raw, "detector_version") or self.DEFAULT_DETECTOR_VERSION
            ),
        )
        return SweepAdapterResult(observation=observation, rejected=None)
