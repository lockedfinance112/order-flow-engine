from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import (
    ArtifactIntegrity,
    LifecycleTransition,
    LiquidityEventResult,
    RejectedSweepInput,
    canonical_hash,
)

logger = logging.getLogger(__name__)

EVENTS_FILENAME = "liquidity_events.csv"
TRANSITIONS_FILENAME = "liquidity_event_transitions.csv"
REJECTED_FILENAME = "liquidity_rejected_inputs.csv"

EVENT_COLUMNS = (
    "event_id",
    "symbol",
    "liquidity_side",
    "classification",
    "reason_code",
    "event_time_ms",
    "detection_time_ms",
    "market_resolution_time_ms",
    "classification_time_ms",
    "confidence",
    "confidence_type",
    "context_coverage",
    "evidence_strength",
    "contradiction_strength",
    "reasons",
    "contradictions",
    "source_observation_hash",
    "policy_hash",
    "model_version",
)

TRANSITION_COLUMNS = (
    "event_id",
    "transition_sequence",
    "transition_time_ms",
    "previous_state",
    "next_state",
    "reason_code",
)

REJECTED_COLUMNS = (
    "source",
    "detection_time_ms",
    "reason_code",
    "reason_detail",
    "source_file_id",
    "source_row_hash",
    "source_observation_hash",
)


@dataclass(frozen=True)
class CanonicalRecord:
    record_type: str  # "TRANSITION" | "EVENT" | "REJECTED_INPUT"
    payload: LifecycleTransition | LiquidityEventResult | RejectedSweepInput

    @property
    def record_type_rank(self) -> int:
        if self.record_type == "TRANSITION" or isinstance(self.payload, LifecycleTransition):
            return 0
        if self.record_type == "EVENT" or isinstance(self.payload, LiquidityEventResult):
            return 1
        return 2

    @property
    def canonical_record_time_ms(self) -> int:
        if isinstance(self.payload, LifecycleTransition):
            return self.payload.transition_time_ms
        if isinstance(self.payload, LiquidityEventResult):
            return self.payload.classification_time_ms
        if isinstance(self.payload, RejectedSweepInput):
            return self.payload.detection_time_ms
        raise ValueError(f"Unknown payload type: {type(self.payload)}")

    @property
    def event_id(self) -> str:
        if isinstance(self.payload, (LifecycleTransition, LiquidityEventResult)):
            return self.payload.event_id
        return ""

    @property
    def transition_sequence(self) -> int:
        if isinstance(self.payload, LifecycleTransition):
            return self.payload.transition_sequence
        return 0

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.payload.to_canonical_dict())

    @property
    def canonical_key(self) -> tuple[int, str, int, int, str]:
        return (
            self.canonical_record_time_ms,
            self.event_id,
            self.transition_sequence,
            self.record_type_rank,
            self.content_hash,
        )


@dataclass
class RecorderTelemetry:
    status: str = "IDLE"
    failure_count: int = 0
    pending_write_count: int = 0
    last_error: str | None = None
    written_event_count: int = 0
    written_transition_count: int = 0
    written_rejected_count: int = 0

    @property
    def pending_writes(self) -> int:
        return self.pending_write_count


@dataclass(frozen=True)
class RenderedArtifacts:
    events_bytes: bytes
    transitions_bytes: bytes
    rejected_inputs_bytes: bytes

    @property
    def all_bytes(self) -> bytes:
        return self.events_bytes + self.transitions_bytes + self.rejected_inputs_bytes


def _format_float(val: float | Decimal | None, decimals: int = 4) -> str:
    if val is None:
        return ""
    return f"{float(val):.{decimals}f}"


def _format_json_array(items: Iterable[str]) -> str:
    return json.dumps(sorted(items), separators=(",", ":"))


def _serialize_event_row(res: LiquidityEventResult) -> list[str]:
    ev = res.evidence
    return [
        res.event_id,
        res.symbol,
        res.liquidity_side.value,
        res.classification.value,
        res.reason_code or "",
        str(res.event_time_ms),
        str(res.detection_time_ms),
        str(res.market_resolution_time_ms),
        str(res.classification_time_ms),
        _format_float(ev.confidence, 4),
        ev.confidence_type.value,
        _format_float(ev.context_coverage, 4),
        _format_float(ev.evidence_strength, 4),
        _format_float(ev.contradiction_strength, 4),
        _format_json_array(ev.reasons),
        _format_json_array(ev.contradictions),
        res.source_observation_hash,
        res.policy_hash,
        res.model_version,
    ]


def _serialize_transition_row(t: LifecycleTransition) -> list[str]:
    return [
        t.event_id,
        str(t.transition_sequence),
        str(t.transition_time_ms),
        t.previous_state.value if t.previous_state else "",
        t.next_state.value,
        t.reason_code or "",
    ]


def _serialize_rejected_row(rej: RejectedSweepInput) -> list[str]:
    return [
        rej.source.value,
        str(rej.detection_time_ms),
        rej.reason_code,
        rej.reason_detail,
        rej.source_file_id or "",
        rej.source_row_hash or "",
        rej.source_observation_hash or "",
    ]


class LiquidityEventRecorder:
    def __init__(
        self,
        mode: str = "live",
        output_dir: Path | str = "./artifacts",
        queue_max_items: int = 10_000,
        auto_flush: bool = False,
    ):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.queue_max_items = queue_max_items
        self.auto_flush = auto_flush
        self.telemetry = RecorderTelemetry(status="RUNNING")
        self._artifact_integrity: ArtifactIntegrity = ArtifactIntegrity.COMPLETE
        self._queue: deque[CanonicalRecord] = deque()
        self._records: list[CanonicalRecord] = []
        self._written_event_ids: set[str] = set()
        self._written_transition_keys: set[tuple[str, int]] = set()

    @property
    def artifact_integrity(self) -> ArtifactIntegrity:
        return self._artifact_integrity

    def enqueue(
        self,
        record: CanonicalRecord | LifecycleTransition | LiquidityEventResult | RejectedSweepInput,
    ) -> bool:
        if self._artifact_integrity is ArtifactIntegrity.QUEUE_OVERFLOW:
            self.telemetry.failure_count += 1
            return False

        if not isinstance(record, CanonicalRecord):
            if isinstance(record, LifecycleTransition):
                rec = CanonicalRecord("TRANSITION", record)
            elif isinstance(record, LiquidityEventResult):
                rec = CanonicalRecord("EVENT", record)
            elif isinstance(record, RejectedSweepInput):
                rec = CanonicalRecord("REJECTED_INPUT", record)
            else:
                raise ValueError(f"Unsupported record type: {type(record)}")
        else:
            rec = record

        current_count = len(self._records) if self.mode == "replay" else len(self._queue)
        if current_count >= self.queue_max_items:
            self.telemetry.failure_count += 1
            self.telemetry.last_error = "queue overflow"
            self._artifact_integrity = ArtifactIntegrity.QUEUE_OVERFLOW
            return False

        if self.mode == "replay":
            self._records.append(rec)
            self.telemetry.pending_write_count = len(self._records)
            if isinstance(rec.payload, LiquidityEventResult):
                self._written_event_ids.add(rec.payload.event_id)
            elif isinstance(rec.payload, LifecycleTransition):
                self._written_transition_keys.add((rec.payload.event_id, rec.payload.transition_sequence))
            return True

        # Live mode
        self._queue.append(rec)
        self.telemetry.pending_write_count = len(self._queue)
        if isinstance(rec.payload, LiquidityEventResult):
            self._written_event_ids.add(rec.payload.event_id)
        elif isinstance(rec.payload, LifecycleTransition):
            self._written_transition_keys.add((rec.payload.event_id, rec.payload.transition_sequence))

        if self.auto_flush:
            self.flush()
        return True

    def flush_replay(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sorted_records = sorted(self._records, key=lambda r: r.canonical_key)

            events_rows: list[list[str]] = []
            transitions_rows: list[list[str]] = []
            rejected_rows: list[list[str]] = []

            for rec in sorted_records:
                if isinstance(rec.payload, LiquidityEventResult):
                    events_rows.append(_serialize_event_row(rec.payload))
                    self._written_event_ids.add(rec.payload.event_id)
                elif isinstance(rec.payload, LifecycleTransition):
                    transitions_rows.append(_serialize_transition_row(rec.payload))
                    self._written_transition_keys.add((rec.payload.event_id, rec.payload.transition_sequence))
                elif isinstance(rec.payload, RejectedSweepInput):
                    rejected_rows.append(_serialize_rejected_row(rec.payload))

            # Write liquidity_events.csv
            with open(self.output_dir / EVENTS_FILENAME, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(EVENT_COLUMNS)
                writer.writerows(events_rows)
            self.telemetry.written_event_count = len(events_rows)

            # Write liquidity_event_transitions.csv
            with open(self.output_dir / TRANSITIONS_FILENAME, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(TRANSITION_COLUMNS)
                writer.writerows(transitions_rows)
            self.telemetry.written_transition_count = len(transitions_rows)

            # Write liquidity_rejected_inputs.csv
            with open(self.output_dir / REJECTED_FILENAME, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(REJECTED_COLUMNS)
                writer.writerows(rejected_rows)
            self.telemetry.written_rejected_count = len(rejected_rows)

            self.telemetry.pending_write_count = 0
        except Exception as e:
            self.telemetry.failure_count += 1
            self.telemetry.last_error = str(e)
            self._artifact_integrity = ArtifactIntegrity.RECORDER_FAILURE
            logger.error("flush_replay error: %s", e)

    def flush(self) -> None:
        if self.mode == "replay":
            self.flush_replay()
            return

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            while self._queue:
                rec = self._queue[0]
                if isinstance(rec.payload, LiquidityEventResult):
                    filepath = self.output_dir / EVENTS_FILENAME
                    is_new = not filepath.exists() or filepath.stat().st_size == 0
                    with open(filepath, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, lineterminator="\n")
                        if is_new:
                            writer.writerow(EVENT_COLUMNS)
                        writer.writerow(_serialize_event_row(rec.payload))
                    self.telemetry.written_event_count += 1
                elif isinstance(rec.payload, LifecycleTransition):
                    filepath = self.output_dir / TRANSITIONS_FILENAME
                    is_new = not filepath.exists() or filepath.stat().st_size == 0
                    with open(filepath, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, lineterminator="\n")
                        if is_new:
                            writer.writerow(TRANSITION_COLUMNS)
                        writer.writerow(_serialize_transition_row(rec.payload))
                    self.telemetry.written_transition_count += 1
                elif isinstance(rec.payload, RejectedSweepInput):
                    filepath = self.output_dir / REJECTED_FILENAME
                    is_new = not filepath.exists() or filepath.stat().st_size == 0
                    with open(filepath, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, lineterminator="\n")
                        if is_new:
                            writer.writerow(REJECTED_COLUMNS)
                        writer.writerow(_serialize_rejected_row(rec.payload))
                    self.telemetry.written_rejected_count += 1

                self._queue.popleft()
                self.telemetry.pending_write_count = len(self._queue)
        except Exception as e:
            self.telemetry.failure_count += 1
            self.telemetry.last_error = str(e)
            self._artifact_integrity = ArtifactIntegrity.RECORDER_FAILURE
            logger.error("flush error: %s", e)

    def certify(
        self,
        expected_event_ids: Sequence[str] | set[str] = (),
        expected_transition_keys: Sequence[tuple[str, int]] | set[tuple[str, int]] = (),
    ) -> ArtifactIntegrity:
        if self._artifact_integrity != ArtifactIntegrity.COMPLETE:
            return self._artifact_integrity

        exp_events = set(expected_event_ids)
        if exp_events and not exp_events.issubset(self._written_event_ids):
            return ArtifactIntegrity.MISSING_EVENT_ROW

        exp_trans = set(expected_transition_keys)
        if exp_trans and not exp_trans.issubset(self._written_transition_keys):
            return ArtifactIntegrity.MISSING_TRANSITION_ROW

        return ArtifactIntegrity.COMPLETE

    def close(self) -> None:
        self.flush()
        self.telemetry.status = "STOPPED"


def render_records(
    output_dir: Path | str,
    ordered_records: Iterable[CanonicalRecord],
) -> RenderedArtifacts:
    p = Path(output_dir)
    recorder = LiquidityEventRecorder(mode="replay", output_dir=p)
    for r in ordered_records:
        recorder.enqueue(r)
    recorder.flush_replay()

    events_bytes = (p / EVENTS_FILENAME).read_bytes() if (p / EVENTS_FILENAME).exists() else b""
    transitions_bytes = (p / TRANSITIONS_FILENAME).read_bytes() if (p / TRANSITIONS_FILENAME).exists() else b""
    rejected_bytes = (p / REJECTED_FILENAME).read_bytes() if (p / REJECTED_FILENAME).exists() else b""

    return RenderedArtifacts(
        events_bytes=events_bytes,
        transitions_bytes=transitions_bytes,
        rejected_inputs_bytes=rejected_bytes,
    )
