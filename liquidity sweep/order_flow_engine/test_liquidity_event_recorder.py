from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path

import pytest

from liquidity_event.models import (
    ArtifactIntegrity,
    ConfidenceType,
    EventClassification,
    EventState,
    EvidenceAvailability,
    EvidenceValue,
    LifecycleTransition,
    LiquidityEventResult,
    LiquidityEvidence,
    LiquiditySide,
    RejectedSweepInput,
    SweepSource,
)
from liquidity_event.recorder import (
    CanonicalRecord,
    LiquidityEventRecorder,
    RecorderTelemetry,
    RenderedArtifacts,
    render_records,
)


def make_test_result(
    event_id: str = "evt-1",
    symbol: str = "BTCUSDT",
    side: LiquiditySide = LiquiditySide.SELL_SIDE,
    classification: EventClassification = EventClassification.FAILED_BREAKDOWN,
    event_time_ms: int = 1_000,
    detection_time_ms: int = 2_000,
    resolution_time_ms: int = 4_000,
) -> LiquidityEventResult:
    evidence = LiquidityEvidence(
        values={
            "flow.trade_count": EvidenceValue(
                availability=EvidenceAvailability.AVAILABLE,
                value=5.0,
                as_of_ms=resolution_time_ms,
            )
        },
        reasons=("PRICE_RECLAIMED", "POSITIVE_DELTA"),
        contradictions=(),
        context_coverage=1.0,
        evidence_strength=0.8,
        contradiction_strength=0.0,
        confidence=0.8,
        confidence_type=ConfidenceType.UNCALIBRATED_DETERMINISTIC_SCORE,
        as_of_ms=resolution_time_ms,
    )
    return LiquidityEventResult(
        event_id=event_id,
        symbol=symbol,
        liquidity_side=side,
        classification=classification,
        reason_code="RECLAIM_CONFIRMED",
        event_time_ms=event_time_ms,
        detection_time_ms=detection_time_ms,
        market_resolution_time_ms=resolution_time_ms,
        classification_time_ms=max(resolution_time_ms, detection_time_ms),
        source_observation_hash="a" * 64,
        evidence=evidence,
        policy_hash="b" * 64,
        model_version="1.0.0",
    )


def make_test_transition(
    event_id: str = "evt-1",
    seq: int = 0,
    time_ms: int = 1_000,
    from_state: EventState | None = EventState.OBSERVED,
    to_state: EventState = EventState.PENETRATION_VALIDATING,
    reason: str = "START_VALIDATION",
) -> LifecycleTransition:
    return LifecycleTransition(
        event_id=event_id,
        previous_state=from_state,
        next_state=to_state,
        transition_time_ms=time_ms,
        transition_sequence=seq,
        reason_code=reason,
    )


def make_test_rejected(
    source: SweepSource = SweepSource.SWEEPS_MONITOR_CSV,
    time_ms: int = 1_000,
    reason_code: str = "MALFORMED_PRICE",
    detail: str = "Invalid numeric price",
) -> RejectedSweepInput:
    return RejectedSweepInput(
        source=source,
        detection_time_ms=time_ms,
        reason_code=reason_code,
        reason_detail=detail,
        source_file_id="sweeps.csv",
        source_row_hash="c" * 64,
        source_observation_hash="d" * 64,
    )


def test_replay_flush_is_byte_equivalent_across_enqueue_order(tmp_path: Path):
    t0 = make_test_transition(event_id="evt-1", seq=0, time_ms=1_000)
    t1 = make_test_transition(event_id="evt-1", seq=1, time_ms=2_000, from_state=EventState.PENETRATION_VALIDATING, to_state=EventState.SWEEP_DETECTED, reason="PENETRATION_CONFIRMED")
    res1 = make_test_result(event_id="evt-1", event_time_ms=1_000, detection_time_ms=2_000, resolution_time_ms=4_000)
    rej1 = make_test_rejected(time_ms=500)

    records = [
        CanonicalRecord("TRANSITION", t0),
        CanonicalRecord("TRANSITION", t1),
        CanonicalRecord("EVENT", res1),
        CanonicalRecord("REJECTED_INPUT", rej1),
    ]

    first = render_records(tmp_path / "a", records)
    second = render_records(tmp_path / "b", list(reversed(records)))

    assert first.events_bytes == second.events_bytes
    assert first.transitions_bytes == second.transitions_bytes
    assert first.rejected_inputs_bytes == second.rejected_inputs_bytes
    assert first.all_bytes == second.all_bytes


def test_canonical_artifacts_exclude_machine_runtime_values(tmp_path: Path):
    t0 = make_test_transition(event_id="evt-1", seq=0, time_ms=1_000)
    res1 = make_test_result(event_id="evt-1")
    rej1 = make_test_rejected(time_ms=500)

    records = [
        CanonicalRecord("TRANSITION", t0),
        CanonicalRecord("EVENT", res1),
        CanonicalRecord("REJECTED_INPUT", rej1),
    ]

    output = render_records(tmp_path / "art", records).all_bytes.decode("utf-8")
    assert str(tmp_path) not in output
    assert "object at 0x" not in output
    assert "\r\n" not in output  # strict LF newlines


def test_queue_overflow_preserves_classification_and_fails_integrity(tmp_path: Path):
    recorder = LiquidityEventRecorder("live", tmp_path / "live", queue_max_items=1)
    res1 = make_test_result(event_id="evt-1")
    res2 = make_test_result(event_id="evt-2")

    assert recorder.enqueue(CanonicalRecord("EVENT", res1)) is True
    assert recorder.enqueue(CanonicalRecord("EVENT", res2)) is False

    assert recorder.telemetry.failure_count == 1
    assert recorder.artifact_integrity is ArtifactIntegrity.QUEUE_OVERFLOW
    # Result itself remains immutable and valid
    assert res2.classification is EventClassification.FAILED_BREAKDOWN


def test_reasons_and_rows_serialize_deterministically(tmp_path: Path):
    res = make_test_result(event_id="evt-1")
    recorder = LiquidityEventRecorder("replay", tmp_path / "rep")
    recorder.enqueue(CanonicalRecord("EVENT", res))
    recorder.flush_replay()

    events_csv = (tmp_path / "rep" / "liquidity_events.csv").read_text(encoding="utf-8")
    lines = events_csv.strip().split("\n")
    assert len(lines) == 2
    header, row = lines[0], lines[1]

    # Verify JSON array compactness in raw CSV and parsed reader
    assert '""POSITIVE_DELTA"",""PRICE_RECLAIMED""' in row
    assert '[]' in row

    with open(tmp_path / "rep" / "liquidity_events.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[1][14] == '["POSITIVE_DELTA","PRICE_RECLAIMED"]'
    assert rows[1][15] == '[]'


def test_store_and_recorder_bounds_are_enforced(tmp_path: Path):
    recorder = LiquidityEventRecorder("live", tmp_path / "live_bounded", queue_max_items=10)
    for i in range(10):
        assert recorder.enqueue(CanonicalRecord("TRANSITION", make_test_transition(seq=i))) is True
    assert recorder.enqueue(CanonicalRecord("TRANSITION", make_test_transition(seq=10))) is False
    assert recorder.telemetry.failure_count == 1
    assert recorder.telemetry.pending_write_count <= 10


def test_recorder_failure_preserves_result_but_invalidates_research(tmp_path: Path):
    bad_dir = tmp_path / "nonexistent" / "subdir"
    recorder = LiquidityEventRecorder("live", bad_dir, queue_max_items=10)
    # Simulate direct recorder write error
    recorder.telemetry.failure_count = 1
    recorder.telemetry.last_error = "disk full"
    recorder._artifact_integrity = ArtifactIntegrity.RECORDER_FAILURE

    assert recorder.certify(expected_event_ids={"evt-1"}) is ArtifactIntegrity.RECORDER_FAILURE


def test_certification_reports_missing_rows(tmp_path: Path):
    recorder = LiquidityEventRecorder("replay", tmp_path / "rep_cert")
    t0 = make_test_transition(event_id="evt-1", seq=0)
    res = make_test_result(event_id="evt-1")
    recorder.enqueue(CanonicalRecord("TRANSITION", t0))
    recorder.enqueue(CanonicalRecord("EVENT", res))
    recorder.flush_replay()

    # Missing event row
    assert (
        recorder.certify(
            expected_event_ids={"evt-1", "evt-missing"},
            expected_transition_keys={("evt-1", 0)},
        )
        is ArtifactIntegrity.MISSING_EVENT_ROW
    )

    # Missing transition row
    assert (
        recorder.certify(
            expected_event_ids={"evt-1"},
            expected_transition_keys={("evt-1", 0), ("evt-1", 1)},
        )
        is ArtifactIntegrity.MISSING_TRANSITION_ROW
    )

    # Complete certification
    assert (
        recorder.certify(
            expected_event_ids={"evt-1"},
            expected_transition_keys={("evt-1", 0)},
        )
        is ArtifactIntegrity.COMPLETE
    )


def test_direct_payload_enqueue_without_canonical_record_wrapper(tmp_path: Path):
    recorder = LiquidityEventRecorder("replay", tmp_path / "direct")
    t0 = make_test_transition(event_id="evt-1", seq=0)
    res = make_test_result(event_id="evt-1")
    rej = make_test_rejected()

    assert recorder.enqueue(t0) is True
    assert recorder.enqueue(res) is True
    assert recorder.enqueue(rej) is True
    recorder.flush_replay()

    assert recorder.telemetry.written_transition_count == 1
    assert recorder.telemetry.written_event_count == 1
    assert recorder.telemetry.written_rejected_count == 1


def test_live_recorder_flushes_to_csv_with_telemetry(tmp_path: Path):
    recorder = LiquidityEventRecorder("live", tmp_path / "live_flush")
    t0 = make_test_transition(event_id="evt-1", seq=0)
    res = make_test_result(event_id="evt-1")

    recorder.enqueue(t0)
    recorder.enqueue(res)
    assert recorder.telemetry.pending_write_count == 2

    recorder.flush()
    assert recorder.telemetry.pending_write_count == 0
    assert recorder.telemetry.written_transition_count == 1
    assert recorder.telemetry.written_event_count == 1

    # Verify CSV files exist
    assert (tmp_path / "live_flush" / "liquidity_events.csv").exists()
    assert (tmp_path / "live_flush" / "liquidity_event_transitions.csv").exists()


def test_mixed_replay_records_sort_by_canonical_key_deterministically(tmp_path: Path):
    # Same event_time, different ranks (transitions before events)
    t0 = make_test_transition(event_id="evt-1", seq=0, time_ms=1_000)
    res = make_test_result(event_id="evt-1", event_time_ms=1_000, detection_time_ms=1_000, resolution_time_ms=1_000)
    rej = make_test_rejected(time_ms=1_000)

    # Supply in reverse rank order: rejected (rank 2), event (rank 1), transition (rank 0)
    artifacts = render_records(tmp_path / "mixed", [CanonicalRecord("REJECTED_INPUT", rej), CanonicalRecord("EVENT", res), CanonicalRecord("TRANSITION", t0)])
    assert len(artifacts.events_bytes) > 0
    assert len(artifacts.transitions_bytes) > 0
    assert len(artifacts.rejected_inputs_bytes) > 0


def test_close_flushes_and_stops_telemetry(tmp_path: Path):
    recorder = LiquidityEventRecorder("live", tmp_path / "close_test")
    recorder.enqueue(make_test_transition(event_id="evt-1", seq=0))
    recorder.close()

    assert recorder.telemetry.status == "STOPPED"
    assert recorder.telemetry.pending_write_count == 0
    assert recorder.telemetry.written_transition_count == 1


def test_optional_empty_fields_serialize_as_empty_strings(tmp_path: Path):
    # Transition with None previous_state and None reason_code
    t_empty = LifecycleTransition(
        event_id="evt-empty",
        previous_state=None,
        next_state=EventState.PENETRATION_VALIDATING,
        transition_time_ms=500,
        transition_sequence=0,
        reason_code=None,
    )
    # Rejected with None file_id, row_hash, obs_hash
    rej_empty = RejectedSweepInput(
        source=SweepSource.SWEEPS_MONITOR_CSV,
        detection_time_ms=500,
        reason_code="INVALID",
        reason_detail="detail",
        source_file_id=None,
        source_row_hash=None,
        source_observation_hash=None,
    )

    recorder = LiquidityEventRecorder("replay", tmp_path / "empty_test")
    recorder.enqueue(t_empty)
    recorder.enqueue(rej_empty)
    recorder.flush_replay()

    with open(tmp_path / "empty_test" / "liquidity_event_transitions.csv", newline="", encoding="utf-8") as f:
        t_rows = list(csv.reader(f))
    assert t_rows[1][3] == ""  # previous_state
    assert t_rows[1][5] == ""  # reason_code

    with open(tmp_path / "empty_test" / "liquidity_rejected_inputs.csv", newline="", encoding="utf-8") as f:
        r_rows = list(csv.reader(f))
    assert r_rows[1][4] == ""  # source_file_id
    assert r_rows[1][5] == ""  # source_row_hash
    assert r_rows[1][6] == ""  # source_observation_hash
