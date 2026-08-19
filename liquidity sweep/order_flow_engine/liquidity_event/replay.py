from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
import gzip
from hashlib import sha256
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

from .classifier import PriceOutcomeTracker
from .engine import LiquidityEventEngine
from .event_store import LiquidityEventStore
from .evidence import LiquidityEvidenceBuilder
from .identity_authority import SQLiteIdentityAuthority
from .models import (
    AggressorSide,
    ArtifactIntegrity,
    DepthObservation,
    EventClassification,
    EventState,
    LifecycleTransition,
    LiquidityEventResult,
    LiquidityEvidence,
    LiquiditySide,
    LiquiditySweepObservation,
    MarketTrade,
    RejectedSweepInput,
    TradeCoverage,
    TradeCoverageProvider,
    _canonical_sequence_component,
    canonical_hash,
)
from .policy import LiquidityClassificationPolicy
from .recorder import (
    CanonicalRecord,
    EVENTS_FILENAME,
    LiquidityEventRecorder,
    REJECTED_FILENAME,
    TRANSITIONS_FILENAME,
)
from .sweep_adapter import SweepsMonitorAdapter

logger = logging.getLogger(__name__)


class ReplayTradeCoverageProvider:
    def __init__(self, coverage: TradeCoverage | None = None) -> None:
        self._cov = coverage or TradeCoverage.safe_zero_activity()

    def coverage(self, symbol: str, start_ms: int, end_ms: int) -> TradeCoverage:
        return self._cov


@dataclass(frozen=True)
class ReplayInput:
    source_type: str  # "TRADE" | "DEPTH" | "SWEEP"
    trade: MarketTrade | None = None
    depth: DepthObservation | None = None
    sweep: LiquiditySweepObservation | None = None
    sweep_callback: dict[str, Any] | None = None

    @classmethod
    def from_trade(cls, trade: MarketTrade) -> "ReplayInput":
        return cls(source_type="TRADE", trade=trade)

    @classmethod
    def from_depth(cls, depth: DepthObservation) -> "ReplayInput":
        return cls(source_type="DEPTH", depth=depth)

    @classmethod
    def from_sweep(cls, sweep: LiquiditySweepObservation) -> "ReplayInput":
        return cls(source_type="SWEEP", sweep=sweep)

    @classmethod
    def from_sweep_callback(cls, callback: dict[str, Any]) -> "ReplayInput":
        return cls(source_type="SWEEP", sweep_callback=callback)

    @property
    def processing_time_ms(self) -> int:
        if self.source_type == "TRADE" and self.trade is not None:
            return self.trade.exchange_time_ms
        if self.source_type == "DEPTH" and self.depth is not None:
            return self.depth.exchange_time_ms
        if self.source_type == "SWEEP":
            if self.sweep is not None:
                return self.sweep.detection_time_ms
            if self.sweep_callback is not None:
                return int(
                    self.sweep_callback.get("detection_time_ms")
                    or self.sweep_callback.get("timestamp_ms")
                    or self.sweep_callback.get("event_time_ms")
                    or 0
                )
        raise ValueError(f"Invalid replay input state: {self}")

    @property
    def exchange_sequence(self) -> tuple[int, int | str]:
        if self.source_type == "TRADE" and self.trade is not None:
            return _canonical_sequence_component(self.trade.sequence_id)
        if self.source_type == "DEPTH" and self.depth is not None:
            return _canonical_sequence_component(self.depth.sequence_id)
        return (0, 0)

    @property
    def source_rank(self) -> int:
        if self.source_type == "TRADE":
            return 0
        if self.source_type == "DEPTH":
            return 1
        if self.source_type == "SWEEP":
            return 2
        raise ValueError(f"Unknown source_type: {self.source_type}")

    @property
    def content_hash(self) -> str:
        if self.source_type == "TRADE" and self.trade is not None:
            return self.trade.content_hash
        if self.source_type == "DEPTH" and self.depth is not None:
            return self.depth.content_hash
        if self.source_type == "SWEEP":
            if self.sweep is not None:
                return self.sweep.source_observation_hash
            if self.sweep_callback is not None:
                return canonical_hash(self.sweep_callback)
        raise ValueError(f"Invalid replay input content: {self}")

    @property
    def processing_key(self) -> tuple[int, tuple[int, int | str], int, str]:
        return (
            self.processing_time_ms,
            self.exchange_sequence,
            self.source_rank,
            self.content_hash,
        )


@dataclass(frozen=True)
class LiquidityReplayResult:
    artifact_integrity: ArtifactIntegrity
    event_count: int
    transition_count: int
    missing_event_row_count: int
    missing_transition_row_count: int
    recorder_failure_count: int
    queue_overflow_count: int
    policy_hash: str
    results: tuple[LiquidityEventResult, ...]
    transitions: tuple[LifecycleTransition, ...]
    artifact_hashes: Mapping[str, str]

    @property
    def exit_code(self) -> int:
        return 0 if self.artifact_integrity is ArtifactIntegrity.COMPLETE else 1


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class LiquidityReplayRunner:
    def __init__(
        self,
        inputs: Iterable[ReplayInput],
        output_dir: Path | str,
        policy: LiquidityClassificationPolicy | None = None,
    ):
        self.inputs = list(inputs)
        self.output_dir = Path(output_dir)
        self.policy = policy or LiquidityClassificationPolicy()

    @classmethod
    def from_recordings(
        cls,
        market_paths: Iterable[Path | str],
        sweep_callbacks: Iterable[dict[str, Any]],
        output_dir: Path | str,
        policy: LiquidityClassificationPolicy | None = None,
    ) -> "LiquidityReplayRunner":
        inputs: list[ReplayInput] = []

        for p_str in market_paths:
            path = Path(p_str)
            opener = gzip.open if path.suffix == ".gz" or str(path).endswith(".jsonl.gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    msg = json.loads(line_str)
                    data = msg.get("data", msg)
                    event_type = data.get("e")

                    if event_type == "aggTrade":
                        symbol = data.get("s", "").upper()
                        price = float(data["p"])
                        size = float(data["q"])
                        is_buyer_maker = data.get("m", False)
                        side = AggressorSide.SELL if is_buyer_maker else AggressorSide.BUY
                        trade_time = int(data.get("T", data.get("E", 0)))
                        seq_id = data.get("a", 0)
                        trade = MarketTrade(
                            symbol=symbol,
                            price=price,
                            quantity=size,
                            aggressor_side=side,
                            exchange_time_ms=trade_time,
                            sequence_id=seq_id,
                        )
                        inputs.append(ReplayInput.from_trade(trade))
                    elif event_type == "depthUpdate":
                        symbol = data.get("s", "").upper()
                        bids = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in data.get("b", []))
                        asks = tuple((Decimal(str(p)), Decimal(str(q))) for p, q in data.get("a", []))
                        depth_time = int(data.get("E", data.get("T", 0)))
                        seq_id = data.get("u", 0)
                        depth = DepthObservation(
                            symbol=symbol,
                            bids=bids,
                            asks=asks,
                            exchange_time_ms=depth_time,
                            sequence_id=seq_id,
                        )
                        inputs.append(ReplayInput.from_depth(depth))

        for cb in sweep_callbacks:
            inputs.append(ReplayInput.from_sweep_callback(cb))

        return cls(inputs=inputs, output_dir=output_dir, policy=policy)

    def run(self) -> LiquidityReplayResult:
        # Stable canonical sort
        sorted_inputs = sorted(self.inputs, key=lambda x: x.processing_key)

        # Fresh isolated SQLite authority per replay run
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "replay_authority.sqlite3"
        authority = SQLiteIdentityAuthority(db_path)

        try:
            store = LiquidityEventStore(self.policy)
            recorder = LiquidityEventRecorder(
                mode="replay",
                output_dir=self.output_dir,
                queue_max_items=self.policy.recorder_queue_max_items,
            )
            adapter = SweepsMonitorAdapter(self.policy)

            emitted_results: list[LiquidityEventResult] = []
            emitted_transitions: list[LifecycleTransition] = []

            def on_transition(t: LifecycleTransition) -> None:
                emitted_transitions.append(t)
                recorder.enqueue(t)

            def on_result(r: LiquidityEventResult) -> None:
                emitted_results.append(r)
                recorder.enqueue(r)

            engine = LiquidityEventEngine(
                policy=self.policy,
                store=store,
                authority=authority,
                on_transition=on_transition,
                on_result=on_result,
            )

            safe_trade_cov = TradeCoverage(
                feed_safe=True,
                known_gap=False,
                buffer_overflow=False,
                unresolved_sequence=False,
                interval_retained=True,
            )

            # Process all canonical inputs
            for inp in sorted_inputs:
                if inp.source_type == "TRADE" and inp.trade is not None:
                    engine.on_trade(inp.trade, safe_trade_cov)
                elif inp.source_type == "DEPTH" and inp.depth is not None:
                    engine.on_depth(inp.depth)
                elif inp.source_type == "SWEEP":
                    obs: LiquiditySweepObservation | None = None
                    if inp.sweep is not None:
                        obs = inp.sweep
                    elif inp.sweep_callback is not None:
                        det_time = int(
                            inp.sweep_callback.get("detection_time_ms")
                            or inp.sweep_callback.get("timestamp_ms")
                            or 0
                        )
                        if det_time == 0 and "timestamp" in inp.sweep_callback:
                            try:
                                from .sweep_adapter import parse_utc_ms
                                det_time = parse_utc_ms(inp.sweep_callback["timestamp"])
                            except Exception:
                                det_time = 0
                        adapted = adapter.adapt(inp.sweep_callback, det_time)
                        if adapted.observation is not None:
                            obs = adapted.observation
                        elif adapted.rejected is not None:
                            recorder.enqueue(adapted.rejected)

                    if obs is not None:
                        engine.on_sweep(obs, ReplayTradeCoverageProvider(TradeCoverage.safe_zero_activity()))

            # Terminal Advance (Symbol-Local) via production engine API
            for symbol in list(engine._runtimes.keys()):
                runtime = engine.symbol_runtime(symbol)
                if runtime.active_trackers:
                    last_time = runtime.max_seen_exchange_time_ms or 0
                    engine.advance_time(
                        symbol=symbol,
                        as_of_exchange_time_ms=last_time,
                        coverage=safe_trade_cov,
                        terminal_input=True,
                    )

            # Flush replay recorder
            recorder.flush_replay()

            # Certify
            expected_event_ids = {r.event_id for r in emitted_results}
            expected_transition_keys = {(t.event_id, t.transition_sequence) for t in emitted_transitions}
            integrity = recorder.certify(
                expected_event_ids=expected_event_ids,
                expected_transition_keys=expected_transition_keys,
            )

            # Calculate artifact hashes
            artifact_hashes: dict[str, str] = {}
            for fname in (EVENTS_FILENAME, TRANSITIONS_FILENAME, REJECTED_FILENAME):
                p = self.output_dir / fname
                if p.exists():
                    artifact_hashes[fname] = _file_sha256(p)

            missing_events = len(expected_event_ids - set(recorder._written_event_ids))
            missing_trans = len(expected_transition_keys - set(recorder._written_transition_keys))

            return LiquidityReplayResult(
                artifact_integrity=integrity,
                event_count=len(emitted_results),
                transition_count=len(emitted_transitions),
                missing_event_row_count=missing_events,
                missing_transition_row_count=missing_trans,
                recorder_failure_count=recorder.telemetry.failure_count,
                queue_overflow_count=1 if integrity is ArtifactIntegrity.QUEUE_OVERFLOW else 0,
                policy_hash=self.policy.policy_hash,
                results=tuple(emitted_results),
                transitions=tuple(emitted_transitions),
                artifact_hashes=artifact_hashes,
            )
        finally:
            authority.close()
            recorder.close()
            try:
                temp_dir.cleanup()
            except Exception:
                pass
