from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Any

from .models import (
    LifecycleTransition,
    LiquidityEventResult,
    LiquiditySweepObservation,
    canonical_hash,
    canonical_json,
)


class IdentityClaimOutcome(str, Enum):
    NEW = "new"
    DUPLICATE_EXISTING = "duplicate_existing"
    IDENTITY_CONFLICT = "identity_conflict"


class IdentityAuthorityUnavailable(RuntimeError):
    pass


class IdentityAlreadyExists(RuntimeError):
    pass


@dataclass(frozen=True)
class PersistedIdentity:
    event_id: str
    identity_hash: str
    identity_payload: MappingProxyType
    observation_payload: MappingProxyType
    source_observation_hash: str
    status: str
    first_detection_time_ms: int


@dataclass(frozen=True)
class IdentityClaimResult:
    outcome: IdentityClaimOutcome
    event_id: str
    existing_identity: PersistedIdentity | None = None
    conflict_reason: str | None = None


@dataclass(frozen=True)
class ClaimedUnresolvedRecovery:
    emitted_transitions: tuple[LifecycleTransition, ...] = ()
    result: LiquidityEventResult | None = None


class SQLiteIdentityAuthority:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self._connection = sqlite3.connect(self.path)
            self._connection.isolation_level = None
            self._initialize_schema()
        except sqlite3.Error as exc:
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable during initialization"
            ) from exc

    def close(self) -> None:
        self._connection.close()

    def claim_observation(
        self, observation: LiquiditySweepObservation
    ) -> IdentityClaimResult:
        identity_payload = _identity_payload(observation)
        identity_hash = canonical_hash(identity_payload)
        observation_payload = observation.to_canonical_dict()
        observation_payload_json = canonical_json(observation_payload)

        try:
            self._begin_immediate()
            row = self._execute(
                """
                SELECT event_id, identity_hash, identity_payload_json,
                       observation_payload_json, source_observation_hash, status,
                       first_detection_time_ms
                FROM event_identity
                WHERE event_id = ?
                """,
                (observation.event_id,),
            ).fetchone()
            if row is None:
                self._execute(
                    """
                    INSERT INTO event_identity (
                        event_id, identity_hash, identity_payload_json,
                        observation_payload_json, source_observation_hash, status,
                        first_detection_time_ms
                    )
                    VALUES (?, ?, ?, ?, ?, 'CLAIMED_UNRESOLVED', ?)
                    """,
                    (
                        observation.event_id,
                        identity_hash,
                        canonical_json(identity_payload),
                        observation_payload_json,
                        observation.source_observation_hash,
                        observation.detection_time_ms,
                    ),
                )
                self._connection.commit()
                return IdentityClaimResult(
                    IdentityClaimOutcome.NEW,
                    observation.event_id,
                )

            existing = _record_from_row(row)
            self._connection.commit()
            if (
                existing.identity_hash == identity_hash
                and existing.source_observation_hash
                == observation.source_observation_hash
            ):
                return IdentityClaimResult(
                    IdentityClaimOutcome.DUPLICATE_EXISTING,
                    observation.event_id,
                    existing_identity=existing,
                )
            return IdentityClaimResult(
                IdentityClaimOutcome.IDENTITY_CONFLICT,
                observation.event_id,
                existing_identity=existing,
                conflict_reason=(
                    "same event_id has different immutable identity or provenance hash"
                ),
            )
        except Exception as exc:
            self._rollback()
            if isinstance(exc, (IdentityAlreadyExists, IdentityAuthorityUnavailable)):
                raise
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable while claiming observation"
            ) from exc

    def claim_transition(self, transition: LifecycleTransition) -> None:
        payload_json = canonical_json(transition.to_canonical_dict())
        try:
            self._begin_immediate()
            self._require_identity_exists(transition.event_id)
            self._execute(
                """
                INSERT INTO event_transition (
                    event_id, transition_sequence, transition_hash,
                    transition_payload_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    transition.event_id,
                    transition.transition_sequence,
                    canonical_hash(transition),
                    payload_json,
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._rollback()
            raise IdentityAlreadyExists(
                "transition identity already exists"
            ) from exc
        except Exception as exc:
            self._rollback()
            if isinstance(exc, IdentityAlreadyExists):
                raise
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable while claiming transition"
            ) from exc

    def claim_result(self, result: LiquidityEventResult) -> None:
        payload_json = canonical_json(result.to_canonical_dict())
        try:
            self._begin_immediate()
            self._require_identity_exists(result.event_id)
            self._execute(
                """
                INSERT INTO event_result (
                    event_id, result_hash, result_payload_json,
                    classification_time_ms
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    result.event_id,
                    canonical_hash(result),
                    payload_json,
                    result.classification_time_ms,
                ),
            )
            status_update = self._execute(
                """
                UPDATE event_identity
                SET status = 'FINALIZED'
                WHERE event_id = ?
                """,
                (result.event_id,),
            )
            if status_update.rowcount != 1:
                raise IdentityAuthorityUnavailable(
                    "result claim missing parent identity"
                )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._rollback()
            raise IdentityAlreadyExists("result identity already exists") from exc
        except Exception as exc:
            self._rollback()
            if isinstance(exc, IdentityAlreadyExists):
                raise
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable while claiming result"
            ) from exc

    def lookup(self, event_id: str) -> PersistedIdentity | None:
        try:
            row = self._execute(
                """
                SELECT event_id, identity_hash, identity_payload_json,
                       observation_payload_json, source_observation_hash, status,
                       first_detection_time_ms
                FROM event_identity
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        except Exception as exc:
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable during lookup"
            ) from exc
        return None if row is None else _record_from_row(row)

    def pending_unresolved(self) -> tuple[PersistedIdentity, ...]:
        try:
            rows = self._execute(
                """
                SELECT event_id, identity_hash, identity_payload_json,
                       observation_payload_json, source_observation_hash, status,
                       first_detection_time_ms
                FROM event_identity
                WHERE status = 'CLAIMED_UNRESOLVED'
                ORDER BY first_detection_time_ms, event_id
                """
            ).fetchall()
        except Exception as exc:
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable while reading pending identities"
            ) from exc
        return tuple(_record_from_row(row) for row in rows)

    def persisted_transition_sequences(self, event_id: str) -> set[int]:
        try:
            rows = self._execute(
                """
                SELECT transition_sequence
                FROM event_transition
                WHERE event_id = ?
                ORDER BY transition_sequence
                """,
                (event_id,),
            ).fetchall()
        except Exception as exc:
            raise IdentityAuthorityUnavailable(
                "identity authority unavailable while reading transitions"
            ) from exc
        return {int(row[0]) for row in rows}

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS event_identity (
                event_id TEXT PRIMARY KEY,
                identity_hash TEXT NOT NULL,
                identity_payload_json TEXT NOT NULL,
                observation_payload_json TEXT NOT NULL,
                source_observation_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('CLAIMED_UNRESOLVED', 'FINALIZED')),
                first_detection_time_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_transition (
                event_id TEXT NOT NULL,
                transition_sequence INTEGER NOT NULL,
                transition_hash TEXT NOT NULL,
                transition_payload_json TEXT NOT NULL,
                PRIMARY KEY (event_id, transition_sequence)
            );

            CREATE TABLE IF NOT EXISTS event_result (
                event_id TEXT PRIMARY KEY,
                result_hash TEXT NOT NULL,
                result_payload_json TEXT NOT NULL,
                classification_time_ms INTEGER NOT NULL
            );
            """
        )

    def _begin_immediate(self) -> None:
        self._execute("BEGIN IMMEDIATE")

    def _execute(self, statement: str, parameters: tuple[Any, ...] = ()):
        return self._connection.execute(statement, parameters)

    def _require_identity_exists(self, event_id: str) -> None:
        row = self._execute(
            """
            SELECT 1
            FROM event_identity
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise IdentityAuthorityUnavailable("missing parent identity claim")

    def _rollback(self) -> None:
        try:
            self._connection.rollback()
        except sqlite3.Error:
            pass


def recover_claimed_unresolved(
    *,
    authority: SQLiteIdentityAuthority,
    pending_identity: PersistedIdentity,
    retained_history: Any,
) -> ClaimedUnresolvedRecovery:
    if _history_get(retained_history, "complete", False):
        persisted = authority.persisted_transition_sequences(pending_identity.event_id)
        emitted = []
        for transition in _history_get(retained_history, "transitions", ()):
            if transition.transition_sequence in persisted:
                continue
            authority.claim_transition(transition)
            emitted.append(transition)
        return ClaimedUnresolvedRecovery(emitted_transitions=tuple(emitted))

    result = _history_get(retained_history, "result")
    if result is None:
        raise IdentityAuthorityUnavailable(
            "missing retained history did not provide a finalization result"
        )
    authority.claim_result(result)
    return ClaimedUnresolvedRecovery(result=result)


def _identity_payload(observation: LiquiditySweepObservation) -> dict[str, Any]:
    payload = observation.to_canonical_dict()
    return {
        "event_time_ms": payload["event_time_ms"],
        "liquidity_side": payload["liquidity_side"],
        "source": payload["source"],
        "source_event_id": payload["source_event_id"],
        "swept_level": payload["swept_level"],
        "symbol": payload["symbol"],
    }


def _record_from_row(row: sqlite3.Row | tuple[Any, ...]) -> PersistedIdentity:
    return PersistedIdentity(
        event_id=row[0],
        identity_hash=row[1],
        identity_payload=_freeze_json(json.loads(row[2])),
        observation_payload=_freeze_json(json.loads(row[3])),
        source_observation_hash=row[4],
        status=row[5],
        first_detection_time_ms=row[6],
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _history_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
