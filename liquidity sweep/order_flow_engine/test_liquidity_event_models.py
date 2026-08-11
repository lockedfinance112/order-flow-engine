from dataclasses import replace
from decimal import Decimal

import pytest


def _required_api():
    try:
        from liquidity_event import (
            EvidenceAvailability,
            EvidenceValue,
            LiquidityClassificationPolicy,
            normalize_price,
        )
    except ImportError:
        pytest.fail("required Phase 1C.1 API is not implemented")
    return (
        EvidenceAvailability,
        EvidenceValue,
        LiquidityClassificationPolicy,
        normalize_price,
    )


def test_v1_policy_hash_is_stable_and_validates_retention():
    _, _, LiquidityClassificationPolicy, _ = _required_api()
    first = LiquidityClassificationPolicy()
    second = LiquidityClassificationPolicy()
    assert first.policy_hash == second.policy_hash
    assert len(first.policy_hash) == 64
    assert first.market_buffer_retention_ms == 180_000
    with pytest.raises(ValueError, match="retention"):
        replace(first, market_buffer_retention_ms=179_999).validate()


def test_price_normalization_is_fixed_half_even_eight_places():
    _, _, _, normalize_price = _required_api()
    assert normalize_price(Decimal("117250.123456785"), 8) == "117250.12345678"
    assert normalize_price(Decimal("117250.123456795"), 8) == "117250.12345680"


def test_models_keep_missing_optional_evidence_distinct_from_zero():
    EvidenceAvailability, EvidenceValue, _, _ = _required_api()
    missing = EvidenceValue(EvidenceAvailability.UNAVAILABLE, None, None)
    zero = EvidenceValue(EvidenceAvailability.AVAILABLE, 0.0, 1000)
    assert missing != zero


def _review_api():
    try:
        from liquidity_event import (
            AggressorSide,
            DepthObservation,
            EventClassification,
            EvidenceAvailability,
            EvidenceValue,
            LiquidityEventResult,
            LiquidityEvidence,
            LiquiditySide,
            MarketTrade,
            canonical_json,
        )
    except ImportError:
        pytest.fail("required Phase 1C.1 API is not implemented")
    return (
        AggressorSide,
        DepthObservation,
        EventClassification,
        EvidenceAvailability,
        EvidenceValue,
        LiquidityEventResult,
        LiquidityEvidence,
        LiquiditySide,
        MarketTrade,
        canonical_json,
    )


def test_canonical_keys_sort_mixed_sequence_id_types_deterministically():
    (
        AggressorSide,
        DepthObservation,
        _,
        _,
        _,
        _,
        _,
        _,
        MarketTrade,
        _,
    ) = _review_api()
    trade = MarketTrade(
        "BTCUSDT", Decimal("100"), Decimal("1"), AggressorSide.BUY, 1_000, 7
    )
    depth = DepthObservation("BTCUSDT", (), (), 1_000, "7")

    assert sorted((depth, trade), key=lambda observation: observation.canonical_key) == [
        trade,
        depth,
    ]


def test_event_result_serializes_resolution_time_and_source_provenance():
    (
        _,
        _,
        EventClassification,
        _,
        _,
        LiquidityEventResult,
        LiquidityEvidence,
        LiquiditySide,
        _,
        _,
    ) = _review_api()
    result = LiquidityEventResult(
        event_id="event-1",
        symbol="BTCUSDT",
        liquidity_side=LiquiditySide.SELL_SIDE,
        classification=EventClassification.FAILED_BREAKDOWN,
        reason_code=None,
        event_time_ms=1_000,
        detection_time_ms=2_000,
        market_resolution_time_ms=5_000,
        classification_time_ms=6_000,
        source_observation_hash="a" * 64,
        evidence=LiquidityEvidence(),
        policy_hash="b" * 64,
        model_version="1C.1-v1",
    )

    assert result.to_canonical_dict()["market_resolution_time_ms"] == 5_000
    assert result.to_canonical_dict()["source_observation_hash"] == "a" * 64


def test_evidence_copies_values_and_canonicalizes_reason_order():
    (
        _,
        _,
        _,
        EvidenceAvailability,
        EvidenceValue,
        _,
        LiquidityEvidence,
        _,
        _,
        canonical_json,
    ) = _review_api()
    supplied_values = {
        "delta": EvidenceValue(EvidenceAvailability.AVAILABLE, 1.0, 1_000)
    }
    evidence = LiquidityEvidence(
        values=supplied_values,
        reasons=("SECOND", "FIRST"),
        contradictions=("Z", "A"),
    )
    equivalent = LiquidityEvidence(
        values=supplied_values,
        reasons=("FIRST", "SECOND"),
        contradictions=("A", "Z"),
    )

    supplied_values["late"] = EvidenceValue(
        EvidenceAvailability.AVAILABLE, 2.0, 2_000
    )

    assert "late" not in evidence.values
    with pytest.raises(TypeError):
        evidence.values["late"] = supplied_values["late"]
    assert canonical_json(evidence.to_canonical_dict()) == canonical_json(
        equivalent.to_canonical_dict()
    )
