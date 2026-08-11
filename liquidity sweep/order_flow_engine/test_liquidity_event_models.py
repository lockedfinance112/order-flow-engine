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


@pytest.mark.parametrize("value, as_of_ms", [(0.0, None), (None, 1_000)])
def test_unavailable_evidence_rejects_value_or_timestamp(value, as_of_ms):
    EvidenceAvailability, EvidenceValue, _, _ = _required_api()

    with pytest.raises(ValueError, match="unavailable evidence"):
        EvidenceValue(EvidenceAvailability.UNAVAILABLE, value, as_of_ms)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_available_evidence_rejects_non_finite_values(value):
    EvidenceAvailability, EvidenceValue, _, _ = _required_api()

    with pytest.raises(ValueError, match="finite"):
        EvidenceValue(EvidenceAvailability.AVAILABLE, value, 1_000)


@pytest.mark.parametrize("value, as_of_ms", [(None, None), (1.0, 1_000)])
def test_unsafe_evidence_requires_timestamp_without_numeric_value(value, as_of_ms):
    EvidenceAvailability, EvidenceValue, _, _ = _required_api()

    with pytest.raises(ValueError, match="unsafe evidence"):
        EvidenceValue(EvidenceAvailability.UNSAFE, value, as_of_ms)


@pytest.mark.parametrize("value, as_of_ms", [(None, 1_000), (1.0, None), (1.0, -1)])
def test_available_evidence_requires_value_and_valid_timestamp(value, as_of_ms):
    EvidenceAvailability, EvidenceValue, _, _ = _required_api()

    with pytest.raises(ValueError, match="available evidence"):
        EvidenceValue(EvidenceAvailability.AVAILABLE, value, as_of_ms)


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


def test_depth_observation_snapshots_caller_owned_levels():
    (
        _,
        DepthObservation,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _review_api()
    bids = [[Decimal("100"), Decimal("1")]]
    asks = [[Decimal("101"), Decimal("2")]]
    observation = DepthObservation("BTCUSDT", bids, asks, 1_000, 7)
    expected_dict = observation.to_canonical_dict()
    expected_hash = observation.content_hash

    bids[0][0] = Decimal("90")
    bids.append([Decimal("89"), Decimal("3")])
    asks[0][1] = Decimal("20")
    asks.clear()

    assert observation.bids == ((Decimal("100"), Decimal("1")),)
    assert observation.asks == ((Decimal("101"), Decimal("2")),)
    assert observation.to_canonical_dict() == expected_dict
    assert observation.content_hash == expected_hash


def _event_result(**overrides):
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
    values = {
        "event_id": "event-1",
        "symbol": "BTCUSDT",
        "liquidity_side": LiquiditySide.SELL_SIDE,
        "classification": EventClassification.FAILED_BREAKDOWN,
        "reason_code": None,
        "event_time_ms": 1_000,
        "detection_time_ms": 2_000,
        "market_resolution_time_ms": 5_000,
        "classification_time_ms": 5_000,
        "source_observation_hash": "a" * 64,
        "evidence": LiquidityEvidence(),
        "policy_hash": "b" * 64,
        "model_version": "1C.1-v1",
    }
    values.update(overrides)
    return LiquidityEventResult(**values)


def test_event_result_serializes_resolution_time_and_source_provenance():
    result = _event_result()

    assert result.to_canonical_dict()["market_resolution_time_ms"] == 5_000
    assert result.to_canonical_dict()["source_observation_hash"] == "a" * 64


def test_event_result_classification_time_uses_later_detection_time():
    result = _event_result(detection_time_ms=9_000, classification_time_ms=9_000)

    assert result.classification_time_ms == 9_000


@pytest.mark.parametrize("classification_time_ms", [4_999, 5_001])
def test_event_result_rejects_incorrect_classification_time(classification_time_ms):
    with pytest.raises(ValueError, match="classification time"):
        _event_result(classification_time_ms=classification_time_ms)


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


def test_canonical_json_serializes_frozen_evidence_directly():
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
    evidence = LiquidityEvidence(
        values={
            "delta": EvidenceValue(
                EvidenceAvailability.AVAILABLE, 1.0, 1_000
            )
        },
        reasons=("SECOND", "FIRST"),
    )

    assert canonical_json(evidence) == canonical_json(evidence.to_canonical_dict())


def test_canonical_hash_serializes_event_result_directly():
    try:
        from liquidity_event import canonical_hash
    except ImportError:
        pytest.fail("required Phase 1C.1 API is not implemented")
    result = _event_result()

    assert canonical_hash(result) == canonical_hash(result.to_canonical_dict())
