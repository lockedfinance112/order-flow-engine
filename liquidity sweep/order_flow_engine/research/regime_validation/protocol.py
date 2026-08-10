import json
import hashlib
from typing import Dict, Any

DEFAULT_PROTOCOL = {
    "schema_version": "1.0",
    "baseline_commit": "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c",
    "regime_model_version": "regime-v1",
    "regime_feature_version": "regime-features-v1",
    "classifier_config_hash": "default",
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    "dataset_date_ranges": {
        "start": "2026-06-01",
        "end": "2026-08-01"
    },
    "warmup_period_days": 22,
    "development_period_pct": 0.60,
    "validation_period_pct": 0.20,
    "holdout_period_pct": 0.20,
    "primary_outcome_horizon_min": 15,
    "secondary_outcome_horizon_min": 60,
    "reference_label_thresholds": {
        "directional_atr": 1.0,
        "efficiency": 0.35,
        "range_atr": 0.50,
        "range_efficiency": 0.25
    },
    "bootstrap_method": "utc_day_block",
    "bootstrap_seed": 1729,
    "bootstrap_repetitions": 1000,
    "minimum_sample_size": 30,
    "cost_scenarios_bps": [0, 2, 5, 10],
    "primary_cost_scenario_bps": 5,
    "signal_regime_join_rules": {
        "max_age_ms": 90000,
        "mode": "as_of_backward"
    },
    "allowed_signal_sources": ["RECORDED_DECISION_TRANSITION", "LEGACY_SIGNAL_LOG"],
    "dataset_quality_requirements": {
        "allow_gaps": False,
        "check_monotonic": True
    }
}

def get_protocol_hash(protocol: Dict[str, Any]) -> str:
    """Computes SHA256 of sorted canonical json protocol."""
    serialized = json.dumps(protocol, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def get_config_hash(config: Dict[str, Any]) -> str:
    """Computes SHA256 of sorted canonical json config."""
    serialized = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def load_or_create_protocol(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        with open(filepath, 'w') as f:
            json.dump(DEFAULT_PROTOCOL, f, indent=4)
        return DEFAULT_PROTOCOL

def verify_protocol_hash(protocol_data: Dict[str, Any], expected_hash: str) -> bool:
    """Verifies that the protocol data has not been mutated."""
    actual_hash = get_protocol_hash(protocol_data)
    return actual_hash == expected_hash
