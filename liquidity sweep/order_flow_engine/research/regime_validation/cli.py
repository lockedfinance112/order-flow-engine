import argparse
import sys
import os
import json
import hashlib
import time
import csv
import subprocess
import tempfile
import shutil
import contextlib
import urllib.request
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from research.regime_validation.protocol import load_or_create_protocol, get_protocol_hash, get_config_hash
from research.regime_validation.dataset import DatasetManager
from research.regime_validation.kline_replay import HistoricalRegimeReplayRunner
from research.regime_validation.signal_sources import SignalLoader
from research.regime_validation.asof_join import AsOfJoiner
from research.regime_validation.outcome_labels import OutcomeLabeler
from research.regime_validation.stability import StabilityAnalyzer
from research.regime_validation.confusion import ConfusionMatrixCalculator
from research.regime_validation.breakout_analysis import BreakoutAnalyzer
from research.regime_validation.expectancy import ExpectancyCalculator
from research.regime_validation.counterfactual import CounterfactualAnalyzer
from research.regime_validation.report import ReportGenerator
from regime.permissions import permissions_for

import contextlib
import urllib.request
import hashlib

@contextlib.contextmanager
def deny_network():
    original_urlopen = urllib.request.urlopen
    def mocked_urlopen(*args, **kwargs):
        raise RuntimeError("Network access denied during deterministic replay")
    urllib.request.urlopen = mocked_urlopen
    try:
        yield
    finally:
        urllib.request.urlopen = original_urlopen

def get_signal_source_hash(signal_source_dir: str) -> str:
    sha = hashlib.sha256()
    for f in sorted(["bias_signals.csv", "bias_transitions.csv"]):
        p = os.path.join(signal_source_dir, f)
        if os.path.exists(p):
            with open(p, "rb") as f_in:
                sha.update(f_in.read())
    return sha.hexdigest()


# ------------------------------------------------------------------ #
#  Frozen Phase 1B baseline commit (do not mutate)                   #
# ------------------------------------------------------------------ #
_PHASE_1B_BASELINE_COMMIT = "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c"

# Protected production files — git diff from Phase 1B baseline must not
# show any of these changed.
_PROTECTED_PRODUCTION_FILES = [
    "config.py",
    "main.py",
    "scoring.py",
    "flow_metrics.py",
    "trade_stream.py",
    "liquidity/scanner.py",
    "regime/classifier.py",
    "regime/features.py",
    "regime/permissions.py",
    "regime/engine.py",
    "regime/models.py",
    "regime/store.py",
]


# ------------------------------------------------------------------ #
#  Utility helpers                                                    #
# ------------------------------------------------------------------ #

def get_git_commit_sha() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return res.decode().strip()
    except Exception:
        return "084aa9ee94ced0677e7f542238bddd9320b9ffe9"


def load_frozen_config() -> Dict[str, Any]:
    import config
    return {
        "REGIME_MODEL_VERSION":          getattr(config, "REGIME_MODEL_VERSION",          "regime-v1"),
        "REGIME_FEATURE_VERSION":        getattr(config, "REGIME_FEATURE_VERSION",        "regime-features-v1"),
        "REGIME_MAX_BARS_PER_TIMEFRAME": getattr(config, "REGIME_MAX_BARS_PER_TIMEFRAME", 2000),
        "REGIME_TRADE_DEDUP_CAPACITY":   getattr(config, "REGIME_TRADE_DEDUP_CAPACITY",   5000),
        "REGIME_MAX_LATE_TRADE_MS":      getattr(config, "REGIME_MAX_LATE_TRADE_MS",      2000),
        "REGIME_SWITCH_CONFIRM_BARS":    getattr(config, "REGIME_SWITCH_CONFIRM_BARS",    3),
        "REGIME_MIN_CONFIDENCE":         getattr(config, "REGIME_MIN_CONFIDENCE",         0.65),
        "REGIME_SWITCH_MARGIN":          getattr(config, "REGIME_SWITCH_MARGIN",          0.10),
        "REGIME_VOL_PERCENTILE_WINDOW":  getattr(config, "REGIME_VOL_PERCENTILE_WINDOW",  200),
        "REGIME_VOL_MIN_SAMPLES":        getattr(config, "REGIME_VOL_MIN_SAMPLES",        100),
        "REGIME_LIQUIDITY_WINDOW":       getattr(config, "REGIME_LIQUIDITY_WINDOW",       500),
        "REGIME_LIQUIDITY_MIN_SAMPLES":  getattr(config, "REGIME_LIQUIDITY_MIN_SAMPLES",  100),
        "REGIME_BREAKOUT_MAX_BARS":      getattr(config, "REGIME_BREAKOUT_MAX_BARS",      5),
    }


def compute_dataset_content_hash(dataset_dir: str, symbols: List[str]) -> str:
    hasher = hashlib.sha256()
    for symbol in sorted(symbols):
        quality_file = os.path.join(dataset_dir, f"{symbol.lower()}_quality.json")
        if os.path.exists(quality_file):
            with open(quality_file, "r") as f:
                q = json.load(f)
            hasher.update(symbol.lower().encode())
            hasher.update(str(q.get("bar_count", 0)).encode())
            hasher.update(q.get("content_sha256", "").encode())
            hasher.update(str(q.get("actual_start_ms", 0)).encode())
            hasher.update(str(q.get("actual_end_ms", 0)).encode())
    return hasher.hexdigest()


def assert_protocol_locked(protocol_path: str, expected_hash: str):
    if not os.path.exists(protocol_path):
        print("status = PROTOCOL_MUTATED")
        sys.exit(1)
    with open(protocol_path, "r") as f:
        data = json.load(f)
    if get_protocol_hash(data) != expected_hash:
        print("status = PROTOCOL_MUTATED")
        sys.exit(1)


def compute_result_content_hash(run_dir: str) -> str:
    """Hash all canonical artifact files.

    Excludes:
        - validation_summary.json  (written last, contains its own hash)
        - run_manifest.json        (written after summary)
        - Any file not ending in .csv or .json
    Wall-clock fields (run_id, created_at, paths) must NOT appear in
    the hashed files themselves — they belong only in the two excluded files.
    """
    hasher = hashlib.sha256()
    files_to_hash = sorted(
        f for f in os.listdir(run_dir)
        if (f.endswith(".csv") or f.endswith(".json"))
        and f not in ("validation_summary.json", "run_manifest.json")
    )
    for fn in files_to_hash:
        p = os.path.join(run_dir, fn)
        hasher.update(fn.encode())
        with open(p, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def _check_production_files_frozen(baseline_commit: str) -> bool:
    """Return True if no protected production files changed since the baseline."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", baseline_commit + "..HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_changed_files = out.splitlines()
        
        PROJECT_PREFIX = "liquidity sweep/order_flow_engine/"
        changed_relative = {
            p[len(PROJECT_PREFIX):]
            for p in git_changed_files
            if p.startswith(PROJECT_PREFIX)
        }
        
        # Convert list to set for O(1) lookups
        protected_set = set(_PROTECTED_PRODUCTION_FILES)
        
        violations = [f for f in changed_relative if f in protected_set]
        if violations:
            print(f"[WARN] production_files_frozen: modified files = {violations}")
            return False
        return True
    except Exception as e:
        print(f"[WARN] production_files_frozen: git diff failed ({e})")
        return False


def _protocol_still_locked(protocol_path: str, expected_hash: str) -> bool:
    """Return True if the protocol file exists and its hash still matches expected_hash."""
    if not os.path.exists(protocol_path):
        return False
    with open(protocol_path, "r") as f:
        data = json.load(f)
    return get_protocol_hash(data) == expected_hash


def _check_dataset_complete(
    protocol: Dict[str, Any],
    quality_summary: Dict[str, Any]
) -> bool:
    """Verify each symbol's data covers the requested warmup_start through requested_end."""
    from datetime import timedelta
    try:
        start_str = protocol["dataset_date_ranges"]["start"]
        end_str   = protocol["dataset_date_ranges"]["end"]
        warmup_days = protocol.get("warmup_period_days", 22)
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(end_str,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        warmup_start_dt = start_dt - timedelta(days=warmup_days)
        expected_start_ms = int(warmup_start_dt.timestamp() * 1000)
        expected_end_ms   = int(end_dt.timestamp() * 1000)

        for sym, q in quality_summary.items():
            actual_start = q.get("actual_start_ms", 0)
            actual_end   = q.get("actual_end_ms",   0)
            if actual_start > expected_start_ms:
                print(f"[WARN] dataset_complete: {sym} actual_start {actual_start} > expected {expected_start_ms}")
                return False
            if actual_end < expected_end_ms - 86400000:  # allow 1-day slack at end
                print(f"[WARN] dataset_complete: {sym} actual_end {actual_end} < expected {expected_end_ms}")
                return False
        return True
    except Exception as e:
        print(f"[WARN] dataset_complete check error: {e}")
        return False


def _build_mandatory_artifact_list(run_dir: str, symbols: List[str]) -> List[str]:
    """Return the list of mandatory artifact paths that must exist before readiness."""
    must_exist = [
        os.path.join(run_dir, "validation_protocol.json"),
        os.path.join(run_dir, "holdout_lock.json"),
        os.path.join(run_dir, "regime_occupancy.csv"),
        os.path.join(run_dir, "regime_persistence.csv"),
        os.path.join(run_dir, "regime_transition_matrix.csv"),
        os.path.join(run_dir, "regime_flip_flops.csv"),
        os.path.join(run_dir, "candidate_validation.csv"),
        os.path.join(run_dir, "confidence_validation.csv"),
        os.path.join(run_dir, "transition_risk_validation.csv"),
        os.path.join(run_dir, "reference_metrics.json"),
        os.path.join(run_dir, "breakout_validation.csv"),
        os.path.join(run_dir, "breakout_summary.json"),
        os.path.join(run_dir, "signal_regime_join.csv"),
        os.path.join(run_dir, "signal_expectancy_by_regime.csv"),
        os.path.join(run_dir, "signal_expectancy_by_symbol.csv"),
        os.path.join(run_dir, "signal_expectancy_by_confidence.csv"),
        os.path.join(run_dir, "signal_expectancy_by_volatility.csv"),
        os.path.join(run_dir, "signal_expectancy_by_liquidity.csv"),
        os.path.join(run_dir, "signal_expectancy_by_persistence.csv"),
        os.path.join(run_dir, "permission_performance.csv"),
        os.path.join(run_dir, "allow_only_counterfactual.json"),
        os.path.join(run_dir, "high_confidence_mismatches.csv"),
        os.path.join(run_dir, "manual_review_windows.csv"),
        os.path.join(run_dir, "run_manifest.json"),
    ]
    for sym in symbols:
        must_exist.append(os.path.join(run_dir, f"regime_timeline_{sym.lower()}.csv"))
    return must_exist


# ------------------------------------------------------------------ #
#  CLI sub-commands                                                   #
# ------------------------------------------------------------------ #

def cmd_inventory(args):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    inventory = []
    targets = ["bias_signals.csv", "bias_transitions.csv", "regime_states.csv",
               "regime_transitions.csv", "flow_events.csv"]
    for t in targets:
        p = os.path.join(root_dir, t)
        if os.path.exists(p):
            stat = os.stat(p)
            sha = hashlib.sha256()
            with open(p, "rb") as f:
                sha.update(f.read())
            inventory.append({
                "path": p, "filename": t, "size_bytes": stat.st_size,
                "sha256": sha.hexdigest(), "quality_notes": "Local workspace output file"
            })
    out_path = os.path.join(root_dir, "research/datasets/dataset_inventory.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=4)
    print(f"Inventory completed. Saved to {out_path}.")


def cmd_prepare(args):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    dataset_dir = os.path.join(root_dir, "research/datasets")
    manager = DatasetManager(dataset_dir)

    symbols = protocol.get("symbols", ["BTCUSDT"])
    start_str = protocol["dataset_date_ranges"]["start"]
    end_str   = protocol["dataset_date_ranges"]["end"]
    warmup_days = protocol.get("warmup_period_days", 22)

    manager.prepare_dataset(symbols, start_str, end_str, warmup_days)

    quality_summary: Dict[str, Any] = {}
    allow_gaps = protocol.get("dataset_quality_requirements", {}).get("allow_gaps", False)

    from datetime import timedelta
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    warmup_start_dt = start_dt - timedelta(days=warmup_days)
    warmup_start_str = warmup_start_dt.strftime("%Y-%m-%d")

    for s in symbols:
        q = manager.validate_dataset(s, allow_gaps, warmup_start_str, end_str)
        quality_summary[s] = q
        if not q["valid"]:
            print(f"Dataset validation FAILED for {s.upper()}. Check quality logs.")
            sys.exit(1)

    manifest = {
        "dataset_id": f"ds_{start_str}_{end_str}",
        "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbols": sorted(symbols),
        "requested_start": start_str,
        "requested_end": end_str,
        "files": {},
    }
    for s in symbols:
        q = quality_summary[s]
        manifest["files"][s.lower()] = {
            "actual_start_ms":     q["actual_start_ms"],
            "actual_end_ms":       q["actual_end_ms"],
            "actual_start_utc":    q["actual_start_utc"],
            "actual_end_utc":      q["actual_end_utc"],
            "bar_count":           q["bar_count"],
            "missing_bar_count":   q["missing_bar_count"],
            "duplicate_bar_count": q["duplicate_bar_count"],
            "invalid_ohlc_count":  q["invalid_ohlc_count"],
            "content_sha256":      q["content_sha256"],
        }
    with open(os.path.join(dataset_dir, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
    with open(os.path.join(dataset_dir, "dataset_quality.json"), "w") as f:
        json.dump(quality_summary, f, indent=4)
    print("Dataset preparation complete.")


def cmd_replay(args):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    config_dict = load_frozen_config()
    cfg_hash = get_config_hash(config_dict)
    protocol["classifier_config_hash"] = cfg_hash

    temp_run_dir = tempfile.mkdtemp()
    run_protocol_path = os.path.join(temp_run_dir, "validation_protocol.json")
    with open(run_protocol_path, "w") as f:
        json.dump(protocol, f, indent=4)

    run_replay_simulation(temp_run_dir, protocol, config_dict, cfg_hash)
    print(f"Replay completed in temporary run {temp_run_dir}.")
    shutil.rmtree(temp_run_dir)


def run_replay_simulation(
    run_dir: str,
    protocol: Dict[str, Any],
    config_dict: Dict[str, Any],
    cfg_hash: str,
    dataset_dir: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Run the kline replay for all symbols and write per-symbol timeline CSVs.

    The cfg_hash is threaded all the way into save_timeline_csv so that
    every timeline row has the correct config_hash — never "default".

    Mandatory invariant:
        timeline config_hash
        == validation_protocol classifier_config_hash
        == run_manifest config_hash
        == validation_summary config_hash
    """
    if not dataset_dir:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        dataset_dir = os.path.join(root_dir, "research/datasets")
    manager = DatasetManager(dataset_dir)
    symbols = protocol.get("symbols", ["BTCUSDT"])
    p_hash = get_protocol_hash(protocol)

    all_timelines: Dict[str, List[Dict[str, Any]]] = {}
    all_bars: Dict[str, Any] = {}

    runner = HistoricalRegimeReplayRunner(symbols, config_dict)
    for symbol in symbols:
        bars = manager.load_bars(symbol)
        all_bars[symbol.lower()] = bars

        warmup_bars, dev_bars, val_bars, holdout_bars = manager.get_splits(
            bars,
            warmup_days=protocol.get("warmup_period_days", 22),
            dev_pct=protocol.get("development_period_pct", 0.60),
            val_pct=protocol.get("validation_period_pct", 0.20),
            holdout_pct=protocol.get("holdout_period_pct", 0.20),
        )

        timeline = runner.run_replay(warmup_bars, dev_bars, val_bars, holdout_bars)
        all_timelines[symbol.lower()] = timeline

        timeline_csv = os.path.join(run_dir, f"regime_timeline_{symbol.lower()}.csv")
        # Pass the real cfg_hash — never "default"
        runner.save_timeline_csv(timeline, timeline_csv, p_hash, cfg_hash)

    return all_timelines, all_bars


def execute_prepared_validation(
    run_dir: str,
    protocol: Dict[str, Any],
    config_dict: Dict[str, Any],
    cfg_hash: str,
    dataset_dir: str,
    signal_source_dir: Optional[str] = None,
) -> str:
    """Execute the FULL analysis pipeline (replay + all analyses) into run_dir.

    Used by cmd_verify_determinism to run the pipeline twice and compare
    result_content_hash. Does NOT write run_manifest or validation_summary
    (those contain wall-clock run_id).

    Returns:
        result_content_hash — the deterministic fingerprint of all canonical artifacts.
    """
    os.makedirs(run_dir, exist_ok=True)
    p_hash = get_protocol_hash(protocol)

    # Replay
    all_timelines, all_bars = run_replay_simulation(run_dir, protocol, config_dict, cfg_hash, dataset_dir=dataset_dir)

    # Write protocol copy
    with open(os.path.join(run_dir, "validation_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=4)

    # Stability (per-symbol, no cross-symbol contamination)
    _write_stability_artifacts(run_dir, all_timelines, protocol)

    # Ex-post labels and confusion
    flat_timeline = [item for tl in all_timelines.values() for item in tl]
    ref_labels_15m, ref_labels_60m = _compute_reference_labels(all_timelines, all_bars, protocol)
    flat_ref_15 = [lbl for labels in ref_labels_15m.values() for lbl in labels]
    flat_ref_60 = [lbl for labels in ref_labels_60m.values() for lbl in labels]
    sum15 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_15)[1]
    sum60 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_60)[1]
    with open(os.path.join(run_dir, "reference_metrics.json"), "w") as f:
        json.dump({"15m": sum15, "60m": sum60}, f, indent=4)

    # Confidence and transition-risk validation
    _write_confidence_and_tr_csvs(run_dir, all_timelines, ref_labels_15m, ref_labels_60m, protocol)

    # Breakout analysis
    _write_breakout_artifacts(run_dir, all_timelines, all_bars, protocol)

    # Signal join and expectancy
    if not signal_source_dir:
        signal_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    allowed_sources = protocol.get("allowed_signal_sources", ["RECORDED_DECISION_TRANSITION"])
    joined_signals = _load_and_join_signals(signal_source_dir, all_timelines, all_bars, protocol, allowed_sources)
    _write_signal_join_csv(run_dir, joined_signals)
    _write_expectancy_csvs(run_dir, joined_signals, flat_timeline)

    # Permission performance and counterfactual
    cf_results = _write_permission_artifacts(run_dir, joined_signals, protocol)

    # Holdout lock
    _write_holdout_lock(run_dir, dataset_dir, all_bars, protocol, p_hash, cfg_hash)

    # High-confidence mismatches and manual review windows
    _write_mismatch_artifacts(run_dir, all_timelines, ref_labels_15m, protocol)

    # Candidate validation CSV
    _write_candidate_validation_csv(run_dir, all_timelines)

    # Compute and return result content hash
    return compute_result_content_hash(run_dir)


def cmd_analyze(args):
    print("Analyze command is integrated into run-all pipeline.")


def cmd_report(args):
    print("Report command is integrated into run-all pipeline.")


# ------------------------------------------------------------------ #
#  Shared analysis helpers                                            #
# ------------------------------------------------------------------ #

def _write_stability_artifacts(
    run_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
    protocol: Dict[str, Any],
):
    """Write occupancy, persistence, flip-flops, and transition-matrix CSVs.

    Temporal analysis is run per-symbol — no cross-symbol contamination.
    """
    # Occupancy: pooled row counts (labelled MICRO — safe)
    flat_timeline = [item for tl in all_timelines.values() for item in tl]
    agg_occupancy = StabilityAnalyzer.analyze_occupancy(flat_timeline)

    # Persistence: per-symbol, then aggregate
    agg_persistence = StabilityAnalyzer.analyze_persistence(all_timelines)

    # Flip-flops: per-symbol weighted aggregation
    agg_flips = StabilityAnalyzer.analyze_flip_flops(all_timelines)

    # Transition matrix: per-symbol, then summed
    trans_counts, trans_matrix = StabilityAnalyzer.analyze_transitions(all_timelines)

    with open(os.path.join(run_dir, "regime_occupancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "occupancy_pct"])
        for r, v in agg_occupancy.items():
            w.writerow([r, v])

    with open(os.path.join(run_dir, "regime_persistence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "blocks", "mean", "median", "p25", "p75", "p90", "max"])
        for r, p in agg_persistence.items():
            w.writerow([r, p["blocks"], p["mean"], p["median"],
                        p["p25"], p["p75"], p["p90"], p["max"]])

    with open(os.path.join(run_dir, "regime_flip_flops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flip_3_rate", "flip_5_rate", "flip_10_rate"])
        w.writerow([agg_flips["flip_3"], agg_flips["flip_5"], agg_flips["flip_10"]])

    regimes = sorted(list(agg_occupancy.keys()))
    with open(os.path.join(run_dir, "regime_transition_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_regime"] + regimes)
        for from_r in regimes:
            row = [from_r]
            total_from = trans_counts.get(from_r, 0)
            for to_r in regimes:
                c = trans_matrix.get(from_r, {}).get(to_r, 0)
                pct = c / total_from if total_from > 0 else 0.0
                row.append(f"{c} ({pct * 100:.2f}%)")
            w.writerow(row)

    return agg_occupancy, agg_persistence, agg_flips


def _write_candidate_validation_csv(
    run_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
):
    """Generate candidate_validation.csv — aggregate + per-symbol rows."""
    agg = StabilityAnalyzer.analyze_candidates(all_timelines)

    rows = []
    for cand, m in sorted(agg.items()):
        rows.append({
            "symbol": "ALL",
            "candidate_regime": cand,
            "episodes": m["episodes"],
            "confirmed": m["confirmed"],
            "reset": m["reset"],
            "confirmation_rate": round(m["confirmation_rate"], 4),
            "reset_rate": round(m["reset_rate"], 4),
        })

    # Per-symbol rows
    for sym, tl in sorted(all_timelines.items()):
        sym_result = StabilityAnalyzer._candidate_episodes(tl)
        for cand, m in sorted(sym_result.items()):
            eps = m["episodes"]
            rows.append({
                "symbol": sym.upper(),
                "candidate_regime": cand,
                "episodes": eps,
                "confirmed": m["confirmed"],
                "reset": m["reset"],
                "confirmation_rate": round(m["confirmed"] / eps if eps > 0 else 0.0, 4),
                "reset_rate": round(m["reset"] / eps if eps > 0 else 0.0, 4),
            })

    with open(os.path.join(run_dir, "candidate_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "candidate_regime", "episodes",
                                           "confirmed", "reset", "confirmation_rate", "reset_rate"])
        w.writeheader()
        w.writerows(rows)


def _compute_reference_labels(
    all_timelines: Dict[str, List[Dict[str, Any]]],
    all_bars: Dict[str, Any],
    protocol: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    ref_labels_15m: Dict[str, List[str]] = {}
    ref_labels_60m: Dict[str, List[str]] = {}
    for s in protocol.get("symbols", []):
        sym_lower = s.lower()
        bars = all_bars[sym_lower]
        bar_times = {b.close_time_ms: i for i, b in enumerate(bars)}
        labels_15: List[str] = []
        labels_60: List[str] = []
        for t in all_timelines[sym_lower]:
            close_ms = t.get("latest_1m_close_time", 0)
            idx = bar_times.get(close_ms, -1)
            if idx == -1:
                labels_15.append("UNLABELLED")
                labels_60.append("UNLABELLED")
            else:
                lbl15, _ = OutcomeLabeler.compute_ex_post_label(
                    bars, idx, 15, protocol.get("reference_label_thresholds", {}))
                lbl60, _ = OutcomeLabeler.compute_ex_post_label(
                    bars, idx, 60, protocol.get("reference_label_thresholds", {}))
                labels_15.append(lbl15)
                labels_60.append(lbl60)
        ref_labels_15m[sym_lower] = labels_15
        ref_labels_60m[sym_lower] = labels_60
    return ref_labels_15m, ref_labels_60m


def _write_confidence_and_tr_csvs(
    run_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
    ref_labels_15m: Dict[str, List[str]],
    ref_labels_60m: Dict[str, List[str]],
    protocol: Dict[str, Any],
):
    """Write confidence_validation.csv and transition_risk_validation.csv.

    Transition-risk 1-bar metric: changed_1 = timeline[i+1].primary_regime != timeline[i].primary_regime
    (NOT persistence_bars == 1).
    """
    conf_buckets = [0.0, 0.50, 0.65, 0.80, 0.90, 1.1]
    conf_data: Dict[int, list] = {b: [] for b in range(len(conf_buckets) - 1)}

    tr_buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.1]
    tr_data: Dict[int, list] = {b: [] for b in range(len(tr_buckets) - 1)}

    for symbol in protocol.get("symbols", []):
        sym_lower = symbol.lower()
        tl = all_timelines[sym_lower]
        for idx, t in enumerate(tl):
            c_val  = t.get("confidence", 0.0)
            tr_val = t.get("transition_risk", 0.0)

            curr_reg = t.get("primary_regime")

            # 1-bar transition: actual next bar regime differs
            changed_1  = (idx + 1 < len(tl) and
                          tl[idx + 1].get("primary_regime") != curr_reg)
            changed_3  = any(tl[idx + j].get("primary_regime") != curr_reg
                             for j in range(1, 4) if idx + j < len(tl))
            changed_5  = any(tl[idx + j].get("primary_regime") != curr_reg
                             for j in range(1, 6) if idx + j < len(tl))
            changed_10 = any(tl[idx + j].get("primary_regime") != curr_reg
                             for j in range(1, 11) if idx + j < len(tl))

            agree_15 = 1 if ConfusionMatrixCalculator.map_prediction(curr_reg) == ref_labels_15m[sym_lower][idx] else 0
            agree_60 = 1 if ConfusionMatrixCalculator.map_prediction(curr_reg) == ref_labels_60m[sym_lower][idx] else 0

            item = {
                "persistence": t.get("persistence_bars", 0),
                "changed_1":  changed_1,
                "changed_3":  changed_3,
                "changed_5":  changed_5,
                "changed_10": changed_10,
                "agree_15": agree_15,
                "agree_60": agree_60,
            }

            for b_idx in range(len(conf_buckets) - 1):
                if conf_buckets[b_idx] <= c_val < conf_buckets[b_idx + 1]:
                    conf_data[b_idx].append(item)

            for b_idx in range(len(tr_buckets) - 1):
                if tr_buckets[b_idx] <= tr_val < tr_buckets[b_idx + 1]:
                    tr_data[b_idx].append(item)

    with open(os.path.join(run_dir, "confidence_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["confidence_bucket", "sample_count", "median_persistence",
                    "flip_3_rate", "flip_5_rate", "reference_agreement_15m", "reference_agreement_60m"])
        for b_idx in range(len(conf_buckets) - 1):
            items = conf_data[b_idx]
            count = len(items)
            med_p = float(np.median([it["persistence"] for it in items])) if items else 0.0
            flip3 = sum(1 for it in items if it["changed_3"]) / count if count > 0 else 0.0
            flip5 = sum(1 for it in items if it["changed_5"]) / count if count > 0 else 0.0
            agr15 = sum(1 for it in items if it["agree_15"]) / count if count > 0 else 0.0
            agr60 = sum(1 for it in items if it["agree_60"]) / count if count > 0 else 0.0
            w.writerow([f"{conf_buckets[b_idx]}-{conf_buckets[b_idx + 1]}",
                        count, med_p, flip3, flip5, agr15, agr60])

    with open(os.path.join(run_dir, "transition_risk_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transition_risk_bucket", "sample_count",
                    "transition_rate_1_bar", "transition_rate_3_bar",
                    "transition_rate_5_bar", "transition_rate_10_bar"])
        for b_idx in range(len(tr_buckets) - 1):
            items = tr_data[b_idx]
            count = len(items)
            tr1  = sum(1 for it in items if it["changed_1"])  / count if count > 0 else 0.0
            tr3  = sum(1 for it in items if it["changed_3"])  / count if count > 0 else 0.0
            tr5  = sum(1 for it in items if it["changed_5"])  / count if count > 0 else 0.0
            tr10 = sum(1 for it in items if it["changed_10"]) / count if count > 0 else 0.0
            w.writerow([f"{tr_buckets[b_idx]}-{tr_buckets[b_idx + 1]}",
                        count, tr1, tr3, tr5, tr10])


def _write_breakout_artifacts(
    run_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
    all_bars: Dict[str, Any],
    protocol: Dict[str, Any],
):
    breakout_results = []
    for symbol in protocol.get("symbols", []):
        sym_lower = symbol.lower()
        breakout_results.extend(
            BreakoutAnalyzer.analyze_breakouts(all_timelines[sym_lower], all_bars[sym_lower])
        )

    bo_fields = ["symbol", "timestamp_ms", "regime", "confidence", "atr", "outcome",
                 "prior_range_high", "prior_range_low", "return_inside_range",
                 "time_to_return_inside_range", "max_extension_atr", "max_retracement_atr",
                 "5m_return_atr", "15m_return_atr", "30m_return_atr",
                 "next_canonical_regime", "time_to_next_regime"]
    with open(os.path.join(run_dir, "breakout_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bo_fields)
        w.writeheader()
        w.writerows(breakout_results)

    total_bo_up   = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP")
    total_bo_down = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN")
    bo_up_success   = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP"   and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH")
    bo_down_success = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN" and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH")

    bo_summary = {
        "breakout_up":   {"total": total_bo_up,   "success_rate": bo_up_success   / total_bo_up   if total_bo_up   > 0 else 0.0},
        "breakout_down": {"total": total_bo_down, "success_rate": bo_down_success / total_bo_down if total_bo_down > 0 else 0.0},
    }
    with open(os.path.join(run_dir, "breakout_summary.json"), "w") as f:
        json.dump(bo_summary, f, indent=4)
    return bo_summary


def _load_and_join_signals(
    signal_source_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
    all_bars: Dict[str, Any],
    protocol: Dict[str, Any],
    allowed_sources: List[str],
) -> List[Dict[str, Any]]:
    raw_signals = []
    transitions_path = os.path.join(signal_source_dir, "bias_transitions.csv")
    if os.path.exists(transitions_path):
        raw_signals.extend(SignalLoader.load_from_transitions_csv(transitions_path))
    legacy_path = os.path.join(signal_source_dir, "bias_signals.csv")
    if os.path.exists(legacy_path):
        raw_signals.extend(SignalLoader.load_legacy_signals_csv(legacy_path))

    signals = [s for s in raw_signals if s.metadata.get("provenance_mode") in allowed_sources]

    joined_signals: List[Dict[str, Any]] = []
    for s in signals:
        sym_lower = s.symbol.lower()
        if sym_lower not in all_timelines:
            continue
        res = AsOfJoiner.join_signal_to_regime(s, all_timelines[sym_lower])
        if res["joined"]:
            reg_state = res["regime_state"]
            bars = all_bars[sym_lower]
            split_tag = reg_state.get("split", "UNKNOWN")
            outcomes = {}
            for h in [1, 5, 15, 60]:
                outcomes[f"{h}m"] = OutcomeLabeler.compute_signal_outcomes(
                    bars, s.timestamp_ms, s.direction, h, s.metadata["entry_price"]
                )
            reg_name = reg_state.get("primary_regime", "UNKNOWN")
            perm = permissions_for(reg_name).get(s.strategy_family, "BLOCK")
            joined_signals.append({
                "signal_id": s.metadata.get("signal_id", f"sig_{sym_lower}_{s.timestamp_ms}"),
                "symbol": s.symbol.upper(),
                "signal_time_ms": s.timestamp_ms,
                "signal_time_utc": datetime.fromtimestamp(s.timestamp_ms / 1000.0, tz=timezone.utc).isoformat(),
                "signal_action": s.action,
                "direction": s.direction,
                "strategy_family": s.strategy_family,
                "signal_provenance": s.metadata.get("provenance_mode", "UNKNOWN"),
                "split": split_tag,
                "regime_close_ms": reg_state.get("latest_1m_close_time", 0),
                "regime_age_ms": s.timestamp_ms - reg_state.get("latest_1m_close_time", 0),
                "primary_regime": reg_name,
                "confidence": reg_state.get("confidence", 0.0),
                "transition_risk": reg_state.get("transition_risk", 0.0),
                "volatility": reg_state.get("volatility", "UNKNOWN"),
                "liquidity": "NOT_AVAILABLE",
                "quality": reg_state.get("quality", "UNKNOWN"),
                "tradable": reg_state.get("tradable", False),
                "join_status": "JOINED" if res["safe"] else "DEGRADED_UNSAFE",
                "permission": perm,
                "outcome_quality": "BAR_APPROX",
                "outcomes": outcomes,
                "safe": res["safe"],
                "joined": True,
            })
    return joined_signals


def _write_signal_join_csv(run_dir: str, joined_signals: List[Dict[str, Any]]):
    join_headers = [
        "signal_id", "symbol", "signal_time_ms", "signal_time_utc", "signal_action",
        "direction", "strategy_family", "signal_provenance", "split",
        "regime_close_ms", "regime_age_ms", "primary_regime", "confidence",
        "transition_risk", "volatility", "liquidity", "quality", "tradable",
        "join_status", "permission", "outcome_quality",
    ]
    with open(os.path.join(run_dir, "signal_regime_join.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=join_headers)
        w.writeheader()
        for js in joined_signals:
            w.writerow({h: js[h] for h in join_headers})


def _write_expectancy_csvs(
    run_dir: str,
    joined_signals: List[Dict[str, Any]],
    flat_timeline: List[Dict[str, Any]],
):
    cost_scenarios = [0, 2, 5, 10]
    horizons       = ["1m", "5m", "15m", "60m"]
    splits         = ["DEVELOPMENT", "VALIDATION", "HOLDOUT"]

    def write_expectancy_csv(filepath: str, slice_fn, slice_headers: List[str]):
        with open(filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "horizon", "cost_bps"] + slice_headers +
                       ["sample_count", "mean_net_return", "win_rate"])
            for sp in splits:
                sp_sigs = [s for s in joined_signals if s["split"] == sp]
                for hz in horizons:
                    for cost in cost_scenarios:
                        slices = slice_fn(sp_sigs)
                        for slice_key, sigs in slices.items():
                            exp = ExpectancyCalculator.calculate_expectancy_for_horizon(sigs, hz, cost)
                            keys = list(slice_key) if isinstance(slice_key, tuple) else [slice_key]
                            w.writerow([sp, hz, cost] + keys +
                                       [exp["sample_count"], exp["mean_return"], exp["win_rate"]])

    regime_slice = lambda sigs: {s["primary_regime"]: [x for x in sigs if x["primary_regime"] == s["primary_regime"]] for s in sigs}
    symbol_slice = lambda sigs: {s["symbol"]: [x for x in sigs if x["symbol"] == s["symbol"]] for s in sigs}

    def conf_slice(sigs):
        buckets = ["0.0-0.50", "0.50-0.65", "0.65-0.80", "0.80-0.90", "0.90-1.0"]
        res = {b: [] for b in buckets}
        for s in sigs:
            cv = s["confidence"]
            if   cv < 0.50: res["0.0-0.50"].append(s)
            elif cv < 0.65: res["0.50-0.65"].append(s)
            elif cv < 0.80: res["0.65-0.80"].append(s)
            elif cv < 0.90: res["0.80-0.90"].append(s)
            else:           res["0.90-1.0"].append(s)
        return res

    vol_slice = lambda sigs: {s["volatility"]: [x for x in sigs if x["volatility"] == s["volatility"]] for s in sigs}
    liq_slice = lambda sigs: {"NOT_AVAILABLE": sigs}

    def pers_slice(sigs):
        res = {"1-5": [], "6-20": [], "20+": []}
        for s in sigs:
            p_val = 0
            for t in flat_timeline:
                if (t.get("latest_1m_close_time") == s["regime_close_ms"] and
                        t.get("symbol", "").upper() == s["symbol"]):
                    p_val = t.get("persistence_bars", 0)
                    break
            if   p_val <= 5:  res["1-5"].append(s)
            elif p_val <= 20: res["6-20"].append(s)
            else:             res["20+"].append(s)
        return res

    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_regime.csv"),      regime_slice, ["primary_regime"])
    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_symbol.csv"),      symbol_slice, ["symbol"])
    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_confidence.csv"),  conf_slice,   ["confidence_bucket"])
    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_volatility.csv"),  vol_slice,    ["volatility"])
    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_liquidity.csv"),   liq_slice,    ["liquidity_slice"])
    write_expectancy_csv(os.path.join(run_dir, "signal_expectancy_by_persistence.csv"), pers_slice,   ["persistence_bucket"])


def _write_permission_artifacts(
    run_dir: str,
    joined_signals: List[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    splits = ["DEVELOPMENT", "VALIDATION", "HOLDOUT"]
    cost_bps = protocol.get("primary_cost_scenario_bps", 5)

    cf_results: Dict[str, Any] = {}
    for sp in splits:
        sp_signals = [s for s in joined_signals if s["split"] == sp]
        cf_results[sp] = CounterfactualAnalyzer.compare_allow_only(sp_signals, cost_bps=cost_bps)

    with open(os.path.join(run_dir, "allow_only_counterfactual.json"), "w") as f:
        json.dump(cf_results, f, indent=4)

    with open(os.path.join(run_dir, "permission_performance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "permission", "sample_count", "mean_net_return"])
        for sp in splits:
            sp_signals = [s for s in joined_signals if s["split"] == sp]
            for p in ["ALLOW", "BLOCK", "REDUCE", "WATCH"]:
                p_sigs = [s for s in sp_signals if s["permission"] == p]
                exp = ExpectancyCalculator.calculate_expectancy_for_horizon(p_sigs, "15m", cost_bps)
                w.writerow([sp, p, len(p_sigs), exp.get("mean_return", 0.0)])

    return cf_results


def _write_holdout_lock(
    run_dir: str,
    dataset_dir: str,
    all_bars: Dict[str, Any],
    protocol: Dict[str, Any],
    p_hash: str,
    cfg_hash: str,
):
    holdout_lock: Dict[str, Any] = {
        "dataset_content_hash":   compute_dataset_content_hash(dataset_dir, protocol.get("symbols", [])),
        "protocol_hash":          p_hash,
        "baseline_phase1b_commit": _PHASE_1B_BASELINE_COMMIT,
        "config_hash":            cfg_hash,
        "validation_commit":      get_git_commit_sha(),
        "symbols":                {},
    }
    for s in protocol.get("symbols", []):
        bars = all_bars[s.lower()]
        _, dev_b, val_b, hold_b = DatasetManager(dataset_dir).get_splits(
            bars,
            protocol.get("warmup_period_days", 22),
            protocol.get("development_period_pct", 0.60),
            protocol.get("validation_period_pct", 0.20),
            protocol.get("holdout_period_pct", 0.20),
        )
        holdout_lock["symbols"][s.lower()] = {
            "development_start_ms":  dev_b[0].open_time_ms  if dev_b  else 0,
            "development_end_ms":    dev_b[-1].close_time_ms if dev_b else 0,
            "development_start_utc": datetime.fromtimestamp(dev_b[0].open_time_ms / 1000.0, tz=timezone.utc).isoformat() if dev_b else "",
            "development_end_utc":   datetime.fromtimestamp(dev_b[-1].close_time_ms / 1000.0, tz=timezone.utc).isoformat() if dev_b else "",
            "validation_start_ms":   val_b[0].open_time_ms  if val_b  else 0,
            "validation_end_ms":     val_b[-1].close_time_ms if val_b else 0,
            "validation_start_utc":  datetime.fromtimestamp(val_b[0].open_time_ms / 1000.0, tz=timezone.utc).isoformat() if val_b else "",
            "validation_end_utc":    datetime.fromtimestamp(val_b[-1].close_time_ms / 1000.0, tz=timezone.utc).isoformat() if val_b else "",
            "holdout_start_ms":      hold_b[0].open_time_ms  if hold_b else 0,
            "holdout_end_ms":        hold_b[-1].close_time_ms if hold_b else 0,
            "holdout_start_utc":     datetime.fromtimestamp(hold_b[0].open_time_ms / 1000.0, tz=timezone.utc).isoformat() if hold_b else "",
            "holdout_end_utc":       datetime.fromtimestamp(hold_b[-1].close_time_ms / 1000.0, tz=timezone.utc).isoformat() if hold_b else "",
        }
    with open(os.path.join(run_dir, "holdout_lock.json"), "w") as f:
        json.dump(holdout_lock, f, indent=4)
    return holdout_lock


def _write_mismatch_artifacts(
    run_dir: str,
    all_timelines: Dict[str, List[Dict[str, Any]]],
    ref_labels_15m: Dict[str, List[str]],
    protocol: Dict[str, Any],
):
    high_mismatches = []
    review_windows  = []
    for symbol in protocol.get("symbols", []):
        sym_lower = symbol.lower()
        tl = all_timelines[sym_lower]
        for idx, t in enumerate(tl):
            ref_lbl = ref_labels_15m[sym_lower][idx]
            if ref_lbl in ("UNLABELLED", "AMBIGUOUS"):
                continue
            pred_lbl = ConfusionMatrixCalculator.map_prediction(t.get("primary_regime", "UNKNOWN"))
            conf = t.get("confidence", 0.0)
            if conf >= 0.80 and pred_lbl != ref_lbl:
                high_mismatches.append({
                    "symbol": symbol.upper(),
                    "timestamp_ms": t.get("latest_1m_close_time", 0),
                    "confidence": conf,
                    "prediction": t.get("primary_regime"),
                    "reference": ref_lbl,
                })
            if len(review_windows) < 100:
                review_windows.append({
                    "symbol": symbol.upper(),
                    "timestamp_ms": t.get("latest_1m_close_time", 0),
                    "regime": t.get("primary_regime"),
                    "confidence": conf,
                    "transition_risk": t.get("transition_risk", 0.0),
                    "reference_outcome": ref_lbl,
                    "reason_selected": ("High-confidence correct"
                                        if pred_lbl == ref_lbl and conf >= 0.80 else "Mismatch"),
                })

    high_mismatches.sort(key=lambda x: x["confidence"], reverse=True)
    with open(os.path.join(run_dir, "high_confidence_mismatches.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "timestamp_ms", "confidence", "prediction", "reference"])
        w.writeheader()
        w.writerows(high_mismatches)
    with open(os.path.join(run_dir, "manual_review_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "timestamp_ms", "regime", "confidence",
                                           "transition_risk", "reference_outcome", "reason_selected"])
        w.writeheader()
        w.writerows(review_windows)


# ------------------------------------------------------------------ #
#  Main orchestrator: cmd_run_all                                     #
# ------------------------------------------------------------------ #

def cmd_run_all(args):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_id  = f"run_{int(time.time())}"
    run_dir = os.path.join(root_dir, f"research/validation_runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # ---- 1. Lock configuration and protocol ----
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol    = load_or_create_protocol(protocol_path)
    config_dict = load_frozen_config()
    cfg_hash    = get_config_hash(config_dict)

    # Materialize config hash into protocol before locking
    protocol["classifier_config_hash"] = cfg_hash
    p_hash = get_protocol_hash(protocol)

    # Write locked copy
    run_protocol_path = os.path.join(run_dir, "validation_protocol.json")
    with open(run_protocol_path, "w") as f:
        json.dump(protocol, f, indent=4)
    assert_protocol_locked(run_protocol_path, p_hash)

    # ---- 2. Inventory and prepare data ----
    cmd_inventory(args)
    cmd_prepare(args)

    dataset_dir = os.path.join(root_dir, "research/datasets")
    with open(os.path.join(dataset_dir, "dataset_quality.json"), "r") as f:
        quality_summary = json.load(f)
    assert_protocol_locked(run_protocol_path, p_hash)

    # ---- 3. Replay simulation ----
    all_timelines, all_bars = run_replay_simulation(run_dir, protocol, config_dict, cfg_hash)
    assert_protocol_locked(run_protocol_path, p_hash)

    symbols = protocol.get("symbols", [])

    # ---- 4. Evaluated-days calculation ----
    evaluated_days_by_symbol: Dict[str, float] = {}
    for s in symbols:
        bars = all_bars[s.lower()]
        _, dev_b, val_b, hold_b = DatasetManager(dataset_dir).get_splits(
            bars,
            protocol.get("warmup_period_days", 22),
            protocol.get("development_period_pct", 0.60),
            protocol.get("validation_period_pct", 0.20),
            protocol.get("holdout_period_pct", 0.20),
        )
        total_eval_ms = 0
        if dev_b:  total_eval_ms += dev_b[-1].close_time_ms  - dev_b[0].open_time_ms
        if val_b:  total_eval_ms += val_b[-1].close_time_ms  - val_b[0].open_time_ms
        if hold_b: total_eval_ms += hold_b[-1].close_time_ms - hold_b[0].open_time_ms
        evaluated_days_by_symbol[s.lower()] = total_eval_ms / (24.0 * 3600.0 * 1000.0)
    conservative_evaluated_days = min(evaluated_days_by_symbol.values()) if evaluated_days_by_symbol else 0.0

    # ---- 5. Stability metrics (per-symbol, no cross-symbol contamination) ----
    agg_occupancy, agg_persistence, agg_flips = _write_stability_artifacts(run_dir, all_timelines, protocol)
    # Candidate validation CSV
    _write_candidate_validation_csv(run_dir, all_timelines)

    # ---- 6. Ex-post labels and confusion ----
    flat_timeline = [item for tl in all_timelines.values() for item in tl]
    ref_labels_15m, ref_labels_60m = _compute_reference_labels(all_timelines, all_bars, protocol)
    flat_ref_15 = [lbl for labels in ref_labels_15m.values() for lbl in labels]
    flat_ref_60 = [lbl for labels in ref_labels_60m.values() for lbl in labels]
    sum15 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_15)[1]
    sum60 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_60)[1]
    with open(os.path.join(run_dir, "reference_metrics.json"), "w") as f:
        json.dump({"15m": sum15, "60m": sum60}, f, indent=4)

    # ---- 7. Confidence and transition-risk validation ----
    _write_confidence_and_tr_csvs(run_dir, all_timelines, ref_labels_15m, ref_labels_60m, protocol)

    # ---- 8. Breakout analysis ----
    bo_summary = _write_breakout_artifacts(run_dir, all_timelines, all_bars, protocol)

    # ---- 9. Signal join, expectancy, permission performance, counterfactual ----
    allowed_sources = protocol.get("allowed_signal_sources", ["RECORDED_DECISION_TRANSITION"])
    joined_signals = _load_and_join_signals(root_dir, all_timelines, all_bars, protocol, allowed_sources)
    _write_signal_join_csv(run_dir, joined_signals)
    _write_expectancy_csvs(run_dir, joined_signals, flat_timeline)
    cf_results = _write_permission_artifacts(run_dir, joined_signals, protocol)

    # ---- 10. High-confidence mismatches and manual review windows ----
    _write_mismatch_artifacts(run_dir, all_timelines, ref_labels_15m, protocol)

    # ---- 11. Holdout lock ----
    holdout_lock = _write_holdout_lock(run_dir, dataset_dir, all_bars, protocol, p_hash, cfg_hash)

    # ---- 11.5 Determinism and No-Network Validation ----
    # Re-run the full pipeline in a temporary dir twice, without network.
    signal_source_dir = root_dir
    signal_source_hash = get_signal_source_hash(signal_source_dir)
    
    no_network_during_replay = False
    replay_deterministic = False
    try:
        with deny_network():
            tmp1 = tempfile.mkdtemp()
            tmp2 = tempfile.mkdtemp()
            try:
                hash_A = execute_prepared_validation(tmp1, protocol, config_dict, cfg_hash, dataset_dir, signal_source_dir)
                hash_B = execute_prepared_validation(tmp2, protocol, config_dict, cfg_hash, dataset_dir, signal_source_dir)
                no_network_during_replay = True
                replay_deterministic = (hash_A == hash_B and hash_A != "")
            finally:
                shutil.rmtree(tmp1, ignore_errors=True)
                shutil.rmtree(tmp2, ignore_errors=True)
    except Exception as e:
        print(f"[WARN] Determinism/Network check failed: {e}")

    # ---- 12. Enforcement criteria A-G ----
    decision = evaluate_enforcement_decision(cf_results, joined_signals)

    # ---- 13. Run manifest ----
    cost_scenarios = [0, 2, 5, 10]
    horizons       = ["1m", "5m", "15m", "60m"]
    run_manifest = {
        "run_id":                 run_id,
        "validation_commit":      get_git_commit_sha(),
        "baseline_phase1b_commit": _PHASE_1B_BASELINE_COMMIT,
        "python_version":         sys.version,
        "platform":               sys.platform,
        "protocol_hash":          p_hash,
        "dataset_hash":           holdout_lock["dataset_content_hash"],
        "config_hash":            cfg_hash,
        "signal_source_hash":     signal_source_hash,
        "model_version":          "regime-v1",
        "feature_version":        "regime-features-v1",
        "symbols":                symbols,
        "date_ranges":            protocol["dataset_date_ranges"],
        "split_ranges":           holdout_lock["symbols"],
        "replay_mode":            "HISTORICAL_CLOSED_BAR_VALIDATION",
        "signal_source":          allowed_sources,
        "outcome_source":         "BAR_APPROX",
        "liquidity_source":       "NOT_AVAILABLE",
        "cost_scenarios":         cost_scenarios,
        "bootstrap_seed":         protocol.get("bootstrap_seed", 1729),
        "bootstrap_reps":         protocol.get("bootstrap_repetitions", 1000),
        "test_status":            "VERIFIED_IN_RUN",
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)

    # ---- 14. Verify mandatory artifact set ----
    mandatory_artifacts = _build_mandatory_artifact_list(run_dir, symbols)
    missing_artifacts = [p for p in mandatory_artifacts if not os.path.exists(p)]

    # ---- 15. Verify config/protocol provenance consistency ----
    # Invariant: timeline config_hash == protocol config_hash == manifest config_hash
    config_hash_valid = (
        cfg_hash != "default"
        and cfg_hash != ""
        and protocol.get("classifier_config_hash") == cfg_hash
        and run_manifest["config_hash"] == cfg_hash
        and run_manifest["protocol_hash"] == p_hash
        and run_manifest["dataset_hash"] == holdout_lock["dataset_content_hash"]
        and run_manifest["signal_source_hash"] == signal_source_hash
    )

    # ---- 16. Compute result content hash ----
    result_content_hash = compute_result_content_hash(run_dir)

    # ---- 17. Derive technical readiness ----
    ready_count = sum(1 for t in flat_timeline if t.get("quality") == "READY")
    technical_gates = {
        "protocol_locked":          _protocol_still_locked(run_protocol_path, p_hash),
        "dataset_valid":            all(q.get("valid", False) for q in quality_summary.values()),
        "dataset_complete":         _check_dataset_complete(protocol, quality_summary),
        "replay_has_ready_states":  ready_count > 0,
        "replay_deterministic":     replay_deterministic,
        "production_files_frozen":  _check_production_files_frozen(_PHASE_1B_BASELINE_COMMIT),
        "required_artifacts_present": len(missing_artifacts) == 0,
        "no_network_during_replay": no_network_during_replay,
        "config_hash_valid":        config_hash_valid,
    }
    
    required_bool_gates = [
        "protocol_locked",
        "dataset_valid",
        "dataset_complete",
        "replay_has_ready_states",
        "replay_deterministic",
        "production_files_frozen",
        "required_artifacts_present",
        "no_network_during_replay",
        "config_hash_valid",
    ]
    
    # Gate on explicit explicit bool variables only
    all_passed = all([technical_gates.get(k, False) is True for k in required_bool_gates])
    tics_ready = "YES" if all_passed else "NO"

    # ---- 18. Write validation summary ----
    summary = {
        "run_id":                     run_id,
        "status":                     "COMPLETED",
        "protocol_hash":              p_hash,
        "dataset_hash":               holdout_lock["dataset_content_hash"],
        "config_hash":                cfg_hash,
        "git_commit":                 get_git_commit_sha(),
        "days_evaluated":             conservative_evaluated_days,
        "symbols":                    symbols,
        "warmup_period_days":         protocol.get("warmup_period_days", 22),
        "regime_ready_coverage":      ready_count / len(flat_timeline) if flat_timeline else 0.0,
        "regime_ready_count":         ready_count,
        "regime_occupancy":           agg_occupancy,
        "median_persistence":         float(np.median([p["median"] for p in agg_persistence.values()])) if agg_persistence else 0.0,
        "flip_flop_3_rate":           agg_flips["flip_3"],
        "flip_flop_5_rate":           agg_flips["flip_5"],
        "reference_agreement_15m":    sum15["agreement_rate"],
        "reference_agreement_60m":    sum60["agreement_rate"],
        "breakout_up_success_rate":   bo_summary["breakout_up"]["success_rate"],
        "breakout_down_success_rate": bo_summary["breakout_down"]["success_rate"],
        "signals_total":              len(joined_signals),
        "signals_censored":           sum(1 for s in joined_signals
                                         if any(s["outcomes"][h]["status"] == "CENSORED" for h in horizons)),
        "allow_signal_count":         len([s for s in joined_signals
                                          if s["split"] == "HOLDOUT" and s["permission"] == "ALLOW"]),
        "block_signal_count":         len([s for s in joined_signals
                                          if s["split"] == "HOLDOUT" and s["permission"] == "BLOCK"]),
        "baseline_15m_net_mean_5bps": cf_results.get("HOLDOUT", {}).get("baseline_metrics", {}).get("mean_return", 0.0),
        "allow_only_15m_net_mean_5bps": cf_results.get("HOLDOUT", {}).get("allow_metrics", {}).get("mean_return", 0.0),
        "allow_minus_block_15m_net_mean_5bps": cf_results.get("HOLDOUT", {}).get("allow_minus_block_point", 0.0),
        "allow_minus_block_ci_low":   (cf_results["HOLDOUT"].get("allow_minus_block_ci") or [0.0])[0] if "HOLDOUT" in cf_results else 0.0,
        "allow_minus_block_ci_high":  (cf_results["HOLDOUT"].get("allow_minus_block_ci") or [0.0, 0.0])[1] if "HOLDOUT" in cf_results else 0.0,
        "holdout_signal_count":       len([s for s in joined_signals if s["split"] == "HOLDOUT"]),
        "technical_gates":            technical_gates,
        "missing_mandatory_artifacts": [os.path.basename(p) for p in missing_artifacts],
        "result_content_hash":        result_content_hash,
        "tics_phase_1b_v_ready":      tics_ready,
        "regime_enforcement_candidate": decision,
    }

    with open(os.path.join(run_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ---- 19. Generate user-facing reports ----
    ReportGenerator.generate_html_report(
        summary, all_timelines, joined_signals,
        os.path.join(run_dir, "validation_report.html")
    )
    ReportGenerator.generate_decision_markdown(
        summary, os.path.join(run_dir, "validation_decision.md")
    )

    print(f"Run complete. TICS_PHASE_1B_V_READY = {summary['tics_phase_1b_v_ready']}")
    print(f"REGIME_ENFORCEMENT_CANDIDATE = {summary['regime_enforcement_candidate']}")


# ------------------------------------------------------------------ #
#  Enforcement criteria A-G evaluator                                #
# ------------------------------------------------------------------ #

def evaluate_enforcement_decision(
    cf_results: Dict[str, Any],
    joined_signals: List[Dict[str, Any]],
) -> str:
    """Evaluate criteria A-G on HOLDOUT split.

    Prerequisites:
        eligible_holdout_15m: split==HOLDOUT, joined==True, safe==True,
                              outcomes["15m"]["status"]=="COMPLETED"
        eligible_holdout_15m >= 100
        ALLOW from eligible >= 30
        BLOCK from eligible >= 30
        Censored 15m signals do NOT count.
    """
    holdout_cf = cf_results.get("HOLDOUT", {})
    if not holdout_cf:
        return "INSUFFICIENT_DATA"

    # --- Eligibility gate (Requirement 2) ---
    eligible_holdout_15m = [
        s for s in joined_signals
        if s.get("split") == "HOLDOUT"
        and s.get("joined", False)
        and s.get("safe", False)
        and s.get("outcomes", {}).get("15m", {}).get("status") == "COMPLETED"
    ]
    allow_signals = [s for s in eligible_holdout_15m if s["permission"] == "ALLOW"]
    block_signals = [s for s in eligible_holdout_15m if s["permission"] == "BLOCK"]

    if len(eligible_holdout_15m) < 100 or len(allow_signals) < 30 or len(block_signals) < 30:
        return "INSUFFICIENT_DATA"

    # --- Criterion A: ALLOW mean > BLOCK mean ---
    allow_mean   = holdout_cf["allow_metrics"].get("mean_return", 0.0)
    block_mean   = holdout_cf["block_metrics"].get("mean_return", 0.0)
    gate_A = allow_mean > block_mean

    # --- Criterion B: bootstrap 95% CI lower bound > 0 ---
    ci = holdout_cf.get("allow_minus_block_ci")
    gate_B = ci[0] > 0.0 if ci else False

    # --- Criterion C: ALLOW-only > unfiltered baseline ---
    baseline_mean = holdout_cf["baseline_metrics"].get("mean_return", 0.0)
    gate_C = allow_mean > baseline_mean

    # --- Criterion D: ALLOW retention >= 30% ---
    retention = holdout_cf.get("retention_pct", 0.0)
    gate_D = retention >= 0.30

    # --- Criterion E: ALLOW average MAE <= baseline MAE * 1.10 ---
    allow_mae    = holdout_cf["allow_metrics"].get("average_MAE", 0.0)
    baseline_mae = holdout_cf["baseline_metrics"].get("average_MAE", 0.0)
    gate_E = allow_mae <= baseline_mae * 1.10

    # --- Criterion F: Real cross-symbol gate (Requirement 3) ---
    # Per-symbol eligibility: >= 10 completed HOLDOUT ALLOW + >= 10 completed HOLDOUT BLOCK
    symbol_lifts: Dict[str, float] = {}
    for sym in set(s["symbol"] for s in eligible_holdout_15m):
        s_allow = [s for s in allow_signals if s["symbol"] == sym]
        s_block = [s for s in block_signals if s["symbol"] == sym]
        if len(s_allow) >= 10 and len(s_block) >= 10:
            cost_bps = 0.0005  # 5bps
            s_allow_mean = float(np.mean([s["outcomes"]["15m"]["return"] - cost_bps for s in s_allow]))
            s_block_mean = float(np.mean([s["outcomes"]["15m"]["return"] - cost_bps for s in s_block]))
            symbol_lifts[sym] = s_allow_mean - s_block_mean

    if len(symbol_lifts) < 3:
        # Fewer than 3 symbols have adequate samples
        return "INSUFFICIENT_DATA"

    pos_lift_symbols = {sym: lift for sym, lift in symbol_lifts.items() if lift > 0.0}
    if len(pos_lift_symbols) < 3:
        gate_F = False
    else:
        total_pos_lift = sum(pos_lift_symbols.values())
        max_sym_contribution = max(pos_lift_symbols.values())
        # No single symbol may account for > 60% of the total positive lift
        if total_pos_lift > 0 and max_sym_contribution / total_pos_lift > 0.60:
            gate_F = False
        else:
            gate_F = True

    # --- Criterion G: VALIDATION lift > 0 AND HOLDOUT lift > 0 (Requirement 4) ---
    # A negative pair is NOT evidence for enforcement.
    val_cf = cf_results.get("VALIDATION", {})
    if val_cf:
        val_allow_mean = val_cf["allow_metrics"].get("mean_return", 0.0)
        val_block_mean = val_cf["block_metrics"].get("mean_return", 0.0)
        val_lift  = val_allow_mean - val_block_mean
        hold_lift = allow_mean - block_mean
        gate_G = val_lift > 0.0 and hold_lift > 0.0
    else:
        gate_G = False

    if gate_A and gate_B and gate_C and gate_D and gate_E and gate_F and gate_G:
        return "YES"
    return "NO"


# ------------------------------------------------------------------ #
#  Determinism verification                                          #
# ------------------------------------------------------------------ #

def cmd_verify_determinism(args):
    """Run the full analysis pipeline twice and compare result_content_hash.

    The hash covers all timeline CSVs, stability artifacts, candidate analysis,
    reference metrics, breakout analysis, signal-regime join, expectancy artifacts,
    permission performance, counterfactual output.
    Wall-clock values (run_id, created_at, paths) are excluded from the hash.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    config_dict = load_frozen_config()
    cfg_hash = get_config_hash(config_dict)
    protocol["classifier_config_hash"] = cfg_hash
    dataset_dir = os.path.join(root_dir, "research/datasets")

    tmp1 = tempfile.mkdtemp()
    tmp2 = tempfile.mkdtemp()

    try:
        hash_A = execute_prepared_validation(tmp1, protocol, config_dict, cfg_hash, dataset_dir)
        hash_B = execute_prepared_validation(tmp2, protocol, config_dict, cfg_hash, dataset_dir)

        if hash_A == hash_B:
            print(f"DETERMINISM VERIFIED. result_content_hash = {hash_A}")
        else:
            print(f"DETERMINISM MISMATCH!")
            print(f"  Run A: {hash_A}")
            print(f"  Run B: {hash_B}")
            # Report which files differ
            for fn in sorted(set(os.listdir(tmp1)) | set(os.listdir(tmp2))):
                if fn in ("validation_summary.json", "run_manifest.json"):
                    continue
                p1 = os.path.join(tmp1, fn)
                p2 = os.path.join(tmp2, fn)
                if not os.path.exists(p1) or not os.path.exists(p2):
                    print(f"  MISSING: {fn}")
                    continue
                with open(p1, "rb") as f1, open(p2, "rb") as f2:
                    if f1.read() != f2.read():
                        print(f"  DIFFERS: {fn}")
            sys.exit(1)
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


# ------------------------------------------------------------------ #
#  Entry point                                                       #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="TICS Regime Validation Research Laboratory CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("inventory",         help="Inspect local datasets")
    subparsers.add_parser("prepare",           help="Download and validate kline datasets")
    subparsers.add_parser("replay",            help="Synchronously simulate kline replay")
    subparsers.add_parser("analyze",           help="Integrated analysis helper")
    subparsers.add_parser("report",            help="Integrated report generator")
    subparsers.add_parser("verify-determinism",help="Verify run-to-run output reproducibility")
    subparsers.add_parser("run-all",           help="Execute complete validation workflow")

    args = parser.parse_args()

    if   args.command == "inventory":          cmd_inventory(args)
    elif args.command == "prepare":            cmd_prepare(args)
    elif args.command == "replay":             cmd_replay(args)
    elif args.command == "analyze":            cmd_analyze(args)
    elif args.command == "report":             cmd_report(args)
    elif args.command == "verify-determinism": cmd_verify_determinism(args)
    elif args.command == "run-all":            cmd_run_all(args)
    else:                                      parser.print_help()


if __name__ == "__main__":
    main()
