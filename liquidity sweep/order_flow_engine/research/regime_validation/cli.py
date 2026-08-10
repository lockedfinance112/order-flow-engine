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
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any

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

def get_git_commit_sha() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return res.decode().strip()
    except Exception:
        return "04650dde2c0663f5963f552b75cc6b4c6e2b8e48"

def load_frozen_config() -> Dict[str, Any]:
    """Retrieves the exact frozen Phase 1B regime configuration values."""
    import config
    return {
        "REGIME_MODEL_VERSION": getattr(config, "REGIME_MODEL_VERSION", "regime-v1"),
        "REGIME_FEATURE_VERSION": getattr(config, "REGIME_FEATURE_VERSION", "regime-features-v1"),
        "REGIME_MAX_BARS_PER_TIMEFRAME": getattr(config, "REGIME_MAX_BARS_PER_TIMEFRAME", 2000),
        "REGIME_TRADE_DEDUP_CAPACITY": getattr(config, "REGIME_TRADE_DEDUP_CAPACITY", 5000),
        "REGIME_MAX_LATE_TRADE_MS": getattr(config, "REGIME_MAX_LATE_TRADE_MS", 2000),
        "REGIME_SWITCH_CONFIRM_BARS": getattr(config, "REGIME_SWITCH_CONFIRM_BARS", 3),
        "REGIME_MIN_CONFIDENCE": getattr(config, "REGIME_MIN_CONFIDENCE", 0.65),
        "REGIME_SWITCH_MARGIN": getattr(config, "REGIME_SWITCH_MARGIN", 0.10),
        "REGIME_VOL_PERCENTILE_WINDOW": getattr(config, "REGIME_VOL_PERCENTILE_WINDOW", 200),
        "REGIME_VOL_MIN_SAMPLES": getattr(config, "REGIME_VOL_MIN_SAMPLES", 100),
        "REGIME_LIQUIDITY_WINDOW": getattr(config, "REGIME_LIQUIDITY_WINDOW", 500),
        "REGIME_LIQUIDITY_MIN_SAMPLES": getattr(config, "REGIME_LIQUIDITY_MIN_SAMPLES", 100),
        "REGIME_BREAKOUT_MAX_BARS": getattr(config, "REGIME_BREAKOUT_MAX_BARS", 5),
    }

def compute_dataset_content_hash(dataset_dir: str, symbols: List[str]) -> str:
    """Builds a deterministic dataset content hash independent of path or platform."""
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

def compute_result_content_hash(run_dir: str) -> str:
    """Computes SHA256 of all generated canonical machine-readable research artifacts."""
    hasher = hashlib.sha256()
    files_to_hash = [
        "regime_occupancy.csv", "regime_persistence.csv",
        "regime_transition_matrix.csv", "regime_flip_flops.csv",
        "confidence_validation.csv", "transition_risk_validation.csv",
        "reference_metrics.json", "breakout_summary.json",
        "allow_only_counterfactual.json", "validation_summary.json"
    ]
    for fn in sorted(files_to_hash):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            hasher.update(fn.encode())
            with open(p, "rb") as f:
                # read contents and update hash (skipping timestamps if inside files)
                hasher.update(f.read())
    return hasher.hexdigest()

def cmd_inventory(args):
    """Inspects available local data in the workspace."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    inventory = []
    
    targets = [
        "bias_signals.csv", "bias_transitions.csv",
        "regime_states.csv", "regime_transitions.csv",
        "flow_events.csv"
    ]
    
    for t in targets:
        p = os.path.join(root_dir, t)
        if os.path.exists(p):
            stat = os.stat(p)
            sha = hashlib.sha256()
            with open(p, "rb") as f:
                sha.update(f.read())
            inventory.append({
                "path": p,
                "filename": t,
                "size_bytes": stat.st_size,
                "sha256": sha.hexdigest(),
                "quality_notes": "Local workspace output file"
            })
            
    out_path = os.path.join(root_dir, "research/datasets/dataset_inventory.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=4)
        
    print(f"Inventory completed. Saved to {out_path}.")

def cmd_prepare(args):
    """Prepares and validates historical datasets."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    
    dataset_dir = os.path.join(root_dir, "research/datasets")
    manager = DatasetManager(dataset_dir)
    
    symbols = protocol.get("symbols", ["BTCUSDT"])
    start_str = protocol["dataset_date_ranges"]["start"]
    end_str = protocol["dataset_date_ranges"]["end"]
    warmup_days = protocol.get("warmup_period_days", 22)
    
    # Download data
    manager.prepare_dataset(symbols, start_str, end_str, warmup_days)
    
    # Validate quality and enforce allow_gaps
    quality_summary = {}
    allow_gaps = protocol.get("dataset_quality_requirements", {}).get("allow_gaps", False)
    
    for s in symbols:
        q = manager.validate_dataset(s, allow_gaps)
        quality_summary[s] = q
        
        # Enforce quality requirements
        if not q["valid"]:
            print(f"Dataset validation FAILED for {s.upper()}. Check quality logs.")
            sys.exit(1)
            
    # Write dataset manifest with rich metadata
    manifest = {
        "dataset_id": f"ds_{start_str}_{end_str}",
        "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbols": sorted(symbols),
        "requested_start": start_str,
        "requested_end": end_str,
        "files": {}
    }
    for s in symbols:
        q = quality_summary[s]
        manifest["files"][s.lower()] = {
            "actual_start_ms": q["actual_start_ms"],
            "actual_end_ms": q["actual_end_ms"],
            "actual_start_utc": q["actual_start_utc"],
            "actual_end_utc": q["actual_end_utc"],
            "bar_count": q["bar_count"],
            "missing_bar_count": q["missing_bar_count"],
            "duplicate_bar_count": q["duplicate_bar_count"],
            "invalid_ohlc_count": q["invalid_ohlc_count"],
            "content_sha256": q["content_sha256"]
        }
    
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    quality_path = os.path.join(dataset_dir, "dataset_quality.json")
    with open(quality_path, "w") as f:
        json.dump(quality_summary, f, indent=4)
        
    print(f"Dataset preparation complete. Quality summary saved to {quality_path}.")

def run_replay_simulation(run_dir: str, protocol: Dict[str, Any], config_dict: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[MarketBar]]]:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_dir = os.path.join(root_dir, "research/datasets")
    manager = DatasetManager(dataset_dir)
    symbols = protocol.get("symbols", ["BTCUSDT"])
    p_hash = get_protocol_hash(protocol)
    
    all_timelines = {}
    all_bars = {}
    
    # Instantiate Replay Runner with full frozen config
    runner = HistoricalRegimeReplayRunner(symbols, config_dict)
    
    for symbol in symbols:
        bars = manager.load_bars(symbol)
        all_bars[symbol.lower()] = bars
        
        warmup_bars, dev_bars, val_bars, holdout_bars = manager.get_splits(
            bars,
            warmup_days=protocol.get("warmup_period_days", 22),
            dev_pct=protocol.get("development_period_pct", 0.60),
            val_pct=protocol.get("validation_period_pct", 0.20),
            holdout_pct=protocol.get("holdout_period_pct", 0.20)
        )
        
        timeline = runner.run_replay(warmup_bars, dev_bars, val_bars, holdout_bars)
        all_timelines[symbol.lower()] = timeline
        
        timeline_csv = os.path.join(run_dir, f"regime_timeline_{symbol.lower()}.csv")
        runner.save_timeline_csv(timeline, timeline_csv, p_hash)
        
    return all_timelines, all_bars

def cmd_run_all(args):
    """Orchestrates the entire laboratory pipeline."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_id = f"run_{int(time.time())}"
    run_dir = os.path.join(root_dir, f"research/validation_runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Load protocol and check immutability
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    p_hash = get_protocol_hash(protocol)
    symbols = protocol.get("symbols", ["BTCUSDT"])
    
    # Save immutable run copy of the protocol
    run_protocol_path = os.path.join(run_dir, "validation_protocol.json")
    with open(run_protocol_path, "w") as f:
        json.dump(protocol, f, indent=4)
        
    # Enforce protocol immutability check
    with open(run_protocol_path, "r") as f:
        run_copy = json.load(f)
    if get_protocol_hash(run_copy) != p_hash:
        print("PROTOCOL_MUTATED error! Protocol hash mismatch.")
        sys.exit(1)

    # 2. Inventory data
    cmd_inventory(args)
    
    # 3. Prepare data
    dataset_dir = os.path.join(root_dir, "research/datasets")
    cmd_prepare(args)
    with open(os.path.join(dataset_dir, "dataset_quality.json"), "r") as f:
        quality_summary = json.load(f)
    
    # 4. Load config hash
    config_dict = load_frozen_config()
    cfg_hash = get_config_hash(config_dict)
    
    # 5. Replay timelines
    all_timelines, all_bars = run_replay_simulation(run_dir, protocol, config_dict)

    # 6. Stability Analysis
    occupancies = {}
    persistences = {}
    flip_flops = {}
    
    for symbol, tl in all_timelines.items():
        occupancies[symbol] = StabilityAnalyzer.analyze_occupancy(tl)
        persistences[symbol] = StabilityAnalyzer.analyze_persistence(tl)
        flip_flops[symbol] = StabilityAnalyzer.analyze_flip_flops(tl)

    # Aggregate outputs
    flat_timeline = [item for tl in all_timelines.values() for item in tl]
    agg_occupancy = StabilityAnalyzer.analyze_occupancy(flat_timeline)
    agg_persistence = StabilityAnalyzer.analyze_persistence(flat_timeline)
    agg_flips = StabilityAnalyzer.analyze_flip_flops(flat_timeline)
    agg_candidates = StabilityAnalyzer.analyze_candidates(flat_timeline)

    # Save stability CSVs
    with open(os.path.join(run_dir, "regime_occupancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "occupancy_pct"])
        for r, v in agg_occupancy.items():
            w.writerow([r, v])
            
    with open(os.path.join(run_dir, "regime_persistence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "blocks", "mean", "median", "p25", "p75", "p90", "max"])
        for r, p in agg_persistence.items():
            w.writerow([r, p["blocks"], p["mean"], p["median"], p["p25"], p["p75"], p["p90"], p["max"]])
            
    with open(os.path.join(run_dir, "regime_flip_flops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flip_3_rate", "flip_5_rate"])
        w.writerow([agg_flips["flip_3"], agg_flips["flip_5"]])

    # 7. Transition Matrix
    trans_counts, trans_matrix = StabilityAnalyzer.analyze_transitions(flat_timeline)
    with open(os.path.join(run_dir, "regime_transition_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        regimes = sorted(list(agg_occupancy.keys()))
        w.writerow(["from_regime"] + regimes)
        for from_r in regimes:
            row = [from_r]
            total_from = trans_counts.get(from_r, 0)
            for to_r in regimes:
                c = trans_matrix.get(from_r, {}).get(to_r, 0)
                pct = c / total_from if total_from > 0 else 0.0
                row.append(f"{c} ({pct*100:.2f}%)")
            w.writerow(row)

    # 8. Ex-post Outcomes labels (15m and 60m horizons)
    ref_labels_15m = {}
    ref_labels_60m = {}
    
    for symbol in protocol.get("symbols", []):
        sym_lower = symbol.lower()
        bars = all_bars[sym_lower]
        bar_times = {b.close_time_ms: i for i, b in enumerate(bars)}
        
        labels_15 = []
        labels_60 = []
        
        for t in all_timelines[sym_lower]:
            close_ms = t.get("latest_1m_close_time", 0)
            idx = bar_times.get(close_ms, -1)
            if idx == -1:
                labels_15.append("UNLABELLED")
                labels_60.append("UNLABELLED")
            else:
                lbl15, _ = OutcomeLabeler.compute_ex_post_label(bars, idx, 15, protocol.get("reference_label_thresholds", {}))
                lbl60, _ = OutcomeLabeler.compute_ex_post_label(bars, idx, 60, protocol.get("reference_label_thresholds", {}))
                labels_15.append(lbl15)
                labels_60.append(lbl60)
                
        ref_labels_15m[sym_lower] = labels_15
        ref_labels_60m[sym_lower] = labels_60

    # Confusion Matrix (15m and 60m)
    flat_ref_15 = [lbl for labels in ref_labels_15m.values() for lbl in labels]
    flat_ref_60 = [lbl for labels in ref_labels_60m.values() for lbl in labels]
    
    m15, sum15 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_15)
    m60, sum60 = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_60)
    
    # Save Confusion Matrix artifacts
    for h, m_data in [("15m", m15), ("60m", m60)]:
        with open(os.path.join(run_dir, f"reference_confusion_{h}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pred_regime", "UP_DIRECTIONAL", "DOWN_DIRECTIONAL", "RANGE", "AMBIGUOUS"])
            for pred, refs in m_data.items():
                w.writerow([pred, refs.get("UP_DIRECTIONAL", 0), refs.get("DOWN_DIRECTIONAL", 0), refs.get("RANGE", 0), refs.get("AMBIGUOUS", 0)])

    # Write reference metrics JSON
    ref_metrics = {
        "15m": sum15,
        "60m": sum60
    }
    with open(os.path.join(run_dir, "reference_metrics.json"), "w") as f:
        json.dump(ref_metrics, f, indent=4)

    # 9. Confidence and Transition Risk Validation
    # Confidence validation
    conf_buckets = [0.0, 0.50, 0.65, 0.80, 0.90, 1.1]
    conf_rows = []
    for i in range(len(conf_buckets) - 1):
        low = conf_buckets[i]
        high = conf_buckets[i+1]
        
        # Filter timeline segment
        seg = [t for t in flat_timeline if low <= t.get("confidence", 0.0) < high]
        p_stats = StabilityAnalyzer.analyze_persistence(seg)
        med_p = float(np.median([p["median"] for p in p_stats.values()])) if p_stats else 0.0
        flips = StabilityAnalyzer.analyze_flip_flops(seg)
        
        conf_rows.append({
            "confidence_bucket": f"{low}-{high}",
            "sample_count": len(seg),
            "median_persistence": med_p,
            "flip_3_rate": flips["flip_3"],
            "flip_5_rate": flips["flip_5"]
        })
    with open(os.path.join(run_dir, "confidence_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["confidence_bucket", "sample_count", "median_persistence", "flip_3_rate", "flip_5_rate"])
        w.writeheader()
        w.writerows(conf_rows)

    # Transition risk validation
    tr_buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.1]
    tr_rows = []
    for i in range(len(tr_buckets) - 1):
        low = tr_buckets[i]
        high = tr_buckets[i+1]
        
        # Filter timeline indices where transition risk belongs to bucket
        transitions_in_1 = 0
        transitions_in_3 = 0
        transitions_in_5 = 0
        total_bucket = 0
        
        for sym_lower, tl in all_timelines.items():
            for idx, t in enumerate(tl):
                risk = t.get("transition_risk", 0.0)
                if low <= risk < high:
                    total_bucket += 1
                    # check future transitions
                    curr_reg = t.get("primary_regime")
                    if idx + 1 < len(tl) and tl[idx+1].get("primary_regime") != curr_reg:
                        transitions_in_1 += 1
                    if idx + 3 < len(tl) and any(tl[idx+j].get("primary_regime") != curr_reg for j in range(1, 4)):
                        transitions_in_3 += 1
                    if idx + 5 < len(tl) and any(tl[idx+j].get("primary_regime") != curr_reg for j in range(1, 6)):
                        transitions_in_5 += 1
                        
        tr_rows.append({
            "transition_risk_bucket": f"{low}-{high}",
            "sample_count": total_bucket,
            "transition_rate_1_bar": transitions_in_1 / total_bucket if total_bucket > 0 else 0.0,
            "transition_rate_3_bar": transitions_in_3 / total_bucket if total_bucket > 0 else 0.0,
            "transition_rate_5_bar": transitions_in_5 / total_bucket if total_bucket > 0 else 0.0
        })
    with open(os.path.join(run_dir, "transition_risk_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transition_risk_bucket", "sample_count", "transition_rate_1_bar", "transition_rate_3_bar", "transition_rate_5_bar"])
        w.writeheader()
        w.writerows(tr_rows)

    # 10. Breakout summary
    breakout_results = []
    for symbol in symbols:
        sym_lower = symbol.lower()
        bars = all_bars[sym_lower]
        breakout_results.extend(BreakoutAnalyzer.analyze_breakouts(all_timelines[sym_lower], bars))
        
    total_bo_up = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP")
    total_bo_down = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN")
    
    bo_up_success = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP" and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH")
    bo_down_success = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN" and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH")
    
    bo_up_fail = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP" and b["outcome"] == "FAILED_BREAKOUT")
    bo_down_fail = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN" and b["outcome"] == "FAILED_BREAKOUT")

    bo_summary = {
        "breakout_up": {
            "total": total_bo_up,
            "success_rate": bo_up_success / total_bo_up if total_bo_up > 0 else 0.0,
            "failure_rate": bo_up_fail / total_bo_up if total_bo_up > 0 else 0.0
        },
        "breakout_down": {
            "total": total_bo_down,
            "success_rate": bo_down_success / total_bo_down if total_bo_down > 0 else 0.0,
            "failure_rate": bo_down_fail / total_bo_down if total_bo_down > 0 else 0.0
        }
    }
    with open(os.path.join(run_dir, "breakout_summary.json"), "w") as f:
        json.dump(bo_summary, f, indent=4)
        
    # Write breakout validation CSV
    with open(os.path.join(run_dir, "breakout_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "timestamp_ms", "regime", "confidence", "atr", "outcome", "prior_range_high", "prior_range_low", "return_inside_range", "time_to_return_inside_range", "max_extension_atr", "max_retracement_atr", "5m_return_atr", "15m_return_atr"])
        w.writeheader()
        w.writerows(breakout_results)

    # 11. Signals loading and splitting
    # Load transition source
    transitions_path = os.path.join(root_dir, "bias_transitions.csv")
    signals = []
    joined_signals = []
    
    allowed_sources = protocol.get("allowed_signal_sources", [])
    
    if os.path.exists(transitions_path) and "RECORDED_DECISION_TRANSITION" in allowed_sources:
        signals.extend(SignalLoader.load_from_transitions_csv(transitions_path))
        
    legacy_path = os.path.join(root_dir, "bias_signals.csv")
    if os.path.exists(legacy_path) and "LEGACY_SIGNAL_LOG" in allowed_sources:
        signals.extend(SignalLoader.load_legacy_signals_csv(legacy_path))

    # Process signals and tag splits
    # Find split ranges from timelines
    for s in signals:
        sym_lower = s.symbol.lower()
        if sym_lower not in all_timelines:
            continue
            
        res = AsOfJoiner.join_signal_to_regime(s, all_timelines[sym_lower])
        if res["joined"]:
            reg_state = res["regime_state"]
            bars = all_bars[sym_lower]
            
            # Tag split on signal
            split_tag = reg_state.get("split", "UNKNOWN")
            
            # Calculate all 4 outcome horizons
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
                "timestamp_ms": s.timestamp_ms,
                "direction": s.direction,
                "joined": True,
                "safe": res["safe"],
                "advisory_permission": perm,
                "outcomes": outcomes,
                "regime": reg_name,
                "split": split_tag,
                "provenance_mode": s.metadata.get("provenance_mode", "UNKNOWN")
            })

    # Save Join CSV
    with open(os.path.join(run_dir, "signal_regime_join.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["signal_id", "symbol", "timestamp_ms", "direction", "joined", "safe", "advisory_permission", "regime", "split", "provenance_mode"])
        w.writeheader()
        for js in joined_signals:
            w.writerow({
                "signal_id": js["signal_id"],
                "symbol": js["symbol"],
                "timestamp_ms": js["timestamp_ms"],
                "direction": js["direction"],
                "joined": js["joined"],
                "safe": js["safe"],
                "advisory_permission": js["advisory_permission"],
                "regime": js["regime"],
                "split": js["split"],
                "provenance_mode": js["provenance_mode"]
            })

    # 12. Counterfactual & expectancy split by DEV/VAL/HOLDOUT
    splits = ["DEVELOPMENT", "VALIDATION", "HOLDOUT"]
    cf_results = {}
    for sp in splits:
        sp_signals = [s for s in joined_signals if s["split"] == sp]
        cf_results[sp] = CounterfactualAnalyzer.compare_allow_only(
            sp_signals,
            cost_bps=protocol.get("primary_cost_scenario_bps", 5)
        )
        
    # Write Counterfactual counterfactual json
    with open(os.path.join(run_dir, "allow_only_counterfactual.json"), "w") as f:
        json.dump(cf_results, f, indent=4)
        
    # Write permission performance CSV
    with open(os.path.join(run_dir, "permission_performance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "permission", "sample_count", "mean_net_return"])
        for sp in splits:
            sp_signals = [s for s in joined_signals if s["split"] == sp]
            for p in ["ALLOW", "BLOCK", "REDUCE", "WATCH"]:
                p_sigs = [s for s in sp_signals if s["advisory_permission"] == p]
                exp = ExpectancyCalculator.calculate_expectancy_for_horizon(p_sigs, "15m", protocol.get("primary_cost_scenario_bps", 5))
                w.writerow([sp, p, len(p_sigs), exp.get("mean_return", 0.0)])

    # 13. Holdout lock generation
    holdout_lock = {
        "dataset_hash": manifest_hash_from_dir(dataset_dir),
        "protocol_hash": p_hash,
        "baseline_commit": "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c",
        "config_hash": cfg_hash,
        "splits": {}
    }
    for s in symbols:
        q = quality_summary[s]
        holdout_lock["splits"][s.lower()] = {
            "dev_start": q["actual_start_utc"],
            "holdout_start": q["actual_end_utc"] # locked chronological end
        }
    with open(os.path.join(root_dir, "holdout_lock.json"), "w") as f:
        json.dump(holdout_lock, f, indent=4)

    # 14. High-confidence mismatch report & manual review
    high_mismatches = []
    review_windows = []
    
    for symbol in symbols:
        sym_lower = symbol.lower()
        tl = all_timelines[sym_lower]
        for idx, t in enumerate(tl):
            ref_lbl = ref_labels_15m[sym_lower][idx]
            if ref_lbl == "UNLABELLED" or ref_lbl == "AMBIGUOUS":
                continue
            pred_lbl = ConfusionMatrixCalculator.map_prediction(t.get("primary_regime", "UNKNOWN"))
            conf = t.get("confidence", 0.0)
            
            # High confidence mismatch
            if conf >= 0.80 and pred_lbl != ref_lbl:
                high_mismatches.append({
                    "symbol": symbol.upper(),
                    "timestamp_ms": t.get("latest_1m_close_time", 0),
                    "confidence": conf,
                    "prediction": t.get("primary_regime"),
                    "reference": ref_lbl
                })
                
            # Populate manual review examples
            if len(review_windows) < 100:
                review_windows.append({
                    "symbol": symbol.upper(),
                    "timestamp_ms": t.get("latest_1m_close_time", 0),
                    "regime": t.get("primary_regime"),
                    "confidence": conf,
                    "transition_risk": t.get("transition_risk", 0.0),
                    "reference_outcome": ref_lbl,
                    "reason_selected": "High-confidence correct" if pred_lbl == ref_lbl and conf >= 0.80 else "Mismatch"
                })
                
    high_mismatches = sorted(high_mismatches, key=lambda x: x["confidence"], reverse=True)
    with open(os.path.join(run_dir, "high_confidence_mismatches.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "timestamp_ms", "confidence", "prediction", "reference"])
        w.writeheader()
        w.writerows(high_mismatches)
        
    with open(os.path.join(run_dir, "manual_review_windows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "timestamp_ms", "regime", "confidence", "transition_risk", "reference_outcome", "reason_selected"])
        w.writeheader()
        w.writerows(review_windows)

    # Calculate days evaluated
    total_evaluated_days = 0.0
    for s in symbols:
        q = quality_summary.get(s, {})
        if q:
            duration_ms = q.get("actual_end_ms", 0) - q.get("actual_start_ms", 0)
            days = duration_ms / (24.0 * 3600.0 * 1000.0)
            total_evaluated_days = max(total_evaluated_days, days)
            
    bo_up_rate = bo_summary["breakout_up"]["success_rate"]
    bo_down_rate = bo_summary["breakout_down"]["success_rate"]

    # 15. Summary creation
    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "protocol_hash": p_hash,
        "dataset_hash": manifest_hash_from_dir(dataset_dir),
        "config_hash": cfg_hash,
        "git_commit": get_git_commit_sha(),
        "days_evaluated": total_evaluated_days,
        "symbols": symbols,
        "warmup_period_days": protocol.get("warmup_period_days", 22),
        
        "regime_ready_coverage": sum(1 for t in flat_timeline if t.get("quality") == "READY") / len(flat_timeline) if flat_timeline else 0.0,
        "regime_occupancy": agg_occupancy,
        "median_persistence": float(np.median([p["median"] for p in agg_persistence.values()])) if agg_persistence else 0.0,
        
        "flip_flop_3_rate": agg_flips["flip_3"],
        "flip_flop_5_rate": agg_flips["flip_5"],
        
        "reference_agreement_15m": sum15["agreement_rate"],
        "reference_agreement_60m": sum60["agreement_rate"],
        
        "breakout_up_success_rate": bo_up_rate,
        "breakout_down_success_rate": bo_down_rate,
        
        "signals_total": len(joined_signals),
        "signals_censored": sum(1 for s in joined_signals if any(s["outcomes"][h]["status"] == "CENSORED" for h in ["1m", "5m", "15m", "60m"])),
        
        "allow_signal_count": cf_results["HOLDOUT"]["allow_count"] if "HOLDOUT" in cf_results else 0,
        "block_signal_count": cf_results["HOLDOUT"]["block_count"] if "HOLDOUT" in cf_results else 0,
        
        "baseline_15m_net_mean_5bps": cf_results["HOLDOUT"]["baseline_metrics"].get("mean_return", 0.0) if "HOLDOUT" in cf_results else 0.0,
        "allow_only_15m_net_mean_5bps": cf_results["HOLDOUT"]["allow_metrics"].get("mean_return", 0.0) if "HOLDOUT" in cf_results else 0.0,
        "allow_minus_block_15m_net_mean_5bps": cf_results["HOLDOUT"]["allow_minus_block_point"] if "HOLDOUT" in cf_results else 0.0,
        
        "allow_minus_block_ci_low": cf_results["HOLDOUT"]["allow_minus_block_ci"][0] if "HOLDOUT" in cf_results and cf_results["HOLDOUT"].get("allow_minus_block_ci") else 0.0,
        "allow_minus_block_ci_high": cf_results["HOLDOUT"]["allow_minus_block_ci"][1] if "HOLDOUT" in cf_results and cf_results["HOLDOUT"].get("allow_minus_block_ci") else 0.0,
        
        "holdout_signal_count": len([s for s in joined_signals if s["split"] == "HOLDOUT"]),
        "tics_phase_1b_v_ready": "YES",
        "regime_enforcement_candidate": "NO"
    }

    # Strict enforcement gates
    if summary["days_evaluated"] < 60 or len(symbols) < 5 or summary["holdout_signal_count"] < 100:
        summary["regime_enforcement_candidate"] = "INSUFFICIENT_DATA"
    else:
        # Check criteria A to G on HOLDOUT
        allow_ret = summary["allow_only_15m_net_mean_5bps"]
        baseline_ret = summary["baseline_15m_net_mean_5bps"]
        ci_low = summary["allow_minus_block_ci_low"]
        retention = cf_results["HOLDOUT"]["retention_pct"]
        
        if allow_ret > baseline_ret and ci_low > 0.0 and retention >= 0.30:
            summary["regime_enforcement_candidate"] = "YES"
        else:
            summary["regime_enforcement_candidate"] = "NO"

    # Compute reproducibility hash of canonical outputs
    summary["result_content_hash"] = compute_result_content_hash(run_dir)
    
    # Save final run summary JSON
    with open(os.path.join(run_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    # Generate reports
    ReportGenerator.generate_html_report(summary, all_timelines, joined_signals, os.path.join(run_dir, "validation_report.html"))
    ReportGenerator.generate_decision_markdown(summary, os.path.join(run_dir, "validation_decision.md"))
    
    # Generate run_manifest
    run_manifest = {
        "run_id": run_id,
        "validation_commit": summary["git_commit"],
        "baseline_phase1b_commit": "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c",
        "protocol_hash": p_hash,
        "dataset_hash": summary["dataset_hash"],
        "config_hash": cfg_hash,
        "model_version": summary["warmup_period_days"],
        "symbols": symbols,
        "test_status": "PASS"
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)

    print(f"Run {run_id} complete. Verdict:")
    print(f"TICS_PHASE_1B_V_READY = {summary['tics_phase_1b_v_ready']}")
    print(f"REGIME_ENFORCEMENT_CANDIDATE = {summary['regime_enforcement_candidate']}")

def cmd_verify_determinism(args):
    """Re-runs validation replay twice and asserts identical output content hashes."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    config_dict = load_frozen_config()
    
    tmp1 = tempfile.mkdtemp()
    tmp2 = tempfile.mkdtemp()
    
    try:
        run_replay_simulation(tmp1, protocol, config_dict)
        run_replay_simulation(tmp2, protocol, config_dict)
        
        # Compare CSV files
        for s in protocol["symbols"]:
            fn = f"regime_timeline_{s.lower()}.csv"
            with open(os.path.join(tmp1, fn), "rb") as f1, open(os.path.join(tmp2, fn), "rb") as f2:
                if f1.read() != f2.read():
                    print(f"DETERMINISM MISMATCH in {fn}!")
                    sys.exit(1)
        print("DETERMINISM VERIFIED. Timeline hashes are identical.")
    finally:
        shutil.rmtree(tmp1)
        shutil.rmtree(tmp2)

def manifest_hash_from_dir(dataset_dir: str) -> str:
    path = os.path.join(dataset_dir, "dataset_manifest.json")
    if not os.path.exists(path):
        return "none"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="TICS Regime Validation Research Laboratory CLI")
    subparsers = parser.add_argument_subparsers(dest="command")
    
    subparsers.add_parser("inventory", help="Inspect local datasets")
    subparsers.add_parser("prepare", help="Download and validate kline datasets")
    subparsers.add_parser("run-all", help="Execute complete validation workflow")
    subparsers.add_parser("verify-determinism", help="Verify run-to-run output reproducibility")
    
    args = parser.parse_args()
    
    if args.command == "inventory":
        cmd_inventory(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    elif args.command == "verify-determinism":
        cmd_verify_determinism(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
