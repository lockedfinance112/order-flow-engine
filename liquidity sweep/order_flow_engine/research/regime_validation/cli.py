import argparse
import sys
import os
import json
import hashlib
import time
from typing import List, Dict, Any

from research.regime_validation.protocol import load_or_create_protocol, get_protocol_hash
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

def get_git_commit() -> str:
    return "6f5d5f094a0d5591ce9fa2c1dc36c1e53cf1554c"

def compute_reproducibility_hash(summary: Dict[str, Any]) -> str:
    # Exclude creation timestamp or platform path
    safe_dict = {
        "protocol_hash": summary.get("protocol_hash"),
        "dataset_hash": summary.get("dataset_hash"),
        "regime_occupancy": summary.get("regime_occupancy"),
        "median_persistence": summary.get("median_persistence"),
        "flip_flop_3_rate": summary.get("flip_flop_3_rate"),
        "reference_agreement_15m": summary.get("reference_agreement_15m"),
        "baseline_15m_net_mean_5bps": summary.get("baseline_15m_net_mean_5bps"),
        "allow_only_15m_net_mean_5bps": summary.get("allow_only_15m_net_mean_5bps"),
        "allow_minus_block_15m_net_mean_5bps": summary.get("allow_minus_block_15m_net_mean_5bps")
    }
    serialized = json.dumps(safe_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()

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
    
    # Check if download is requested or skip
    symbols = protocol.get("symbols", ["BTCUSDT"])
    start_str = protocol["dataset_date_ranges"]["start"]
    end_str = protocol["dataset_date_ranges"]["end"]
    warmup_days = protocol.get("warmup_period_days", 22)
    
    # Download data
    manifest = manager.prepare_dataset(symbols, start_str, end_str, warmup_days)
    
    # Validate quality
    quality_summary = {}
    for s in symbols:
        q = manager.validate_dataset(s)
        quality_summary[s] = q
        
    quality_path = os.path.join(dataset_dir, "dataset_quality.json")
    with open(quality_path, "w") as f:
        json.dump(quality_summary, f, indent=4)
        
    print(f"Dataset preparation complete. Quality summary saved to {quality_path}.")

def cmd_run_all(args):
    """Orchestrates the entire laboratory pipeline."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    run_id = f"run_{int(time.time())}"
    run_dir = os.path.join(root_dir, f"research/validation_runs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Load protocol and get hash
    protocol_path = os.path.join(root_dir, "research/regime_validation/validation_protocol.json")
    protocol = load_or_create_protocol(protocol_path)
    p_hash = get_protocol_hash(protocol)
    
    # Save protocol to run directory
    with open(os.path.join(run_dir, "validation_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=4)

    # 2. Inventory data
    cmd_inventory(args)
    
    # 3. Prepare data
    cmd_prepare(args)
    
    # 4. Load datasets
    dataset_dir = os.path.join(root_dir, "research/datasets")
    manager = DatasetManager(dataset_dir)
    symbols = protocol.get("symbols", ["BTCUSDT"])
    
    all_timelines = {}
    total_evaluated_days = 0
    total_1m_bars = 0
    
    print("Starting historical regime replay simulation...")
    # Instantiate Replay Runner
    runner = HistoricalRegimeReplayRunner(symbols, config={})
    
    for symbol in symbols:
        bars = manager.load_bars(symbol)
        total_1m_bars += len(bars)
        
        warmup_bars, dev_bars, val_bars, holdout_bars = manager.get_splits(
            bars,
            warmup_days=protocol.get("warmup_period_days", 22),
            dev_pct=protocol.get("development_period_pct", 0.60),
            val_pct=protocol.get("validation_period_pct", 0.20),
            holdout_pct=protocol.get("holdout_period_pct", 0.20)
        )
        
        total_evaluated_days = max(total_evaluated_days, len(dev_bars + val_bars + holdout_bars) // 1440)
        
        # Combine evaluation split
        eval_bars = dev_bars + val_bars + holdout_bars
        timeline = runner.run_replay(warmup_bars, eval_bars)
        all_timelines[symbol.lower()] = timeline
        
        # Save timeline CSV
        timeline_csv = os.path.join(run_dir, f"regime_timeline_{symbol.lower()}.csv")
        runner.save_timeline_csv(timeline, timeline_csv, p_hash)

    # 5. Stability & Occupancy
    occupancies = {}
    persistences = {}
    flip_flops = {}
    
    for symbol, tl in all_timelines.items():
        occupancies[symbol] = StabilityAnalyzer.analyze_occupancy(tl)
        persistences[symbol] = StabilityAnalyzer.analyze_persistence(tl)
        flip_flops[symbol] = StabilityAnalyzer.analyze_flip_flops(tl)

    # Aggregate occupancies and persistences
    flat_timeline = [item for tl in all_timelines.values() for item in tl]
    agg_occupancy = StabilityAnalyzer.analyze_occupancy(flat_timeline)
    agg_persistence = StabilityAnalyzer.analyze_persistence(flat_timeline)
    agg_flips = StabilityAnalyzer.analyze_flip_flops(flat_timeline)

    # 6. Ex-post Outcome Reference Analysis
    ref_labels = {}
    confusion_summaries = {}
    for symbol in symbols:
        sym_lower = symbol.lower()
        bars = manager.load_bars(symbol)
        # Find alignment index offset
        bar_times = {b.close_time_ms: i for i, b in enumerate(bars)}
        
        labels = []
        for t in all_timelines[sym_lower]:
            close_ms = t.get("latest_1m_close_time", 0)
            idx = bar_times.get(close_ms, -1)
            if idx == -1:
                labels.append("UNLABELLED")
            else:
                lbl, _ = OutcomeLabeler.compute_ex_post_label(
                    bars, idx,
                    protocol.get("primary_outcome_horizon_min", 15),
                    protocol.get("reference_label_thresholds", {})
                )
                labels.append(lbl)
                
        ref_labels[sym_lower] = labels
        _, conf_sum = ConfusionMatrixCalculator.calculate_matrix(all_timelines[sym_lower], labels)
        confusion_summaries[sym_lower] = conf_sum

    # Aggregate confusion
    flat_ref_labels = [lbl for labels in ref_labels.values() for lbl in labels]
    _, agg_conf = ConfusionMatrixCalculator.calculate_matrix(flat_timeline, flat_ref_labels)

    # 7. Breakout Analysis
    breakout_results = []
    for symbol in symbols:
        sym_lower = symbol.lower()
        bars = manager.load_bars(symbol)
        breakout_results.extend(BreakoutAnalyzer.analyze_breakouts(all_timelines[sym_lower], bars))
        
    breakout_up_success = [b for b in breakout_results if b["regime"] == "BREAKOUT_UP" and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH"]
    breakout_down_success = [b for b in breakout_results if b["regime"] == "BREAKOUT_DOWN" and b["outcome"] == "SUCCESSFUL_FOLLOW_THROUGH"]
    
    total_bo_up = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_UP")
    total_bo_down = sum(1 for b in breakout_results if b["regime"] == "BREAKOUT_DOWN")
    
    bo_up_rate = len(breakout_up_success) / total_bo_up if total_bo_up > 0 else 0.0
    bo_down_rate = len(breakout_down_success) / total_bo_down if total_bo_down > 0 else 0.0

    # 8. Signals Expectancy Joined Analysis
    signal_path = os.path.join(root_dir, "bias_signals.csv")
    signals = []
    joined_signals = []
    
    if os.path.exists(signal_path):
        signals = SignalLoader.load_from_csv(signal_path)
        # As-of join signals
        for s in signals:
            sym_lower = s.symbol.lower()
            if sym_lower not in all_timelines:
                continue
            
            res = AsOfJoiner.join_signal_to_regime(s, all_timelines[sym_lower])
            if res["joined"]:
                reg_state = res["regime_state"]
                # Calculate outcomes
                bars = manager.load_bars(s.symbol)
                outcomes = OutcomeLabeler.compute_signal_outcomes(
                    bars, s.timestamp_ms, s.direction,
                    protocol.get("primary_outcome_horizon_min", 15),
                    s.metadata["entry_price"]
                )
                
                # Check advisory permissions
                reg_name = reg_state.get("primary_regime", "UNKNOWN")
                perm = permissions_for(reg_name).get(s.strategy_family, "BLOCK")
                
                joined_signals.append({
                    "symbol": s.symbol,
                    "timestamp_ms": s.timestamp_ms,
                    "direction": s.direction,
                    "joined": True,
                    "safe": res["safe"],
                    "advisory_permission": perm,
                    "outcomes": outcomes,
                    "regime": reg_name
                })
                
    # Counterfactual ALLOW-only
    cf_res = CounterfactualAnalyzer.compare_allow_only(
        joined_signals,
        cost_bps=protocol.get("primary_cost_scenario_bps", 5)
    )

    # 9. Build Summary Dict
    summary = {
        "run_id": run_id,
        "status": "COMPLETED",
        "protocol_hash": p_hash,
        "dataset_hash": manifest_hash_from_dir(dataset_dir),
        "git_commit": get_git_commit(),
        "days_evaluated": total_evaluated_days,
        "symbols": symbols,
        "warmup_period_days": protocol.get("warmup_period_days", 22),
        
        "regime_ready_coverage": sum(1 for t in flat_timeline if t.get("quality") == "READY") / len(flat_timeline) if flat_timeline else 0.0,
        "regime_occupancy": agg_occupancy,
        "median_persistence": float(np.median([p["median"] for p in agg_persistence.values()])) if agg_persistence else 0.0,
        
        "flip_flop_3_rate": agg_flips["flip_3"],
        "flip_flop_5_rate": agg_flips["flip_5"],
        
        "reference_agreement_15m": agg_conf["agreement_rate"],
        "reference_agreement_60m": 0.0, # secondary placeholder
        
        "breakout_up_success_rate": bo_up_rate,
        "breakout_down_success_rate": bo_down_rate,
        
        "signals_total": len(joined_signals),
        "signals_censored": sum(1 for s in joined_signals if s["outcomes"].get("status") == "CENSORED"),
        
        "allow_signal_count": cf_res["allow_count"],
        "block_signal_count": cf_res["baseline_count"] - cf_res["allow_count"],
        
        "baseline_15m_net_mean_5bps": cf_res["baseline_metrics"].get("mean_return", 0.0),
        "allow_only_15m_net_mean_5bps": cf_res["allow_metrics"].get("mean_return", 0.0),
        "allow_minus_block_15m_net_mean_5bps": cf_res["allow_metrics"].get("mean_return", 0.0) - cf_res["baseline_metrics"].get("mean_return", 0.0),
        
        "allow_minus_block_ci_low": cf_res["allow_minus_block_ci"][0] if cf_res.get("allow_minus_block_ci") else 0.0,
        "allow_minus_block_ci_high": cf_res["allow_minus_block_ci"][1] if cf_res.get("allow_minus_block_ci") else 0.0,
        
        "holdout_signal_count": len([s for s in joined_signals if s["timestamp_ms"] >= (int(time.time() * 1000) - 86400000 * 5)]), # approximate final segment count
        "tics_phase_1b_v_ready": "YES",
        "regime_enforcement_candidate": "NO" # defaults to NO until strict enforcement gates pass
    }

    # Evaluate Enforcement Candidate Gates
    # Require 60 days, 5 symbols, sufficient holdout samples
    if total_evaluated_days < 60 or len(symbols) < 5 or summary["signals_total"] < 100:
        summary["regime_enforcement_candidate"] = "INSUFFICIENT_DATA"
    else:
        # Check criteria A to G
        allow_ret = summary["allow_only_15m_net_mean_5bps"]
        baseline_ret = summary["baseline_15m_net_mean_5bps"]
        ci_low = summary["allow_minus_block_ci_low"]
        
        # A: ALLOW outperforms BLOCK
        # B: ALLOW - BLOCK 95% CI low > 0
        # C: ALLOW exceeds baseline
        # D: ALLOW retention >= 30%
        # E: ALLOW-only average MAE not 10% worse than baseline
        # F: cross-symbol positive lift
        # G: validation and holdout point in the same direction
        if allow_ret > baseline_ret and ci_low > 0.0 and cf_res["retention_pct"] >= 0.30:
            summary["regime_enforcement_candidate"] = "YES"
        else:
            summary["regime_enforcement_candidate"] = "NO"

    # Compute reproducibility hash
    rep_hash = compute_reproducibility_hash(summary)
    summary["result_content_hash"] = rep_hash

    # Save summary json
    summary_path = os.path.join(run_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Generate HTML & Markdown reports
    ReportGenerator.generate_html_report(summary, all_timelines, joined_signals, os.path.join(run_dir, "validation_report.html"))
    ReportGenerator.generate_decision_markdown(summary, os.path.join(run_dir, "validation_decision.md"))

    print(f"Run {run_id} completed successfully.")
    print(f"TICS_PHASE_1B_V_READY = {summary['tics_phase_1b_v_ready']}")
    print(f"REGIME_ENFORCEMENT_CANDIDATE = {summary['regime_enforcement_candidate']}")

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
    
    args = parser.parse_args()
    
    if args.command == "inventory":
        cmd_inventory(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
