import os
import json
import csv
from datetime import datetime
from research.phase2b.data_loader import load_canonical_signals, split_dataset
from research.phase2b.statistics import calculate_stats, bootstrap_ci, float_val
from research.phase2b.hypothesis_registry import HYPOTHESES
from research.phase2b.counterfactual_engine import CounterfactualEngine

def pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.4f}%"

def run_research():
    base_dir = os.path.dirname(__file__)
    signals_csv = os.path.join(os.path.dirname(base_dir), "phase2", "signal_outcomes_v2.csv")
    output_md = os.path.join(base_dir, "phase2b_report.md")
    output_json = os.path.join(base_dir, "phase2b_results.json")
    
    signals = load_canonical_signals(signals_csv)
    n = len(signals)
    
    # Apply sample size gates
    if n < 20:
        status = "INSTRUMENTATION VALIDATION ONLY"
    elif n < 50:
        status = "EARLY EXPLORATORY"
    elif n < 100:
        status = "HYPOTHESIS GENERATION"
    elif n < 200:
        status = "PRELIMINARY COUNTERFACTUAL RESEARCH"
    else:
        status = "PHASE 2B FULL RESEARCH ELIGIBLE"

    # Compute LONG/SHORT baseline sizes
    long_sigs = [s for s in signals if s.get("direction") == "LONG"]
    short_sigs = [s for s in signals if s.get("direction") == "SHORT"]
    
    completed_15m = sum(1 for s in signals if s.get("horizon_15m_status") == "CAPTURED")
    interrupted = sum(1 for s in signals if s.get("completion_status") == "INTERRUPTED")
    usable_excursions = sum(1 for s in signals if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE")

    # Splits
    dev, val, holdout = split_dataset(signals)
    splits_status = "HOLDOUT SPLITS ACTIVATED" if n >= 100 else "HOLDOUT NOT ACTIVATED — INSUFFICIENT SAMPLE"

    results = {
        "canonical_signals_count": n,
        "long_count": len(long_sigs),
        "short_count": len(short_sigs),
        "completed_15m": completed_15m,
        "interrupted_signals": interrupted,
        "usable_excursions": usable_excursions,
        "sample_status": status,
        "splits": {
            "dev_count": len(dev),
            "val_count": len(val),
            "holdout_count": len(holdout),
            "status": splits_status
        },
        "hypotheses": {}
    }

    # Evaluate baseline
    def get_outcomes(sigs):
        ret_15m = []
        mfes, maes = [], []
        for s in sigs:
            if s.get("horizon_15m_status") == "CAPTURED":
                v = float_val(s.get("return_15m_pct"))
                if v is not None: ret_15m.append(v)
            if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE":
                f = float_val(s.get("max_favorable_pct"))
                a = float_val(s.get("max_adverse_pct"))
                if f is not None: mfes.append(f)
                if a is not None: maes.append(a)
        return ret_15m, mfes, maes

    b_ret, b_mfe, b_mae = get_outcomes(signals)
    baseline_stats = calculate_stats(b_ret, b_mfe, b_mae)
    results["baseline"] = baseline_stats

    # CVD alignment alignment baseline checks
    engine = CounterfactualEngine(signals)
    for name, filter_func in HYPOTHESES.items():
        try:
            results["hypotheses"][name] = engine.evaluate_hypothesis(filter_func)
        except Exception as e:
            results["hypotheses"][name] = {"error": str(e)}

    # Generate MD Report
    report_md = f"""# PHASE 2B COUNTERFACTUAL SIGNAL RESEARCH

**GROSS DIRECTIONAL RETURNS**
*Note: Returns are calculated gross of fees, slippage, execution latency, and funding rates.*

## Dataset Status & Exclusions
- Total Canonical Signals: {n} (LONG={len(long_sigs)} | SHORT={len(short_sigs)})
- Completed 15m outcomes: {completed_15m}
- Interrupted signals: {interrupted}
- Usable MFE/MAE excursion tracks: {usable_excursions}

## Sample-Size Eligibility
- Current Status: **{status}**
- Research Splits Status: **{splits_status}**
- Split sizes: DEV={len(dev)} | VAL={len(val)} | HOLDOUT={len(holdout)}

## Baseline Stats
- Win Rate (15m): {pct(baseline_stats['win_rate'])}
- Expectancy (15m): {baseline_stats['expectancy']:.6f}
- Profit Factor: {baseline_stats['profit_factor']:.4f}
- Win/Loss Ratio: {baseline_stats['win_loss_ratio']:.4f}

## Counterfactual Hypotheses Performance
"""
    for name, h_res in results["hypotheses"].items():
        if "error" in h_res:
            report_md += f"### {name}\n- Error: {h_res['error']}\n\n"
            continue
        c_stats = h_res["candidate"]
        report_md += f"""### {name}
- **Retention**: {pct(h_res['retention_pct'])} (N={h_res['candidate_count']} | Removed={h_res['removed_count']})
- **Sacrificed Winners**: {h_res['winners_sacrificed']} | **Avoided Losers**: {h_res['losers_avoided']}
- **Candidate Win Rate**: {pct(c_stats['win_rate'])} (vs baseline {pct(baseline_stats['win_rate'])})
- **Candidate Expectancy**: {c_stats['expectancy']:.6f} (vs baseline {baseline_stats['expectancy']:.6f})
- **Candidate Profit Factor**: {c_stats['profit_factor']:.4f} (vs baseline {baseline_stats['profit_factor']:.4f})

"""

    report_md += """
## Walk-forward Stability
- Walk-forward validation not yet activated due to insufficient sample.

## Bootstrap Uncertainty
- Bootstrap confidence intervals not yet activated due to insufficient sample.

## Holdout Status
- Holdout remains untouched and locked.

## Phase 2C Promotion Candidate Selection
- **Status: INSUFFICIENT DATA FOR STRATEGY SELECTION**
- Continue collecting canonical data under the frozen Phase 1 strategy.

NO PRODUCTION STRATEGY CHANGES HAVE BEEN APPLIED.
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- Phase 2B Research Executed Successfully ---")
    print(f"Sample Status: {status}")
    print(f"Total Canonical Signals: {n}")
    print(f"LONG Count: {len(long_sigs)} | SHORT Count: {len(short_sigs)}")
    print(f"Results written to {output_md}")

if __name__ == "__main__":
    run_research()
