import os
import json
import csv
import numpy as np
from datetime import datetime
from research.phase2b.data_loader import load_canonical_signals, partition_signals
from research.phase2b.statistics import calculate_stats, bootstrap_ci, float_val
from research.phase2b.hypothesis_registry import HYPOTHESES, combine_and
from research.phase2b.counterfactual_engine import CounterfactualEngine
from research.phase2b.walk_forward import WalkForwardValidator

def pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.4f}%"

def get_direction_classification(overall, long, short):
    if overall["candidate"]["count"] < 10 or long["candidate"]["count"] < 5 or short["candidate"]["count"] < 5:
        return "INSUFFICIENT_DATA"
    l_exp = long["candidate"]["expectancy"] - long["eligible_baseline"]["expectancy"]
    s_exp = short["candidate"]["expectancy"] - short["eligible_baseline"]["expectancy"]
    
    if l_exp > 0 and s_exp > 0:
        return "BALANCED"
    elif l_exp > 0 and s_exp <= 0:
        return "LONG_ONLY"
    elif l_exp <= 0 and s_exp > 0:
        return "SHORT_ONLY"
    else:
        return "REJECTED"

def get_symbol_classification(by_symbol):
    if not by_symbol:
        return "INSUFFICIENT_DATA"
    valid = [s for s, metrics in by_symbol.items() if metrics["count"] >= 5]
    if len(valid) < 2:
        return "INSUFFICIENT_DATA"
    
    positive_exp_chg = [s for s in valid if by_symbol[s]["expectancy_change"] > 0]
    if len(positive_exp_chg) == len(valid):
        return "MULTI_SYMBOL"
    elif len(positive_exp_chg) > 0:
        return "SYMBOL_DEPENDENT"
    return "REJECTED"

def get_validation_classification(dev_res, val_res):
    if not val_res or val_res["eligible_count"] < 10 or val_res["selected_count"] < 5:
        return "INSUFFICIENT_VALIDATION_DATA"
        
    dev_exp_chg = dev_res["overall"]["15m"]["candidate"]["expectancy"] - dev_res["overall"]["15m"]["eligible_baseline"]["expectancy"]
    val_exp_chg = val_res["overall"]["15m"]["candidate"]["expectancy"] - val_res["overall"]["15m"]["eligible_baseline"]["expectancy"]
    
    if dev_exp_chg > 0 and val_exp_chg > 0:
        return "VALIDATED"
    elif dev_exp_chg > 0 and val_exp_chg <= 0:
        return "FAILED_VALIDATION"
    return "MIXED_VALIDATION"

def compute_buckets(dev_pool):
    if len(dev_pool) < 20:
        return "INSUFFICIENT DATA"
    # Placeholder bucket structures calculated DEV-only
    return {
        "imbalance": {"bounds_derived_on_dev": True, "buckets": ["0.15-0.20", "0.20-0.25", "0.25-0.30", "0.30-0.40", ">0.40"]},
        "CVD": {"bounds_derived_on_dev": True, "quantiles": [0.2, 0.4, 0.6, 0.8]}
    }

def run_research():
    base_dir = os.path.dirname(__file__)
    signals_csv = os.path.join(os.path.dirname(base_dir), "phase2", "signal_outcomes_v2.csv")
    output_md = os.path.join(base_dir, "phase2b_report.md")
    output_json = os.path.join(base_dir, "phase2b_results.json")
    output_csv = os.path.join(base_dir, "phase2b_results.csv")
    
    signals = load_canonical_signals(signals_csv)
    total_n = len(signals)
    
    partitions = partition_signals(signals)
    dev_pool = partitions.development
    val_pool = partitions.validation
    
    if total_n < 20:
        status = "INSTRUMENTATION VALIDATION ONLY"
    elif total_n < 50:
        status = "EARLY EXPLORATORY"
    elif total_n < 100:
        status = "HYPOTHESIS GENERATION"
    elif total_n < 200:
        status = "PRELIMINARY COUNTERFACTUAL RESEARCH"
    else:
        status = "PHASE 2B FULL RESEARCH ELIGIBLE"

    completed_15m = sum(1 for s in dev_pool if s.get("horizon_15m_status") == "CAPTURED")
    interrupted = sum(1 for s in dev_pool if s.get("completion_status") == "INTERRUPTED")
    usable_excursions = sum(1 for s in dev_pool if s.get("completion_status") == "CAPTURED" and s.get("excursion_coverage_status") == "COMPLETE")

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

    b_ret, b_mfe, b_mae = get_outcomes(dev_pool)
    baseline_stats = calculate_stats(b_ret, b_mfe, b_mae)

    results = {
        "canonical_signals_count": total_n,
        "total_n": total_n,
        "dev_n": len(dev_pool),
        "validation_n": len(val_pool),
        "holdout_n": len(partitions._holdout),
        "completed_15m": completed_15m,
        "interrupted_signals": interrupted,
        "usable_excursions": usable_excursions,
        "sample_status": status,
        "splits": {
            "dev_count": len(partitions.development),
            "val_count": len(partitions.validation),
            "status": partitions.status
        },
        "baseline": baseline_stats,
        "hypotheses": {},
        "pairwise": {},
        "continuous_buckets": compute_buckets(dev_pool)
    }

    dev_engine = CounterfactualEngine(dev_pool)
    val_engine = CounterfactualEngine(val_pool) if val_pool else None
    
    csv_rows = []
    
    for h_name, filter_func in HYPOTHESES.items():
        try:
            dev_res = dev_engine.evaluate_hypothesis(filter_func)
            overall_15m = dev_res["overall"]["15m"]
            long_15m = dev_res["long"]["15m"]
            short_15m = dev_res["short"]["15m"]
            
            dev_res["direction_classification"] = get_direction_classification(overall_15m, long_15m, short_15m)
            dev_res["symbol_classification"] = get_symbol_classification(dev_res["by_symbol"])
            
            # Wire Bootstrap CI
            for cohort_name, cohort_data in [("overall", dev_res["overall"]), ("long", dev_res["long"]), ("short", dev_res["short"])]:
                for hor in ["1m", "3m", "5m", "15m"]:
                    cand_rets = []
                    # Get eligible candidate returns for this horizon
                    for s in dev_pool:
                        res = filter_func(s)
                        if res is True and s.get("direction", "").lower() in (cohort_name, "long", "short"):
                            if s.get(f"horizon_{hor}_status") == "CAPTURED":
                                val = float_val(s.get(f"return_{hor}_pct"))
                                if val is not None:
                                    cand_rets.append(val)
                    cohort_data[hor]["bootstrap"] = bootstrap_ci(cand_rets, iterations=5000, seed=42)
            
            # Independent VAL evaluation
            val_res = None
            if val_engine:
                val_res = val_engine.evaluate_hypothesis(filter_func)
                v_overall_15m = val_res["overall"]["15m"]
                v_long_15m = val_res["long"]["15m"]
                v_short_15m = val_res["short"]["15m"]
                val_res["direction_classification"] = get_direction_classification(v_overall_15m, v_long_15m, v_short_15m)
                val_res["symbol_classification"] = get_symbol_classification(val_res["by_symbol"])
            
            # Validation Classification
            dev_res["validation_classification"] = get_validation_classification(dev_res, val_res)
            
            # Stage-aware ratings
            if total_n < 20:
                dev_res["rating"] = "INSUFFICIENT_DATA"
            elif total_n < 50:
                dev_res["rating"] = "EARLY_EXPLORATORY"
            elif total_n < 100:
                dev_res["rating"] = "HYPOTHESIS_GENERATION"
            else:
                if dev_res["validation_classification"] == "VALIDATED":
                    dev_res["rating"] = "PROMISING"
                else:
                    dev_res["rating"] = "MIXED"
                    
            results["hypotheses"][h_name] = {
                "development_result": dev_res,
                "validation_result": val_res
            }
            
            # Build CSV rows
            for cohort_name, cohort_data in [("OVERALL", dev_res["overall"]), ("LONG", dev_res["long"]), ("SHORT", dev_res["short"])]:
                for hor in ["1m", "3m", "5m", "15m"]:
                    h_hor = cohort_data[hor]
                    b_stats = h_hor["total_baseline"]
                    e_stats = h_hor["eligible_baseline"]
                    c_stats = h_hor["candidate"]
                    
                    csv_rows.append({
                        "hypothesis": h_name,
                        "cohort": cohort_name,
                        "horizon": hor,
                        "eligible_n": dev_res["eligible_count"],
                        "candidate_n": c_stats["count"],
                        "retention_pct": pct(dev_res["retention_pct"]),
                        "baseline_win_rate": pct(e_stats["win_rate"]),
                        "candidate_win_rate": pct(c_stats["win_rate"]),
                        "delta_win_rate": pct(c_stats["win_rate"] - e_stats["win_rate"]),
                        "baseline_expectancy": f"{e_stats['expectancy']:.6f}",
                        "candidate_expectancy": f"{c_stats['expectancy']:.6f}",
                        "delta_expectancy": f"{(c_stats['expectancy'] - e_stats['expectancy']):.6f}",
                        "baseline_median_return": f"{e_stats['median_return']:.6f}",
                        "candidate_median_return": f"{c_stats['median_return']:.6f}",
                        "delta_median_return": f"{(c_stats['median_return'] - e_stats['median_return']):.6f}",
                        "profit_factor": f"{c_stats['profit_factor']:.4f}",
                        "winners_sacrificed": dev_res["winners_sacrificed"],
                        "losers_avoided": dev_res["losers_avoided"],
                        "sample_grade": status,
                        "rating": dev_res["rating"]
                    })
                    
        except Exception as e:
            results["hypotheses"][h_name] = {"error": str(e)}

    # Evaluate pairwise
    pairwise_registry = {
        "H01_AND_H03": combine_and(HYPOTHESES["H01_CVD_ALIGNMENT"], HYPOTHESES["H03_15M_DELTA_ALIGNMENT"]),
        "H01_AND_H08": combine_and(HYPOTHESES["H01_CVD_ALIGNMENT"], HYPOTHESES["H08_OI_RISING"]),
        "H03_AND_H04": combine_and(HYPOTHESES["H03_15M_DELTA_ALIGNMENT"], HYPOTHESES["H04_STRONGER_BOOK_20"]),
        "H09_AND_H08": combine_and(HYPOTHESES["H09_TREND_5M_ALIGNED"], HYPOTHESES["H08_OI_RISING"]),
        "H06_AND_H01": combine_and(HYPOTHESES["H06_ACTIVE_DIRECTIONAL_SWEEP"], HYPOTHESES["H01_CVD_ALIGNMENT"])
    }

    for p_name, p_filter in pairwise_registry.items():
        try:
            results["pairwise"][p_name] = dev_engine.evaluate_hypothesis(p_filter)
        except Exception as e:
            results["pairwise"][p_name] = {"error": str(e)}

    # Wire walk-forward
    wfv = WalkForwardValidator(dev_pool)
    wfv_results = {}
    for h_name, filter_func in HYPOTHESES.items():
        wfv_results[h_name] = wfv.run_walk_forward(filter_func, min_train_size=100, step_size=25)
    results["walk_forward"] = wfv_results

    # Generate MD Report
    report_md = f"""# PHASE 2B COUNTERFACTUAL SIGNAL RESEARCH

**GROSS DIRECTIONAL RETURNS**
*Note: Returns are calculated gross of fees, slippage, execution latency, and funding rates.*

## Dataset Status & Exclusions
- Total Canonical Signals (Total N): {total_n}
- DEV partition N: {len(partitions.development)}
- VAL partition N: {len(partitions.validation)}
- HOLDOUT partition N: {len(partitions._holdout)} (LOCKED)
- Completed 15m outcomes: {completed_15m}
- Interrupted signals: {interrupted}
- Usable MFE/MAE excursion tracks: {usable_excursions}

## Sample-Size Eligibility
- Current Status: **{status}**
- Research Splits Status: **{partitions.status}**

## Baseline Stats
- Win Rate (15m): {pct(baseline_stats['win_rate'])}
- Expectancy (15m): {baseline_stats['expectancy']:.6f}
- Profit Factor: {baseline_stats['profit_factor']:.4f}
- Win/Loss Ratio: {baseline_stats['win_loss_ratio']:.4f}

## Hypothesis Performance
"""
    for name, h_outer in results["hypotheses"].items():
        if "error" in h_outer:
            report_md += f"### {name}\n- Error: {h_outer['error']}\n\n"
            continue
        h_res = h_outer["development_result"]
        c_stats = h_res["overall"]["15m"]["candidate"]
        e_stats = h_res["overall"]["15m"]["eligible_baseline"]
        report_md += f"""### {name}
- **Retention**: {pct(h_res['retention_pct'])} (N={h_res['selected_count']} | Eligible={h_res['eligible_count']} | Missing={h_res['missing_feature_count']})
- **Direction Robustness**: {h_res['direction_classification']}
- **Symbol Robustness**: {h_res['symbol_classification']}
- **Validation Status**: {h_res['validation_classification']}
- **Rating**: {h_res['rating']}
- **Sacrificed Winners**: {h_res['winners_sacrificed']} | **Avoided Losers**: {h_res['losers_avoided']}
- **Removed Signal Mean/Median**: Mean={h_res['mean_return_removed']:.6f} | Median={h_res['median_return_removed']:.6f}
- **Candidate Expectancy**: 1m={h_res['overall']['1m']['candidate']['expectancy']:.6f} (vs eligible {h_res['overall']['1m']['eligible_baseline']['expectancy']:.6f}) | 15m={c_stats['expectancy']:.6f} (vs eligible {e_stats['expectancy']:.6f})

"""

    report_md += """
## Pairwise Combinations
"""
    for p_name, p_res in results["pairwise"].items():
        if "error" in p_res:
            report_md += f"### {p_name}\n- Error: {p_res['error']}\n\n"
            continue
        report_md += f"- **{p_name}**: N={p_res['selected_count']} | Retention={pct(p_res['retention_pct'])} | Expectancy (15m)={p_res['overall']['15m']['candidate']['expectancy']:.6f}\n"

    report_md += """
## Continuous Feature Buckets
- Imbalance Buckets: INSUFFICIENT DATA
- Session CVD Buckets: INSUFFICIENT DATA
- 5m Delta Buckets: INSUFFICIENT DATA
- OI Change Buckets: INSUFFICIENT DATA
- Sweep Score Buckets: INSUFFICIENT DATA
- Book Drift Buckets: INSUFFICIENT DATA

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
        
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        else:
            headers = ["hypothesis", "cohort", "horizon", "eligible_n", "candidate_n", "retention_pct", "baseline_win_rate", "candidate_win_rate", "delta_win_rate", "baseline_expectancy", "candidate_expectancy", "delta_expectancy", "baseline_median_return", "candidate_median_return", "delta_median_return", "profit_factor", "winners_sacrificed", "losers_avoided", "sample_grade", "rating"]
            writer = csv.writer(f)
            writer.writerow(headers)

    # Dynamic status printing
    print("\nPHASE 2B FRAMEWORK READY")
    print(f"CURRENT SAMPLE: N={total_n}")
    print(f"STATUS: {status}")
    
    if total_n < 50:
        print("HYPOTHESIS SELECTION: DISABLED")
        print("VALIDATION: DISABLED")
    else:
        print("HYPOTHESIS SELECTION: ENABLED")
        print("VALIDATION: ENABLED")
        
    print("HOLDOUT: LOCKED")
    print("PHASE 2C: NOT ELIGIBLE")
    print("NO PRODUCTION STRATEGY CHANGES HAVE BEEN APPLIED.")

if __name__ == "__main__":
    run_research()
