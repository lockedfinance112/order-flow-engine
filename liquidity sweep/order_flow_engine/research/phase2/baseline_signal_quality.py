import os
import csv
import json
import numpy as np
from datetime import datetime

def pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.4f}%"

def float_val(v):
    if v is None or v == "" or v == "None" or v == "N/A":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def sample_grade(n):
    if n < 20:
        return "VERY LOW SAMPLE"
    elif n < 50:
        return "LOW SAMPLE"
    elif n < 100:
        return "MODERATE SAMPLE"
    return "USEFUL SAMPLE"

def calculate_stats(returns, mfes=None, maes=None):
    n = len(returns)
    if n == 0:
        return {
            "count": 0, "win_rate": 0.0, "avg_return": 0.0, "median_return": 0.0,
            "avg_winner": 0.0, "avg_loser": 0.0, "win_loss_ratio": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "grade": sample_grade(0)
        }
    
    positives = [r for r in returns if r > 0]
    negatives = [r for r in returns if r < 0]
    
    win_rate = len(positives) / n
    avg_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    
    avg_winner = float(np.mean(positives)) if positives else 0.0
    avg_loser = float(np.mean(negatives)) if negatives else 0.0
    
    win_loss_ratio = (avg_winner / abs(avg_loser)) if avg_loser != 0 else 0.0
    expectancy = (win_rate * avg_winner) - ((1 - win_rate) * abs(avg_loser))
    
    sum_pos = sum(positives)
    sum_neg = abs(sum(negatives))
    profit_factor = (sum_pos / sum_neg) if sum_neg != 0 else (float('inf') if sum_pos > 0 else 1.0)
    
    stats = {
        "count": n,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "win_loss_ratio": win_loss_ratio,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "grade": sample_grade(n)
    }
    
    if mfes is not None and len(mfes) > 0:
        stats["mfe_mean"] = float(np.mean(mfes))
        stats["mfe_median"] = float(np.median(mfes))
        stats["mfe_p25"] = float(np.percentile(mfes, 25))
        stats["mfe_p75"] = float(np.percentile(mfes, 75))
        stats["mfe_p90"] = float(np.percentile(mfes, 90))
        
    if maes is not None and len(maes) > 0:
        stats["mae_mean"] = float(np.mean(maes))
        stats["mae_median"] = float(np.median(maes))
        stats["mae_p25"] = float(np.percentile(maes, 25))
        stats["mae_p75"] = float(np.percentile(maes, 75))
        stats["mae_p90"] = float(np.percentile(maes, 90))
        
    return stats

def run_analysis():
    base_dir = os.path.dirname(__file__)
    phase2_file = os.path.join(base_dir, "signal_outcomes_v2.csv")
    transitions_file = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "bias_transitions.csv")
    output_md = os.path.join(base_dir, "baseline_report.md")
    output_json = os.path.join(base_dir, "baseline_report.json")

    results = {
        "overall": {}, "long": {}, "short": {}, "by_symbol": {},
        "splits": {"DEVELOPMENT": 0, "VALIDATION": 0, "HOLDOUT": 0, "status": "HOLDOUT NOT YET ACTIVATED — INSUFFICIENT SAMPLE"},
        "cvd_alignment": {}, "delta_alignment": {}, "sweep_confluence": {}, "events_context": {}, "imbalance_buckets": {},
        "watch_conversion": {
            "total_watch_episodes": 0,
            "direct_conversion_rate": 0.0,
            "episode_conversion_rate": 0.0,
            "reversal_rate": 0.0,
            "expiry_rate": 0.0,
            "censored_count": 0,
            "median_seconds_to_confirmation": 0.0
        },
        "binance_context": {}
    }

    if not os.path.exists(phase2_file):
        # Create empty report
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# PHASE 2 BASELINE SIGNAL QUALITY REPORT\n\n**INSUFFICIENT DATA** - No Phase 2 canonical signals recorded yet.\n")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Baseline analysis complete: INSUFFICIENT DATA")
        return

    # Load Phase 2 signals
    signals = []
    with open(phase2_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset_class") == "CANONICAL_PHASE2":
                signals.append(row)

    if not signals:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# PHASE 2 BASELINE SIGNAL QUALITY REPORT\n\n**INSUFFICIENT DATA** - No Phase 2 canonical signals found.\n")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Baseline analysis complete: 0 canonical signals")
        return

    # Chronological split
    signals.sort(key=lambda x: float(x.get("entry_time", 0.0)))
    total_n = len(signals)
    dev_split = int(total_n * 0.6)
    val_split = int(total_n * 0.8)
    
    results["splits"]["DEVELOPMENT"] = dev_split
    results["splits"]["VALIDATION"] = val_split - dev_split
    results["splits"]["HOLDOUT"] = total_n - val_split
    if total_n >= 100:
        results["splits"]["status"] = "HOLDOUT SPLITS ACTIVATED"

    # Helpers to filter eligible prices
    def get_returns_for_horizon(sigs, horizon):
        ret_list = []
        for s in sigs:
            if s.get(f"horizon_{horizon}_status") == "CAPTURED":
                val = float_val(s.get(f"return_{horizon}_pct"))
                if val is not None:
                    ret_list.append(val)
        return ret_list

    def get_excursions(sigs):
        mfes, maes = [], []
        for s in sigs:
            if s.get("completion_status") == "CAPTURED":
                f = float_val(s.get("max_favorable_pct"))
                a = float_val(s.get("max_adverse_pct"))
                if f is not None: mfes.append(f)
                if a is not None: maes.append(a)
        return mfes, maes

    # 1. Overall & Direction stats
    mfes_all, maes_all = get_excursions(signals)
    results["overall"] = calculate_stats(get_returns_for_horizon(signals, "15m"), mfes_all, maes_all)
    
    long_sigs = [s for s in signals if s.get("direction") == "LONG"]
    mfes_l, maes_l = get_excursions(long_sigs)
    results["long"] = calculate_stats(get_returns_for_horizon(long_sigs, "15m"), mfes_l, maes_l)
    
    short_sigs = [s for s in signals if s.get("direction") == "SHORT"]
    mfes_s, maes_s = get_excursions(short_sigs)
    results["short"] = calculate_stats(get_returns_for_horizon(short_sigs, "15m"), mfes_s, maes_s)

    # 2. Per-symbol stats
    symbols = set(s.get("symbol") for s in signals)
    for sym in symbols:
        sym_sigs = [s for s in signals if s.get("symbol") == sym]
        mfes_sym, maes_sym = get_excursions(sym_sigs)
        results["by_symbol"][sym] = calculate_stats(get_returns_for_horizon(sym_sigs, "15m"), mfes_sym, maes_sym)

    # 3. CVD Alignment stats
    aligned_sigs, opposing_sigs = [], []
    for s in signals:
        direction = s.get("direction")
        cvd = float_val(s.get("session_cvd_usdt", 0.0))
        if cvd is not None:
            if (direction == "LONG" and cvd > 0) or (direction == "SHORT" and cvd < 0):
                aligned_sigs.append(s)
            else:
                opposing_sigs.append(s)
    results["cvd_alignment"]["aligned"] = calculate_stats(get_returns_for_horizon(aligned_sigs, "15m"))
    results["cvd_alignment"]["opposing"] = calculate_stats(get_returns_for_horizon(opposing_sigs, "15m"))

    # 4. Multi-window Delta Alignment
    full_align, partial_align = [], []
    for s in signals:
        direction = s.get("direction")
        d1 = float_val(s.get("delta_1m_usdt", 0.0))
        d5 = float_val(s.get("delta_5m_usdt", 0.0))
        d15 = float_val(s.get("delta_15m_usdt", 0.0))
        if d1 is not None and d5 is not None and d15 is not None:
            if direction == "LONG" and d1 > 0 and d5 > 0 and d15 > 0:
                full_align.append(s)
            elif direction == "SHORT" and d1 < 0 and d5 < 0 and d15 < 0:
                full_align.append(s)
            else:
                partial_align.append(s)
    results["delta_alignment"]["full"] = calculate_stats(get_returns_for_horizon(full_align, "15m"))
    results["delta_alignment"]["partial"] = calculate_stats(get_returns_for_horizon(partial_align, "15m"))

    # 5. Sweep confluences
    sweep_yes, sweep_no = [], []
    for s in signals:
        active = s.get("sweep_active") == "True" or s.get("sweep_active") is True
        if active:
            sweep_yes.append(s)
        else:
            sweep_no.append(s)
    results["sweep_confluence"]["sweep"] = calculate_stats(get_returns_for_horizon(sweep_yes, "15m"))
    results["sweep_confluence"]["no_sweep"] = calculate_stats(get_returns_for_horizon(sweep_no, "15m"))

    # 6. Imbalance buckets
    buckets = {"0.15_0.20": [], "0.20_0.30": [], "0.30_0.40": [], "above_0.40": []}
    for s in signals:
        imb = float_val(s.get("imbalance", 0.0))
        if imb is not None:
            val = abs(imb)
            if 0.15 <= val < 0.20:
                buckets["0.15_0.20"].append(s)
            elif 0.20 <= val < 0.30:
                buckets["0.20_0.30"].append(s)
            elif 0.30 <= val < 0.40:
                buckets["0.30_0.40"].append(s)
            elif val >= 0.40:
                buckets["above_0.40"].append(s)
    for k, v in buckets.items():
        results["imbalance_buckets"][k] = calculate_stats(get_returns_for_horizon(v, "15m"))

    # 7. WATCH -> CONFIRMED Transition analysis
    if os.path.exists(transitions_file):
        try:
            episodes = []
            active_episodes = {} # {symbol: {"start_time": float, "type": str, "steps": list}}
            
            with open(transitions_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None) # skip headers
                rows = list(reader)
                rows.sort(key=lambda x: x[0]) # sort by timestamp
                
                direct_conversions = 0
                total_episodes_started = 0
                conversions = 0
                reversals = 0
                expirations = 0
                censored = 0
                durations = []
                
                for r in rows:
                    ts = datetime.fromisoformat(r[0].replace("Z", "+00:00")).timestamp()
                    sym = r[1].lower()
                    old_a = r[2]
                    new_a = r[3]
                    
                    # End conditions check
                    if sym in active_episodes:
                        ep = active_episodes[sym]
                        ep["steps"].append(new_a)
                        target_direction = "LONG" if ep["type"] == "WATCH_LONG" else "SHORT"
                        
                        if new_a == f"CONFIRMED_{target_direction}":
                            conversions += 1
                            durations.append(ts - ep["start_time"])
                            if len(ep["steps"]) == 1:
                                direct_conversions += 1
                            del active_episodes[sym]
                        elif new_a in (f"WATCH_{'SHORT' if target_direction == 'LONG' else 'LONG'}", f"CONFIRMED_{'SHORT' if target_direction == 'LONG' else 'LONG'}"):
                            reversals += 1
                            del active_episodes[sym]
                        elif new_a == "WAITING":
                            expirations += 1
                            del active_episodes[sym]
                        elif new_a in ("WARMING_UP", "DATA_INVALID", "DATA_STALE"):
                            censored += 1
                            del active_episodes[sym]
                            
                    # Start conditions check
                    if new_a in ("WATCH_LONG", "WATCH_SHORT") and old_a != new_a:
                        active_episodes[sym] = {
                            "start_time": ts,
                            "type": new_a,
                            "steps": []
                        }
                        total_episodes_started += 1
                        
                results["watch_conversion"] = {
                    "total_watch_episodes": total_episodes_started,
                    "direct_conversion_rate": (direct_conversions / total_episodes_started) if total_episodes_started > 0 else 0.0,
                    "episode_conversion_rate": (conversions / total_episodes_started) if total_episodes_started > 0 else 0.0,
                    "reversal_rate": (reversals / total_episodes_started) if total_episodes_started > 0 else 0.0,
                    "expiry_rate": (expirations / total_episodes_started) if total_episodes_started > 0 else 0.0,
                    "censored_count": censored,
                    "median_seconds_to_confirmation": float(np.median(durations)) if durations else 0.0
                }
        except Exception as e:
            print(f"Error parsing transitions: {e}")

    # Generate MD Baseline Report
    report_md = f"""# PHASE 2 BASELINE SIGNAL QUALITY REPORT

**GROSS DIRECTIONAL RETURNS**
*Note: Returns are calculated gross of fees, slippage, execution latency, and funding rates.*

## Dataset Integrity
- Total CANONICAL_PHASE2 signals: {total_n}
- Data quality splits status: {results['splits']['status']}
- Split allocation: DEV={results['splits']['DEVELOPMENT']} | VAL={results['splits']['VALIDATION']} | HOLDOUT={results['splits']['HOLDOUT']}

## Overall Performance
- Confirmed signals overall: {results['overall']['count']}
- Win Rate (15m): {pct(results['overall']['win_rate'])}
- Expectancy (15m): {results['overall']['expectancy']:.6f}
- Profit Factor: {results['overall']['profit_factor']:.4f}
- Win/Loss Ratio: {results['overall']['win_loss_ratio']:.4f}

## Directional Performance
### CONFIRMED_LONG
- Count: {results['long']['count']} ({sample_grade(results['long']['count'])})
- Win Rate (15m): {pct(results['long']['win_rate'])}
- Average Return (15m): {pct(results['long']['avg_return'])}
- Median Return (15m): {pct(results['long']['median_return'])}

### CONFIRMED_SHORT
- Count: {results['short']['count']} ({sample_grade(results['short']['count'])})
- Win Rate (15m): {pct(results['short']['win_rate'])}
- Average Return (15m): {pct(results['short']['avg_return'])}
- Median Return (15m): {pct(results['short']['median_return'])}

## Performance by Symbol
"""
    for sym, stats in results["by_symbol"].items():
        report_md += f"- **{sym}**: N={stats['count']} | 15m Win Rate={pct(stats['win_rate'])} | Avg Return={pct(stats['avg_return'])} | Median Return={pct(stats['median_return'])}\n"

    report_md += f"""
## CVD Alignment Analysis
- **CVD Aligned**: N={results['cvd_alignment'].get('aligned', {}).get('count', 0)} | Win Rate={pct(results['cvd_alignment'].get('aligned', {}).get('win_rate'))} | Expectancy={results['cvd_alignment'].get('aligned', {}).get('expectancy', 0.0):.6f}
- **CVD Opposing**: N={results['cvd_alignment'].get('opposing', {}).get('count', 0)} | Win Rate={pct(results['cvd_alignment'].get('opposing', {}).get('win_rate'))} | Expectancy={results['cvd_alignment'].get('opposing', {}).get('expectancy', 0.0):.6f}

## Delta Alignment Analysis
- **Full Alignment (1m+5m+15m)**: N={results['delta_alignment'].get('full', {}).get('count', 0)} | Win Rate={pct(results['delta_alignment'].get('full', {}).get('win_rate'))}
- **Partial/Opposing Alignment**: N={results['delta_alignment'].get('partial', {}).get('count', 0)} | Win Rate={pct(results['delta_alignment'].get('partial', {}).get('win_rate'))}

## Sweep confluences
- **Active Sweep Confluence**: N={results['sweep_confluence'].get('sweep', {}).get('count', 0)} | Win Rate={pct(results['sweep_confluence'].get('sweep', {}).get('win_rate'))}
- **No Sweep Confluence**: N={results['sweep_confluence'].get('no_sweep', {}).get('count', 0)} | Win Rate={pct(results['sweep_confluence'].get('no_sweep', {}).get('win_rate'))}

## Imbalance buckets
"""
    for k, v in results["imbalance_buckets"].items():
        report_md += f"- **Bucket {k}**: N={v['count']} | Win Rate={pct(v['win_rate'])} | Avg Return={pct(v['avg_return'])}\n"

    report_md += f"""
## WATCH → CONFIRMED Conversion
- Total WATCH Episodes: {results['watch_conversion']['total_watch_episodes']}
- Direct Conversion Rate: {pct(results['watch_conversion']['direct_conversion_rate'])}
- Episode Conversion Rate: {pct(results['watch_conversion']['episode_conversion_rate'])}
- Reversal Rate: {pct(results['watch_conversion']['reversal_rate'])}
- Expiry Rate: {pct(results['watch_conversion']['expiry_rate'])}
- Censored Count: {results['watch_conversion']['censored_count']}
- Median Seconds to Confirmation: {results['watch_conversion']['median_seconds_to_confirmation']:.2f}s

## HYPOTHESES ONLY — NO STRATEGY CHANGES APPLIED
"""

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Baseline report generated successfully.")

if __name__ == "__main__":
    run_analysis()
