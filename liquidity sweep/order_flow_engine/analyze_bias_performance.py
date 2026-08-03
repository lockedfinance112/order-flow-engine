import os
import csv
import sys

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "bias_signals.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Performance log not found at {csv_path}")
        sys.exit(1)

    signals = []
    try:
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                signals.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    total_signals = len(signals)
    if total_signals == 0:
        print("=======================================================")
        print("BIAS SIGNAL PERFORMANCE ANALYZER (V2.4.1)")
        print("=======================================================")
        print("\nNo completed signals have been logged yet.")
        print("Signals are tracked for exactly 15 minutes before finalization.")
        print("Please check back once the engine has run for at least 15 minutes.")
        sys.exit(0)

    long_signals = [s for s in signals if s["direction"] == "LONG"]
    short_signals = [s for s in signals if s["direction"] == "SHORT"]
    
    completed_count = 0
    total_return = 0.0
    wins = 0
    total_max_fav = 0.0
    total_max_adv = 0.0
    
    symbol_stats = {}

    for s in signals:
        try:
            ret = float(s["final_return_pct"])
            fav = float(s["max_favorable_pct"])
            adv = float(s["max_adverse_pct"])
            symbol = s["symbol"].upper()
            
            total_return += ret
            total_max_fav += fav
            total_max_adv += adv
            completed_count += 1
            if ret > 0:
                wins += 1

            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"count": 0, "win_count": 0, "sum_return": 0.0}
            
            symbol_stats[symbol]["count"] += 1
            symbol_stats[symbol]["sum_return"] += ret
            if ret > 0:
                symbol_stats[symbol]["win_count"] += 1
        except (ValueError, TypeError):
            continue

    if completed_count == 0:
        print("No completed records with valid numeric outcomes.")
        sys.exit(0)

    win_rate = (wins / completed_count) * 100.0
    avg_return = (total_return / completed_count) * 100.0
    avg_max_fav = (total_max_fav / completed_count) * 100.0
    avg_max_adv = (total_max_adv / completed_count) * 100.0

    print("=======================================================")
    print("BIAS SIGNAL PERFORMANCE ANALYZER (V2.4.1)")
    print(f"Target file: {csv_path}")
    print("=======================================================")
    print(f"\n--- General Statistics ---")
    print(f"Total Completed Signals Tracked: {completed_count}")
    print(f"  - LONG  Signals: {len(long_signals)}")
    print(f"  - SHORT Signals: {len(short_signals)}")
    print(f"Overall Win Rate         : {win_rate:.2f}%")
    print(f"Average Final Return     : {avg_return:+.3f}%")
    print(f"Average Max Favorable Excursion: {avg_max_fav:.3f}%")
    print(f"Average Max Adverse Excursion  : {avg_max_adv:.3f}%")

    print(f"\n--- Symbol Breakdown ---")
    print(f"{'Symbol':<10} | {'Count':<6} | {'Win Rate':<10} | {'Avg Return':<12}")
    print("-" * 47)
    for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        sym_win_rate = (stats["win_count"] / stats["count"]) * 100.0
        sym_avg_return = (stats["sum_return"] / stats["count"]) * 100.0
        print(f"{sym:<10} | {stats['count']:<6} | {sym_win_rate:.1f}% | {sym_avg_return:+.3f}%")

if __name__ == "__main__":
    main()
