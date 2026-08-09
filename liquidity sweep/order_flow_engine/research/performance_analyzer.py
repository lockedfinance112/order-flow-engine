import os
import csv
from typing import List, Dict, Any

class PerformanceAnalyzer:
    """
    Evaluates signal performance (CONFIRMED_LONG / CONFIRMED_SHORT) against subsequent
    price action, computing win rates, MFE (Favourable Excursion), MAE (Adverse Excursion),
    drawdowns, and average returns.
    """
    def __init__(self, symbol: str, output_dir: str = "research"):
        self.symbol = symbol.lower()
        self.output_dir = output_dir
        self.results_filepath = os.path.join(self.output_dir, f"signal_performance_{self.symbol}.csv")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def analyze_signals(self, transitions: List[dict], trades: List[dict]) -> List[dict]:
        """
        Scans forward from each signal transition timestamp through chronological trade list.
        Lookback windows: 5 minutes (300s), 15 minutes (900s), 1 hour (3600s).
        """
        evaluated_signals = []
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x["timestamp"])
        
        # Filter for confirmed signals
        signals = [t for t in transitions if t["new_action"] in ("CONFIRMED_LONG", "CONFIRMED_SHORT")]

        for sig in signals:
            sig_time = sig["timestamp"]
            sig_price = sig["price"]
            sig_type = sig["new_action"]  # "CONFIRMED_LONG" or "CONFIRMED_SHORT"
            
            # Find the forward windows: 5m, 15m, 1h
            windows = [300.0, 900.0, 3600.0]
            win_stats = {}

            for w in windows:
                w_trades = [t for t in sorted_trades if sig_time < t["timestamp"] <= sig_time + w]
                if not w_trades:
                    win_stats[int(w)] = {
                        "mfe_pct": 0.0,
                        "mae_pct": 0.0,
                        "pnl_pct": 0.0,
                        "win": False
                    }
                    continue

                prices = [t["price"] for t in w_trades]
                max_price = max(prices)
                min_price = min(prices)
                final_price = prices[-1]

                if sig_type == "CONFIRMED_LONG":
                    mfe = ((max_price - sig_price) / sig_price) * 100.0
                    mae = ((min_price - sig_price) / sig_price) * 100.0
                    pnl = ((final_price - sig_price) / sig_price) * 100.0
                else: # CONFIRMED_SHORT
                    mfe = ((sig_price - min_price) / sig_price) * 100.0
                    mae = ((sig_price - max_price) / sig_price) * 100.0
                    pnl = ((sig_price - final_price) / sig_price) * 100.0

                win_stats[int(w)] = {
                    "mfe_pct": round(mfe, 4),
                    "mae_pct": round(mae, 4),
                    "pnl_pct": round(pnl, 4),
                    "win": pnl > 0.0
                }

            evaluated_signals.append({
                "timestamp": sig_time,
                "type": sig_type,
                "price": sig_price,
                "reason": sig["reason"],
                "5m": win_stats[300],
                "15m": win_stats[900],
                "1h": win_stats[3600]
            })

        # Save to CSV
        self.save_to_csv(evaluated_signals)
        return evaluated_signals

    def save_to_csv(self, evaluated_signals: List[dict]):
        """Exports results to a CSV performance report."""
        headers = [
            "timestamp", "signal_type", "price", "reason",
            "5m_mfe_pct", "5m_mae_pct", "5m_pnl_pct", "5m_win",
            "15m_mfe_pct", "15m_mae_pct", "15m_pnl_pct", "15m_win",
            "1h_mfe_pct", "1h_mae_pct", "1h_pnl_pct", "1h_win"
        ]
        
        try:
            with open(self.results_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for s in evaluated_signals:
                    writer.writerow([
                        s["timestamp"],
                        s["type"],
                        s["price"],
                        s["reason"],
                        s["5m"]["mfe_pct"], s["5m"]["mae_pct"], s["5m"]["pnl_pct"], str(s["5m"]["win"]),
                        s["15m"]["mfe_pct"], s["15m"]["mae_pct"], s["15m"]["pnl_pct"], str(s["15m"]["win"]),
                        s["1h"]["mfe_pct"], s["1h"]["mae_pct"], s["1h"]["pnl_pct"], str(s["1h"]["win"])
                    ])
        except Exception as e:
            print(f"Error saving performance CSV: {e}")
            
    def compile_aggregate_report(self, evaluated_signals: List[dict]) -> dict:
        """Computes Precision (win rate), Avg Drawdown, and Avg PnL statistics."""
        total = len(evaluated_signals)
        if total == 0:
            return {"total_signals": 0}

        report = {"total_signals": total}
        for w_str, w_val in [("5m", 300), ("15m", 900), ("1h", 3600)]:
            wins = sum(1 for s in evaluated_signals if s[w_str]["win"])
            mfes = [s[w_str]["mfe_pct"] for s in evaluated_signals]
            maes = [s[w_str]["mae_pct"] for s in evaluated_signals]
            pnls = [s[w_str]["pnl_pct"] for s in evaluated_signals]
            
            report[f"{w_str}_precision"] = round((wins / total) * 100.0, 2)
            report[f"{w_str}_avg_mfe_pct"] = round(sum(mfes) / total, 4)
            report[f"{w_str}_avg_mae_pct"] = round(sum(maes) / total, 4)
            report[f"{w_str}_avg_pnl_pct"] = round(sum(pnls) / total, 4)
            
        return report
