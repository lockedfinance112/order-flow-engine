import os
import csv
import json
from datetime import datetime

def run_audit():
    base_dir = os.path.dirname(__file__)
    # Corrected path to engine root
    legacy_file = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "bias_signals.csv")
    phase2_file = os.path.join(base_dir, "signal_outcomes_v2.csv")
    output_json = os.path.join(base_dir, "data_quality_report.json")

    report = {
        "legacy_bias_signals_rows": 0,
        "phase2_canonical_signals_rows": 0,
        "phase2_long_count": 0,
        "phase2_short_count": 0,
        "phase2_by_symbol": {},
        "first_timestamp": None,
        "last_timestamp": None,
        "missing_fields": {
            "entry_price": 0,
            "session_cvd_usdt": 0,
            "delta_5m_usdt": 0,
            "gate_5m_delta_bias": 0,
            "funding_rate_pct": 0,
            "open_interest": 0,
            "context_quality": 0,
            "price_after_15m": 0
        },
        "duplicate_signal_ids": 0,
        "duplicate_canonical_event_keys": 0,
        "duplicate_transitions": 0,
        "impossible_timestamps": 0,
        "incomplete_15m_signals": 0,
        "interrupted_signals": 0
    }

    # 1. Audit legacy bias_signals.csv if it exists
    if os.path.exists(legacy_file):
        try:
            with open(legacy_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    rows = list(reader)
                    report["legacy_bias_signals_rows"] = len(rows)
        except Exception as e:
            print(f"Error reading legacy signals: {e}")

    # 2. Audit Phase 2 signal_outcomes_v2.csv
    if os.path.exists(phase2_file):
        try:
            with open(phase2_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                report["phase2_canonical_signals_rows"] = len(rows)
                
                signal_ids = set()
                event_keys = set()
                transitions_seen = set()
                
                for row in rows:
                    direction = row.get("direction", "")
                    if direction == "LONG":
                        report["phase2_long_count"] += 1
                    elif direction == "SHORT":
                        report["phase2_short_count"] += 1
                        
                    sym = row.get("symbol", "").upper()
                    if sym:
                        report["phase2_by_symbol"][sym] = report["phase2_by_symbol"].get(sym, 0) + 1
                        
                    entry_time_raw = row.get("entry_time")
                    if entry_time_raw:
                        try:
                            ts = float(entry_time_raw)
                            dt = datetime.fromtimestamp(ts)
                            dt_str = dt.isoformat()
                            if report["first_timestamp"] is None or dt_str < report["first_timestamp"]:
                                report["first_timestamp"] = dt_str
                            if report["last_timestamp"] is None or dt_str > report["last_timestamp"]:
                                report["last_timestamp"] = dt_str
                        except Exception:
                            pass
                            
                    for field in report["missing_fields"]:
                        val = row.get(field)
                        if val is None or val == "" or val == "None" or val == "N/A" or val == "MISSING":
                            report["missing_fields"][field] += 1
                            
                    sig_id = row.get("signal_id")
                    if sig_id:
                        if sig_id in signal_ids:
                            report["duplicate_signal_ids"] += 1
                        signal_ids.add(sig_id)

                    event_key = row.get("canonical_event_key")
                    if event_key:
                        if event_key in event_keys:
                            report["duplicate_canonical_event_keys"] += 1
                        event_keys.add(event_key)
                        
                    trans_key = (row.get("symbol"), row.get("entry_time"), row.get("action"))
                    if trans_key in transitions_seen:
                        report["duplicate_transitions"] += 1
                    transitions_seen.add(trans_key)
                    
                    comp_time_raw = row.get("completed_time")
                    if entry_time_raw and comp_time_raw:
                        try:
                            entry_ts = float(entry_time_raw)
                            comp_dt = datetime.fromisoformat(comp_time_raw.replace("Z", "+00:00"))
                            if entry_ts > comp_dt.timestamp():
                                report["impossible_timestamps"] += 1
                        except Exception:
                            pass
                            
                    status = row.get("completion_status")
                    if status == "INTERRUPTED":
                        report["interrupted_signals"] += 1
                    elif status == "PENDING":
                        report["incomplete_15m_signals"] += 1
                        
        except Exception as e:
            print(f"Error auditing phase2 signals: {e}")

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("\n--- Data Quality Audit Report ---")
        print(f"Legacy bias_signals.csv rows: {report['legacy_bias_signals_rows']}")
        print(f"Phase 2 canonical_signals_v2.csv rows: {report['phase2_canonical_signals_rows']}")
        print(f"Phase 2: LONG={report['phase2_long_count']} | SHORT={report['phase2_short_count']}")
        print(f"First timestamp: {report['first_timestamp']}")
        print(f"Last timestamp: {report['last_timestamp']}")
        print(f"Missing price_after_15m count: {report['missing_fields']['price_after_15m']}")
        print(f"Duplicate IDs: {report['duplicate_signal_ids']}")
        print(f"Duplicate Event Keys: {report['duplicate_canonical_event_keys']}")
        print(f"Interrupted signals: {report['interrupted_signals']}")
        print("Data Quality status: COMPLETED")
    except Exception as e:
        print(f"Failed to write quality report: {e}")

if __name__ == "__main__":
    run_audit()
