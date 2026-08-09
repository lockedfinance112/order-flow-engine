import os
import csv
import json
import logging

logger = logging.getLogger("OrderFlow.Phase2b.DataLoader")

def load_canonical_signals(signals_csv_path: str) -> list:
    signals = []
    if not os.path.exists(signals_csv_path):
        return signals
        
    try:
        with open(signals_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("dataset_class") == "CANONICAL_PHASE2":
                    signals.append(row)
    except Exception as e:
        logger.error(f"Error loading signals: {e}")
    return signals

def split_dataset(signals: list) -> tuple:
    # Sort chronologically by entry_time
    sorted_signals = sorted(signals, key=lambda x: float(x.get("entry_time", 0.0)))
    total = len(sorted_signals)
    
    dev_cutoff = int(total * 0.6)
    val_cutoff = int(total * 0.8)
    
    dev = sorted_signals[:dev_cutoff]
    val = sorted_signals[dev_cutoff:val_cutoff]
    holdout = sorted_signals[val_cutoff:]
    
    return dev, val, holdout
