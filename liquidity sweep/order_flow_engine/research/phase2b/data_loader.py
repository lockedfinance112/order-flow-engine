import os
import csv
import logging

logger = logging.getLogger("OrderFlow.Phase2b.DataLoader")

class ResearchPartitions:
    def __init__(self, development: list, validation: list, holdout: list, status: str):
        self.development = development
        self.validation = validation
        self._holdout = holdout
        self.status = status

    def unlock_holdout_for_final_evaluation(self, purpose: str) -> list:
        if purpose != "PHASE2C_FINAL_EVALUATION":
            raise PermissionError("Access Denied: Holdout partition remains locked unless purpose is PHASE2C_FINAL_EVALUATION.")
        logger.warning("Caution: Unlocking holdout dataset partition!")
        return self._holdout

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

def partition_signals(signals: list) -> ResearchPartitions:
    sorted_signals = sorted(signals, key=lambda x: float(x.get("entry_time", 0.0)))
    total = len(sorted_signals)
    
    if total < 100:
        return ResearchPartitions(
            development=sorted_signals,
            validation=[],
            holdout=[],
            status="NOT ACTIVATED — INSUFFICIENT SAMPLE"
        )
        
    dev_cutoff = int(total * 0.6)
    val_cutoff = int(total * 0.8)
    
    return ResearchPartitions(
        development=sorted_signals[:dev_cutoff],
        validation=sorted_signals[dev_cutoff:val_cutoff],
        holdout=sorted_signals[val_cutoff:],
        status="HOLDOUT SPLITS ACTIVATED"
    )
