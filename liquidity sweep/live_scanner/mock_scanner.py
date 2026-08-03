import csv
import os
import time
from datetime import datetime, timezone

# Resolve paths relative to scanner location
SWEEPS_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "sweeps.csv"))

def ensure_headers():
    os.makedirs(os.path.dirname(SWEEPS_CSV_PATH), exist_ok=True)
    if not os.path.exists(SWEEPS_CSV_PATH):
        print(f"Creating sweeps.csv at {SWEEPS_CSV_PATH}...")
        with open(SWEEPS_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "type", "sweep_level", "sweep_id"])

def write_sweep(sweep_type: str, level: float, sweep_id: str):
    ensure_headers()
    timestamp = datetime.now(timezone.utc).isoformat()
    row = [timestamp, "btcusdt", sweep_type, f"{level:.2f}", sweep_id]
    
    with open(SWEEPS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Simulated {sweep_type} sweep written to CSV: Level={level} | ID={sweep_id}")

def main():
    print("MOCK SCANNER STARTED - Press Ctrl+C to exit")
    print("------------------------------------------")
    try:
        # Initial wait
        time.sleep(2)
        
        # 1. Trigger Bullish Sweep
        bullish_id = f"swp_mock_{int(time.time())}_bull"
        print(f"\nTriggering a Bullish Sweep (Swept Low) with ID: {bullish_id}...")
        write_sweep("bullish", 58450.0, bullish_id)
        
        # Wait 15 seconds
        time.sleep(15)
        
        # 2. Trigger Bearish Sweep
        bearish_id = f"swp_mock_{int(time.time())}_bear"
        print(f"\nTriggering a Bearish Sweep (Swept High) with ID: {bearish_id}...")
        write_sweep("bearish", 58550.0, bearish_id)
        
        print("\nMock events completed. You can restart this script to trigger them again.")
    except KeyboardInterrupt:
        print("\nExiting Mock Scanner.")

if __name__ == "__main__":
    main()
