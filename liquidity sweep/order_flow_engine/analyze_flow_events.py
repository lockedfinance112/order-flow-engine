import csv
import os
import sys
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(__file__), "flow_events.csv")

def analyze():
    print("=======================================================")
    print("ORDER FLOW EVENT ANALYZER (V2.3 Normalized)")
    print(f"Target: {CSV_PATH}")
    print("=======================================================\n")

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please ensure the scanner has run and logged some events.")
        sys.exit(1)

    total_rows = 0
    mock_count = 0
    symbols = []
    event_types = []
    confluence_scores = []
    
    sweep_confluence_count = 0
    low_confluence_count = 0
    
    # Track notional volume flow (buy_notional + sell_notional) per symbol
    symbol_notional_flow = Counter()
    
    large_trade_symbols = []
    absorption_symbols = []
    divergence_symbols = []
    burst_symbols = []

    try:
        with open(CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) # skip header
            
            for row in reader:
                if not row or len(row) < 11:
                    continue
                
                sweep_id = row[11] if len(row) > 11 else ""
                
                # Exclude mock sweeps from confluence stats
                if sweep_id.startswith("mock_"):
                    mock_count += 1
                    continue
                
                total_rows += 1
                symbol = row[1].upper()
                event_type = row[3]
                
                symbols.append(symbol)
                event_types.append(event_type)
                
                # Check confluence status
                if event_type == "SWEEP_CONFLUENCE":
                    sweep_confluence_count += 1
                elif event_type == "LOW_CONFLUENCE":
                    low_confluence_count += 1
                
                # Parse confluence score if present
                if len(row) > 13 and row[13]:
                    try:
                        confluence_scores.append(int(row[13]))
                    except ValueError:
                        pass
                
                # Parse notional volume flow: buy_vol_usdt (row[4]) + sell_vol_usdt (row[5])
                try:
                    buy_usdt = float(row[4])
                    sell_usdt = float(row[5])
                    symbol_notional_flow[symbol] += (buy_usdt + sell_usdt)
                except (ValueError, IndexError):
                    pass
                
                # Categorize alerts by symbol
                if event_type == "LARGE_TRADE":
                    large_trade_symbols.append(symbol)
                elif "ABSORPTION" in event_type:
                    absorption_symbols.append(symbol)
                elif "DIVERGENCE" in event_type:
                    divergence_symbols.append(symbol)
                elif "BURST" in event_type:
                    burst_symbols.append(symbol)

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    print(f"Mock sweep rows excluded: {mock_count}\n")

    if total_rows == 0:
        print("CSV file has no non-mock events. Waiting for events to log...")
        sys.exit(0)

    # 1. General Summary
    print("--- 1. GENERAL STATISTICS ---")
    print(f"Total Logged Alerts: {total_rows}")
    print(f"Unique Symbols Tracked: {len(set(symbols))}")
    print(f"SWEEP_CONFLUENCE Alerts: {sweep_confluence_count}")
    print(f"LOW_CONFLUENCE Alerts: {low_confluence_count}")
    
    if confluence_scores:
        avg_score = sum(confluence_scores) / len(confluence_scores)
        print(f"Average Confluence Score: {avg_score:.2f}/10")
    else:
        print("Average Confluence Score: N/A (no sweep events logged)")
    print()

    # 2. Top Events by Type
    print("--- 2. ALERTS BY TYPE ---")
    event_counter = Counter(event_types)
    for ev, count in event_counter.most_common():
        pct = (count / total_rows) * 100
        print(f"  {ev:<30} : {count:<5} ({pct:.1f}%)")
    print()

    # 3. Top Active Symbols by Alert Count
    print("--- 3. MOST ACTIVE SYMBOLS (ALL ALERTS) ---")
    symbol_counter = Counter(symbols)
    for sym, count in symbol_counter.most_common(5):
        pct = (count / total_rows) * 100
        print(f"  {sym:<10} : {count:<5} ({pct:.1f}%)")
    print()

    # 4. Top Symbols by USDT Notional Flow Volume
    print("--- 4. TOP SYMBOLS BY NOTIONAL FLOW VOLUME ---")
    if symbol_notional_flow:
        total_flow = sum(symbol_notional_flow.values())
        for sym, flow in symbol_notional_flow.most_common(5):
            pct = (flow / total_flow) * 100 if total_flow > 0 else 0
            if flow >= 1_000_000:
                flow_str = f"${flow/1_000_000:.2f}M"
            elif flow >= 1_000:
                flow_str = f"${flow/1_000:.1f}K"
            else:
                flow_str = f"${flow:.1f}"
            print(f"  {sym:<10} : {flow_str:<10} ({pct:.1f}%)")
    else:
        print("  No volume flow metrics recorded yet.")
    print()

    # 5. Absorption Hotspots
    print("--- 5. ABSORPTION EVENTS BY SYMBOL ---")
    abs_counter = Counter(absorption_symbols)
    if abs_counter:
        for sym, count in abs_counter.most_common(5):
            print(f"  {sym:<10} : {count:<5} events")
    else:
        print("  No absorption events logged yet.")
    print()

    # 6. Large Trade Hotspots
    print("--- 6. LARGE TRADES BY SYMBOL ---")
    lt_counter = Counter(large_trade_symbols)
    if lt_counter:
        for sym, count in lt_counter.most_common(5):
            print(f"  {sym:<10} : {count:<5} large trades")
    else:
        print("  No large trades logged yet.")
    print()

    # 7. Divergence Hotspots
    print("--- 7. CVD DIVERGENCES BY SYMBOL ---")
    div_counter = Counter(divergence_symbols)
    if div_counter:
        for sym, count in div_counter.most_common(5):
            print(f"  {sym:<10} : {count:<5} divergences")
    else:
        print("  No divergence events logged yet.")
    print()

if __name__ == "__main__":
    analyze()
