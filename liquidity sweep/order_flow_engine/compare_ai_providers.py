import os
import sys
import json
import asyncio
import time
from datetime import datetime, timezone

# Ensure we import config and providers correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import ai_providers
import ai_prompts

MOCK_SNAPSHOT = {
  "timestamp": datetime.now(timezone.utc).isoformat(),
  "symbols": {
    "BTCUSDT": {
      "price": 61250.0,
      "delta_1m_usdt": 120000.0,
      "delta_5m_usdt": 250000.0,
      "delta_15m_usdt": 450000.0,
      "session_cvd_usdt": 1200000.0,
      "imbalance": 0.28,
      "next_action": "LONG_BIAS",
      "latest_event": "BUY_AGGRESSION"
    },
    "ETHUSDT": {
      "price": 1645.0,
      "delta_1m_usdt": -50000.0,
      "delta_5m_usdt": -80000.0,
      "delta_15m_usdt": 120000.0,
      "session_cvd_usdt": -150000.0,
      "imbalance": -0.05,
      "next_action": "WAITING",
      "latest_event": "POSSIBLE_BEARISH_ABSORPTION"
    },
    "SOLUSDT": {
      "price": 81.50,
      "delta_1m_usdt": -120000.0,
      "delta_5m_usdt": -320000.0,
      "delta_15m_usdt": -450000.0,
      "session_cvd_usdt": -890000.0,
      "imbalance": -0.32,
      "next_action": "SHORT_BIAS",
      "latest_event": "SELL_AGGRESSION"
    }
  },
  "recent_transitions": [
    {"symbol": "SOLUSDT", "old_action": "WAITING", "new_action": "SHORT_BIAS", "suppression_reason": "NONE"},
    {"symbol": "ETHUSDT", "old_action": "LONG_BIAS", "new_action": "WAITING", "suppression_reason": "BEARISH_CONFLICT:POSSIBLE_BEARISH_ABSORPTION"}
  ],
  "recent_completed_signals": [
    {"symbol": "BTCUSDT", "direction": "LONG", "final_return_pct": "0.45"}
  ],
  "scanner_mode": "visual_only_no_execution"
}

async def test_provider(provider_name):
    # Retrieve provider and configure temporary keys if environment is mocked
    provider = ai_providers.get_provider(provider_name, timeout=20)
    valid, err = provider.validate_config()
    if not valid:
        print(f"[-] Skip {provider_name.upper()}: {err}")
        return None
        
    print(f"[+] Running {provider_name.upper()} ({provider.model})...")
    user_prompt = ai_prompts.USER_PROMPT_TEMPLATE.format(
        snapshot_json=json.dumps(MOCK_SNAPSHOT, indent=2)
    )
    system_prompt = ai_prompts.SYSTEM_PROMPT
    
    start_time = time.time()
    res = await provider.interpret(system_prompt, user_prompt)
    latency = int((time.time() - start_time) * 1000)
    
    return {
        "provider": provider_name.upper(),
        "model": provider.model,
        "latency_ms": latency,
        "result": res
    }

async def main():
    print("==================================================")
    print("      AI PROVIDER SIDE-BY-SIDE COMPARISON         ")
    print("==================================================")
    
    # Try loading dotenv file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
        
    providers = ["openai", "gemini", "claude", "deepseek"]
    tasks = [test_provider(p) for p in providers]
    results = await asyncio.gather(*tasks)
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("\nNo providers were configured/executed. Please add API keys to .env.")
        return
        
    print("\n=================== COMPARISON SUMMARY ===================")
    print(f"{'PROVIDER':<10} | {'MODEL':<20} | {'LATENCY':<8} | {'RISK':<6} | {'REGIME':<10} | {'OK':<5} | {'ERROR':<15}")
    print("-" * 85)
    
    for r in valid_results:
        res = r["result"]
        ok = res.get("ok", False)
        err = res.get("error", "")
        if ok and res.get("json"):
            interpretation = res["json"]
            risk = interpretation.get("overall_risk", "N/A")
            regime = interpretation.get("regime", "N/A")
            err_str = "-"
        else:
            risk = "N/A"
            regime = "N/A"
            err_str = str(err)[:20]
            
        print(f"{r['provider']:<10} | {r['model'][:20]:<20} | {r['latency_ms']:>6}ms | {risk:<6} | {regime:<10} | {str(ok):<5} | {err_str:<15}")
        
    # Detail explanations if ok
    for r in valid_results:
        res = r["result"]
        if res.get("ok") and res.get("json"):
            print(f"\n>>> {r['provider']} ({r['model']}) Detailed Output:")
            interpretation = res["json"]
            print(f"  Market Summary: {interpretation.get('market_summary')}")
            print(f"  Cleanest Biases: {interpretation.get('cleanest_bias_symbols')}")
            print(f"  Suppressed Signals: {interpretation.get('suppressed_symbols')}")
            print("  Symbol Explanations:")
            syms = interpretation.get("symbol_interpretations", {})
            for sym, details in syms.items():
                print(f"    - {sym}: Action={details.get('bias')} | Confidence={details.get('confidence')} | Watch={details.get('what_to_watch_next')}")
                print(f"      Reasoning: {details.get('explanation')}")

if __name__ == "__main__":
    asyncio.run(main())
