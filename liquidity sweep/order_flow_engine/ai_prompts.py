# Order Flow Engine V2.5 - AI Prompts & JSON Schema

# JSON Schema definition for structured model outputs
AI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "market_summary": {
            "type": "string",
            "description": "A high-level synthesis of general market conditions, buying/selling aggression dominance, and notional flow trends."
        },
        "symbol_interpretations": {
            "type": "object",
            "description": "Detailed per-symbol analyses keyed by uppercase symbols (e.g. BTCUSDT). Only include symbols present in the snapshot.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "bias": {
                        "type": "string",
                        "description": "Must match the symbol's next_action EXACTLY as provided in the snapshot."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Clear reasoning of the flow dynamics (deltas, CVD, book imbalances, recent events) leading to this state."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0.0 (uncertain) and 1.0 (highly confident flow alignment)."
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any contradictory events, depth stale states, or extreme imbalances."
                    },
                    "what_to_watch_next": {
                        "type": "string",
                        "description": "Specific metrics or levels to monitor for next action changes (e.g. sweep expiry, absorption confirmation)."
                    }
                },
                "required": ["bias", "explanation", "confidence", "risk_flags", "what_to_watch_next"]
            }
        },
        "overall_risk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Overall risk level assessment of the current market regime."
        },
        "cleanest_bias_symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of symbols showing the cleanest order-flow alignment and highest confidence biases (exclude WAITING)."
        },
        "suppressed_symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of symbols currently demoted to WAITING due to active conflict suppression."
        },
        "regime": {
            "type": "string",
            "enum": ["trending", "choppy", "absorbing", "sweep-heavy", "mixed"],
            "description": "Dominant market structure regime based on current order flow dynamics."
        },
        "data_quality_warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Warnings about missing metrics, stale orderbook updates, or server connection issues."
        },
        "not_trade_advice": {
            "type": "boolean",
            "description": "Must be set to true at all times to affirm this is not financial/trade advice."
        }
    },
    "required": [
        "market_summary", "symbol_interpretations", "overall_risk", 
        "cleanest_bias_symbols", "suppressed_symbols", "regime", 
        "data_quality_warnings", "not_trade_advice"
    ]
}

SYSTEM_PROMPT = """You are a professional quantitative trading research assistant analyzing order flow metrics.
Your role is to explain, interpret, and summarize the market state and scanner actions in plain English.

CRITICAL RULES:
1. You are NOT allowed to change, override, or invent next_action states for any symbol. Use the next_action provided in the data snapshot exactly as is.
2. You must never give execution or financial advice. Do not say "buy", "sell", "enter", "long", "short", or "take the trade" as an recommendation. Use terms like "scanner bias", "flow conditions", "risk flags", and "watch next".
3. Validate that the next_action and suppression_reason fields are correctly explained based on recent events list.
4. Output your response as a single, valid JSON object matching the JSON schema.
5. Binance context, when present, is read-only confirmation/warning context. Use it to explain quality, funding, premium, open interest, and trend conflicts, but never use it to override scanner next_action.
6. Do not include markdown code block wrappers (such as ```json) or any conversational text before or after the JSON payload. Return only raw JSON."""

USER_PROMPT_TEMPLATE = """Current Market Data Snapshot:
{snapshot_json}

Return exactly one JSON object with these top-level keys and value types:
{{
  "market_summary": "string",
  "symbol_interpretations": {{
    "BTCUSDT": {{
      "bias": "must equal that symbol's next_action from the snapshot",
      "explanation": "string",
      "confidence": 0.0,
      "risk_flags": ["string"],
      "what_to_watch_next": "string"
    }}
  }},
  "overall_risk": "low | medium | high",
  "cleanest_bias_symbols": ["SYMBOL"],
  "suppressed_symbols": ["SYMBOL"],
  "regime": "trending | choppy | absorbing | sweep-heavy | mixed",
  "data_quality_warnings": ["string"],
  "not_trade_advice": true
}}

Rules for the JSON:
- Include every top-level key exactly as written above.
- Use empty arrays or objects when there is nothing to report.
- Do not rename keys, add a wrapper object, or place the summary under another key.
- Keep all analysis descriptive and visual-only; do not recommend executions.
- Treat binance_context as confirmation/warning evidence only. The bias field must still match each symbol's scanner next_action exactly.

Please interpret the current snapshot and return only that valid JSON object."""
