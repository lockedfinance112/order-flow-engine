# Order Flow Engine V2.5 - AI Interpretation Orchestrator

import os
import csv
import time
import json
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import config
import ai_providers
import ai_prompts

logger = logging.getLogger("OrderFlow.AIInterpreter")


def _safe_text(value: Optional[str]) -> Optional[str]:
    return ai_providers.redact_sensitive(value)


def _timestamp_age_seconds(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))
    except Exception:
        return None


def _stringify_ai_value(value: Any) -> str:
    """Converts provider-specific summary shapes into compact display text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _stringify_ai_value(item)
            if text:
                parts.append(f"{key}: {text}")
        return " | ".join(parts)
    if isinstance(value, list):
        return "; ".join(_stringify_ai_value(item) for item in value if _stringify_ai_value(item))
    return str(value).strip()


def _listify_ai_value(value: Any) -> list:
    """Normalizes optional provider fields into schema-compatible arrays."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _normalize_risk(value: Any) -> Optional[str]:
    text = _stringify_ai_value(value).lower()
    for risk in ("low", "medium", "high"):
        if risk in text:
            return risk
    return None


def _normalize_regime(value: Any) -> Optional[str]:
    text = _stringify_ai_value(value).lower()
    for regime in ("trending", "choppy", "absorbing", "sweep-heavy", "mixed"):
        if regime in text:
            return regime
    return None


def repair_ai_response_schema(resp: dict) -> Tuple[dict, bool]:
    """
    Repairs harmless provider naming drift into the required V2.5 dashboard schema.
    This is display/audit only; it never changes scanner or scoring decisions.
    """
    if not isinstance(resp, dict):
        return resp, False

    repaired = dict(resp)
    changed = False

    if "market_summary" not in repaired:
        for key in ("overall_market_assessment", "market_assessment", "market_read", "summary", "market_conditions"):
            summary = _stringify_ai_value(resp.get(key))
            if summary:
                repaired["market_summary"] = summary
                changed = True
                break
        if "market_summary" not in repaired:
            fallback_parts = [
                _stringify_ai_value(resp.get("recent_activity")),
                _stringify_ai_value(resp.get("risk_flags")),
            ]
            fallback_summary = " | ".join(part for part in fallback_parts if part)
            repaired["market_summary"] = fallback_summary or "Provider returned a partial interpretation; schema repaired for display."
            changed = True

    if "symbol_interpretations" not in repaired:
        symbol_source = None
        for key in ("symbol_insights", "symbol_analysis", "symbols"):
            if isinstance(resp.get(key), dict):
                symbol_source = resp[key]
                break

        symbol_interpretations = {}
        if symbol_source:
            for symbol, details in symbol_source.items():
                if isinstance(details, dict):
                    symbol_interpretations[str(symbol).upper()] = {
                        "bias": _stringify_ai_value(details.get("bias") or details.get("scanner_bias") or details.get("next_action") or "WAITING"),
                        "explanation": _stringify_ai_value(details.get("explanation") or details.get("analysis") or details),
                        "confidence": details.get("confidence", 0.0),
                        "risk_flags": _listify_ai_value(details.get("risk_flags") or details.get("risks")),
                        "what_to_watch_next": _stringify_ai_value(details.get("what_to_watch_next") or details.get("watch_next")),
                    }
                else:
                    symbol_interpretations[str(symbol).upper()] = {
                        "bias": "WAITING",
                        "explanation": _stringify_ai_value(details),
                        "confidence": 0.0,
                        "risk_flags": [],
                        "what_to_watch_next": "",
                    }
        repaired["symbol_interpretations"] = symbol_interpretations
        changed = True

    if repaired.get("overall_risk") not in ("low", "medium", "high"):
        repaired["overall_risk"] = (
            _normalize_risk(resp.get("overall_risk"))
            or _normalize_risk(resp.get("risk"))
            or _normalize_risk(resp.get("risk_level"))
            or _normalize_risk(resp.get("risk_flags"))
            or "medium"
        )
        changed = True

    if repaired.get("regime") not in ("trending", "choppy", "absorbing", "sweep-heavy", "mixed"):
        repaired["regime"] = (
            _normalize_regime(resp.get("regime"))
            or _normalize_regime(resp.get("market_regime"))
            or _normalize_regime(resp.get("overall_market_assessment"))
            or "mixed"
        )
        changed = True

    if "cleanest_bias_symbols" not in repaired:
        repaired["cleanest_bias_symbols"] = _listify_ai_value(
            resp.get("cleanest_bias_symbols")
            or resp.get("cleanest_biases")
            or resp.get("cleanest_symbols")
        )
        changed = True

    if "suppressed_symbols" not in repaired:
        repaired["suppressed_symbols"] = _listify_ai_value(
            resp.get("suppressed_symbols")
            or resp.get("suppressed_signals")
        )
        changed = True

    if "data_quality_warnings" not in repaired:
        repaired["data_quality_warnings"] = _listify_ai_value(
            resp.get("data_quality_warnings")
            or resp.get("warnings")
        )
        changed = True

    return repaired, changed


def read_csv_tail(filepath: str, n: int) -> List[Dict[str, str]]:
    """Reads the last N rows of a CSV file safely."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[-n:]
    except Exception as e:
        logger.error(f"Error reading CSV tail for {filepath}: {_safe_text(str(e))}")
        return []

def validate_ai_response(resp: dict) -> Tuple[bool, Optional[str]]:
    """Validates the response dictionary against the V2.5 required JSON schema."""
    if not isinstance(resp, dict):
        return False, "Response is not a dictionary."
    
    # Required keys check
    required_keys = [
        "market_summary", "symbol_interpretations", "overall_risk", 
        "cleanest_bias_symbols", "suppressed_symbols", "regime", 
        "data_quality_warnings", "not_trade_advice"
    ]
    for key in required_keys:
        if key not in resp:
            return False, f"Missing required JSON schema key: '{key}'."
            
    # Value constraints validation
    if resp["overall_risk"] not in ["low", "medium", "high"]:
        return False, f"Invalid overall_risk value: '{resp['overall_risk']}' (must be low, medium, or high)."
        
    if resp["regime"] not in ["trending", "choppy", "absorbing", "sweep-heavy", "mixed"]:
        return False, f"Invalid regime value: '{resp['regime']}' (must be trending, choppy, absorbing, sweep-heavy, or mixed)."
        
    if not isinstance(resp["symbol_interpretations"], dict):
        return False, "symbol_interpretations is not an object."
        
    if not isinstance(resp["cleanest_bias_symbols"], list):
        return False, "cleanest_bias_symbols is not an array."
        
    if not isinstance(resp["suppressed_symbols"], list):
        return False, "suppressed_symbols is not an array."
        
    if resp["not_trade_advice"] is not True:
        return False, "not_trade_advice must be set to true."
        
    return True, None


def scanner_actions_from_snapshot(prompt_snapshot: dict) -> Dict[str, str]:
    symbols = (prompt_snapshot or {}).get("symbols", {})
    return {
        str(symbol).upper(): str(data.get("next_action", "WAITING"))
        for symbol, data in symbols.items()
        if isinstance(data, dict)
    }


def enforce_scanner_biases(resp: dict, scanner_actions: Dict[str, str]) -> Tuple[dict, bool]:
    """
    Keeps AI explanations subordinate to the scanner. The provider may explain,
    but displayed/cached bias must match scanner next_action exactly.
    """
    if not isinstance(resp, dict):
        return resp, False
    interpretations = resp.get("symbol_interpretations")
    if not isinstance(interpretations, dict):
        return resp, False

    changed = False
    for symbol, scanner_action in scanner_actions.items():
        info = interpretations.get(symbol)
        if not isinstance(info, dict):
            continue
        if info.get("bias") != scanner_action:
            info["ai_reported_bias"] = info.get("bias")
            info["bias"] = scanner_action
            info["scanner_action_enforced"] = True
            changed = True
    return resp, changed

class AIInterpreter:
    def __init__(self, lock: Optional[asyncio.Lock] = None, runtime_config=None):
        self.cache_file = os.path.join(os.path.dirname(__file__), "ai_cache.json")
        self.log_csv = os.path.join(os.path.dirname(__file__), "ai_interpretations.csv")
        self.cache: Dict[str, Any] = self._load_cache()
        self._ensure_log_headers()
        self.lock = lock or asyncio.Lock()
        self.runtime_config = runtime_config

    def _load_cache(self) -> Dict[str, Any]:
        """Loads cached interpretation from file if it exists and is valid."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {_safe_text(str(e))}")
        return {
            "timestamp": "",
            "provider": "",
            "model": "",
            "input_hash": "",
            "expires_at": 0.0,
            "ok": False,
            "interpretation": {},
            "error": "No cache loaded",
            "compliance_warning": False,
            "schema_repaired": False,
            "fallback_used": False,
            "stale_cache_timestamp": None
        }

    def _save_cache(self, cache_obj: dict):
        """Saves current interpretation cache status to file."""
        self.cache = cache_obj
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {_safe_text(str(e))}")

    def _ensure_log_headers(self):
        """Creates or migrates ai_interpretations.csv with correct schema headers if missing."""
        headers = [
            "timestamp", "provider", "model", "scope", "input_hash", 
            "symbols_count", "overall_risk", "regime", "interpretation_json", 
            "latency_ms", "ok", "error", "compliance_warning", "schema_repaired", "fallback_used"
        ]
        if not os.path.exists(self.log_csv):
            try:
                with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            except Exception as e:
                logger.error(f"Failed to initialize CSV log: {_safe_text(str(e))}")
            return

        # Migration logic if file exists but lacks new columns
        try:
            with open(self.log_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if rows:
                current_headers = rows[0]
                missing_headers = [h for h in headers if h not in current_headers]
                if missing_headers:
                    logger.info(f"Migrating ai_interpretations.csv to include audit columns: {', '.join(missing_headers)}")
                    defaults = {
                        "compliance_warning": "False",
                        "schema_repaired": "False",
                        "fallback_used": "False",
                    }
                    new_rows = [headers]
                    for r in rows[1:]:
                        row_map = {
                            current_headers[i]: r[i]
                            for i in range(min(len(current_headers), len(r)))
                        }
                        new_rows.append([row_map.get(h, defaults.get(h, "")) for h in headers])
                    with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(new_rows)
        except Exception as e:
            logger.error(f"Failed to migrate CSV log: {_safe_text(str(e))}")

    def _log_interpretation(self, provider: str, model: str, input_hash: str, symbols_count: int,
                            overall_risk: str, regime: str, interpretation_json: str, 
                            latency_ms: int, ok: bool, error: Optional[str],
                            compliance_warning: bool, schema_repaired: bool, fallback_used: bool):
        """Logs a single interpretation attempt to CSV."""
        timestamp = datetime.now(timezone.utc).isoformat()
        row = [
            timestamp, provider, model, "market_summary", input_hash,
            symbols_count, overall_risk, regime, _safe_text(interpretation_json) or "",
            latency_ms, str(ok), _safe_text(error) or "",
            str(compliance_warning), str(schema_repaired), str(fallback_used)
        ]
        try:
            with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write interpretation log: {_safe_text(str(e))}")


    def build_compact_snapshot(self, symbol_data: dict, binance_context: Optional[dict] = None) -> Tuple[dict, dict]:
        """
        Builds both a compact, rounded snapshot for stable hashing,
        and a full snapshot suitable for the AI prompt context.
        """
        transitions = read_csv_tail(os.path.join(os.path.dirname(__file__), "bias_transitions.csv"), 20)
        signals = read_csv_tail(os.path.join(os.path.dirname(__file__), "bias_signals.csv"), 20)

        compact_symbols = {}
        full_symbols = {}
        
        binance_symbols = {}
        if isinstance(binance_context, dict):
            binance_symbols = binance_context.get("symbols", {}) or {}

        for sym, data in sorted(symbol_data.items()):
            price = data.get("price", 0.0)
            rounded_price = round(price, 2) if price >= 1.0 else round(price, 4)
            
            # Aggregate flow deltas to nearest $1K and imbalance to 2 decimals
            delta_1m = round(data.get("delta_1m_usdt", 0.0) / 1000.0) * 1000
            delta_5m = round(data.get("delta_5m_usdt", 0.0) / 1000.0) * 1000
            delta_15m = round(data.get("delta_15m_usdt", 0.0) / 1000.0) * 1000
            cvd_sess = round(data.get("session_cvd_usdt", 0.0) / 1000.0) * 1000
            imbalance = round(data.get("imbalance", 0.0), 2)
            
            ctx = binance_symbols.get(sym.upper(), {})
            compact_context = {}
            prompt_context = {}
            if ctx:
                compact_context = {
                    "mark_premium_pct": ctx.get("mark_premium_pct"),
                    "funding_rate_pct": ctx.get("funding_rate_pct"),
                    "open_interest_change_pct": ctx.get("open_interest_change_pct"),
                    "trend_5m": ctx.get("trend_5m"),
                    "trend_15m": ctx.get("trend_15m"),
                    "context_quality": ctx.get("context_quality"),
                    "context_confirm": ctx.get("context_confirm"),
                }
                prompt_context = {
                    "mark_price": ctx.get("mark_price"),
                    "index_price": ctx.get("index_price"),
                    "mark_premium_pct": ctx.get("mark_premium_pct"),
                    "funding_rate_pct": ctx.get("funding_rate_pct"),
                    "current_open_interest": ctx.get("current_open_interest"),
                    "open_interest_change_pct": ctx.get("open_interest_change_pct"),
                    "trend_1m": ctx.get("trend_1m"),
                    "trend_5m": ctx.get("trend_5m"),
                    "trend_15m": ctx.get("trend_15m"),
                    "orderbook_snapshot_imbalance": ctx.get("orderbook_snapshot_imbalance"),
                    "book_drift_vs_ws_imbalance": ctx.get("book_drift_vs_ws_imbalance"),
                    "context_quality": ctx.get("context_quality"),
                    "context_confirm": ctx.get("context_confirm"),
                    "warnings": ctx.get("warnings", []),
                }

            # 1. Compact dictionary for stable input hash computation
            compact_symbols[sym.upper()] = {
                "price": rounded_price,
                "delta_5m_usdt": delta_5m,
                "session_cvd_usdt": cvd_sess,
                "imbalance": imbalance,
                "next_action": data.get("next_action", "WAITING"),
                "latest_event": data.get("latest_event") or "-",
                "binance_context": compact_context
            }

            # 2. Detailed dictionary for rich prompt details
            full_symbols[sym.upper()] = {
                "price": rounded_price,
                "delta_1m_usdt": delta_1m,
                "delta_5m_usdt": delta_5m,
                "delta_15m_usdt": delta_15m,
                "session_cvd_usdt": cvd_sess,
                "imbalance": imbalance,
                "next_action": data.get("next_action", "WAITING"),
                "latest_event": data.get("latest_event") or "-",
                "binance_context": prompt_context
            }

        compact_transitions = []
        for t in transitions:
            compact_transitions.append({
                "symbol": t.get("symbol", ""),
                "old_action": t.get("old_action", ""),
                "new_action": t.get("new_action", ""),
                "suppression_reason": t.get("suppression_reason", "NONE")
            })

        compact_signals = []
        for s in signals:
            compact_signals.append({
                "symbol": s.get("symbol", ""),
                "direction": s.get("direction", ""),
                "final_return_pct": s.get("final_return_pct", "0.0")
            })

        hash_snapshot = {
            "symbols": compact_symbols,
            "recent_transitions": compact_transitions,
            "recent_completed_signals": compact_signals
        }
        
        prompt_snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": full_symbols,
            "recent_transitions": compact_transitions,
            "recent_completed_signals": compact_signals,
            "scanner_mode": "visual_only_no_execution",
            "binance_context_status": (binance_context or {}).get("status", {})
        }
        
        return hash_snapshot, prompt_snapshot

    def get_latest(self, current_symbol_data: Optional[dict] = None) -> Dict[str, Any]:
        """Returns the current cached interpretation state directly (no generation)."""
        latest = dict(self.cache)
        if not latest.get("ai_snapshot_timestamp"):
            latest["ai_snapshot_timestamp"] = latest.get("timestamp")
        if not latest.get("scanner_snapshot_timestamp"):
            latest["scanner_snapshot_timestamp"] = latest.get("stale_cache_timestamp") or latest.get("timestamp")
        cache_age_seconds = _timestamp_age_seconds(latest.get("ai_snapshot_timestamp") or latest.get("timestamp"))
        latest["cache_age_seconds"] = cache_age_seconds
        latest["stale"] = bool(latest.get("fallback_used")) or (
            cache_age_seconds is not None and cache_age_seconds > config.AI_CACHE_TTL_SECONDS
        )
        latest["latest_scanner_timestamp"] = datetime.now(timezone.utc).isoformat()
        if current_symbol_data:
            latest["current_scanner_actions"] = {
                str(sym).upper(): str(data.get("next_action", "WAITING"))
                for sym, data in current_symbol_data.items()
                if isinstance(data, dict)
            }
        return latest

    async def interpret(self, symbol_data: dict, force: bool = False, binance_context: Optional[dict] = None) -> Dict[str, Any]:
        """
        Evaluates the current snapshot. Caches calls using TTL and input hashes.
        Uses a task lock to prevent concurrent auto-loop and manual POST updates.
        """
        async with self.lock:
            return await self._interpret_locked(symbol_data, force, binance_context)

    async def _interpret_locked(self, symbol_data: dict, force: bool = False, binance_context: Optional[dict] = None) -> Dict[str, Any]:
        """Runs the locked interpretation logic."""
        now = time.time()
        
        # 1. Build snapshots
        hash_snap, prompt_snap = self.build_compact_snapshot(symbol_data, binance_context)
        scanner_actions = scanner_actions_from_snapshot(prompt_snap)
        
        # 2. Calculate stable hash of the metrics state
        snap_bytes = json.dumps(hash_snap, sort_keys=True).encode("utf-8")
        input_hash = hashlib.sha256(snap_bytes).hexdigest()
        
        # 3. Check cache hit
        if not force and self.cache.get("ok", False) and self.cache.get("input_hash") == input_hash:
            if now < self.cache.get("expires_at", 0.0):
                logger.info("AI interpretation cache hit (TTL valid and hash matched).")
                return self.cache

        # 4. Initialize client provider
        provider_name = self.runtime_config.provider if self.runtime_config is not None else config.AI_PROVIDER
        provider = ai_providers.get_provider(provider_name, config.AI_TIMEOUT_SECONDS, self.runtime_config)
        valid, err_msg = provider.validate_config()
        if not valid:
            err_obj = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ai_snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "scanner_snapshot_timestamp": prompt_snap.get("timestamp"),
                "scanner_actions": scanner_actions,
                "provider": provider_name,
                "model": provider.model,
                "input_hash": input_hash,
                "expires_at": now + 10.0,  # Fast retry on config errors
                "ok": False,
                "interpretation": {},
                "error": _safe_text(err_msg or "AI Provider settings validation failed."),
                "compliance_warning": False,
                "schema_repaired": False,
                "fallback_used": False,
                "stale_cache_timestamp": None
            }
            self._log_interpretation(
                provider=provider_name,
                model=provider.model,
                input_hash=input_hash,
                symbols_count=len(symbol_data),
                overall_risk="",
                regime="",
                interpretation_json="",
                latency_ms=0,
                ok=False,
                error=err_obj["error"],
                compliance_warning=False,
                schema_repaired=False,
                fallback_used=False
            )
            self._save_cache(err_obj)
            return err_obj

        # 5. Format prompt and make async call
        user_prompt = ai_prompts.USER_PROMPT_TEMPLATE.format(
            snapshot_json=json.dumps(prompt_snap, indent=2)
        )
        system_prompt = ai_prompts.SYSTEM_PROMPT
        
        if config.AI_LOG_PROMPTS:
            logger.info(f"[AI Prompt Log] System:\n{_safe_text(system_prompt)}\nUser:\n{_safe_text(user_prompt)}")

        logger.info(f"Triggering fresh AI interpretation using provider '{provider_name}'...")
        res = await provider.interpret(system_prompt, user_prompt)
        
        # 6. Evaluate response result
        ok = res.get("ok", False)
        err = _safe_text(res.get("error"))
        latency = res.get("latency_ms", 0)
        parsed_json = res.get("json")
        raw_text = _safe_text(res.get("raw_text"))

        compliance_warning = False
        schema_repaired = False
        fallback_used = False

        # Validate schema structure on success
        if ok and parsed_json:
            parsed_json, repaired_shape = repair_ai_response_schema(parsed_json)
            if repaired_shape:
                schema_repaired = True
                logger.warning("AI response schema naming drift repaired before validation.")

            parsed_json, bias_repaired = enforce_scanner_biases(parsed_json, scanner_actions)
            if bias_repaired:
                schema_repaired = True
                logger.warning("AI response bias mismatch repaired to scanner next_action.")

            # Audit compliance disclaimer
            llm_advice = parsed_json.get("not_trade_advice")
            if llm_advice is not True:
                compliance_warning = True
                schema_repaired = True
                logger.warning(f"Compliance violation: provider returned not_trade_advice={llm_advice}. Repairing schema.")
            
            # Enforce safety
            parsed_json["not_trade_advice"] = True
            
            schema_ok, schema_err = validate_ai_response(parsed_json)
            if not schema_ok:
                ok = False
                err = f"Schema Validation Failure: {schema_err}"
                parsed_json = None
                logger.error(f"AI response schema validation failed: {_safe_text(err)}")

        # Handle fallback if call failed but we have a valid previous cache
        if not ok and self.cache.get("ok", False):
            logger.warning(f"AI call failed: {_safe_text(err)}. Falling back to previous cached interpretation.")
            fallback_used = True
            
            # Log the failed attempt with fallback flag set
            self._log_interpretation(
                provider=provider_name,
                model=provider.model,
                input_hash=input_hash,
                symbols_count=len(symbol_data),
                overall_risk="",
                regime="",
                interpretation_json=raw_text or str(err),
                latency_ms=latency,
                ok=ok,
                error=err,
                compliance_warning=compliance_warning,
                schema_repaired=schema_repaired,
                fallback_used=fallback_used
            )
            
            # Extend cache TTL so we don't spam requests immediately on outage.
            fallback_cache = dict(self.cache)
            fallback_cache["expires_at"] = now + config.AI_CACHE_TTL_SECONDS
            fallback_cache["fallback_used"] = True
            fallback_cache["fallback_at"] = datetime.now(timezone.utc).isoformat()
            fallback_cache["fallback_reason"] = _safe_text(err)
            fallback_cache["stale_cache_timestamp"] = self.cache.get("timestamp")
            fallback_cache["ai_snapshot_timestamp"] = self.cache.get("ai_snapshot_timestamp") or self.cache.get("timestamp")
            self._save_cache(fallback_cache)
            return fallback_cache

        # 7. Write to CSV audit log
        if not fallback_used:
            overall_risk = parsed_json.get("overall_risk", "") if parsed_json else ""
            regime = parsed_json.get("regime", "") if parsed_json else ""
            logged_json = json.dumps(parsed_json) if parsed_json else (raw_text or "")
            self._log_interpretation(
                provider=provider_name,
                model=provider.model,
                input_hash=input_hash,
                symbols_count=len(symbol_data),
                overall_risk=overall_risk,
                regime=regime,
                interpretation_json=logged_json,
                latency_ms=latency,
                ok=ok,
                error=err,
                compliance_warning=compliance_warning,
                schema_repaired=schema_repaired,
                fallback_used=fallback_used
            )

        # 8. Update cache JSON object
        expires_at = now + config.AI_CACHE_TTL_SECONDS
        cache_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ai_snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
            "scanner_snapshot_timestamp": prompt_snap.get("timestamp"),
            "scanner_actions": scanner_actions,
            "provider": provider_name,
            "model": provider.model,
            "input_hash": input_hash,
            "expires_at": expires_at,
            "ok": ok,
            "interpretation": parsed_json or {},
            "error": _safe_text(err),
            "compliance_warning": compliance_warning,
            "schema_repaired": schema_repaired,
            "fallback_used": False,
            "stale_cache_timestamp": None
        }
        
        self._save_cache(cache_obj)
        logger.info(f"AI interpretation completed. Success: {ok} | Latency: {latency}ms")
        return cache_obj
