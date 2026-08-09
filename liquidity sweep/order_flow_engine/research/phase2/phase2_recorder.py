import os
import csv
import time
import uuid
import json
import copy
import logging
import subprocess
from datetime import datetime, timezone

logger = logging.getLogger("OrderFlow.Phase2Recorder")

COLUMNS = [
    "signal_id", "symbol", "direction", "action", "entry_time", "entry_price",
    "scanner_cycle_id", "action_version", "decision_reason",
    "delta_1m_usdt", "delta_5m_usdt", "delta_15m_usdt", "session_cvd_usdt",
    "buy_ratio_1m", "buy_ratio_5m", "imbalance", "spread_bps",
    "gate_1m_delta", "gate_5m_delta_bias", "gate_aggression", "gate_imbalance", "gate_stacking", "gate_no_opposite_conflict",
    "sweep_active", "sweep_direction", "sweep_score", "sweep_age_seconds",
    "latest_event", "recent_events",
    "funding_rate_pct", "mark_premium_pct", "open_interest", "open_interest_change_pct",
    "trend_1m", "trend_5m", "trend_15m", "book_snapshot_imbalance", "book_drift", "context_quality",
    "binance_context_timestamp", "binance_context_age_seconds", "binance_last_slow_context_time", "binance_context_confirm",
    "price_after_1m", "price_after_3m", "price_after_5m", "price_after_15m",
    "price_after_1m_time", "price_after_3m_time", "price_after_5m_time", "price_after_15m_time",
    "horizon_1m_delay_seconds", "horizon_3m_delay_seconds", "horizon_5m_delay_seconds", "horizon_15m_delay_seconds",
    "return_1m_pct", "return_3m_pct", "return_5m_pct", "return_15m_pct",
    "max_favorable_pct", "max_adverse_pct",
    "completion_status", "horizon_1m_status", "horizon_3m_status", "horizon_5m_status", "horizon_15m_status",
    "recovered_after_restart", "excursion_coverage_status",
    "research_schema_version", "strategy_version", "engine_commit_sha", "dataset_class",
    "completed_time"
]

def _get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"

class Phase2SignalRecorder:
    def __init__(self, output_dir="research/phase2", signals_csv="signal_outcomes_v2.csv", active_json="active_signals_v2.json"):
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), output_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.signals_csv = os.path.join(self.base_dir, signals_csv)
        self.active_json = os.path.join(self.base_dir, active_json)
        
        self.active_tracks = []
        self.registered_idempotency_keys = set()
        self.last_checkpoint_time = time.time()
        self.dirty = False
        
        self._ensure_headers()
        self._load_active_signals()

    def _ensure_headers(self):
        if not os.path.exists(self.signals_csv):
            try:
                with open(self.signals_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(COLUMNS)
            except Exception as e:
                logger.error(f"Failed to create CSV headers: {e}")

    def _load_active_signals(self):
        if os.path.exists(self.active_json):
            try:
                with open(self.active_json, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    now = time.time()
                    for t in loaded:
                        elapsed = now - t["entry_time"]
                        
                        t["recovered_after_restart"] = True
                        t["excursion_coverage_status"] = "INTERRUPTED"
                        
                        # Populate idempotency key from loaded track
                        ik = f"{t['symbol'].lower()}_{t['action'].lower()}_{t['scanner_cycle_id']}_{t['action_version']}"
                        self.registered_idempotency_keys.add(ik)
                        
                        horizons = [("1m", 60.0), ("3m", 180.0), ("5m", 300.0), ("15m", 900.0)]
                        for h, target in horizons:
                            if t.get(f"horizon_{h}_status") == "PENDING" and elapsed >= target:
                                t[f"horizon_{h}_status"] = "MISSED_DURING_DOWNTIME"
                        
                        if elapsed >= 900.0:
                            t["completion_status"] = "INTERRUPTED"
                            # Attempt write; if it fails, keep in active tracks to retry later
                            if self._append_to_csv(t):
                                continue
                        
                        self.active_tracks.append(t)
                self.dirty = True
                self.flush_checkpoint(force=True)
            except Exception as e:
                logger.error(f"Failed to load active signals: {e}")

    def checkpoint_due(self, now: float) -> bool:
        return self.dirty and (now - self.last_checkpoint_time >= 5.0)

    def flush_checkpoint(self, force=False):
        if not self.dirty and not force:
            return
        try:
            temp_path = self.active_json + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.active_tracks, f, indent=2)
            os.replace(temp_path, self.active_json)
            self.last_checkpoint_time = time.time()
            self.dirty = False
        except Exception as e:
            logger.error(f"Failed to write active signals checkpoint: {e}")

    def register_signal_change(self, symbol: str, old_action: str, new_action: str, price: float,
                                decision_snapshot: dict, metrics_1m: dict, metrics_5m: dict, metrics_15m: dict,
                                imbalance: float, session_cvd_usdt: float, spread_bps: float, latest_event: str | None,
                                recent_events: list, binance_context: dict):
        try:
            if new_action not in ("CONFIRMED_LONG", "CONFIRMED_SHORT"):
                return
            if old_action == new_action:
                return

            decision = copy.deepcopy(decision_snapshot)
            scanner_cycle = decision.get("scanner_cycle_id", 0)
            action_version = decision.get("action_version", 1)

            # 1. Enforce Idempotency check using symbol + action + cycle + version
            ik = f"{symbol.lower()}_{new_action.lower()}_{scanner_cycle}_{action_version}"
            if ik in self.registered_idempotency_keys:
                logger.info(f"[Research] Idempotent key {ik} already registered. Skipping.")
                return
            self.registered_idempotency_keys.add(ik)

            # Anchor entry_time to the committed decision timestamp
            entry_time = decision.get("timestamp", time.time())
            direction = "LONG" if "LONG" in new_action else "SHORT"
            signal_id = str(uuid.uuid4())
            commit_sha = _get_git_commit_sha()

            gates = decision.get("gates", {})
            sweep = decision.get("active_sweep", {})

            if direction == "LONG":
                gate_1m_delta = gates.get("1m_delta_positive", "FAIL")
            else:
                gate_1m_delta = gates.get("1m_delta_negative", "FAIL")

            bc = copy.deepcopy(binance_context or {})
            bc_ts = bc.get("timestamp")
            bc_age = None
            if bc_ts:
                try:
                    dt = datetime.fromisoformat(bc_ts.replace("Z", "+00:00"))
                    bc_age = entry_time - dt.timestamp()
                except Exception:
                    pass

            track = {
                "signal_id": signal_id,
                "symbol": symbol.upper(),
                "direction": direction,
                "action": new_action,
                "entry_time": entry_time,
                "entry_price": price,
                "scanner_cycle_id": scanner_cycle,
                "action_version": action_version,
                "decision_reason": decision.get("reason", ""),
                
                # Flow Metrics Snapshot
                "delta_1m_usdt": metrics_1m.get("delta_usdt", 0.0),
                "delta_5m_usdt": metrics_5m.get("delta_usdt", 0.0),
                "delta_15m_usdt": metrics_15m.get("delta_usdt", 0.0),
                "session_cvd_usdt": session_cvd_usdt,
                "buy_ratio_1m": metrics_1m.get("buy_ratio", 0.5),
                "buy_ratio_5m": metrics_5m.get("buy_ratio", 0.5),
                "imbalance": imbalance,
                "spread_bps": spread_bps,
                
                # Gates Snapshot
                "gate_1m_delta": gate_1m_delta,
                "gate_5m_delta_bias": gates.get("5m_delta_bias", "FAIL"),
                "gate_aggression": gates.get("buy_aggression", "FAIL") if direction == "LONG" else gates.get("sell_aggression", "FAIL"),
                "gate_imbalance": gates.get("imbalance_bullish", "FAIL") if direction == "LONG" else gates.get("imbalance_bearish", "FAIL"),
                "gate_stacking": gates.get("stacking_bullish", "FAIL") if direction == "LONG" else gates.get("stacking_bearish", "FAIL"),
                "gate_no_opposite_conflict": gates.get("no_bearish_conflict", "FAIL") if direction == "LONG" else gates.get("no_bullish_conflict", "FAIL"),
                
                # Sweep Snapshot
                "sweep_active": sweep.get("active", False),
                "sweep_direction": sweep.get("direction", ""),
                "sweep_score": sweep.get("score", 0),
                "sweep_age_seconds": sweep.get("age_seconds", 999.0),
                
                # Events Context
                "latest_event": latest_event or "",
                "recent_events": json.dumps(list(recent_events or [])),
                
                # Binance Context
                "funding_rate_pct": bc.get("funding_rate_pct"),
                "mark_premium_pct": bc.get("mark_premium_pct"),
                "open_interest": bc.get("current_open_interest"),
                "open_interest_change_pct": bc.get("open_interest_change_pct"),
                "trend_1m": bc.get("trend_1m"),
                "trend_5m": bc.get("trend_5m"),
                "trend_15m": bc.get("trend_15m"),
                "book_snapshot_imbalance": bc.get("orderbook_snapshot_imbalance"),
                "book_drift": bc.get("book_drift_vs_ws_imbalance"),
                "context_quality": bc.get("context_quality", "MISSING"),
                
                "binance_context_timestamp": bc_ts,
                "binance_context_age_seconds": bc_age,
                "binance_last_slow_context_time": bc.get("last_slow_context_time"),
                "binance_context_confirm": bc.get("context_confirm"),
                
                # Forward Horizon Tracking Outputs
                "price_after_1m": None, "price_after_3m": None, "price_after_5m": None, "price_after_15m": None,
                "price_after_1m_time": None, "price_after_3m_time": None, "price_after_5m_time": None, "price_after_15m_time": None,
                "horizon_1m_delay_seconds": None, "horizon_3m_delay_seconds": None, "horizon_5m_delay_seconds": None, "horizon_15m_delay_seconds": None,
                "return_1m_pct": None, "return_3m_pct": None, "return_5m_pct": None, "return_15m_pct": None,
                
                "max_favorable_pct": 0.0,
                "max_adverse_pct": 0.0,
                "completion_status": "PENDING",
                
                "horizon_1m_status": "PENDING",
                "horizon_3m_status": "PENDING",
                "horizon_5m_status": "PENDING",
                "horizon_15m_status": "PENDING",
                
                # Excursion coverage flags
                "recovered_after_restart": False,
                "excursion_coverage_status": "COMPLETE",
                
                # Provenance
                "research_schema_version": 2,
                "strategy_version": "PHASE1_FROZEN",
                "engine_commit_sha": commit_sha,
                "dataset_class": "CANONICAL_PHASE2",
                "completed_time": ""
            }

            self.active_tracks.append(track)
            self.dirty = True
            self.flush_checkpoint(force=True)
            logger.info(f"[Research] Registered track {new_action} for {symbol.upper()} @ {price}")
        except Exception as e:
            logger.error(f"Error registering Phase 2 track: {e}", exc_info=True)

    def update_price(self, symbol: str, price: float):
        try:
            now = time.time()
            sym_upper = symbol.upper()
            
            for track in self.active_tracks:
                if track["symbol"] == sym_upper:
                    entry_price = track["entry_price"]
                    direction = track["direction"]
                    
                    if direction == "LONG":
                        fav = (price - entry_price) / entry_price
                        adv = (entry_price - price) / entry_price
                    else:
                        fav = (entry_price - price) / entry_price
                        adv = (price - entry_price) / entry_price
                        
                    # Set dirty = True if excursions updated
                    if track["completion_status"] == "PENDING" and track.get("excursion_coverage_status") == "COMPLETE":
                        old_fav = track["max_favorable_pct"]
                        old_adv = track["max_adverse_pct"]
                        
                        track["max_favorable_pct"] = max(old_fav, max(0.0, fav))
                        track["max_adverse_pct"] = max(old_adv, max(0.0, adv))
                        
                        if track["max_favorable_pct"] != old_fav or track["max_adverse_pct"] != old_adv:
                            self.dirty = True
                    
                    elapsed = now - track["entry_time"]
                    
                    # Horizon updates
                    self._check_horizon(track, "1m", 60.0, elapsed, price, now)
                    self._check_horizon(track, "3m", 180.0, elapsed, price, now)
                    self._check_horizon(track, "5m", 300.0, elapsed, price, now)
                    self._check_horizon(track, "15m", 900.0, elapsed, price, now)
        except Exception as e:
            logger.error(f"Error in Phase 2 update_price: {e}", exc_info=True)

    def _check_horizon(self, track: dict, h: str, target: float, elapsed: float, price: float, now: float):
        if elapsed >= target and track[f"price_after_{h}"] is None and track[f"horizon_{h}_status"] == "PENDING":
            track[f"price_after_{h}"] = price
            track[f"price_after_{h}_time"] = now
            delay = elapsed - target
            track[f"horizon_{h}_delay_seconds"] = delay
            
            entry_price = track["entry_price"]
            if track["direction"] == "LONG":
                ret = (price - entry_price) / entry_price
            else:
                ret = (entry_price - price) / entry_price
            track[f"return_{h}_pct"] = ret
            track[f"horizon_{h}_status"] = "CAPTURED"
            self.dirty = True

    def finalize_expired_signals(self):
        try:
            now = time.time()
            to_remove = []
            
            for track in self.active_tracks:
                elapsed = now - track["entry_time"]
                if elapsed >= 900.0:  # 15m
                    
                    # Determine completed/interrupted outcome statuses
                    if track["horizon_15m_status"] == "PENDING":
                        track["horizon_15m_status"] = "INTERRUPTED"
                        track["completion_status"] = "INTERRUPTED"
                    elif any(track[f"horizon_{h}_status"] in ("MISSED_DURING_DOWNTIME", "INTERRUPTED") for h in ["1m", "3m", "5m", "15m"]):
                        track["completion_status"] = "INTERRUPTED"
                    else:
                        track["completion_status"] = "CAPTURED"
                        
                    track["completed_time"] = datetime.now(timezone.utc).isoformat()
                    
                    # Mark any remaining pending horizons as interrupted
                    for h in ["1m", "3m", "5m", "15m"]:
                        if track[f"horizon_{h}_status"] == "PENDING":
                            track[f"horizon_{h}_status"] = "INTERRUPTED"
                    
                    # Safe CSV append: do not delete track from active list if write fails!
                    if self._append_to_csv(track):
                        to_remove.append(track)
                    
            if to_remove:
                for track in to_remove:
                    self.active_tracks.remove(track)
                self.dirty = True
                self.flush_checkpoint(force=True)
        except Exception as e:
            logger.error(f"Error finalising Phase 2 signals: {e}", exc_info=True)

    def _append_to_csv(self, track: dict) -> bool:
        try:
            row = [track.get(col, "") for col in COLUMNS]
            with open(self.signals_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            logger.info(f"[Research] Logged signal outcome: {track['symbol']} | {track['direction']} | {track['completion_status']}")
            return True
        except Exception as e:
            logger.error(f"Failed to append to CSV: {e}")
            return False

    def shutdown(self):
        try:
            logger.info("Graceful shutdown: checkpointing active signals...")
            self.flush_checkpoint(force=True)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
