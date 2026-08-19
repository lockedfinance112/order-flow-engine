# TICS Phase 1C.1 — Task 9 (Shadow-Only Application Integration) Walkthrough

## Summary
Task 9 integrates the Phase 1C.1 Liquidity Event Engine into the live application environment under strictly observational shadow mode with zero trading, sizing, or scoring authority.

## Frozen Authority Flags
Verified in `config.py` and across all integration tests:
- `LIQUIDITY_EVENT_ENGINE_ENABLED = True`
- `LIQUIDITY_EVENT_ENFORCEMENT_ENABLED = False`
- `REGIME_ENFORCEMENT_ENABLED = False`
- `EXECUTION_DISABLED = True`

## Architecture & Data Flow
1. **Trade Feed**: Incoming market trades from `TradeStream` are fed to `OrderFlowEngine._process_trades_loop`, creating canonical `MarketTrade` instances with explicit Guardian-derived `TradeCoverage` and forwarding to `LiquidityEventEngine.on_trade`.
2. **Depth Feed**: WebSocket depth updates are converted to canonical `DepthObservation` objects and forwarded to `LiquidityEventEngine.on_depth`.
3. **Sweep Feed**: Detected sweeps from `SweepsMonitor` are evaluated by `SweepsMonitorAdapter` and opened as shadow `LiquiditySweepObservation` via `LiquidityEventEngine.on_sweep`.
4. **Context Adapters**: Legacy depth context is consumed strictly via `LegacyLiquidityContextAdapter`.
5. **Read-Only API**: `GET /api/liquidity-events` provides read-only active events, recent resolved events, and engine telemetry. All POST/mutation requests are strictly rejected.
6. **Dashboard**: Added a compact "Liquidity Events (Shadow)" panel explicitly designated as uncalibrated and non-executing.
7. **Scorer & Execution Invariance**: Scorer outputs and paper trader portfolio logic remain mathematically and behaviorally identical whether the liquidity engine is enabled or disabled.
8. **Shutdown Safety**: Guaranteed flush of recorder artifacts and closing of `SQLiteIdentityAuthority`.

## Verification
- Task 9 Integration Suite: 26 passed, 0 failed
- Full Phase 1C.1 Suite: 333 passed, 0 failed
- Full Repository Suite: 418 passed, 0 failed
- `compileall` and `git diff --check` clean.
