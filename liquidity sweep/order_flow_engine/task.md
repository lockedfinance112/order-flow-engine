# Task List: Order Flow Engine V2.4 Action Indicator

- [x] Add Flow Bias / Next Action logic to `scoring.py`.
- [x] Update `dashboard.py` to display and style the new `Next Action` column in terminal and Web HTML templates.
- [x] Update `main.py` to parse, cache, and transmit the symbol action signals.
- [x] Verify compilation and launch the engine.
- [x] Inject sweeps and execute analysis to verify the indicators trigger correctly.

# Task List: Order Flow Engine V2.4.1 (Conflict Suppression & Signal Tracker)
- [x] Add BEARISH/BULLISH conflict suppression sets to `scoring.py`.
- [x] Implement conflict-aware `get_bias_action()` evaluation in `scoring.py`.
- [x] Create `signal_tracker.py` for persistent signal outcomes and transition logging.
- [x] Hook tracker updates and transition listeners into `main.py`.
- [x] Build signal outcome research analyzer `analyze_bias_performance.py`.
- [x] Compile check and start live soak run of V2.4.1.

# Task List: Order Flow Engine V2.4.2 (Strict Suppression & Reason Logging)
- [x] Refactor conflict checking in `scoring.py` to check all active alerts inside a 300-second lookback list.
- [x] Return the specific suppression reason as a tuple `(action, suppression_reason)` from `get_bias_action()`.
- [x] Add self-healing check to reset transitions CSV header if legacy schema exists.
- [x] Log the specific `suppression_reason` to `bias_transitions.csv` on state changes.
- [x] Update version headers/labels to `V2.4.2` in console panels and Web HTML template in `dashboard.py`.
- [x] Verify compile check, launch background task, and test inject sweeps.

# Task List: Order Flow Engine V2.5A (Adapter and Endpoint Layer)
- [x] Add config/env keys in config.py
- [x] Create .env.example
- [x] Add ai_prompts.py schema
- [x] Add ai_providers.py adapters
- [x] Add provider health validation
- [x] Add ai_interpreter.py snapshot/hash/cache/logging
- [x] Add POST /api/ai/interpret endpoint in main.py
- [x] Add GET /api/ai/latest endpoint in main.py
- [x] Add GET /api/ai/providers endpoint in main.py
- [x] Run compile checks
- [x] Verify execution with AI_ENABLED=false (default)
- [x] Test live provider with mock/actual API key

# Task List: Order Flow Engine V2.5B (Dashboard AI Panel)
- [x] Setup side column panel for terminal console dashboard layout in dashboard.py
- [x] Add console AI readout panel generation matching cache state
- [x] Setup three-column grid layout for web dashboard HTML rendering
- [x] Add AI metadata badges, summary text, and cleanest bias/suppressed lists
- [x] Integrate manual click trigger refresh and async polling GET /api/ai/latest

# Task List: Order Flow Engine V2.5C (Auto Interpretation Loop)
- [x] Implement async auto background loop (_ai_loop) running every interval
- [x] Safeguard loop with AI_ENABLED flag and non-crashing exceptions block
- [x] Verify compile check, run codebase and test endpoints responses

# Task List: Order Flow Engine V2.5 Hardening Pass
- [x] Integrate manual interpret request cooldown (30 seconds) returning 429 status code in main.py
- [x] Programmatically enforce not_trade_advice = True in data parser prior to schema validation
- [x] Set fallback cache retrieval logic in ai_interpreter.py to preserve valid cache when API requests fail
- [x] Create comparative testing script compare_ai_providers.py to compare LLM readouts side-by-side

# Task List: Order Flow Scanner Upgrade v3.0 (Phase 1: Data Integrity)
- [x] Split WebSocket streams into separate trade (`/market`) and depth (`/public`) connections.
- [x] Build local diff-depth order book sync using REST snapshots and validator.
- [x] Add stream health metrics tracking (latencies, updates, gaps, duplicates).
- [x] Standardize independent rolling window aggregations on canonical deques.
- [x] Display warming countdowns and multi-dimension connection health badges.
- [x] Write comprehensive unit tests in `test_data_integrity.py`.

# Task List: Order Flow Scanner Upgrade v3.0 (Phase 2: Explainable Signals & Liquidity Intelligence)
- [x] Calculate multi-level book imbalances (0-5, 5-15, 15-30 bps) and microprice stats.
- [x] Create wall tracking module checking age, stability, and replenishment.
- [x] Create stacking/pulling, absorption, iceberg, spoof, and vacuum detectors.
- [x] Replaced simple WAITING with an auditable checklist gate state machine.
- [x] Enforce 30-second AI data freshness constraint and read-only explanations.
- [x] Write comprehensive unit tests in `test_liquidity.py`.

# Task List: Order Flow Scanner Upgrade v3.0 (Phase 3: Research and Replay)
- [x] Implement raw trade and depth event recording to gzip-compressed JSONLines.
- [x] Implement deterministic replay runner feeding events chronologically into FlowMetrics.
- [x] Build performance analyzer reporting win rates, MFE, MAE, drawdowns, and returns.
- [x] Write comprehensive unit tests in `test_replay.py`.

# Task List: Order Flow Scanner Upgrade v3.0 (Phase 4: Liquidity Map)
- [x] Add symbol details overlay panel HTML and CSS layout to dashboard.py.
- [x] Create real-time L2 order book ladder rendering.
- [x] Build HTML5 Canvas liquidity heatmap drawing time series grids.
- [x] Implement zooming controls and hover tooltip indicators.
- [x] Add click triggers to table rows and test endpoints.
