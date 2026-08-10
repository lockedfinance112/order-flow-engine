import json
import os
import time
from typing import Dict, Any, List

class ReportGenerator:
    """Generates the HTML validation report and validation_decision.md."""
    @staticmethod
    def generate_html_report(
        summary: Dict[str, Any],
        timelines: Dict[str, List[Dict[str, Any]]],
        signals: List[Dict[str, Any]],
        output_filepath: str
    ):
        """Generates self-contained HTML report with premium HSL styling and embedded SVG plots."""
        svg_content = ""
        for symbol, timeline in timelines.items():
            if not timeline:
                continue
                
            width = 800
            height = 200
            n = len(timeline[-500:])
            
            bands = []
            colors = {
                "TREND_UP": "#10b981",
                "TREND_DOWN": "#ef4444",
                "RANGE": "#3b82f6",
                "BREAKOUT_UP": "#f59e0b",
                "BREAKOUT_DOWN": "#8b5cf6",
                "TRANSITION": "#6b7280",
                "UNKNOWN": "#9ca3af"
            }
            
            w_step = width / max(n, 1)
            for i, t in enumerate(timeline[-500:]):
                r = t.get("primary_regime", "UNKNOWN")
                color = colors.get(r, "#9ca3af")
                x = i * w_step
                bands.append(f'<rect x="{x}" y="0" width="{w_step + 0.5}" height="{height}" fill="{color}" opacity="0.15" />')
                
            svg_content += f"""
            <h3>{symbol.upper()} - Last 500 Closed Minutes (Regime Tracks)</h3>
            <svg width="{width}" height="{height}" style="background: #1e1e2e; border-radius: 8px; margin-bottom: 20px;">
                {"".join(bands)}
                <text x="10" y="20" fill="#ffffff" font-family="sans-serif" font-size="12">Green: Trend Up | Red: Trend Down | Blue: Range | Yellow/Purple: Breakout</text>
            </svg>
            """

        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TICS Phase 1B-V Regime Validation Report</title>
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0f0f16;
            color: #e2e8f0;
            margin: 40px;
        }}
        .card {{
            background: #151521;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #2d2d44;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        h1, h2, h3 {{
            color: #38bdf8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2d2d44;
        }}
        th {{
            background-color: #1e1e2f;
            color: #38bdf8;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-ready {{ background-color: #059669; color: white; }}
        .badge-degraded {{ background-color: #d97706; color: white; }}
    </style>
</head>
<body>
    <h1>TICS Phase 1B-V Validation Report</h1>
    <div class="card">
        <h2>Run Summary</h2>
        <p><strong>Run ID:</strong> {summary.get("run_id")}</p>
        <p><strong>TICS_PHASE_1B_V_READY:</strong> <span class="badge badge-ready">{summary.get("tics_phase_1b_v_ready")}</span></p>
        <p><strong>REGIME_ENFORCEMENT_CANDIDATE:</strong> <span class="badge badge-degraded">{summary.get("regime_enforcement_candidate")}</span></p>
        <p><strong>Protocol SHA:</strong> {summary.get("protocol_hash")}</p>
        <p><strong>Result Content SHA:</strong> {summary.get("result_content_hash")}</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>Regime Occupancy</h2>
            <table>
                <tr><th>Regime</th><th>Occupancy %</th></tr>
                {"".join(f"<tr><td>{r}</td><td>{v*100:.2f}%</td></tr>" for r, v in summary.get("regime_occupancy", {}).items())}
            </table>
        </div>
        <div class="card">
            <h2>Breakout Follow-through</h2>
            <p><strong>BREAKOUT_UP Success Rate:</strong> {summary.get("breakout_up_success_rate", 0.0)*100:.2f}%</p>
            <p><strong>BREAKOUT_DOWN Success Rate:</strong> {summary.get("breakout_down_success_rate", 0.0)*100:.2f}%</p>
        </div>
    </div>

    <div class="card">
        <h2>Strategy Expectancy & Counterfactual Comparison (15m Horizon at 5bps)</h2>
        <table>
            <tr><th>Metric</th><th>Baseline</th><th>ALLOW-Only</th></tr>
            <tr><td>Signal Count</td><td>{summary.get("signals_total")}</td><td>{summary.get("allow_signal_count")}</td></tr>
            <tr><td>Mean Net Return</td><td>{summary.get("baseline_15m_net_mean_5bps", 0.0)*100:.4f}%</td><td>{summary.get("allow_only_15m_net_mean_5bps", 0.0)*100:.4f}%</td></tr>
            <tr><td>ALLOW - BLOCK Mean Diff</td><td colspan="2">{summary.get("allow_minus_block_15m_net_mean_5bps", 0.0)*100:.4f}%</td></tr>
            <tr><td>ALLOW - BLOCK 95% Bootstrap CI</td><td colspan="2">[{summary.get("allow_minus_block_ci_low", 0.0)*100:.4f}%, {summary.get("allow_minus_block_ci_high", 0.0)*100:.4f}%]</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Timeline Visualization</h2>
        {svg_content}
    </div>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html_template)

    @staticmethod
    def generate_decision_markdown(summary: Dict[str, Any], output_filepath: str):
        md = f"""# TICS PHASE 1B-V VALIDATION DECISION

## DATASET
- Evaluated Days: {summary.get("days_evaluated")}
- Symbols Evaluated: {", ".join(summary.get("symbols", [])) if "symbols" in summary else "None"}
- Warmup Period: {summary.get("warmup_period_days", 22)} days

## REPRODUCIBILITY
- Protocol Hash: `{summary.get("protocol_hash")}`
- Result Content Hash: `{summary.get("result_content_hash")}`
- Git Commit: `{summary.get("git_commit")}`

## REGIME COVERAGE
- Regime Ready Coverage: {summary.get("regime_ready_coverage", 0.0)*100:.2f}%
- Occupancy Matrix:
{chr(10).join(f"  - {r}: {v*100:.2f}%" for r, v in summary.get("regime_occupancy", {}).items())}

## STABILITY
- Median Persistence: {summary.get("median_persistence", 0.0)} bars
- 3-Bar Flip-Flop Rate: {summary.get("flip_flop_3_rate", 0.0)*100:.2f}%
- 5-Bar Flip-Flop Rate: {summary.get("flip_flop_5_rate", 0.0)*100:.2f}%

## REFERENCE AGREEMENT
- 15m Horizon Agreement Rate: {summary.get("reference_agreement_15m", 0.0)*100:.2f}%
- 60m Horizon Agreement Rate: {summary.get("reference_agreement_60m", 0.0)*100:.2f}%

## BREAKOUT VALIDATION
- BREAKOUT_UP Success Rate: {summary.get("breakout_up_success_rate", 0.0)*100:.2f}%
- BREAKOUT_DOWN Success Rate: {summary.get("breakout_down_success_rate", 0.0)*100:.2f}%

## SIGNAL EXPECTANCY
- Total Signals: {summary.get("signals_total")}
- Signals Censored: {summary.get("signals_censored")}
- Baseline Mean Net Return (15m, 5bps): {summary.get("baseline_15m_net_mean_5bps", 0.0)*100:.4f}%

## PERMISSION MATRIX PERFORMANCE
- ALLOW Signal Count: {summary.get("allow_signal_count")}
- BLOCK Signal Count: {summary.get("block_signal_count")}
- ALLOW - BLOCK Mean Diff: {summary.get("allow_minus_block_15m_net_mean_5bps", 0.0)*100:.4f}%
- ALLOW - BLOCK Bootstrap 95% CI: `[{summary.get("allow_minus_block_ci_low", 0.0)*100:.4f}%, {summary.get("allow_minus_block_ci_high", 0.0)*100:.4f}%]`

## OUT-OF-SAMPLE HOLDOUT
- Holdout Signal Count: {summary.get("holdout_signal_count")}

## LIMITATIONS
- Fixed horizon returns proxy lifecycle position PnL.
- Replay relies on historical closed 1m bars approximation for signal metrics when high resolution trade datasets are missing.

## ENFORCEMENT DECISION
- **TICS_PHASE_1B_V_READY**: `{summary.get("tics_phase_1b_v_ready")}`
- **REGIME_ENFORCEMENT_CANDIDATE**: `{summary.get("regime_enforcement_candidate")}`
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(md)
