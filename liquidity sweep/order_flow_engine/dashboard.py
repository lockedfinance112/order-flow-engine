import os
import json
import time
import collections
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

def format_usdt(val: float) -> str:
    """Formats a USDT value into a clean, compact string (e.g. +$104K, -$1.2M)."""
    sign = "+" if val >= 0 else "-"
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val/1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{sign}${abs_val/1_000:.1f}K"
    else:
        return f"{sign}${abs_val:.1f}"

class OrderFlowDashboard:
    """
    Renders a rich, multi-panel terminal dashboard for the Order Flow Engine.
    Supports single-symbol fallback mode and a normalized multi-symbol radar table.
    """
    def __init__(self, symbols: List[str]):
        self.console = Console()
        self.layout = Layout()
        self.symbols = [s.lower() for s in symbols]
        self.is_multi = len(self.symbols) > 1
        
        # Historical lists across all tracked symbols
        self.recent_events = collections.deque(maxlen=10)
        self.recent_large_trades = collections.deque(maxlen=10)
        
        # Setup initial layout structure
        if self.is_multi:
            self._setup_layout_multi()
        else:
            self._setup_layout()

    def _setup_layout(self):
        """Single-symbol dashboard layout (V2.1 fallback)."""
        self.layout.split(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=8)
        )
        self.layout["body"].split_row(
            Layout(name="left_column", ratio=2),
            Layout(name="alerts", ratio=1)
        )
        self.layout["left_column"].split(
            Layout(name="metrics", ratio=1),
            Layout(name="orderbook", size=8)
        )

    def _setup_layout_multi(self):
        """Multi-symbol dashboard layout with side column split for Alerts and AI readout (V2.5)."""
        self.layout.split(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=8)
        )
        self.layout["body"].split_row(
            Layout(name="main_table", ratio=3),
            Layout(name="side_column", ratio=1)
        )
        self.layout["side_column"].split(
            Layout(name="alerts", ratio=1),
            Layout(name="ai_readout", ratio=1)
        )

    def add_event(self, symbol: str, event_type: str, window: str, price: float, notes: str):
        time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        sym_prefix = f"[{symbol.upper()}] " if self.is_multi else ""
        self.recent_events.append(f"[{time_str}] {sym_prefix}{event_type} ({window}) @ {price:.2f} - {notes}")

    def add_large_trade(self, symbol: str, side: str, qty: float, price: float, notional_usdt: float):
        time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        color = "green" if side == "BUY" else "red"
        sym_prefix = f"{symbol.upper()} " if self.is_multi else ""
        self.recent_large_trades.append(
            f"[{time_str}] {sym_prefix}[{color}]{side}[/{color}] ${notional_usdt:,.0f} @ {price:.2f}"
        )

    def _render_console_ai_panel(self) -> Panel:
        """Helper to read cache JSON and format a real-time console display of the AI summary."""
        cache_file = os.path.join(os.path.dirname(__file__), "ai_cache.json")
        ai_text = Text()
        ai_text.append("AI Market Summary:\n", style="bold yellow")
        
        if not os.path.exists(cache_file):
            ai_text.append("  No AI interpretations generated yet.\n", style="dim gray")
            return Panel(ai_text, border_style="magenta", title="AI Interpretation")
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cache_ts = data.get("timestamp") or "-"
            if data.get("fallback_used"):
                stale_ts = data.get("stale_cache_timestamp") or cache_ts
                ai_text.append(Text.from_markup(f"  [yellow]Stale cache:[/] {stale_ts}\n"))
                if data.get("fallback_at"):
                    ai_text.append(Text.from_markup(f"  [yellow]Fallback at:[/] {data.get('fallback_at')}\n"))
            else:
                ai_text.append(f"  Cache: {cache_ts}\n")
                
            if not data.get("ok", False):
                ai_text.append(Text.from_markup(f"  [red]Error:[/] {data.get('error', 'AI inactive')}\n"))
                return Panel(ai_text, border_style="magenta", title="AI Interpretation")
                
            interpretation = data.get("interpretation", {})
            regime = interpretation.get("regime", "mixed").upper()
            risk = interpretation.get("overall_risk", "medium").upper()
            
            risk_color = "green" if risk == "LOW" else ("yellow" if risk == "MEDIUM" else "red")
            
            ai_text.append(Text.from_markup(f"  Regime: [cyan]{regime}[/]  |  Risk: [{risk_color}]{risk}[/]\n\n"))
            summary = interpretation.get("market_summary", "No summary.")
            
            # Wrap summary text cleanly
            ai_text.append(f"{summary}\n", style="white")
            
            cleanest = interpretation.get("cleanest_bias_symbols", [])
            if cleanest:
                clean_str = ", ".join([s.replace("USDT", "") for s in cleanest])
                ai_text.append(Text.from_markup(f"\n  Cleanest: [green]{clean_str}[/]\n"))

                
        except Exception as e:
            ai_text.append(f"  Failed to parse cache: {str(e)}\n", style="red")
            
        return Panel(ai_text, border_style="magenta", title="AI Interpretation")

    def render(self, price: float, running_cvd: float, session_cvd: float, 
                metrics_1m: dict, metrics_5m: dict, metrics_15m: dict,
                best_bid: float = 0.0, best_ask: float = 0.0, spread: float = 0.0,
                bid_depth: float = 0.0, ask_depth: float = 0.0, imbalance: float = 0.0,
                microprice: float = 0.0,
                burst_status: Optional[dict] = None, 
                absorption_status_1m: Tuple[Optional[str], str] = (None, ""),
                absorption_status_5m: Tuple[Optional[str], str] = (None, ""),
                divergence_status: Tuple[Optional[str], str] = (None, ""),
                trade_ws_status: str = "DISCONNECTED",
                depth_ws_status: str = "DISCONNECTED") -> Layout:
        """
        Renders the single-symbol dashboard panel (fallback mode).
        """
        # --- 1. HEADER PANEL ---
        header_text = Text()
        header_text.append("ORDER FLOW SCANNER V2.5  ", style="bold cyan")
        header_text.append("|  BINANCE USD-M FUTURES  ", style="dim white")
        header_text.append(f"|  {self.symbols[0].upper()}  \n", style="bold yellow")
        
        # Current Aggression estimation based on 5m buy ratio
        ratio_5m = metrics_5m.get("buy_ratio", 0.5)
        if ratio_5m >= 0.58:
            aggression = "[bold green]BUYERS (AGGRESSIVE)[/bold green]"
        elif ratio_5m <= 0.42:
            aggression = "[bold red]SELLERS (AGGRESSIVE)[/bold red]"
        else:
            aggression = "[bold yellow]NEUTRAL[/bold yellow]"

        def get_status_style(status: str) -> str:
            if status == "CONNECTED":
                return "bold green"
            elif status == "RECONNECTING":
                return "bold yellow"
            return "bold red"

        header_text.append(Text.from_markup(f"Price: {price:,.2f} USD  |  Aggression: {aggression}  |  "))
        header_text.append(Text.from_markup(f"Trades WS: [{get_status_style(trade_ws_status)}]{trade_ws_status}[/]  |  "))
        header_text.append(Text.from_markup(f"Depth WS: [{get_status_style(depth_ws_status)}]{depth_ws_status}[/]"))


        self.layout["header"].update(Panel(header_text, border_style="cyan"))

        # --- 2. METRICS PANEL (Sliding Windows) ---
        metrics_table = Table(title="Order Flow Sliding Windows", expand=True)
        metrics_table.add_column("Window", justify="center", style="bold cyan")
        metrics_table.add_column("Trades Count", justify="right")
        metrics_table.add_column("Buy Vol ($)", justify="right", style="green")
        metrics_table.add_column("Sell Vol ($)", justify="right", style="red")
        metrics_table.add_column("Delta ($)", justify="right")
        metrics_table.add_column("Buy/Sell Ratio", justify="center")

        def add_window_row(table, name, m):
            delta_usdt = m.get("delta_usdt", 0.0)
            delta_str = f"[green]+${delta_usdt:,.0f}[/green]" if delta_usdt >= 0 else f"[red]-${abs(delta_usdt):,.0f}[/red]"
            b_ratio = m.get("buy_ratio", 0.5) * 100
            s_ratio = m.get("sell_ratio", 0.5) * 100
            ratio_str = f"[green]{b_ratio:.1f}%[/green] / [red]{s_ratio:.1f}%[/red]"
            table.add_row(
                name,
                f"{m.get('trade_count', 0):,}",
                f"${m.get('buy_volume_usdt', 0.0):,.0f}",
                f"${m.get('sell_volume_usdt', 0.0):,.0f}",
                delta_str,
                ratio_str
            )

        add_window_row(metrics_table, "1 Min", metrics_1m)
        add_window_row(metrics_table, "5 Min", metrics_5m)
        add_window_row(metrics_table, "15 Min", metrics_15m)

        self.layout["metrics"].update(Panel(metrics_table, border_style="white"))

        # --- 3. ORDER BOOK PANEL ---
        ob_text = Text()
        ob_text.append("Order Book Pressure (Top 5 Levels in USDT Notional):\n", style="bold cyan")
        
        # Color coding the imbalance value
        if imbalance >= 0.15:
            imbalance_style = "bold green"
        elif imbalance <= -0.15:
            imbalance_style = "bold red"
        else:
            imbalance_style = "yellow"

        ob_text.append(f"  Best Bid: {best_bid:,.2f}  |  Best Ask: {best_ask:,.2f}  |  Spread: {spread:.2f} USD\n", style="white")
        ob_text.append(f"  Bid Depth top-5 ($): {format_usdt(bid_depth)}  |  Ask Depth top-5 ($): {format_usdt(ask_depth)}\n", style="white")
        ob_text.append(f"  Bid/Ask Imbalance: [{imbalance_style}]{imbalance:+.4f}[/{imbalance_style}]  |  ", style="white")
        ob_text.append(f"Microprice: {microprice:,.2f} USD\n", style="bold yellow")
        
        # Visual progress bar for imbalance
        bar_length = 40
        num_green = int((imbalance + 1.0) / 2.0 * bar_length)
        num_green = max(0, min(bar_length, num_green))
        num_red = bar_length - num_green
        bar_str = "[green]" + "█" * num_green + "[/green]" + "[red]" + "█" * num_red + "[/red]"
        ob_text.append(f"  [{bar_str}]")

        self.layout["orderbook"].update(Panel(ob_text, border_style="blue", title="Order Book Snapshot"))

        # --- 4. ALERTS PANEL ---
        alerts_text = Text()
        alerts_text.append("Recent Large Trades (>= $100K):\n", style="bold yellow")
        if not self.recent_large_trades:
            alerts_text.append("  No large trades detected yet.\n", style="dim gray")
        for lt in reversed(self.recent_large_trades):
            alerts_text.append(Text.from_markup(f"  {lt}\n"))

        
        alerts_text.append("\nOrder Flow State Alerts:\n", style="bold yellow")
        
        # Trade Burst
        if burst_status:
            alerts_text.append(
                f"  ⚠️ [bold orange1]BURST DETECTED![/bold orange1] {burst_status['current_count']} trades/10s (Avg: {burst_status['average_count']})\n"
            )
        else:
            alerts_text.append("  - Burst Status: NORMAL\n", style="dim gray")

        # Absorption 1m
        abs_1m_type, abs_1m_notes = absorption_status_1m
        if abs_1m_type:
            color = "green" if "BULLISH" in abs_1m_type else "red"
            alerts_text.append(f"  ⚠️ [bold {color}]{abs_1m_type} (1m)[/bold {color}] detected!\n")
        
        # Absorption 5m
        abs_5m_type, abs_5m_notes = absorption_status_5m
        if abs_5m_type:
            color = "green" if "BULLISH" in abs_5m_type else "red"
            alerts_text.append(f"  ⚠️ [bold {color}]{abs_5m_type} (5m)[/bold {color}] detected!\n")

        # CVD Divergence
        div_type, div_notes = divergence_status
        if div_type:
            color = "green" if "BULLISH" in div_type else "red"
            alerts_text.append(f"  ⚠️ [bold {color}]{div_type}[/bold {color}] detected!\n")

        self.layout["alerts"].update(Panel(alerts_text, border_style="yellow", title="Alerts"))

        # --- 5. FOOTER PANEL ---
        footer_text = Text()
        footer_text.append("SYSTEM STATUS: RUNNING (SCANNER ONLY)  |  ", style="bold green")
        footer_text.append("WARNING: AUTO-TRADING & ORDER EXECUTION DISABLED\n", style="blink bold red")
        footer_text.append("Recent Event Logs:\n", style="bold cyan")

        if not self.recent_events:
            footer_text.append("  Waiting for order flow events to log...", style="dim gray")
        else:
            for ev in reversed(self.recent_events):
                footer_text.append(f"  {ev}\n")

        self.layout["footer"].update(Panel(footer_text, border_style="red"))
        
        return self.layout

    def render_multi(self, symbol_data: Dict[str, dict], 
                      trade_ws_status: str = "DISCONNECTED", 
                      depth_ws_status: str = "DISCONNECTED") -> Layout:
        """
        Renders the multi-symbol radar table sorted by USDT notional activity score.
        """
        # --- 1. HEADER PANEL ---
        header_text = Text()
        header_text.append("ORDER FLOW SCANNER V3.0  ", style="bold cyan")
        header_text.append("|  BINANCE USD-M FUTURES  ", style="dim white")
        header_text.append(f"|  TOP {len(self.symbols)} RADAR (Normalized)\n", style="bold yellow")
        
        # Aggregate states across active symbols
        all_synced = True
        any_gap = False
        any_resync = False
        any_warming = False
        for sym, s in symbol_data.items():
            book_state = s.get("book_state", "INITIALISING")
            if book_state == "SEQUENCE_GAP": any_gap = True
            if book_state == "RESYNCING": any_resync = True
            if book_state != "HEALTHY": all_synced = False
            if s.get("metrics_15m", {}).get("status") == "WARMING_UP": any_warming = True

        book_status = "SEQUENCE_GAP" if any_gap else ("RESYNCING" if any_resync else ("SYNCED" if all_synced and len(symbol_data) > 0 else "INITIALISING"))
        window_status = "WARMING" if any_warming else "VALID"

        def get_status_style(status: str) -> str:
            if status in ("CONNECTED", "SYNCED", "VALID", "READY", "FRESH"):
                return "bold green"
            elif status in ("RECONNECTING", "RESYNCING", "WARMING", "STALE"):
                return "bold yellow"
            return "bold red"

        header_text.append("Radar Mode: SCANNER ONLY (No Execution)  |  ", style="white")
        header_text.append(Text.from_markup(f"Trades: [{get_status_style(trade_ws_status)}]{trade_ws_status}[/]  |  "))
        header_text.append(Text.from_markup(f"Depth: [{get_status_style(depth_ws_status)}]{depth_ws_status}[/]  |  "))
        header_text.append(Text.from_markup(f"Book: [{get_status_style(book_status)}]{book_status}[/]  |  "))
        header_text.append(Text.from_markup(f"Windows: [{get_status_style(window_status)}]{window_status}[/]"))

        self.layout["header"].update(Panel(header_text, border_style="cyan"))

        # --- 2. MULTI-SYMBOL MAIN TABLE ---
        table = Table(title="Order Flow Notional Radar Dashboard", expand=True)
        table.add_column("Symbol", justify="center", style="bold cyan")
        table.add_column("Price", justify="right")
        table.add_column("Aggression", justify="center")
        table.add_column("1m Δ ($)", justify="right")
        table.add_column("5m Δ ($)", justify="right")
        table.add_column("15m Δ ($)", justify="right")
        table.add_column("Buy/Sell %", justify="center")
        table.add_column("CVD (Sess)", justify="right")
        table.add_column("Book Imbalance", justify="center")
        table.add_column("Next Action", justify="center")
        table.add_column("Latest Event", justify="left")

        # Score and sort symbols dynamically by USDT notional activity
        ranked_symbols = []
        now = time.time()
        for symbol in self.symbols:
            data = symbol_data.get(symbol, {})
            d1_usdt = data.get("delta_1m_usdt", 0.0)
            d5_usdt = data.get("delta_5m_usdt", 0.0)
            imb = data.get("imbalance", 0.0)
            action = data.get("next_action", "WAITING")
            
            # score bonuses
            large_trade_age = now - data.get("last_large_trade_time", 0.0)
            lt_bonus = 50.0 if large_trade_age <= 60 else 0.0
            
            event_age = now - data.get("last_event_time", 0.0)
            ev_bonus = 100.0 if event_age <= 60 else 0.0

            # Huge bonus to place active sweep confluences at the top of the radar
            sweep_bonus = 500.0 if "SWEEP" in action else 0.0
            
            # Activity Score = |1m_delta_usdt| / 10000 + |5m_delta_usdt| / 25000 + |imbalance|*100 + bonuses
            score = abs(d1_usdt) / 10000.0 + abs(d5_usdt) / 25000.0 + abs(imb) * 100.0 + lt_bonus + ev_bonus + sweep_bonus
            ranked_symbols.append((score, symbol, data))

        # Sort descending by activity score
        ranked_symbols.sort(key=lambda x: x[0], reverse=True)

        for _, sym, data in ranked_symbols:
            price = data.get("price", 0.0)
            if price >= 1.0:
                price_str = f"{price:,.2f}"
            else:
                price_str = f"{price:.4f}"
                
            # Aggression
            r5m = data.get("buy_ratio_5m", 0.5)
            if r5m >= 0.58:
                aggression = "[bold green]BUYERS[/bold green]"
            elif r5m <= 0.42:
                aggression = "[bold red]SELLERS[/bold red]"
            else:
                aggression = "[yellow]NEUTRAL[/yellow]"
                
            # Delta fields (Formatted in compact USDT format)
            m1 = data.get("metrics_1m", {})
            m5 = data.get("metrics_5m", {})
            m15 = data.get("metrics_15m", {})
            
            if m1.get("status") == "WARMING_UP":
                d1_str = f"[yellow]{m1.get('warmup_text')}[/yellow]"
            else:
                d1 = m1.get("delta_usdt", 0.0)
                d1_str = f"[green]+{format_usdt(d1)}[/green]" if d1 >= 0 else f"[red]{format_usdt(d1)}[/red]"

            if m5.get("status") == "WARMING_UP":
                d5_str = f"[yellow]{m5.get('warmup_text')}[/yellow]"
            else:
                d5 = m5.get("delta_usdt", 0.0)
                d5_str = f"[green]+{format_usdt(d5)}[/green]" if d5 >= 0 else f"[red]{format_usdt(d5)}[/red]"

            if m15.get("status") == "WARMING_UP":
                d15_str = f"[yellow]{m15.get('warmup_text')}[/yellow]"
            else:
                d15 = m15.get("delta_usdt", 0.0)
                d15_str = f"[green]+{format_usdt(d15)}[/green]" if d15 >= 0 else f"[red]{format_usdt(d15)}[/red]"
            
            # Ratios
            b1 = data.get("buy_ratio_1m", 0.5) * 100.0
            s1 = data.get("sell_ratio_1m", 0.5) * 100.0
            ratio_str = f"[green]{b1:.0f}[/green]/[red]{s1:.0f}[/red]"
            
            # CVD
            cvd = data.get("session_cvd_usdt", 0.0)
            cvd_str = f"[green]{format_usdt(cvd)}[/green]" if cvd >= 0 else f"[red]{format_usdt(cvd)}[/red]"
            
            # Imbalance formatting
            imb = data.get("imbalance", 0.0)
            if abs(imb) >= 0.15:
                imb_color = "bold green" if imb >= 0 else "bold red"
                imb_str = f"[{imb_color}]{imb:+.2f}[/{imb_color}]"
            else:
                imb_str = f"[dim]{imb:+.2f}[/dim]"
                
            # Style the Next Action Suggestion Column
            action = data.get("next_action", "WAITING")
            if action == "LONG (SWEEP)":
                action_str = "[bold green]LONG (SWEEP)[/bold green]"
            elif action == "SHORT (SWEEP)":
                action_str = "[bold red]SHORT (SWEEP)[/bold red]"
            elif action == "LONG_BIAS":
                action_str = "[green]LONG_BIAS[/green]"
            elif action == "SHORT_BIAS":
                action_str = "[red]SHORT_BIAS[/red]"
            else:
                action_str = "[dim]WAITING[/dim]"
                
            latest_ev = "[bold red]WINDOW_DUPLICATION_SUSPECTED[/bold red]" if data.get("duplication_suspected") else data.get("latest_event", "")
            
            table.add_row(
                sym.upper().replace("USDT", ""),
                price_str,
                aggression,
                d1_str,
                d5_str,
                d15_str,
                ratio_str,
                cvd_str,
                imb_str,
                action_str,
                latest_ev
            )

        self.layout["main_table"].update(Panel(table, border_style="white"))

        # --- 3. ALERTS PANEL (Right) ---
        alerts_text = Text()
        alerts_text.append("Recent Large Trades:\n", style="bold yellow")
        if not self.recent_large_trades:
            alerts_text.append("  No large trades detected yet.\n", style="dim gray")
        for lt in list(self.recent_large_trades)[-5:]:
            alerts_text.append(Text.from_markup(f"  {lt}\n"))

            
        alerts_text.append("\nOrder Book Stale Check:\n", style="bold yellow")
        for symbol in self.symbols:
            data = symbol_data.get(symbol, {})
            last_depth = data.get("last_depth_timestamp", 0.0)
            age = (now - last_depth) if last_depth > 0 else 9999.0
            if age > 3.0:
                alerts_text.append(f"  ⚠️ [red]{symbol.upper()}[/] DEPTH STALE ({age:.1f}s)\n", style="red")

        self.layout["alerts"].update(Panel(alerts_text, border_style="yellow", title="Live Radar Feed"))

        # --- 4. AI READOUT PANEL ---
        self.layout["ai_readout"].update(self._render_console_ai_panel())

        # --- 5. FOOTER PANEL (Logs scroll) ---
        footer_text = Text()
        footer_text.append("SYSTEM STATUS: RADAR RUNNING (SCANNER ONLY)  |  ", style="bold green")
        footer_text.append("WARNING: EXECUTION DISABLED\n", style="blink bold red")
        footer_text.append("Recent Event Logs:\n", style="bold cyan")

        if not self.recent_events:
            footer_text.append("  Waiting for order flow events to log...", style="dim gray")
        else:
            for ev in list(self.recent_events)[-4:]:
                footer_text.append(f"  {ev}\n")

        self.layout["footer"].update(Panel(footer_text, border_style="red"))
        
        return self.layout

    def render_html(self) -> str:
        """Returns a premium dark-mode web status dashboard supporting V2.5 Next Action and AI read panel."""
        return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Order Flow Scanner V2.5</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0f172a;
            --bg-surface: #1e293b;
            --bg-card: #0f172a;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
            --magenta: #d946ef;
        }
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 24px;
            min-width: 1180px;
        }
        .container {
            max-width: 1840px;
            margin: 0 auto;
        }
        .header-panel {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-left h1 {
            font-size: 26px;
            font-weight: 700;
            color: var(--primary);
            margin: 0;
        }
        .header-left p {
            color: var(--text-muted);
            margin: 4px 0 0 0;
            font-size: 14px;
        }
        .header-right {
            display: flex;
            gap: 16px;
        }
        .header-status-badge {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-status-badge span {
            font-family: 'JetBrains Mono', monospace;
        }
        .header-status-badge span.connected { color: var(--green); }
        .header-status-badge span.reconnecting { color: var(--yellow); }
        .header-status-badge span.stale { color: var(--red); }
        .header-status-badge span.disconnected { color: var(--red); }

        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
            align-items: start;
        }
        .column {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        .panel {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            overflow: hidden;
        }
        .panel-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-main);
            margin-top: 0;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            table-layout: fixed;
        }
        th, td {
            text-align: right;
            padding: 10px 10px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        th:first-child, td:first-child {
            text-align: center;
        }
        th {
            color: var(--text-muted);
            font-weight: 500;
            white-space: normal;
            line-height: 1.2;
        }
        th {
            overflow: visible;
            text-overflow: clip;
        }
        th:nth-child(1), td:nth-child(1) { width: 6%; }
        th:nth-child(2), td:nth-child(2) { width: 9%; }
        th:nth-child(3), td:nth-child(3) { width: 9%; }
        th:nth-child(4), td:nth-child(4),
        th:nth-child(5), td:nth-child(5),
        th:nth-child(6), td:nth-child(6),
        th:nth-child(8), td:nth-child(8) { width: 8%; }
        th:nth-child(7), td:nth-child(7) { width: 9%; }
        th:nth-child(9), td:nth-child(9) { width: 10%; }
        th:nth-child(10), td:nth-child(10) { width: 8%; }
        th:nth-child(11), td:nth-child(11),
        th:nth-child(12), td:nth-child(12) { width: 8%; }
        th:nth-child(13), td:nth-child(13) {
            width: 13%;
            text-align: right;
        }
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .neutral { color: var(--yellow); }
        .bold { font-weight: bold; }
        .dim { color: var(--text-muted); opacity: 0.6; }
        
        .list-items {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }
        .list-items li {
            padding: 8px 12px;
            border-radius: 8px;
            background-color: var(--bg-card);
            border-left: 4px solid var(--border);
            line-height: 1.4;
            overflow-wrap: anywhere;
        }
        .list-items li.event-confluence {
            border-left-color: var(--magenta);
            background-color: rgba(217, 70, 239, 0.05);
        }
        .list-items li.event-large-buy {
            border-left-color: var(--green);
        }
        .list-items li.event-large-sell {
            border-left-color: var(--red);
        }
        .list-items li.event-warning {
            border-left-color: var(--yellow);
        }
        .activity-card-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        .ai-under-row {
            display: grid;
            grid-template-columns: minmax(360px, 0.8fr) minmax(520px, 1.2fr);
            gap: 16px;
            align-items: start;
        }
        .activity-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            min-width: 0;
            overflow: hidden;
        }
        .activity-card .panel-title {
            font-size: 16px;
            margin-bottom: 14px;
        }
        .activity-card .list-items {
            max-height: 360px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .ai-settings-card .ai-settings-form {
            max-width: 760px;
        }
        .ai-market-card #ai-symbol-analysis {
            max-height: 380px;
        }
        .badge {
            font-size: 11px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            display: inline-flex;
            align-items: center;
            min-height: 20px;
            line-height: 1.15;
        }
        .badge.low { color: var(--green); border-color: var(--green); }
        .badge.medium { color: var(--yellow); border-color: var(--yellow); }
        .badge.high { color: var(--red); border-color: var(--red); }
        
        .badge.regime-trending { color: var(--primary); border-color: var(--primary); }
        .badge.regime-choppy { color: var(--yellow); border-color: var(--yellow); }
        .badge.regime-absorbing { color: var(--magenta); border-color: var(--magenta); }
        .badge.regime-sweep-heavy { color: #f43f5e; border-color: #f43f5e; }
        .badge.regime-mixed { color: var(--text-muted); border-color: var(--border); }
        .context-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 76px;
            padding: 3px 6px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            line-height: 1.1;
        }
        .context-confirmed { color: var(--green); border-color: var(--green); }
        .context-conflicting { color: var(--red); border-color: var(--red); }
        .context-mixed { color: var(--yellow); border-color: var(--yellow); }
        .context-insufficient { color: var(--text-muted); border-color: var(--border); }
        .quality-ok { color: var(--green); border-color: var(--green); }
        .quality-stale { color: var(--yellow); border-color: var(--yellow); }
        .quality-error { color: var(--red); border-color: var(--red); }
        .health-healthy { color: var(--green); border-color: var(--green); background: rgba(0, 230, 115, 0.05); }
        .health-degraded { color: var(--yellow); border-color: var(--yellow); background: rgba(255, 204, 0, 0.05); }
        .health-bad { color: var(--red); border-color: var(--red); background: rgba(255, 51, 51, 0.05); }
        
        .ai-sym-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        #ai-high-probability .ai-sym-grid {
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        }
        #ai-high-probability .ai-sym-card {
            min-height: 170px;
        }
        .probability-panel {
            min-height: 250px;
        }
        .panel-subtitle {
            color: var(--text-muted);
            font-size: 12px;
            margin-top: -12px;
            margin-bottom: 16px;
        }
        .ai-sym-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 11px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 8px;
            aspect-ratio: 1 / 1;
            color: inherit;
            cursor: pointer;
            text-align: left;
            transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
            width: 100%;
            min-width: 0;
        }
        .ai-sym-card:hover,
        .ai-sym-card:focus-visible {
            border-color: var(--primary);
            background: rgba(56, 189, 248, 0.08);
            transform: translateY(-1px);
            outline: none;
        }
        .ai-sym-card.ai-mismatch {
            border-color: var(--yellow);
            background: rgba(245, 158, 11, 0.06);
        }
        .ai-sym-card.ai-stale {
            opacity: 0.82;
            border-style: dashed;
        }
        .ai-sym-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            font-weight: 700;
            font-size: 12px;
            gap: 8px;
        }
        .ai-sym-symbol {
            color: var(--primary);
            font-size: 15px;
            letter-spacing: 0;
        }
        .ai-sym-bias {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            text-align: right;
            line-height: 1.2;
            max-width: 68px;
            overflow-wrap: anywhere;
        }
        .ai-sym-desc {
            font-size: 11px;
            color: #cbd5e1;
            line-height: 1.4;
            overflow-wrap: anywhere;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .ai-sym-warning {
            color: var(--yellow);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .ai-sym-warning.mismatch {
            color: var(--red);
        }
        .ai-sym-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
        }
        .ai-sym-open {
            color: var(--primary);
            font-weight: 700;
        }
        .ai-detail-overlay {
            position: fixed;
            inset: 0;
            z-index: 50;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(2, 6, 23, 0.72);
            padding: 24px;
        }
        .ai-detail-overlay.open {
            display: flex;
        }
        .ai-detail-card {
            width: min(760px, 100%);
            max-height: min(82vh, 720px);
            overflow-y: auto;
            background: #172033;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
        }
        .ai-detail-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
            margin-bottom: 18px;
        }
        .ai-detail-title {
            margin: 0;
            color: var(--primary);
            font-size: 24px;
            line-height: 1.1;
        }
        .ai-detail-bias {
            margin-top: 8px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }
        .ai-detail-warning {
            display: none;
            margin: 0 0 14px;
            padding: 10px 12px;
            border: 1px solid rgba(245, 158, 11, 0.55);
            border-radius: 8px;
            color: var(--yellow);
            background: rgba(245, 158, 11, 0.08);
            font-size: 12px;
            line-height: 1.4;
            overflow-wrap: anywhere;
        }
        .ai-detail-close {
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            min-width: 36px;
            height: 36px;
            font-size: 20px;
            cursor: pointer;
        }
        .ai-detail-close:hover,
        .ai-detail-close:focus-visible {
            border-color: var(--primary);
            outline: none;
        }
        .ai-detail-section {
            margin-top: 16px;
        }
        .ai-detail-label {
            color: var(--primary);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 6px;
        }
        .ai-detail-body {
            color: #e2e8f0;
            font-size: 15px;
            line-height: 1.6;
            overflow-wrap: anywhere;
        }
        .ai-detail-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        .ai-detail-context-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
        }
        .ai-detail-context-item {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 9px 10px;
            min-width: 0;
        }
        .ai-detail-context-label {
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            margin-bottom: 4px;
        }
        .ai-detail-context-value {
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .footer-banner {
            background-color: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: var(--red);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
            margin-top: 24px;
            letter-spacing: 0.5px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        #ai-refresh-btn,
        .ai-action-btn {
            background: var(--primary); 
            color: #0f172a; 
            border: none; 
            padding: 6px 12px; 
            border-radius: 8px; 
            font-family: inherit; 
            font-size: 12px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: all 0.2s ease;
            min-height: 32px;
        }
        #ai-refresh-btn {
            white-space: nowrap;
        }
        #ai-refresh-btn:hover,
        .ai-action-btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        #ai-refresh-btn:disabled,
        .ai-action-btn:disabled {
            background: var(--border);
            color: var(--text-muted);
            cursor: not-allowed;
            transform: none;
        }
        .ai-settings-form {
            display: grid;
            gap: 10px;
        }
        .ai-settings-row {
            display: grid;
            grid-template-columns: 86px minmax(0, 1fr);
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: var(--text-muted);
        }
        .ai-settings-row input,
        .ai-settings-row select {
            width: 100%;
            box-sizing: border-box;
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            min-width: 0;
        }
        .ai-settings-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 4px;
        }
        .ai-settings-actions .ai-action-btn:first-child {
            grid-column: span 2;
        }
        .ai-settings-status {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
            font-size: 11px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.35;
        }
        .ai-settings-status span.configured { color: var(--green); }
        .ai-settings-status span.not-configured { color: var(--yellow); }
        .ai-panel-heading {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .ai-panel-heading span {
            min-width: 0;
            line-height: 1.2;
        }
        .ai-meta {
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 12px;
            font-family: 'JetBrains Mono', monospace;
            overflow-wrap: anywhere;
        }
        .ai-cache-state {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 8px;
            padding: 5px 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.2;
        }
        .ai-cache-state.fresh {
            color: var(--green);
            border-color: rgba(16, 185, 129, 0.55);
        }
        .ai-cache-state.stale {
            color: var(--yellow);
            border-color: rgba(245, 158, 11, 0.55);
        }
        .ai-cache-state.error {
            color: var(--red);
            border-color: rgba(239, 68, 68, 0.55);
        }
        .ai-meta-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 16px;
        }
        .ai-section-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 6px;
            border-left: 3px solid var(--primary);
            padding-left: 8px;
        }
        #ai-summary {
            font-size: 12px;
            line-height: 1.55;
            margin: 0 0 16px 0;
            color: #cbd5e1;
            font-family: 'Outfit', sans-serif;
            overflow-wrap: anywhere;
        }
        .ai-chip-list {
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 16px;
        }
        #ai-symbol-analysis {
            display: block;
            max-height: 280px;
            overflow-y: auto;
            padding-right: 4px;
        }
        @media (max-width: 1500px) {
            body { padding: 18px; }
            table {
                font-size: 12px;
            }
            th, td {
                padding: 9px 7px;
                }
        }
        
        /* Notification & Alerts Panel Styles */
        .panel-header-flex {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }
        .notification-controls {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        .control-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
        }
        .control-toggle input[type="checkbox"] {
            width: auto;
            cursor: pointer;
        }
        .clear-alerts-btn {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }
        .clear-alerts-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.25);
        }
        .alert-log-container {
            max-height: 180px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-right: 4px;
        }
        .no-alerts-msg {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 12px;
            font-style: italic;
        }
        .alert-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12.5px;
            font-family: 'JetBrains Mono', monospace;
            border: 1px solid transparent;
            animation: slideInAlert 0.25s cubic-bezier(0.1, 0.8, 0.3, 1);
        }
        @keyframes slideInAlert {
            from { opacity: 0; transform: translateY(-8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .alert-buy {
            background: rgba(0, 230, 118, 0.06);
            border-color: rgba(0, 230, 118, 0.25);
            color: #00e676;
            box-shadow: 0 0 10px rgba(0, 230, 118, 0.05);
        }
        .alert-sell {
            background: rgba(255, 23, 68, 0.06);
            border-color: rgba(255, 23, 68, 0.25);
            color: #ff1744;
            box-shadow: 0 0 10px rgba(255, 23, 68, 0.05);
        }
        .alert-watch-long {
            background: rgba(255, 215, 0, 0.04);
            border-color: rgba(255, 215, 0, 0.2);
            color: #ffd700;
        }
        .alert-watch-short {
            background: rgba(255, 140, 0, 0.04);
            border-color: rgba(255, 140, 0, 0.2);
            color: #ff8c00;
        }
        .alert-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .alert-badge {
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
        }
        .alert-buy .alert-badge { background: rgba(0, 230, 118, 0.2); }
        .alert-sell .alert-badge { background: rgba(255, 23, 68, 0.2); }
        .alert-watch-long .alert-badge { background: rgba(255, 215, 0, 0.15); }
        .alert-watch-short .alert-badge { background: rgba(255, 140, 0, 0.15); }
        
        .alert-symbol {
            font-weight: bold;
            color: var(--text);
        }
        .alert-time {
            color: var(--text-muted);
            font-size: 11px;
        }
        
        /* Custom Toast Container */
        #toast-container {
            position: fixed;
            top: 24px;
            right: 24px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        }
        .toast-item {
            pointer-events: auto;
            background: #151a26;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 16px 20px;
            color: var(--text);
            min-width: 280px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            gap: 6px;
            animation: toastIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            transition: all 0.3s;
        }
        @keyframes toastIn {
            from { transform: translateX(100%) scale(0.9); opacity: 0; }
            to { transform: translateX(0) scale(1); opacity: 1; }
        }
        .toast-item.toast-buy {
            border-left: 4px solid #00e676;
        }
        .toast-item.toast-sell {
            border-left: 4px solid #ff1744;
        }
        .toast-item.toast-watch-long {
            border-left: 4px solid #ffd700;
        }
        .toast-item.toast-watch-short {
            border-left: 4px solid #ff8c00;
        }
        .toast-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .toast-title {
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }
        .toast-buy .toast-title { color: #00e676; }
        .toast-sell .toast-title { color: #ff1744; }
        .toast-watch-long .toast-title { color: #ffd700; }
        .toast-watch-short .toast-title { color: #ff8c00; }
        .toast-close {
            cursor: pointer;
            color: var(--text-muted);
            font-size: 12px;
            border: none;
            background: none;
        }
        .toast-close:hover { color: var(--text); }
        .toast-body {
            font-size: 12px;
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
        }

        @media (max-width: 1250px) {
            body {
                min-width: 0;
                padding: 14px;
            }
            .header-panel {
                align-items: flex-start;
                gap: 16px;
            }
            .header-right {
                flex-wrap: wrap;
                justify-content: flex-end;
            }
            .grid {
                grid-template-columns: 1fr;
            }
            .panel {
                padding: 18px;
            }
            .column {
                min-width: 0;
            }
            .grid > .column:first-child {
                grid-row: auto;
                overflow-x: auto;
            }
            .activity-card-row,
            .ai-under-row {
                grid-template-columns: 1fr;
            }
            table {
                min-width: 980px;
            }
        }
    </style>
</head>
<body>
    <!-- TOAST CONTAINER FOR ALERTS -->
    <div id="toast-container"></div>
    <div class="container">
        <!-- HEADER -->
        <div class="header-panel">
            <div class="header-left">
                <h1 id="header-title-text">ORDER FLOW SCANNER V3.0</h1>
                <p>Binance USD-M Futures Normalized Notional Radar & Liquidity Intelligence</p>
            </div>
            <div class="header-right">
                <div class="header-status-badge">Trades: <span id="ws-trade-status" class="disconnected">DISCONNECTED</span></div>
                <div class="header-status-badge">Depth: <span id="ws-depth-status" class="disconnected">DISCONNECTED</span></div>
                <div class="header-status-badge">Book: <span id="ws-book-status" class="disconnected">INITIALISING</span></div>
                <div class="header-status-badge">Windows: <span id="ws-window-status" class="disconnected">WARMING</span></div>
                <div class="header-status-badge">Context: <span id="ws-context-status" class="disconnected">NOT_READY</span></div>
                <div class="header-status-badge">AI: <span id="ws-ai-status" class="disconnected">STALE</span></div>
            </div>
        </div>

        <!-- GRID -->
        <div class="grid">
            <!-- LEFT COLUMN -->
            <div class="column">
                <!-- MAIN TABLE -->
                <div class="panel">
                    <div class="panel-title">Order Flow Notional Radar & Bias Dashboard</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Aggression</th>
                                <th>1m Delta ($)</th>
                                <th>5m Delta ($)</th>
                                <th>15m Delta ($)</th>
                                <th>Buy/Sell Ratio</th>
                                <th>CVD ($)</th>
                                <th>Book Imbalance</th>
                                <th>Next Action</th>
                                <th>Context</th>
                                <th>Trades Health</th>
                                <th>Depth Health</th>
                                <th>Book Sync</th>
                                <th>Window Health</th>
                                <th>Context Health</th>
                                <th>Latest Event</th>
                            </tr>
                        </thead>
                        <tbody id="multi-table-body">
                            <tr><td colspan="17" style="text-align: center; color: var(--text-muted);">Waiting for metrics payload...</td></tr>
                        </tbody>
                    </table>
                </div>

                <!-- NOTIFICATION CONTROL & LOG PANEL -->
                <div class="panel notification-panel">
                    <div class="panel-header-flex">
                        <div class="panel-title" style="margin-bottom: 0;">Order Flow Buy/Sell Alerts & Signals</div>
                        <div class="notification-controls">
                            <label class="control-toggle" title="Push desktop notifications when BUY/SELL signals trigger">
                                <input type="checkbox" id="alert-enable" onchange="toggleBrowserNotifications()">
                                <span>Desktop Push</span>
                            </label>
                            <label class="control-toggle">
                                <input type="checkbox" id="alert-sound" checked>
                                <span>Sound Alerts</span>
                            </label>
                            <label class="control-toggle">
                                <input type="checkbox" id="alert-watch" checked>
                                <span>Include Watch Signals</span>
                            </label>
                            <button class="clear-alerts-btn" onclick="clearAlertHistory()">Clear Logs</button>
                        </div>
                    </div>
                    <div class="alert-log-container" id="alert-log-container">
                        <div class="no-alerts-msg">No BUY/SELL signals triggered in this session. Watching 20 symbols...</div>
                    </div>
                </div>

                <div class="panel probability-panel">
                    <div class="panel-title">Highest Probability AI Reads</div>
                    <div class="panel-subtitle">Top symbols ranked by AI confidence from the latest market read.</div>
                    <div id="ai-high-probability">
                        <span style="font-size: 12px; color: var(--text-muted);">Waiting for AI confidence data...</span>
                    </div>
                </div>

                <div class="activity-card-row">
                    <div class="activity-card">
                        <div class="panel-title">Recent Large Trades (>= $100K)</div>
                        <ul id="large-trades-list" class="list-items">
                            <li style="color: var(--text-muted);">No large trades detected yet.</li>
                        </ul>
                    </div>

                    <div class="activity-card">
                        <div class="panel-title">Recent Event Logs</div>
                        <ul id="events-list" class="list-items">
                            <li style="color: var(--text-muted);">Waiting for order flow events...</li>
                        </ul>
                    </div>
                </div>

                <div class="ai-under-row">
                    <div class="activity-card ai-settings-card">
                        <div class="panel-title">AI Settings</div>
                        <div class="ai-settings-form">
                        <label class="ai-settings-row">
                            <span>Provider</span>
                            <select id="ai-settings-provider">
                                <option value="openai">OpenAI</option>
                                <option value="gemini">Gemini</option>
                                <option value="claude">Claude</option>
                                <option value="deepseek">DeepSeek</option>
                            </select>
                        </label>
                        <label class="ai-settings-row">
                            <span>Model</span>
                            <input id="ai-settings-model" list="ai-model-options" value="deepseek-v4-pro" autocomplete="off">
                            <datalist id="ai-model-options">
                                <option value="gpt-5.4-mini"></option>
                                <option value="gemini-3-flash"></option>
                                <option value="claude-sonnet-4-5"></option>
                                <option value="deepseek-v4-pro"></option>
                            </datalist>
                        </label>
                        <label class="ai-settings-row">
                            <span>API Key</span>
                            <input id="ai-settings-key" type="password" autocomplete="off" placeholder="Paste key for this session">
                        </label>
                        <label class="ai-settings-row">
                            <span>Auto-loop</span>
                            <input id="ai-settings-enabled" type="checkbox" style="width: auto;">
                        </label>
                        <div class="ai-settings-actions">
                            <button class="ai-action-btn" id="ai-settings-save-btn" onclick="saveAiSettings()">Save for Session</button>
                            <button class="ai-action-btn" id="ai-settings-test-btn" onclick="testAiProvider()">Test Provider</button>
                            <button class="ai-action-btn" onclick="triggerAiRefresh()">Refresh AI</button>
                        </div>
                        <div class="ai-settings-status">
                            <span>Status: <span id="ai-settings-status" class="not-configured">Not configured</span></span>
                            <span id="ai-settings-mode">Mode: Manual</span>
                        </div>
                        <div id="ai-settings-message" style="font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;"></div>
                    </div>
                </div>

                    <div class="activity-card ai-market-card">
                    <div class="panel-title ai-panel-heading">
                        <span>AI Market Read</span>
                        <button id="ai-refresh-btn" onclick="triggerAiRefresh()">Refresh AI</button>
                    </div>
                    
                    <div class="ai-meta">
                        Provider: <span id="ai-provider" class="bold" style="color: var(--text-main);">-</span> | Model: <span id="ai-model" class="bold" style="color: var(--text-main);">-</span>
                    </div>
                    <div class="ai-meta">
                        <span id="ai-cache-timestamp">Cache: -</span>
                    </div>
                    <div id="ai-cache-state" class="ai-cache-state stale">AI cache: unknown</div>
                    <div id="ai-sync-warning" class="ai-detail-warning"></div>
                    
                    <div class="ai-meta-badges">
                        <span id="ai-regime-badge" class="badge">Regime: -</span>
                        <span id="ai-risk-badge" class="badge">Risk: -</span>
                    </div>
                    
                    <div class="ai-section-title">Market Summary</div>
                    <p id="ai-summary">Waiting for AI summary data...</p>
                    
                    <div class="ai-section-title">Cleanest Biases</div>
                    <div id="ai-cleanest-biases" class="ai-chip-list">
                        <span style="color: var(--text-muted);">None</span>
                    </div>
                    
                    <div class="ai-section-title">Suppressed Signals</div>
                    <div id="ai-suppressed-signals" class="ai-chip-list">
                        <span style="color: var(--text-muted);">None</span>
                    </div>

                    <div class="ai-section-title">Symbol Explanations</div>
                    <div id="ai-symbol-analysis">
                        <span style="font-size: 12px; color: var(--text-muted);">No symbol explanations available.</span>
                    </div>
                </div>
            </div>

            <!-- PAPER TRADING PORTFOLIO & TERMINAL -->
            <div class="panel paper-trading-panel" style="margin-top: 16px;">
                <div class="panel-header-flex">
                    <div class="panel-title" style="margin-bottom: 0;">Simulated Paper Trading Terminal</div>
                    <div class="notification-controls">
                        <label class="control-toggle" title="Auto-trade signals using 10% of portfolio equity per trade">
                            <input type="checkbox" id="paper-auto-trade" onchange="togglePaperAutoTrade()">
                            <span>Auto-Trade Signals</span>
                        </label>
                        <button class="clear-alerts-btn" onclick="resetPaperTrader()">Reset Account ($10K)</button>
                    </div>
                </div>
                
                <div class="paper-grid" style="display: grid; grid-template-columns: 1fr 1.5fr; gap: 20px;">
                    <!-- Left side: Portfolio Summary and Manual trade Form -->
                    <div style="display: flex; flex-direction: column; gap: 16px;">
                        <div class="paper-stats-card" style="background: var(--bg-base); padding: 16px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: var(--text-muted); font-size: 13px;">Net Asset Value (NAV):</span>
                                <span id="paper-nav" style="font-weight: bold; font-size: 18px; color: #00e676;">$10,000.00</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: var(--text-muted); font-size: 13px;">Cash Balance:</span>
                                <span id="paper-cash" style="font-weight: bold;">$10,000.00 USDT</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: var(--text-muted); font-size: 13px;">Realized PnL:</span>
                                <span id="paper-realized" style="font-weight: bold;">$0.00</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: var(--text-muted); font-size: 13px;">Unrealized PnL:</span>
                                <span id="paper-unrealized" style="font-weight: bold;">$0.00</span>
                            </div>
                        </div>
                        
                        <!-- Manual Trade Entry Form -->
                        <div class="paper-order-form" style="background: var(--bg-base); padding: 16px; border-radius: 8px; border: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px;">
                            <div style="font-size: 13px; font-weight: bold; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 4px;">Manual Execution Order</div>
                            <label style="display: flex; justify-content: space-between; align-items: center; font-size: 12.5px;">
                                <span>Select Symbol</span>
                                <select id="paper-order-symbol" style="background: var(--bg-surface); color: var(--text); border: 1px solid var(--border); padding: 4px; border-radius: 4px; font-family: inherit;">
                                    <!-- Will populate dynamically -->
                                </select>
                            </label>
                            <label style="display: flex; justify-content: space-between; align-items: center; font-size: 12.5px;">
                                <span>Leverage (Fixed)</span>
                                <span style="color: var(--primary); font-weight: bold;">5.0x Futures</span>
                            </label>
                            <label style="display: flex; justify-content: space-between; align-items: center; font-size: 12.5px;">
                                <span>Quantity (USDT Value)</span>
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <input type="number" id="paper-order-qty" value="1000" min="10" step="10" style="width: 100px; background: var(--bg-surface); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; text-align: right; font-family: inherit;">
                                    <span>USDT</span>
                                </div>
                            </label>
                            <div style="display: flex; gap: 10px; margin-top: 6px;">
                                <button class="clear-alerts-btn" onclick="submitPaperOrder('BUY')" style="flex: 1; background: rgba(0, 230, 118, 0.15); color: #00e676; border-color: rgba(0, 230, 118, 0.3); font-weight: bold; padding: 6px;">BUY (LONG)</button>
                                <button class="clear-alerts-btn" onclick="submitPaperOrder('SELL')" style="flex: 1; background: rgba(255, 23, 68, 0.15); color: #ff1744; border-color: rgba(255, 23, 68, 0.3); font-weight: bold; padding: 6px;">SELL (SHORT)</button>
                            </div>
                            <div id="paper-order-msg" style="font-size: 11px; text-align: center; min-height: 14px; font-family: 'JetBrains Mono', monospace; font-weight: bold;"></div>
                        </div>
                    </div>
                    
                    <!-- Right side: Active Positions and Trade Log -->
                    <div style="display: flex; flex-direction: column; gap: 12px; min-width: 0;">
                        <!-- Active Positions Table -->
                        <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; flex: 1;">
                            <div style="font-size: 13px; font-weight: bold; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 8px;">Active Positions</div>
                            <div style="overflow-x: auto;">
                                <table style="min-width: 100%; font-size: 11.5px; border-collapse: collapse;">
                                    <thead>
                                        <tr style="border-bottom: 1px solid var(--border); color: var(--text-muted); text-align: left;">
                                            <th style="padding: 4px;">Symbol</th>
                                            <th style="padding: 4px;">Side</th>
                                            <th style="padding: 4px;">Size</th>
                                            <th style="padding: 4px;">Entry</th>
                                            <th style="padding: 4px;">Mark</th>
                                            <th style="padding: 4px; text-align: right;">PnL</th>
                                            <th style="padding: 4px; text-align: center;">Close</th>
                                        </tr>
                                    </thead>
                                    <tbody id="paper-positions-body">
                                        <tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 16px;">No active positions open.</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Trade History logs -->
                        <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; flex: 1;">
                            <div style="font-size: 13px; font-weight: bold; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 6px;">Executions Log</div>
                            <ul id="paper-trades-list" style="list-style: none; margin: 0; padding: 0; font-family: 'JetBrains Mono', monospace; font-size: 11px; max-height: 80px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px;">
                                <li style="color: var(--text-muted);">No executions recorded yet.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="ai-detail-overlay" class="ai-detail-overlay" aria-hidden="true">
            <div class="ai-detail-card" role="dialog" aria-modal="true" aria-labelledby="ai-detail-symbol">
                <div class="ai-detail-header">
                    <div>
                        <h2 id="ai-detail-symbol" class="ai-detail-title">Symbol</h2>
                        <div id="ai-detail-bias" class="ai-detail-bias">Bias: -</div>
                    </div>
                    <button class="ai-detail-close" type="button" onclick="closeAiSymbolDetail()" aria-label="Close symbol detail">&times;</button>
                </div>
                <div class="ai-detail-meta">
                    <span id="ai-detail-confidence" class="badge">Confidence: -</span>
                    <span id="ai-detail-risk" class="badge medium">Risk flags: -</span>
                </div>
                <div id="ai-detail-warning" class="ai-detail-warning"></div>
                <div class="ai-detail-section">
                    <div class="ai-detail-label">Explanation</div>
                    <div id="ai-detail-explanation" class="ai-detail-body">-</div>
                </div>
                <div class="ai-detail-section">
                    <div class="ai-detail-label">What To Watch Next</div>
                    <div id="ai-detail-watch" class="ai-detail-body">-</div>
                </div>
                <div class="ai-detail-section">
                    <div class="ai-detail-label">Binance Context</div>
                    <div id="ai-detail-binance" class="ai-detail-context-grid"></div>
                </div>
            </div>
        </div>

        <!-- SYMBOL DETAILS MODAL (v3.0 HEATMAP & LADDER) -->
        <div id="symbol-details-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(15, 23, 42, 0.96); z-index: 1000; padding: 40px; box-sizing: border-box; overflow-y: auto;">
            <div style="max-width: 1400px; margin: 0 auto; background: var(--bg-surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; position: relative; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
                 <!-- Header -->
                 <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 15px;">
                      <h2 id="modal-title" style="margin: 0; color: var(--primary); font-size: 24px;">SYMBOL DETAILS</h2>
                      <button onclick="closeSymbolDetails()" style="background: var(--red); color: white; border: none; border-radius: 6px; padding: 8px 20px; cursor: pointer; font-family: inherit; font-weight: bold; transition: opacity 0.2s;">CLOSE</button>
                 </div>
                 
                 <div style="display: flex; gap: 24px;">
                      <!-- Left Column: Live Ladder -->
                      <div style="flex: 0 0 350px; background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 16px;">
                           <h3 style="margin-top: 0; margin-bottom: 15px; font-size: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px;">L2 Order Book Ladder</h3>
                           <div style="max-height: 550px; overflow-y: auto; scrollbar-width: thin;">
                                <table style="width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12px;">
                                     <thead>
                                          <tr style="border-bottom: 1px solid var(--border); text-align: right; color: var(--text-muted);">
                                               <th style="text-align: left; padding: 6px 0;">Bid Size</th>
                                               <th style="text-align: center;">Price</th>
                                               <th style="text-align: right; padding: 6px 0;">Ask Size</th>
                                          </tr>
                                     </thead>
                                     <tbody id="ladder-body">
                                          <tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">Selecting symbol...</td></tr>
                                     </tbody>
                                </table>
                           </div>
                      </div>
                      
                      <!-- Right Column: Interactive Heatmap -->
                      <div style="flex: 1; display: flex; flex-direction: column; gap: 16px;">
                           <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 16px; position: relative;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                     <h3 style="margin: 0; font-size: 16px;">Historical Liquidity Heatmap & Order Flow</h3>
                                     <div style="display: flex; gap: 8px;">
                                          <button onclick="zoomHeatmap(1.2)" style="background: var(--bg-surface); border: 1px solid var(--border); color: white; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 11px;">Zoom In</button>
                                          <button onclick="zoomHeatmap(0.8)" style="background: var(--bg-surface); border: 1px solid var(--border); color: white; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 11px;">Zoom Out</button>
                                          <button onclick="resetHeatmap()" style="background: var(--bg-surface); border: 1px solid var(--border); color: white; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 11px;">Reset</button>
                                     </div>
                                </div>
                                <canvas id="heatmap-canvas" width="850" height="400" style="width: 100%; height: 400px; display: block; background: #080d1a; border-radius: 6px; border: 1px solid var(--border);"></canvas>
                                <div id="heatmap-tooltip" style="position: absolute; display: none; background: rgba(15, 23, 42, 0.95); border: 1px solid var(--primary); padding: 8px; border-radius: 4px; font-size: 11px; z-index: 10; pointer-events: none; color: white; font-family: 'JetBrains Mono', monospace;"></div>
                           </div>
                           
                           <!-- Microstructure Stats -->
                           <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                                <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center;">
                                     <div style="font-size: 12px; color: var(--text-muted);">Spread (BPS)</div>
                                     <div id="detail-spread" style="font-size: 18px; font-weight: bold; color: var(--primary); margin-top: 4px;">-</div>
                                </div>
                                <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center;">
                                     <div style="font-size: 12px; color: var(--text-muted);">Microprice Deviation</div>
                                     <div id="detail-microprice" style="font-size: 18px; font-weight: bold; color: var(--yellow); margin-top: 4px;">-</div>
                                </div>
                                <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center;">
                                     <div style="font-size: 12px; color: var(--text-muted);">Weighted Imbalance</div>
                                     <div id="detail-weighted-imbalance" style="font-size: 18px; font-weight: bold; color: var(--green); margin-top: 4px;">-</div>
                                </div>
                                <div style="background: var(--bg-base); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center;">
                                     <div style="font-size: 12px; color: var(--text-muted);">Depth Imbalances (0-5 / 5-15 / 15-30 bps)</div>
                                     <div id="detail-depth-imbalances" style="font-size: 14px; font-weight: bold; color: var(--text-main); margin-top: 6px;">-</div>
                                </div>
                           </div>
                      </div>
                 </div>
            </div>
        </div>

        <!-- FOOTER WARNING -->
        <div class="footer-banner">
            WARNING: AUTO-TRADING & ORDER EXECUTION DISABLED (SCANNER ONLY MODE)
        </div>
    </div>

    <!-- UPDATE SCRIPT -->
    <script>
        const defaultAiModels = {
            openai: 'gpt-5.4-mini',
            gemini: 'gemini-3-flash',
            claude: 'claude-sonnet-4-5',
            deepseek: 'deepseek-v4-pro'
        };
        let currentAiSymbolDetails = {};
        let currentScannerActions = {};
        let currentScannerSymbols = {};
        let currentScannerTimestamp = null;
        let latestAiPayload = null;
        let fixedRadarSymbolOrder = [];

        function escapeHtml(value) {
            return String(value ?? '').replace(/[&<>"']/g, (char) => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            }[char]));
        }

        function normalizeAiList(value) {
            if (!value) return [];
            if (Array.isArray(value)) return value.filter(Boolean).map(String);
            return [String(value)];
        }

        function normalizeSymbolKey(value) {
            return String(value || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
        }

        function scannerActionFor(symbol, aiData) {
            const key = normalizeSymbolKey(symbol);
            const stripped = key.endsWith('USDT') ? key.slice(0, -4) : key;
            const candidates = [key, stripped, stripped + 'USDT'];
            const aiActions = (aiData && aiData.current_scanner_actions) || {};
            const cachedActions = (aiData && aiData.scanner_actions) || {};
            for (const candidate of candidates) {
                if (currentScannerActions[candidate]) return currentScannerActions[candidate];
                if (aiActions[candidate]) return aiActions[candidate];
                if (cachedActions[candidate]) return cachedActions[candidate];
            }
            return null;
        }

        function scannerSymbolFor(symbol) {
            const key = normalizeSymbolKey(symbol);
            const stripped = key.endsWith('USDT') ? key.slice(0, -4) : key;
            return currentScannerSymbols[key] || currentScannerSymbols[stripped] || currentScannerSymbols[stripped + 'USDT'] || {};
        }

        function enrichAiSymbolInfo(symbol, info, aiData) {
            const enriched = { ...(info || {}) };
            const aiBias = String(enriched.bias || 'WAITING');
            const scannerAction = scannerActionFor(symbol, aiData);
            const scannerSymbol = scannerSymbolFor(symbol);
            enriched.scanner_data = scannerSymbol;
            enriched.binance_context = scannerSymbol.binance_context || enriched.binance_context || {};
            if (scannerAction) {
                enriched.scanner_action = scannerAction;
                const ignoreStates = ["WARMING_UP", "DATA_INVALID", "DATA_STALE"];
                if (aiBias !== scannerAction && !ignoreStates.includes(scannerAction)) {
                    enriched.ai_reported_bias = enriched.ai_reported_bias || aiBias;
                    enriched.bias = scannerAction;
                    enriched.ai_mismatch = true;
                }
            }
            return enriched;
        }

        function formatNullable(value, formatter) {
            if (value === null || value === undefined || value === '') return 'N/A';
            if (typeof value === 'number' && !Number.isFinite(value)) return 'N/A';
            return formatter ? formatter(value) : String(value);
        }

        function formatPercentValue(value, digits = 4) {
            return formatNullable(value, v => {
                const n = Number(v);
                return Number.isFinite(n) ? n.toFixed(digits) + '%' : 'N/A';
            });
        }

        function formatLargeNumber(value) {
            return formatNullable(value, v => {
                const n = Number(v);
                if (!Number.isFinite(n)) return 'N/A';
                if (Math.abs(n) >= 1000000000) return (n / 1000000000).toFixed(2) + 'B';
                if (Math.abs(n) >= 1000000) return (n / 1000000).toFixed(2) + 'M';
                if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + 'K';
                return n.toFixed(2);
            });
        }

        function renderBinanceDetailGrid(ctx) {
            const warnings = Array.isArray(ctx.warnings) ? ctx.warnings : [];
            const rows = [
                ['Funding', formatPercentValue(ctx.funding_rate_pct)],
                ['Premium %', formatPercentValue(ctx.mark_premium_pct)],
                ['Open Interest', formatLargeNumber(ctx.current_open_interest)],
                ['1m Trend', formatNullable(ctx.trend_1m)],
                ['5m Trend', formatNullable(ctx.trend_5m)],
                ['15m Trend', formatNullable(ctx.trend_15m)],
                ['Book Snapshot Imbalance', formatNullable(ctx.orderbook_snapshot_imbalance, v => {
                    const n = Number(v);
                    return Number.isFinite(n) ? n.toFixed(2) : 'N/A';
                })],
                ['Context Warning', warnings.length ? warnings.join(', ') : (ctx.error || 'N/A')]
            ];
            return rows.map(([label, value]) => `
                <div class="ai-detail-context-item">
                    <div class="ai-detail-context-label">${escapeHtml(label)}</div>
                    <div class="ai-detail-context-value">${escapeHtml(value)}</div>
                </div>
            `).join('');
        }

        function aiCacheTimestamp(data) {
            return data ? (data.ai_snapshot_timestamp || data.timestamp || data.stale_cache_timestamp) : null;
        }

        function timestampAgeSeconds(value) {
            if (!value) return null;
            const parsed = new Date(value);
            if (Number.isNaN(parsed.getTime())) return null;
            return Math.max(0, Math.floor((Date.now() - parsed.getTime()) / 1000));
        }

        function formatAge(seconds) {
            if (seconds === null || seconds === undefined) return 'unknown age';
            if (seconds < 60) return seconds + 's old';
            const mins = Math.floor(seconds / 60);
            if (mins < 60) return mins + 'm old';
            const hours = Math.floor(mins / 60);
            return hours + 'h ' + (mins % 60) + 'm old';
        }

        function closeAiSymbolDetail() {
            const overlay = document.getElementById('ai-detail-overlay');
            overlay.classList.remove('open');
            overlay.setAttribute('aria-hidden', 'true');
        }

        function openAiSymbolDetail(symbol) {
            const info = currentAiSymbolDetails[symbol];
            if (!info) return;
            const confidence = Number(info.confidence || 0);
            const risks = normalizeAiList(info.risk_flags);
            const scannerAction = info.scanner_action || info.bias || 'WAITING';
            const warning = document.getElementById('ai-detail-warning');
            document.getElementById('ai-detail-symbol').innerText = symbol.replace('USDT', '');
            document.getElementById('ai-detail-bias').innerText = 'Scanner action: ' + scannerAction;
            document.getElementById('ai-detail-confidence').innerText = 'Confidence: ' + Math.round(confidence * 100) + '%';
            document.getElementById('ai-detail-risk').innerText = 'Risk flags: ' + (risks.length ? risks.join(', ') : 'None');
            document.getElementById('ai-detail-explanation').innerText = info.explanation || 'No explanation provided.';
            document.getElementById('ai-detail-watch').innerText = info.what_to_watch_next || 'No watch condition provided.';
            document.getElementById('ai-detail-binance').innerHTML = renderBinanceDetailGrid(info.binance_context || {});
            if (info.ai_mismatch) {
                warning.style.display = 'block';
                warning.innerText = 'AI cached bias was ' + (info.ai_reported_bias || 'unknown') + '; showing current scanner action ' + scannerAction + '.';
            } else if (info.ai_cache_stale) {
                warning.style.display = 'block';
                warning.innerText = 'AI cache is stale; scanner and Binance fields are current.';
            } else {
                warning.style.display = 'none';
                warning.innerText = '';
            }
            const overlay = document.getElementById('ai-detail-overlay');
            overlay.classList.add('open');
            overlay.setAttribute('aria-hidden', 'false');
            document.querySelector('.ai-detail-close').focus();
        }

        function createAiSymbolCard(symbol, info) {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'ai-sym-card' + (info.ai_mismatch ? ' ai-mismatch' : '') + (info.ai_cache_stale ? ' ai-stale' : '');

            let actionColor = 'var(--text-muted)';
            const bias = String(info.bias || 'WAITING');
            if (bias.includes('LONG')) actionColor = 'var(--green)';
            else if (bias.includes('SHORT')) actionColor = 'var(--red)';
            const confidence = Number(info.confidence || 0);
            const explanation = info.explanation || 'No explanation provided.';
            const warningLabel = info.ai_mismatch ? 'AI MISMATCH' : (info.ai_cache_stale ? 'AI STALE' : '');
            const mismatchHtml = warningLabel
                ? `<div class="ai-sym-warning ${info.ai_mismatch ? 'mismatch' : ''}">${escapeHtml(warningLabel)}</div>`
                : '';

            card.innerHTML = `
                <div class="ai-sym-header">
                    <span class="ai-sym-symbol">${escapeHtml(symbol.replace("USDT", ""))}</span>
                    <span class="ai-sym-bias" style="color: ${actionColor};">${escapeHtml(bias)}</span>
                </div>
                <div class="ai-sym-desc">${escapeHtml(explanation)}</div>
                ${mismatchHtml}
                <div class="ai-sym-footer">
                    <span>${Math.round(confidence * 100)}%</span>
                    <span class="ai-sym-open">Open</span>
                </div>
            `;
            card.addEventListener('click', () => openAiSymbolDetail(symbol));
            return card;
        }

        function renderAiCardGrid(container, rankedItems, emptyText) {
            container.innerHTML = '';
            if (!rankedItems.length) {
                container.innerHTML = `<span style="font-size: 12px; color: var(--text-muted);">${escapeHtml(emptyText)}</span>`;
                return;
            }
            const grid = document.createElement('div');
            grid.className = 'ai-sym-grid';
            rankedItems.forEach(({ sym, info }) => {
                grid.appendChild(createAiSymbolCard(sym, info));
            });
            container.appendChild(grid);
        }

        async function fetchMetrics() {
            try {
                const response = await fetch('/api');
                const data = await response.json();
                updateUI(data);
            } catch (err) {
                console.error("Failed to fetch order flow metrics: ", err);
            }
        }

        async function fetchLatestAI() {
            try {
                const response = await fetch('/api/ai/latest');
                const data = await response.json();
                updateAiUI(data);
            } catch (err) {
                console.error("Failed to fetch latest AI: ", err);
            }
        }

        async function fetchAiSettings() {
            try {
                const response = await fetch('/api/ai/settings');
                const data = await response.json();
                updateAiSettingsUI(data);
            } catch (err) {
                updateAiSettingsMessage('Settings unavailable.');
                console.error("Failed to fetch AI settings: ", err);
            }
        }

        async function saveAiSettings() {
            const btn = document.getElementById('ai-settings-save-btn');
            btn.disabled = true;
            try {
                const keyInput = document.getElementById('ai-settings-key');
                const payload = {
                    provider: document.getElementById('ai-settings-provider').value,
                    model: document.getElementById('ai-settings-model').value,
                    enabled: document.getElementById('ai-settings-enabled').checked
                };
                if (keyInput.value.trim()) {
                    payload.api_key = keyInput.value.trim();
                }
                const response = await fetch('/api/ai/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                keyInput.value = '';
                updateAiSettingsUI(data);
                updateAiSettingsMessage(response.ok ? 'Saved for this engine session.' : (data.error || 'Save failed.'));
            } catch (err) {
                updateAiSettingsMessage('Save failed.');
                console.error("Failed to save AI settings: ", err);
            } finally {
                btn.disabled = false;
            }
        }

        async function testAiProvider() {
            const btn = document.getElementById('ai-settings-test-btn');
            btn.disabled = true;
            try {
                const response = await fetch('/api/ai/test', { method: 'POST' });
                const data = await response.json();
                updateAiSettingsMessage(data.ok ? 'Provider config is ready.' : (data.error || 'Provider is not configured.'));
            } catch (err) {
                updateAiSettingsMessage('Provider test failed.');
                console.error("AI provider test failed: ", err);
            } finally {
                btn.disabled = false;
            }
        }

        async function triggerAiRefresh() {
            const btn = document.getElementById('ai-refresh-btn');
            btn.disabled = true;
            btn.innerText = 'Analyzing...';
            try {
                const response = await fetch('/api/ai/interpret', { method: 'POST' });
                const data = await response.json();
                updateAiUI(data);
            } catch (err) {
                console.error("AI Refresh failed: ", err);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Refresh AI';
            }
        }

        function updateAiSettingsMessage(message) {
            document.getElementById('ai-settings-message').innerText = message || '';
        }

        function updateAiSettingsUI(data) {
            if (!data) return;
            const provider = data.provider || 'deepseek';
            document.getElementById('ai-settings-provider').value = provider;
            document.getElementById('ai-settings-model').value = data.model || defaultAiModels[provider] || '';
            document.getElementById('ai-settings-enabled').checked = Boolean(data.enabled);
            const status = document.getElementById('ai-settings-status');
            if (data.configured) {
                status.innerText = 'Configured';
                status.className = 'configured';
            } else {
                status.innerText = 'Not configured';
                status.className = 'not-configured';
            }
            document.getElementById('ai-settings-mode').innerText = data.enabled ? 'Mode: Auto enabled' : 'Mode: Manual / Auto disabled';
        }

        function formatUSDT(val) {
            const sign = val >= 0 ? '+' : '-';
            const absVal = Math.abs(val);
            if (absVal >= 1000000) {
                return sign + '$' + (absVal / 1000000).toFixed(2) + 'M';
            } else if (absVal >= 1000) {
                return sign + '$' + (absVal / 1000).toFixed(1) + 'K';
            } else {
                return sign + '$' + absVal.toFixed(1);
            }
        }

        function contextClass(value) {
            const label = String(value || 'INSUFFICIENT').toLowerCase();
            if (label === 'confirmed') return 'context-confirmed';
            if (label === 'conflicting') return 'context-conflicting';
            if (label === 'mixed') return 'context-mixed';
            return 'context-insufficient';
        }

        function qualityClass(value) {
            const label = String(value || 'STALE').toLowerCase();
            if (label === 'ok') return 'quality-ok';
            if (label === 'error') return 'quality-error';
            return 'quality-stale';
        }

        function healthClass(value) {
            const label = String(value || '').toLowerCase();
            if (label === 'healthy' || label === 'valid' || label === 'synced' || label === 'ready' || label === 'ok') return 'health-healthy';
            if (label === 'degraded' || label === 'warning' || label === 'warming' || label === 'warming_up' || label === 'syncing') return 'health-degraded';
            return 'health-bad'; // stale, error, sequence_gap, invalid
        }

        function updateUI(data) {
            // Update connection health
            document.getElementById('ws-trade-status').innerText = data.trade_ws_status.toUpperCase();
            document.getElementById('ws-trade-status').className = data.trade_ws_status.toLowerCase();
            document.getElementById('ws-depth-status').innerText = data.depth_ws_status.toUpperCase();
            document.getElementById('ws-depth-status').className = data.depth_ws_status.toLowerCase();

            // Calculate global aggregates for new health badges
            let allSynced = true;
            let anyGap = false;
            let anyResync = false;
            let anyWarming = false;
            
            Object.keys(data.symbols || {}).forEach(sym => {
                const s = data.symbols[sym];
                if (s.book_state === 'SEQUENCE_GAP') anyGap = true;
                if (s.book_state === 'RESYNCING') anyResync = true;
                if (s.book_state !== 'HEALTHY') allSynced = false;
                if (s.metrics_15m && s.metrics_15m.status === 'WARMING_UP') anyWarming = true;
            });
            
            const bookStatusEl = document.getElementById('ws-book-status');
            if (bookStatusEl) {
                if (anyGap) { bookStatusEl.innerText = 'SEQUENCE_GAP'; bookStatusEl.className = 'disconnected'; }
                else if (anyResync) { bookStatusEl.innerText = 'RESYNCING'; bookStatusEl.className = 'stale'; }
                else if (allSynced && Object.keys(data.symbols || {}).length > 0) { bookStatusEl.innerText = 'SYNCED'; bookStatusEl.className = 'connected'; }
                else { bookStatusEl.innerText = 'INITIALISING'; bookStatusEl.className = 'disconnected'; }
            }
            
            const winStatusEl = document.getElementById('ws-window-status');
            if (winStatusEl) {
                if (anyWarming) { winStatusEl.innerText = 'WARMING'; winStatusEl.className = 'stale'; }
                else { winStatusEl.innerText = 'VALID'; winStatusEl.className = 'connected'; }
            }
            
            const contextStatusEl = document.getElementById('ws-context-status');
            if (contextStatusEl) {
                if (data.binance_context_status && data.binance_context_status.ok) {
                    contextStatusEl.innerText = 'READY';
                    contextStatusEl.className = 'connected';
                } else {
                    contextStatusEl.innerText = 'NOT_READY';
                    contextStatusEl.className = 'disconnected';
                }
            }
            
            const aiStatusEl = document.getElementById('ws-ai-status');
            if (aiStatusEl) {
                if (latestAiPayload && latestAiPayload.ok) {
                    const age = Number(latestAiPayload.cache_age_seconds);
                    const isStale = latestAiPayload.stale || latestAiPayload.fallback_used || (age > 120);
                    aiStatusEl.innerText = isStale ? 'STALE' : 'FRESH';
                    aiStatusEl.className = isStale ? 'stale' : 'connected';
                } else {
                    aiStatusEl.innerText = 'DISABLED';
                    aiStatusEl.className = 'disconnected';
                }
            }
            currentScannerTimestamp = new Date().toISOString();
            currentScannerActions = {};
            currentScannerSymbols = {};
            Object.keys(data.symbols || {}).forEach(sym => {
                const s = data.symbols[sym] || {};
                const action = s.next_action || 'WAITING';
                const key = normalizeSymbolKey(sym);
                const stripped = key.endsWith('USDT') ? key.slice(0, -4) : key;
                currentScannerActions[key] = action;
                currentScannerActions[stripped] = action;
                currentScannerActions[stripped + 'USDT'] = action;
                currentScannerSymbols[key] = s;
                currentScannerSymbols[stripped] = s;
                currentScannerSymbols[stripped + 'USDT'] = s;
            });

            // Keep table rows pinned so symbols do not jump around between refreshes.
            const incomingSymbols = Object.keys(data.symbols || {});
            if (fixedRadarSymbolOrder.length === 0) {
                fixedRadarSymbolOrder = incomingSymbols;
            } else {
                incomingSymbols.forEach(sym => {
                    if (!fixedRadarSymbolOrder.includes(sym)) {
                        fixedRadarSymbolOrder.push(sym);
                    }
                });
            }
            const orderedSymbols = fixedRadarSymbolOrder
                .filter(sym => data.symbols && data.symbols[sym])
                .concat(incomingSymbols.filter(sym => !fixedRadarSymbolOrder.includes(sym)));
            const sorted = orderedSymbols.map(sym => {
                const s = data.symbols[sym];
                return { symbol: sym, data: s };
            });

            // Populate table
            const tbody = document.getElementById('multi-table-body');
            tbody.innerHTML = '';
            sorted.forEach(item => {
                const symUpper = item.symbol.toUpperCase().replace("USDT", "");
                const price = item.data.price;
                const priceStr = price >= 1.0 ? price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : price.toFixed(4);
                const d1m = item.data.metrics_1m.delta_usdt;
                const d5m = item.data.metrics_5m.delta_usdt;
                const d15m = item.data.metrics_15m.delta_usdt;

                const d1mWarming = item.data.metrics_1m.status === 'WARMING_UP';
                const d5mWarming = item.data.metrics_5m.status === 'WARMING_UP';
                const d15mWarming = item.data.metrics_15m.status === 'WARMING_UP';

                const d1m_text = d1mWarming ? item.data.metrics_1m.warmup_text : formatUSDT(d1m);
                const d5m_text = d5mWarming ? item.data.metrics_5m.warmup_text : formatUSDT(d5m);
                const d15m_text = d15mWarming ? item.data.metrics_15m.warmup_text : formatUSDT(d15m);

                const d1mClass = d1mWarming ? 'neutral' : (d1m >= 0 ? 'positive' : 'negative');
                const d5mClass = d5mWarming ? 'neutral' : (d5m >= 0 ? 'positive' : 'negative');
                const d15mClass = d15mWarming ? 'neutral' : (d15m >= 0 ? 'positive' : 'negative');
                
                const imb = item.data.imbalance;
                let imbClass = '';
                if (Math.abs(imb) >= 0.15) {
                    imbClass = imb >= 0 ? 'positive' : 'negative';
                }
                
                const r5m = item.data.metrics_5m.buy_ratio;
                let aggression = 'NEUTRAL';
                let aggClass = 'neutral';
                if (r5m >= 0.58) { aggression = 'BUYERS'; aggClass = 'positive'; }
                else if (r5m <= 0.42) { aggression = 'SELLERS'; aggClass = 'negative'; }
                
                const action = item.data.next_action || 'WAITING';
                let actionClass = 'dim';
                if (action === 'LONG (SWEEP)') actionClass = 'positive bold';
                else if (action === 'SHORT (SWEEP)') actionClass = 'negative bold';
                else if (action === 'LONG_BIAS') actionClass = 'positive';
                else if (action === 'SHORT_BIAS') actionClass = 'negative';
                const binanceCtx = item.data.binance_context || {};
                const contextLabel = binanceCtx.context_confirm || 'INSUFFICIENT';
                
                const tradesHealth = item.data.stream_health_status || 'STALE';
                const depthHealth = item.data.stream_health_status || 'STALE';
                const bookSync = item.data.book_state || 'INITIALISING';
                const windowHealth = d15mWarming ? 'WARMING' : 'VALID';
                
                const eventText = item.data.duplication_suspected ? 'WINDOW_DUPLICATION_SUSPECTED' : (item.data.latest_event || '-');
                const eventStyle = item.data.duplication_suspected ? 'color: var(--red); font-weight: bold;' : 'color: var(--text-muted); font-weight: bold;';
                
                const row = document.createElement('tr');
                row.style.cursor = 'pointer';
                row.onclick = () => selectSymbol(item.symbol);
                row.innerHTML = `
                    <td style="font-weight: bold; color: var(--primary);">${symUpper}</td>
                    <td>${priceStr}</td>
                    <td class="${aggClass}" style="font-weight: bold;">${aggression}</td>
                    <td class="${d1mClass}">${d1m_text}</td>
                    <td class="${d5mClass}">${d5m_text}</td>
                    <td class="${d15mClass}">${d15m_text}</td>
                    <td><span class="positive">${Math.round(item.data.metrics_1m.buy_ratio * 100)}</span>/<span class="negative">${Math.round(item.data.metrics_1m.sell_ratio * 100)}</span></td>
                    <td class="${item.data.session_cvd_usdt >= 0 ? 'positive' : 'negative'}">${formatUSDT(item.data.session_cvd_usdt)}</td>
                    <td class="${imbClass}" style="font-weight: bold;">${imb >= 0 ? '+' : ''}${imb.toFixed(2)}</td>
                    <td class="${actionClass}">${action}</td>
                    <td><span class="context-pill ${contextClass(contextLabel)}">${contextLabel}</span></td>
                    <td><span class="context-pill ${healthClass(tradesHealth)}">${tradesHealth}</span></td>
                    <td><span class="context-pill ${healthClass(depthHealth)}">${depthHealth}</span></td>
                    <td><span class="context-pill ${healthClass(bookSync)}">${bookSync}</span></td>
                    <td><span class="context-pill ${healthClass(windowHealth)}">${windowHealth}</span></td>
                    <td><span class="context-pill ${contextClass(contextLabel)}">${contextLabel}</span></td>
                    <td style="font-size: 11px; ${eventStyle}">${eventText}</td>
                `;
                tbody.appendChild(row);
            });

            // Large Trades list
            const ltList = document.getElementById('large-trades-list');
            ltList.innerHTML = '';
            if (data.recent_large_trades.length === 0) {
                ltList.innerHTML = '<li style="color: var(--text-muted);">No large trades detected yet.</li>';
            } else {
                [...data.recent_large_trades].reverse().forEach(lt => {
                    const li = document.createElement('li');
                    if (lt.includes("BUY")) {
                        li.className = "event-large-buy";
                    } else if (lt.includes("SELL")) {
                        li.className = "event-large-sell";
                    }
                    let cleanLt = lt.replace(/\[\/?\w+.*?\]/g, "");
                    li.innerText = cleanLt;
                    ltList.appendChild(li);
                });
            }

            // Event Logs
            const evList = document.getElementById('events-list');
            evList.innerHTML = '';
            if (data.recent_events.length === 0) {
                evList.innerHTML = '<li style="color: var(--text-muted);">Waiting for order flow events...</li>';
            } else {
                [...data.recent_events].reverse().forEach(ev => {
                    const li = document.createElement('li');
                    if (ev.includes("CONFLUENCE")) {
                        li.className = "event-confluence";
                    } else if (ev.includes("BURST") || ev.includes("ABSORPTION") || ev.includes("DIVERGENCE")) {
                        li.className = "event-warning";
                    }
                    li.innerText = ev;
                    evList.appendChild(li);
                });
            }

            if (latestAiPayload) {
                updateAiUI(latestAiPayload, true);
            }
            
            // Process real-time BUY/SELL notifications
            processSignalAlerts(data);
            
            // Refresh paper trading UI
            updatePaperTraderUI(data);
        }

        function formatAiTimestamp(value) {
            if (!value) return '-';
            const parsed = new Date(value);
            if (Number.isNaN(parsed.getTime())) return value;
            return parsed.toLocaleString();
        }

        function updateAiCacheMeta(data) {
            const el = document.getElementById('ai-cache-timestamp');
            const stateEl = document.getElementById('ai-cache-state');
            if (!data) {
                el.innerText = 'Cache: -';
                el.style.color = 'var(--text-muted)';
                stateEl.innerText = 'AI cache: unknown';
                stateEl.className = 'ai-cache-state stale';
                return;
            }
            const cacheTs = aiCacheTimestamp(data);
            const age = Number.isFinite(Number(data.cache_age_seconds)) ? Number(data.cache_age_seconds) : timestampAgeSeconds(cacheTs);
            const stale = Boolean(data.stale) || Boolean(data.fallback_used) || (age !== null && age > 120);
            stateEl.innerText = (stale ? 'STALE' : 'FRESH') + ' AI cache - ' + formatAge(age);
            stateEl.className = 'ai-cache-state ' + (data.ok === false ? 'error' : (stale ? 'stale' : 'fresh'));
            if (data.fallback_used) {
                const staleTs = data.stale_cache_timestamp || data.timestamp;
                const fallbackAt = data.fallback_at ? ' | fallback: ' + formatAiTimestamp(data.fallback_at) : '';
                el.innerText = 'AI snapshot: ' + formatAiTimestamp(staleTs) + fallbackAt;
                el.style.color = 'var(--yellow)';
                return;
            }
            const scannerTs = data.latest_scanner_timestamp || currentScannerTimestamp;
            el.innerText = 'AI snapshot: ' + formatAiTimestamp(cacheTs) + ' | Scanner: ' + formatAiTimestamp(scannerTs);
            el.style.color = 'var(--text-muted)';
        }

        function updateAiUI(data, fromScannerTick = false) {
            if (!fromScannerTick) {
                latestAiPayload = data;
            }
            updateAiCacheMeta(data);
            if (!data || !data.ok) {
                currentAiSymbolDetails = {};
                document.getElementById('ai-provider').innerText = data ? (data.provider || '-') : '-';
                document.getElementById('ai-model').innerText = data ? (data.model || '-') : '-';
                document.getElementById('ai-regime-badge').innerText = 'Regime: ERROR';
                document.getElementById('ai-regime-badge').className = 'badge';
                document.getElementById('ai-risk-badge').innerText = 'Risk: ERROR';
                document.getElementById('ai-risk-badge').className = 'badge';
                document.getElementById('ai-summary').innerText = data ? (data.error || 'AI generation failed.') : 'No cache loaded.';
                document.getElementById('ai-cleanest-biases').innerHTML = '<span style="color: var(--text-muted);">None</span>';
                document.getElementById('ai-suppressed-signals').innerHTML = '<span style="color: var(--text-muted);">None</span>';
                document.getElementById('ai-symbol-analysis').innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">No symbol analysis available.</span>';
                document.getElementById('ai-high-probability').innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">No AI confidence data available.</span>';
                document.getElementById('ai-sync-warning').style.display = 'none';
                return;
            }

            const interpretation = data.interpretation;
            document.getElementById('ai-provider').innerText = data.provider.toUpperCase();
            document.getElementById('ai-model').innerText = data.model;

            const regime = interpretation.regime || 'mixed';
            const regimeBadge = document.getElementById('ai-regime-badge');
            regimeBadge.innerText = 'Regime: ' + regime.toUpperCase();
            regimeBadge.className = 'badge regime-' + regime;

            const risk = interpretation.overall_risk || 'medium';
            const riskBadge = document.getElementById('ai-risk-badge');
            riskBadge.innerText = 'Risk: ' + risk.toUpperCase();
            riskBadge.className = 'badge ' + risk;

            document.getElementById('ai-summary').innerText = interpretation.market_summary || 'No summary provided.';

            const cleanestDiv = document.getElementById('ai-cleanest-biases');
            cleanestDiv.innerHTML = '';
            const cleanest = interpretation.cleanest_bias_symbols || [];
            if (cleanest.length === 0) {
                cleanestDiv.innerHTML = '<span style="color: var(--text-muted);">None</span>';
            } else {
                cleanest.forEach(sym => {
                    const span = document.createElement('span');
                    span.className = 'badge low';
                    span.innerText = sym.replace("USDT", "");
                    cleanestDiv.appendChild(span);
                });
            }

            const suppressedDiv = document.getElementById('ai-suppressed-signals');
            suppressedDiv.innerHTML = '';
            const suppressed = interpretation.suppressed_symbols || [];
            if (suppressed.length === 0) {
                suppressedDiv.innerHTML = '<span style="color: var(--text-muted);">None</span>';
            } else {
                suppressed.forEach(sym => {
                    const span = document.createElement('span');
                    span.className = 'badge medium';
                    span.innerText = sym.replace("USDT", "");
                    suppressedDiv.appendChild(span);
                });
            }

            const symAnalysisDiv = document.getElementById('ai-symbol-analysis');
            symAnalysisDiv.innerHTML = '';
            const rawSyms = interpretation.symbol_interpretations || {};
            const aiStale = Boolean(data.stale) || Boolean(data.fallback_used);
            const syms = {};
            Object.keys(rawSyms).forEach(sym => {
                syms[sym] = enrichAiSymbolInfo(sym, rawSyms[sym] || {}, data);
                syms[sym].ai_cache_stale = aiStale;
            });
            const symKeys = Object.keys(syms);
            currentAiSymbolDetails = syms;
            const mismatchCount = symKeys.filter(sym => syms[sym] && syms[sym].ai_mismatch).length;
            const syncWarning = document.getElementById('ai-sync-warning');
            if (mismatchCount > 0) {
                syncWarning.style.display = 'block';
                syncWarning.innerText = mismatchCount + ' AI card' + (mismatchCount === 1 ? '' : 's') + ' had cached bias different from the live radar. Showing current scanner Next Action on cards.';
            } else {
                syncWarning.style.display = 'none';
                syncWarning.innerText = '';
            }
            const rankedByConfidence = symKeys
                .map(sym => ({ sym, info: syms[sym] || {} }))
                .sort((a, b) => Number(b.info.confidence || 0) - Number(a.info.confidence || 0));

            renderAiCardGrid(
                document.getElementById('ai-high-probability'),
                rankedByConfidence.slice(0, 6),
                'No AI confidence data available.'
            );

            if (symKeys.length === 0) {
                symAnalysisDiv.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">No symbol explanations available.</span>';
            } else {
                renderAiCardGrid(
                    symAnalysisDiv,
                    rankedByConfidence,
                    'No symbol explanations available.'
                );
            }
        }

        document.getElementById('ai-detail-overlay').addEventListener('click', (event) => {
            if (event.target.id === 'ai-detail-overlay') {
                closeAiSymbolDetail();
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeAiSymbolDetail();
                closeSymbolDetails();
            }
        });

        // ==========================================
        // SYMBOL DETAILS & HEATMAP (v3.0)
        // ==========================================
        let activeSymbol = null;
        let detailsInterval = null;
        let heatmapZoomFactor = 1.0;
        let hoverPrice = null;
        let hoverTime = null;

        // ==========================================
        // SIGNAL ALERTS & NOTIFICATIONS LOGIC
        // ==========================================
        let lastSignalStates = {}; // Map of symbol -> last seen signal state
        let audioCtx = null;

        function processSignalAlerts(data) {
            if (!data || !data.symbols) return;
            
            const includeWatch = document.getElementById('alert-watch').checked;
            
            Object.keys(data.symbols).forEach(sym => {
                const s = data.symbols[sym];
                const symbolUpper = sym.toUpperCase();
                const action = s.next_action || 'WAITING';
                const lastAction = lastSignalStates[sym];
                
                // We only alert on transitions (when the action changes)
                if (lastAction !== undefined && lastAction !== action) {
                    let triggerAlert = false;
                    let type = ''; // BUY, SELL, WATCH_LONG, WATCH_SHORT
                    let message = '';
                    
                    if (action === 'CONFIRMED_LONG' || action === 'LONG (SWEEP)' || action === 'LONG_BIAS') {
                        triggerAlert = true;
                        type = 'BUY';
                        message = `${symbolUpper} Order Flow BUY signal confirmed! Price: ${s.price}`;
                    } else if (action === 'CONFIRMED_SHORT' || action === 'SHORT (SWEEP)' || action === 'SHORT_BIAS') {
                        triggerAlert = true;
                        type = 'SELL';
                        message = `${symbolUpper} Order Flow SELL signal confirmed! Price: ${s.price}`;
                    } else if (includeWatch) {
                        if (action === 'WATCH_LONG') {
                            triggerAlert = true;
                            type = 'WATCH_LONG';
                            message = `${symbolUpper} Watch Buy: Delta/Aggression turning bullish. Price: ${s.price}`;
                        } else if (action === 'WATCH_SHORT') {
                            triggerAlert = true;
                            type = 'WATCH_SHORT';
                            message = `${symbolUpper} Watch Sell: Delta/Aggression turning bearish. Price: ${s.price}`;
                        }
                    }
                    
                    if (triggerAlert) {
                        triggerSystemAlert(symbolUpper, type, message);
                    }
                }
                
                // Save current state
                lastSignalStates[sym] = action;
            });
        }

        function triggerSystemAlert(symbol, type, message) {
            // 1. Add to HTML signal log panel
            const logContainer = document.getElementById('alert-log-container');
            const noAlertsMsg = logContainer.querySelector('.no-alerts-msg');
            if (noAlertsMsg) {
                logContainer.innerHTML = '';
            }
            
            const timeStr = new Date().toLocaleTimeString();
            const alertEl = document.createElement('div');
            alertEl.className = `alert-item alert-${type.toLowerCase().replace('_', '-')}`;
            alertEl.innerHTML = `
                <div class="alert-left">
                    <span class="alert-badge">${type}</span>
                    <span class="alert-symbol">${symbol}</span>
                    <span>${message}</span>
                </div>
                <div class="alert-time">${timeStr}</div>
            `;
            logContainer.insertBefore(alertEl, logContainer.firstChild);
            
            // Limit log history to last 50 items
            if (logContainer.children.length > 50) {
                logContainer.lastChild.remove();
            }

            // 2. Play Audio feedback
            if (document.getElementById('alert-sound').checked) {
                playAlertSound(type);
            }

            // 3. Show Toast Notification
            showToastNotification(type, symbol, message);

            // 4. Desktop Push Notification
            if (document.getElementById('alert-enable').checked) {
                showPushNotification(type, message);
            }
        }

        function playAlertSound(type) {
            try {
                if (!audioCtx) {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }
                if (audioCtx.state === 'suspended') {
                    audioCtx.resume();
                }
                
                const now = audioCtx.currentTime;
                
                if (type === 'BUY') {
                    // C5 (523.25Hz) -> E5 (659.25Hz) ascending
                    playTone(523.25, 0.1, now);
                    playTone(659.25, 0.15, now + 0.1);
                } else if (type === 'SELL') {
                    // G4 (392.00Hz) -> Eb4 (311.13Hz) descending
                    playTone(392.00, 0.1, now);
                    playTone(311.13, 0.15, now + 0.1);
                } else {
                    // Watch/Warning: A single neutral chime tone A4 (440Hz)
                    playTone(440.00, 0.15, now);
                }
            } catch (e) {
                console.error("Audio beep failed: ", e);
            }
        }

        function playTone(freq, duration, startTime) {
            if (!audioCtx) return;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, startTime);
            
            gain.gain.setValueAtTime(0.15, startTime);
            // Exponential decay to avoid clicking sounds
            gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
            
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.start(startTime);
            osc.stop(startTime + duration);
        }

        function showToastNotification(type, symbol, message) {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast-item toast-${type.toLowerCase().replace('_', '-')}`;
            
            toast.innerHTML = `
                <div class="toast-header">
                    <span class="toast-title">${type} ALERT</span>
                    <button class="toast-close" onclick="this.parentElement.parentElement.remove()">✕</button>
                </div>
                <div class="toast-body">${message}</div>
            `;
            
            container.appendChild(toast);
            
            // Auto remove after 6 seconds
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%) scale(0.9)';
                setTimeout(() => toast.remove(), 300);
            }, 6000);
        }

        function toggleBrowserNotifications() {
            const checkbox = document.getElementById('alert-enable');
            if (checkbox.checked) {
                if (!("Notification" in window)) {
                    alert("This browser does not support desktop notifications.");
                    checkbox.checked = false;
                    return;
                }
                
                if (Notification.permission === "default") {
                    Notification.requestPermission().then(permission => {
                        if (permission !== "granted") {
                            checkbox.checked = false;
                        }
                    });
                } else if (Notification.permission === "denied") {
                    alert("Notification permission has been denied. Please enable them in your browser settings.");
                    checkbox.checked = false;
                }
            }
        }

        function showPushNotification(type, message) {
            if ("Notification" in window && Notification.permission === "granted") {
                new Notification(`${type} Signal - Order Flow Scanner`, {
                    body: message
                });
            }
        }

        function clearAlertHistory() {
            const logContainer = document.getElementById('alert-log-container');
            logContainer.innerHTML = '<div class="no-alerts-msg">No BUY/SELL signals triggered in this session. Watching 20 symbols...</div>';
        }

        // ==========================================
        // PAPER TRADING TERMINAL LOGIC
        // ==========================================
        let paperSymbolSelectorInitialized = false;
        let togglingAutoTrade = false;

        function updatePaperTraderUI(data) {
            if (!data || !data.paper_portfolio) return;
            const p = data.paper_portfolio;
            
            // 1. Initialize manual order entry symbols list once
            if (!paperSymbolSelectorInitialized && data.symbols) {
                const selector = document.getElementById('paper-order-symbol');
                if (selector) {
                    selector.innerHTML = '';
                    Object.keys(data.symbols).forEach(sym => {
                        const opt = document.createElement('option');
                        opt.value = sym;
                        opt.innerText = sym.toUpperCase();
                        selector.appendChild(opt);
                    });
                    paperSymbolSelectorInitialized = true;
                }
            }
            
            // 2. Set auto-trade status badge
            const autoChk = document.getElementById('paper-auto-trade');
            if (autoChk && !togglingAutoTrade) autoChk.checked = Boolean(p.auto_trade_enabled);
            
            // 3. Update stats card
            const nav = p.equity;
            const cash = p.cash;
            const realized = p.realized_pnl;
            const unrealized = p.unrealized_pnl;
            
            const navEl = document.getElementById('paper-nav');
            if (navEl) {
                navEl.innerText = '$' + nav.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                navEl.style.color = nav >= 10000.0 ? '#00e676' : '#ff1744';
            }
            
            const cashEl = document.getElementById('paper-cash');
            if (cashEl) cashEl.innerText = '$' + cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' USDT';
            
            const realEl = document.getElementById('paper-realized');
            if (realEl) {
                realEl.innerText = (realized >= 0 ? '+' : '') + '$' + realized.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                realEl.style.color = realized >= 0 ? '#00e676' : '#ff1744';
            }
            
            const unrealEl = document.getElementById('paper-unrealized');
            if (unrealEl) {
                unrealEl.innerText = (unrealized >= 0 ? '+' : '') + '$' + unrealized.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                unrealEl.style.color = unrealized >= 0 ? '#00e676' : '#ff1744';
            }
            
            // 4. Update Open Positions Table
            const tbody = document.getElementById('paper-positions-body');
            if (tbody) {
                tbody.innerHTML = '';
                if (p.positions.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 16px;">No active positions open.</td></tr>';
                } else {
                    p.positions.forEach(pos => {
                        const row = document.createElement('tr');
                        row.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
                        
                        const pnlSign = pos.pnl >= 0 ? '+' : '';
                        const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
                        
                        row.innerHTML = `
                            <td style="padding: 6px; font-weight: bold; color: var(--text);">${pos.symbol}</td>
                            <td style="padding: 6px; font-weight: bold;" class="${pos.side === 'BUY' ? 'positive' : 'negative'}">${pos.side === 'BUY' ? 'LONG' : 'SHORT'}</td>
                            <td style="padding: 6px; font-family: monospace;">${pos.qty.toFixed(4)}</td>
                            <td style="padding: 6px; font-family: monospace;">$${pos.entry_price.toFixed(4)}</td>
                            <td style="padding: 6px; font-family: monospace;">$${pos.mark_price.toFixed(4)}</td>
                            <td style="padding: 6px; font-family: monospace; text-align: right;" class="${pnlClass}">${pnlSign}$${pos.pnl.toFixed(2)} (${pnlSign}${pos.pnl_pct.toFixed(2)}%)</td>
                            <td style="padding: 6px; text-align: center;">
                                <button class="clear-alerts-btn" onclick="closePaperPosition('${pos.symbol}', '${pos.side}', ${pos.qty})" style="background: rgba(255, 23, 68, 0.15); color: #ff1744; border-color: rgba(255, 23, 68, 0.3); padding: 2px 6px; font-size: 10px;">CLOSE</button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                }
            }
            
            // 5. Update Executions Log
            const tradesList = document.getElementById('paper-trades-list');
            if (tradesList) {
                tradesList.innerHTML = '';
                if (p.trades.length === 0) {
                    tradesList.innerHTML = '<li style="color: var(--text-muted);">No executions recorded yet.</li>';
                } else {
                    [...p.trades].reverse().forEach(t => {
                        const li = document.createElement('li');
                        li.style.borderBottom = '1px solid rgba(255, 255, 255, 0.02)';
                        li.style.paddingBottom = '4px';
                        
                        const timeStr = new Date(t.timestamp * 1000).toLocaleTimeString();
                        const actionType = t.type === 'CLOSE' ? 'CLOSE' : (t.type === 'PARTIAL_CLOSE' ? 'PART_CLOSE' : 'OPEN');
                        const sideColor = t.side === 'BUY' ? '#00e676' : '#ff1744';
                        
                        let pnlText = '';
                        if (t.realized_pnl !== 0.0) {
                            pnlText = ` | PnL: ${t.realized_pnl >= 0 ? '+' : ''}$${t.realized_pnl.toFixed(2)}`;
                        }
                        
                        li.innerHTML = `
                            <span style="color: var(--text-muted);">${timeStr}</span> | 
                            <span style="font-weight: bold; color: ${sideColor};">${t.side}</span> | 
                            <span style="font-weight: bold; color: var(--text);">${t.symbol}</span> | 
                            <span>${t.qty.toFixed(4)} @ $${t.price.toFixed(4)}</span> | 
                            <span style="font-size: 10px; color: var(--primary); font-weight: bold;">[${actionType}]</span>${pnlText}
                        `;
                        tradesList.appendChild(li);
                    });
                }
            }
        }

        async function submitPaperOrder(side) {
            const msgEl = document.getElementById('paper-order-msg');
            msgEl.innerText = 'Sending order...';
            msgEl.style.color = 'var(--text-muted)';
            
            try {
                const symbol = document.getElementById('paper-order-symbol').value;
                const usdtValue = parseFloat(document.getElementById('paper-order-qty').value);
                
                if (isNaN(usdtValue) || usdtValue <= 0) {
                    msgEl.innerText = 'Error: Invalid USDT amount.';
                    msgEl.style.color = '#ff1744';
                    return;
                }
                
                // Fetch the current price from our client-cached data
                const key = symbol.toUpperCase();
                const symbolInfo = currentScannerSymbols[key] || currentScannerSymbols[key + 'USDT'] || currentScannerSymbols[key.replace('USDT', '')];
                if (!symbolInfo) {
                    msgEl.innerText = 'Error: Price unavailable.';
                    msgEl.style.color = '#ff1744';
                    return;
                }
                
                const price = symbolInfo.price;
                if (!price || price <= 0) {
                    msgEl.innerText = 'Error: Symbol price zero.';
                    msgEl.style.color = '#ff1744';
                    return;
                }
                
                // Calculate size in contracts
                const quantity = usdtValue / price;
                
                const response = await fetch('/api/paper/order', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, side, quantity })
                });
                
                const result = await response.json();
                if (result.ok) {
                    msgEl.innerText = `Success: ${side} ${symbol.toUpperCase()} executed.`;
                    msgEl.style.color = '#00e676';
                    
                    // Refresh immediately
                    fetchMetrics();
                } else {
                    msgEl.innerText = `Rejected: ${result.error}`;
                    msgEl.style.color = '#ff1744';
                }
            } catch (err) {
                msgEl.innerText = `Error: ${err.message}`;
                msgEl.style.color = '#ff1744';
            }
            
            setTimeout(() => { msgEl.innerText = ''; }, 5000);
        }

        async function closePaperPosition(symbol, side, qty) {
            const closeSide = side === 'BUY' ? 'SELL' : 'BUY';
            try {
                const response = await fetch('/api/paper/order', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, side: closeSide, quantity: qty })
                });
                const result = await response.json();
                if (result.ok) {
                    fetchMetrics();
                } else {
                    alert(`Failed to close position: ${result.error}`);
                }
            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        }

        async function togglePaperAutoTrade() {
            togglingAutoTrade = true;
            const enabled = document.getElementById('paper-auto-trade').checked;
            try {
                const response = await fetch('/api/paper/toggle_auto', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled })
                });
                const result = await response.json();
                if (result.ok) {
                    fetchMetrics();
                }
            } catch (err) {
                console.error("Auto trade toggle failed: ", err);
            } finally {
                togglingAutoTrade = false;
            }
        }

        async function resetPaperTrader() {
            if (!confirm("Are you sure you want to reset your Paper Trading account back to $10,000 USDT? All positions will be closed.")) return;
            try {
                const response = await fetch('/api/paper/reset', { method: 'POST' });
                const result = await response.json();
                if (result.ok) {
                    fetchMetrics();
                }
            } catch (err) {
                alert(`Reset failed: ${err.message}`);
            }
        }

        function selectSymbol(sym) {
            activeSymbol = sym.toLowerCase();
            document.getElementById('symbol-details-modal').style.display = 'block';
            document.getElementById('modal-title').innerText = sym.toUpperCase() + ' Details';
            
            // Clear prior stats
            document.getElementById('ladder-body').innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">Fetching book ladder...</td></tr>';
            
            // Trigger loop
            if (detailsInterval) clearInterval(detailsInterval);
            fetchSymbolDetails();
            detailsInterval = setInterval(fetchSymbolDetails, 1000);
        }

        function closeSymbolDetails() {
            activeSymbol = null;
            if (detailsInterval) {
                clearInterval(detailsInterval);
                detailsInterval = null;
            }
            document.getElementById('symbol-details-modal').style.display = 'none';
        }

        function zoomHeatmap(factor) {
            heatmapZoomFactor *= factor;
            drawHeatmap();
        }

        function resetHeatmap() {
            heatmapZoomFactor = 1.0;
            drawHeatmap();
        }

        let lastDetailsData = null;

        async function fetchSymbolDetails() {
            if (!activeSymbol) return;
            try {
                const response = await fetch(`/api/symbol?sym=${activeSymbol}`);
                const data = await response.json();
                lastDetailsData = data;
                
                // Update stats
                document.getElementById('detail-spread').innerText = `${data.spread.toFixed(2)} (${data.spread_bps.toFixed(2)} bps)`;
                document.getElementById('detail-microprice').innerText = `${data.microprice_dev.toFixed(2)} bps (Micro: ${data.microprice.toFixed(2)})`;
                document.getElementById('detail-weighted-imbalance').innerText = data.depth_weighted_imbalance.toFixed(2);
                document.getElementById('detail-depth-imbalances').innerText = 
                    `${data.imbalance_0_5_bps.toFixed(2)} | ${data.imbalance_5_15_bps.toFixed(2)} | ${data.imbalance_15_30_bps.toFixed(2)}`;
                
                // Update L2 ladder
                renderLadder(data);
                
                // Update Canvas Heatmap
                drawHeatmap();
            } catch (err) {
                console.error("Error fetching symbol details:", err);
            }
        }

        function renderLadder(data) {
            const tbody = document.getElementById('ladder-body');
            tbody.innerHTML = '';
            
            if (!data.book_history || data.book_history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">No depth data available.</td></tr>';
                return;
            }
            
            const lastSnap = data.book_history[data.book_history.length - 1];
            const bids = Object.entries(lastSnap.bids).map(([p, sz]) => [parseFloat(p), sz]).sort((a,b) => b[0] - a[0]).slice(0, 15);
            const asks = Object.entries(lastSnap.asks).map(([p, sz]) => [parseFloat(p), sz]).sort((a,b) => a[0] - b[0]).slice(0, 15);
            
            // Asks (descending price)
            [...asks].reverse().forEach(([price, size]) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="color: var(--text-muted);"></td>
                    <td style="text-align: center; color: var(--red); font-weight: bold; padding: 4px 0;">${price.toFixed(2)}</td>
                    <td style="text-align: right; color: var(--text-main);">${size.toLocaleString(undefined, {maximumFractionDigits: 3})}</td>
                `;
                tbody.appendChild(tr);
            });
            
            // Spread divider
            const sprTr = document.createElement('tr');
            sprTr.innerHTML = `
                <td colspan="3" style="text-align: center; background: rgba(56, 189, 248, 0.1); font-weight: bold; color: var(--primary); padding: 6px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);">
                    SPREAD: ${data.spread.toFixed(2)} (${data.spread_bps.toFixed(2)} bps)
                </td>
            `;
            tbody.appendChild(sprTr);
            
            // Bids (descending price)
            bids.forEach(([price, size]) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="color: var(--text-main);">${size.toLocaleString(undefined, {maximumFractionDigits: 3})}</td>
                    <td style="text-align: center; color: var(--green); font-weight: bold; padding: 4px 0;">${price.toFixed(2)}</td>
                    <td style="color: var(--text-muted);"></td>
                `;
                tbody.appendChild(tr);
            });
        }

        function drawHeatmap() {
            const canvas = document.getElementById('heatmap-canvas');
            if (!canvas || !lastDetailsData) return;
            const ctx = canvas.getContext('2d');
            
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);
            
            const history = lastDetailsData.book_history;
            if (history.length === 0) return;
            
            let allPrices = [];
            history.forEach(snap => {
                Object.keys(snap.bids).forEach(p => allPrices.push(parseFloat(p)));
                Object.keys(snap.asks).forEach(p => allPrices.push(parseFloat(p)));
            });
            
            if (allPrices.length === 0) return;
            const mid = lastDetailsData.price;
            let maxPrice = Math.max(...allPrices);
            let minPrice = Math.min(...allPrices);
            
            const originalRange = maxPrice - minPrice;
            const range = (originalRange * heatmapZoomFactor) || 1.0;
            maxPrice = mid + range / 2;
            minPrice = mid - range / 2;
            
            const minTime = history[0].timestamp;
            const maxTime = history[history.length - 1].timestamp;
            const timeRange = (maxTime - minTime) || 1.0;
            
            function getX(t) {
                return ((t - minTime) / timeRange) * (width - 80) + 40;
            }
            function getY(p) {
                return height - (((p - minPrice) / range) * (height - 60) + 30);
            }
            
            const cellWidth = Math.max(1, (width - 80) / history.length);
            
            history.forEach((snap, idx) => {
                const tX = getX(snap.timestamp);
                
                Object.entries(snap.bids).forEach(([pStr, sz]) => {
                    const price = parseFloat(pStr);
                    if (price < minPrice || price > maxPrice) return;
                    const pY = getY(price);
                    const sizeNorm = Math.min(1.0, sz / 50.0);
                    ctx.fillStyle = `rgba(56, 189, 248, ${sizeNorm * 0.7})`;
                    ctx.fillRect(tX, pY - 2, cellWidth + 1, 4);
                });
                
                Object.entries(snap.asks).forEach(([pStr, sz]) => {
                    const price = parseFloat(pStr);
                    if (price < minPrice || price > maxPrice) return;
                    const pY = getY(price);
                    const sizeNorm = Math.min(1.0, sz / 50.0);
                    ctx.fillStyle = `rgba(239, 68, 68, ${sizeNorm * 0.7})`;
                    ctx.fillRect(tX, pY - 2, cellWidth + 1, 4);
                });
            });
            
            ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
            ctx.font = "10px 'JetBrains Mono', monospace";
            ctx.textAlign = "right";
            
            const intervals = 8;
            for (let i = 0; i <= intervals; i++) {
                const targetPrice = minPrice + (range / intervals) * i;
                const pY = getY(targetPrice);
                ctx.fillText(targetPrice.toFixed(2), width - 5, pY + 3);
                
                ctx.strokeStyle = "rgba(51, 65, 85, 0.15)";
                ctx.beginPath();
                ctx.moveTo(40, pY);
                ctx.lineTo(width - 80, pY);
                ctx.stroke();
            }
            
            const trades = lastDetailsData.trades || [];
            trades.forEach(t => {
                if (t.timestamp < minTime || t.timestamp > maxTime) return;
                if (t.price < minPrice || t.price > maxPrice) return;
                
                const tX = getX(t.timestamp);
                const tY = getY(t.price);
                const r = Math.max(3, Math.min(15, Math.sqrt(t.notional) / 100));
                
                ctx.beginPath();
                ctx.arc(tX, tY, r, 0, 2 * Math.PI);
                if (t.side === "BUY") {
                    ctx.fillStyle = "rgba(16, 185, 129, 0.6)";
                    ctx.strokeStyle = "rgba(16, 185, 129, 0.9)";
                } else {
                    ctx.fillStyle = "rgba(239, 68, 68, 0.6)";
                    ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
                }
                ctx.fill();
                ctx.stroke();
            });
            
            if (hoverPrice !== null && hoverTime !== null) {
                const tX = getX(hoverTime);
                const tY = getY(hoverPrice);
                
                ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
                ctx.setLineDash([4, 4]);
                
                ctx.beginPath();
                ctx.moveTo(40, tY);
                ctx.lineTo(width - 80, tY);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.moveTo(tX, 30);
                ctx.lineTo(tX, height - 30);
                ctx.stroke();
                
                ctx.setLineDash([]);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('heatmap-canvas');
            if (!canvas) return;
            const tooltip = document.getElementById('heatmap-tooltip');
            
            canvas.addEventListener('mousemove', (e) => {
                if (!lastDetailsData || lastDetailsData.book_history.length === 0) return;
                
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                const width = canvas.width;
                const height = canvas.height;
                
                const history = lastDetailsData.book_history;
                const minTime = history[0].timestamp;
                const maxTime = history[history.length - 1].timestamp;
                const timeRange = (maxTime - minTime) || 1.0;
                
                const pctX = (mouseX - 40) / (width - 80);
                if (pctX < 0 || pctX > 1.0) {
                    tooltip.style.display = 'none';
                    hoverPrice = null;
                    hoverTime = null;
                    drawHeatmap();
                    return;
                }
                
                const targetTime = minTime + pctX * timeRange;
                let bestSnap = history[0];
                let bestTimeDiff = Math.abs(bestSnap.timestamp - targetTime);
                
                history.forEach(snap => {
                    const diff = Math.abs(snap.timestamp - targetTime);
                    if (diff < bestTimeDiff) {
                        bestTimeDiff = diff;
                        bestSnap = snap;
                    }
                });
                
                let allPrices = [];
                Object.keys(bestSnap.bids).forEach(p => allPrices.push(parseFloat(p)));
                Object.keys(bestSnap.asks).forEach(p => allPrices.push(parseFloat(p)));
                
                if (allPrices.length === 0) return;
                const mid = lastDetailsData.price;
                const maxPrice = Math.max(...allPrices);
                const minPrice = Math.min(...allPrices);
                const range = ((maxPrice - minPrice) * heatmapZoomFactor) || 1.0;
                const limitMax = mid + range / 2;
                const limitMin = mid - range / 2;
                
                const pctY = 1.0 - (mouseY - 30) / (height - 60);
                const targetPrice = limitMin + pctY * range;
                
                let closestPrice = null;
                let closestDiff = Infinity;
                let closestSize = 0;
                let isBid = true;
                
                Object.entries(bestSnap.bids).forEach(([pStr, sz]) => {
                    const p = parseFloat(pStr);
                    const diff = Math.abs(p - targetPrice);
                    if (diff < closestDiff) {
                        closestDiff = diff;
                        closestPrice = p;
                        closestSize = sz;
                        isBid = true;
                    }
                });
                Object.entries(bestSnap.asks).forEach(([pStr, sz]) => {
                    const p = parseFloat(pStr);
                    const diff = Math.abs(p - targetPrice);
                    if (diff < closestDiff) {
                        closestDiff = diff;
                        closestPrice = p;
                        closestSize = sz;
                        isBid = false;
                    }
                });
                
                if (closestPrice !== null) {
                    hoverPrice = closestPrice;
                    hoverTime = bestSnap.timestamp;
                    
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                    tooltip.style.top = (e.clientY - rect.top + 15) + 'px';
                    
                    const sideText = isBid ? '<span style="color: var(--green);">BID</span>' : '<span style="color: var(--red);">ASK</span>';
                    tooltip.innerHTML = `
                        <strong>Price:</strong> ${closestPrice.toFixed(2)}<br>
                        <strong>Side:</strong> ${sideText}<br>
                        <strong>Depth:</strong> ${closestSize.toLocaleString()} Qty
                    `;
                    drawHeatmap();
                }
            });
            
            canvas.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
                hoverPrice = null;
                hoverTime = null;
                drawHeatmap();
            });
        });

        // Metrics polling every 1s
        setInterval(fetchMetrics, 1000);
        fetchMetrics();

        // AI Cache polling every 5s
        setInterval(fetchLatestAI, 5000);
        fetchLatestAI();

        document.getElementById('ai-settings-provider').addEventListener('change', (event) => {
            const provider = event.target.value;
            document.getElementById('ai-settings-model').value = defaultAiModels[provider] || '';
        });
        fetchAiSettings();
    </script>
</body>
</html>
"""
