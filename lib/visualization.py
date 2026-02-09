from __future__ import annotations
import os
from pathlib import Path
from typing import Mapping
from numbers import Number
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot
import matplotlib.pyplot as plt

# Matplotlib settings
plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
    }
)

def build_plotly_dashboard(
    hist_df: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    output_path: str | Path,
) -> Path:
    """Create an interactive Plotly dashboard and write it to disk."""
    composite = hist_df["Composite"]
    realtime = hist_df["Composite_real_time"]
    pressure_z = hist_df["Pressure_z"]
    trigger_z = hist_df["Trigger_z"]
    pressure_raw = hist_df["Pressure_raw"]
    trigger_raw = hist_df["Trigger_raw"]
    drawdown = hist_df["SPX_Drawdown"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
    )

    # Regime shading on composite panel
    x_start = composite.index.min()
    x_end = composite.index.max()
    regime_shapes = [
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": -4,
            "y1": 1,
            "fillcolor": "rgba(46, 125, 50, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": 1,
            "y1": 2,
            "fillcolor": "rgba(249, 199, 79, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "x1",
            "yref": "y1",
            "x0": x_start,
            "x1": x_end,
            "y0": 2,
            "y1": 4,
            "fillcolor": "rgba(239, 35, 60, 0.18)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    fig.add_trace(
        go.Scatter(
            x=composite.index,
            y=composite,
            name="Composite (z)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Composite (z): %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if realtime.notna().any():
        fig.add_trace(
            go.Scatter(
                x=realtime.index,
                y=realtime,
                name="Composite pseudo real-time",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                hovertemplate="Composite RT (z): %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=1, col=1)

    if event_dates:
        event_points = hist_df.loc[hist_df.index.isin(event_dates)]
        fig.add_trace(
            go.Scatter(
                x=event_points.index,
                y=event_points["Composite"],
                mode="markers",
                name="Signal → drawdown hit",
                marker=dict(color="#ef233c", size=8, symbol="x"),
                hovertemplate="Event: %{x|%Y-%m}<br>Composite: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=1, col=1)

    # Pressure/Trigger traces (z scores)
    fig.add_trace(
        go.Scatter(
            x=pressure_z.index,
            y=pressure_z,
            name="Pressure (z)",
            line=dict(color="#264653", width=2),
            hovertemplate="Pressure (z): %{y:.2f}<br>Raw: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([pressure_raw]),
            visible=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trigger_z.index,
            y=trigger_z,
            name="Trigger (z)",
            line=dict(color="#e76f51", width=2),
            hovertemplate="Trigger (z): %{y:.2f}<br>Raw: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([trigger_raw]),
            visible=True,
        ),
        row=2,
        col=1,
    )

    # Raw traces (initially hidden)
    fig.add_trace(
        go.Scatter(
            x=pressure_raw.index,
            y=pressure_raw,
            name="Pressure (raw)",
            line=dict(color="#264653", width=2, dash="dot"),
            hovertemplate="Pressure (raw): %{y:.2f}<br>Z-score: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([pressure_z]),
            visible=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trigger_raw.index,
            y=trigger_raw,
            name="Trigger (raw)",
            line=dict(color="#e76f51", width=2, dash="dot"),
            hovertemplate="Trigger (raw): %{y:.2f}<br>Z-score: %{customdata[0]:.2f}<extra></extra>",
            customdata=np.column_stack([trigger_z]),
            visible=False,
        ),
        row=2,
        col=1,
    )

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="S&P Drawdown",
            fill="tozeroy",
            line=dict(color="rgba(214, 39, 40, 0.7)", width=1.5),
            fillcolor="rgba(214, 39, 40, 0.3)",
            hovertemplate="Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    vis_z = [True, True, bool(event_dates), True, True, False, False, True]
    vis_raw = [True, True, bool(event_dates), False, False, True, True, True]

    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            x=0,
            y=1.18,
            xanchor="left",
            buttons=[
                dict(
                    label="Z-score",
                    method="update",
                    args=[
                        {"visible": vis_z},
                        {"yaxis2": {"title": "Pressure / Trigger (z-score)"}},
                    ],
                ),
                dict(
                    label="Raw",
                    method="update",
                    args=[
                        {"visible": vis_raw},
                        {"yaxis2": {"title": "Pressure / Trigger (raw units)"}},
                    ],
                ),
            ],
        )
    ]

    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=70, r=40, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        updatemenus=updatemenus,
        shapes=regime_shapes,
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Composite (z)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure / Trigger (z-score)", row=2, col=1)
    fig.update_yaxes(
        title_text="S&P 500 Drawdown",
        row=3,
        col=1,
        tickformat=".0%",
        range=[min(drawdown.min() * 1.1, -1.0), 0.05],
    )

    # Threshold lines
    for thresh in [1, 2]:
        fig.add_hline(
            y=thresh,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            row=1,
            col=1,
        )
    fig.add_hline(
        y=-0.15,
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
        row=3,
        col=1,
    )

    output_path = Path(output_path)
    plotly_plot(
        fig,
        filename=str(output_path),
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "bubbly_dashboard", "scale": 2},
        },
    )
    return output_path

def _format_timestamp(ts) -> str:
    if pd.isna(ts):
        return "N/A"
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d")
    return str(ts)

def save_html_report(
    overview_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    source_dates: Mapping[str, pd.Timestamp],
    composite_value: float,
    phase: str,
    pressure: float,
    trigger: float,
    data_coverage: pd.Timestamp,
    output_dir: str | Path,
    history_chart_filename: str = "bubbly_backtest.png",
) -> Path:
    """Render the HTML report with themed styling and save it to disk."""
    output_path = Path(output_dir) / "bubbly_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overview_html = overview_df.to_html(
        classes="data-table overview-table",
        border=0,
        index=True,
        justify="center",
        float_format=lambda x: f"{x:.2f}",
    )

    if validation_summary.empty:
        validation_html = "<p class=\"empty-note\">No composite signals above threshold are available yet.</p>"
    else:
        validation_html = validation_summary.to_html(
            classes="data-table validation-table",
            border=0,
            index=False,
            justify="center",
            float_format=lambda x: f"{x:.3f}" if isinstance(x, Number) else x,
        )

    rows = []
    for name, ts in source_dates.items():
        if pd.isna(ts):
            sort_key = pd.NaT
        elif isinstance(ts, pd.Timestamp):
            sort_key = ts
        else:
            try:
                sort_key = pd.to_datetime(ts)
            except Exception:
                sort_key = pd.NaT
        rows.append((name, _format_timestamp(ts), sort_key))

    source_df = pd.DataFrame(rows, columns=["Indicator", "Last Update", "_sort"]).sort_values(
        "_sort", na_position="first"
    )
    sources_html = source_df.drop(columns="_sort").to_html(
        classes="data-table sources-table",
        border=0,
        index=False,
        justify="center",
    )

    hero_metrics = f"""
        <div class=\"metrics\">
            <div class=\"metric-card\">
                <span class=\"label\">Composite Phase</span>
                <span class=\"value\">{composite_value:.2f}</span>
                <span class=\"meta\">{phase}</span>
            </div>
            <div class=\"metric-card\">
                <span class=\"label\">Valuation Pressure</span>
                <span class=\"value\">{pressure:.2f}</span>
            </div>
            <div class=\"metric-card\">
                <span class=\"label\">Liquidity Trigger</span>
                <span class=\"value\">{trigger:.2f}</span>
            </div>
            <div class=\"metric-card\">
                <span class=\"label\">Data Coverage</span>
                <span class=\"value\">{_format_timestamp(data_coverage)}</span>
            </div>
        </div>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>🍾 Bubbly Market Monitor</title>
        <style>
            :root {{
                color-scheme: dark;
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }}
            body {{
                margin: 0;
                padding: 0;
                background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.08), transparent 40%),
                            radial-gradient(circle at 80% 30%, rgba(255,255,255,0.05), transparent 45%),
                            #0b0f16;
                color: #f6f7fb;
                min-height: 100vh;
            }}
            header {{
                padding: 2.5rem 1.5rem 1.5rem;
                text-align: center;
                position: relative;
            }}
            header::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: radial-gradient(circle, rgba(255,215,128,0.18) 0%, transparent 65%);
                opacity: 0.6;
                pointer-events: none;
            }}
            .title {{
                font-size: clamp(2.5rem, 4vw, 3.2rem);
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
                z-index: 1;
                position: relative;
            }}
            .subtitle {{
                font-size: 1rem;
                color: rgba(255, 255, 255, 0.7);
                letter-spacing: 0.18em;
                text-transform: uppercase;
                position: relative;
                z-index: 1;
            }}
            main {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 0 1.5rem 4rem;
            }}
            section {{
                background: rgba(12, 18, 30, 0.82);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 24px;
                padding: 2rem;
                margin-bottom: 2rem;
                backdrop-filter: blur(14px);
                box-shadow: 0 20px 45px rgba(5, 10, 22, 0.45);
            }}
            h2 {{
                margin-top: 0;
                font-size: 1.5rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
            }}
            .metric-card {{
                background: linear-gradient(145deg, rgba(38, 52, 74, 0.9), rgba(14, 22, 34, 0.9));
                border-radius: 18px;
                padding: 1.2rem;
                border: 1px solid rgba(255,255,255,0.08);
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }}
            .metric-card .label {{
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: rgba(255, 255, 255, 0.6);
            }}
            .metric-card .value {{
                font-size: 1.8rem;
                font-weight: 600;
                color: #ffd782;
            }}
            .metric-card .meta {{
                font-size: 0.95rem;
                color: rgba(255, 255, 255, 0.7);
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1.5rem;
                border-radius: 12px;
                overflow: hidden;
                font-size: 0.95rem;
            }}
            .data-table thead {{
                background: rgba(255, 215, 130, 0.12);
                color: #ffd782;
            }}
            .data-table tbody tr {{
                background: rgba(15, 22, 32, 0.65);
            }}
            .data-table tbody tr:nth-child(even) {{
                background: rgba(20, 28, 42, 0.8);
            }}
            .data-table th,
            .data-table td {{
                padding: 0.75rem 1rem;
                border: none;
                text-align: center;
            }}
            .data-table tbody tr:hover {{
                background: rgba(255, 215, 130, 0.16);
                transition: background 0.2s ease-in-out;
            }}
            .empty-note {{
                color: rgba(255, 255, 255, 0.6);
                font-style: italic;
                margin-top: 1rem;
            }}
            .chart-wrapper {{
                text-align: center;
                margin-top: 1.5rem;
            }}
            .chart-wrapper img {{
                width: min(100%, 720px);
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 35px rgba(0, 0, 0, 0.45);
            }}
            footer {{
                text-align: center;
                padding: 2rem 1rem;
                color: rgba(255, 255, 255, 0.45);
                font-size: 0.85rem;
            }}
            a {{
                color: #ffd782;
            }}
        </style>
    </head>
    <body>
        <header>
            <div class=\"title\">🍾 Bubbly Market Monitor</div>
            <div class=\"subtitle\">A whimsical watch on market froth</div>
        </header>
        <main>
            <section>
                <h2>Current Pulse</h2>
                {hero_metrics}
            </section>
            <section>
                <h2>Indicator Dashboard</h2>
                {overview_html}
            </section>
            <section>
                <h2>Historical Backdrop</h2>
                <div class=\"chart-wrapper\">
                    <img src=\"{history_chart_filename}\" alt=\"Composite vs S&amp;P 500 drawdown\" />
                </div>
            </section>
            <section>
                <h2>Validation Signals</h2>
                {validation_html}
            </section>
            <section>
                <h2>Data Freshness</h2>
                {sources_html}
            </section>
        </main>
        <footer>
            Crafted with bubbles and benchmarks · {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </footer>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")
    return output_path

def generate_static_plots(hist_df: pd.DataFrame, event_dates: list, output_dir: Path | str = "docs"):
    """Generate static matplotlib plots (backtest and realtime validation)."""
    
    event_points = hist_df.loc[event_dates] if event_dates else pd.DataFrame()

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.8], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_top)

    regime_bands = [
        (-4, 1, "#2d6a4f"),
        (1, 2, "#f9c74f"),
        (2, 4, "#ef233c"),
    ]
    for y0, y1, color in regime_bands:
        ax_top.axhspan(y0, y1, color=color, alpha=0.12, lw=0)

    ax_top.plot(
        hist_df.index,
        hist_df["Composite"],
        label="Composite (z)",
        color="#1f77b4",
        linewidth=1.8,
    )
    if hist_df["Composite_real_time"].notna().any():
        ax_top.plot(
            hist_df.index,
            hist_df["Composite_real_time"],
            label="Composite pseudo real-time",
            color="#ff7f0e",
            linewidth=1.4,
            linestyle="--",
        )

    if not event_points.empty:
        ax_top.scatter(
            event_points.index,
            event_points["Composite"],
            color="#ef233c",
            marker="o",
            edgecolors="white",
            linewidths=0.4,
            s=36,
            label="Signal → drawdown hit",
            zorder=5,
        )

    ax_top.axhline(0, color="white", alpha=0.3, linewidth=0.8)
    for thresh, style in [(1, "--"), (2, ":")]:
        ax_top.axhline(thresh, color="white", alpha=0.4, linewidth=0.8, linestyle=style)

    ax_top.set_ylabel("Composite (z)")
    ax_top.set_title("Bubbly Composite Index")
    ax_top.legend(loc="upper left", frameon=False)
    ax_top.grid(alpha=0.15, linestyle="--")

    # Middle panel: pressure vs trigger
    ax_mid.plot(
        hist_df.index,
        hist_df["Pressure_z"],
        label="Valuation pressure (z)",
        color="#264653",
        linewidth=1.6,
    )
    ax_mid.plot(
        hist_df.index,
        hist_df["Trigger_z"],
        label="Liquidity / Vol trigger (z)",
        color="#e76f51",
        linewidth=1.6,
    )
    ax_mid.axhline(0, color="white", alpha=0.3, linewidth=0.8)
    ax_mid.set_ylabel("Pressure & Trigger (z)")
    ax_mid.legend(loc="upper left", frameon=False)
    ax_mid.grid(alpha=0.15, linestyle="--")

    # Bottom panel: drawdown
    ax_bot.fill_between(
        hist_df.index,
        hist_df["SPX_Drawdown"],
        0,
        color="#d62828",
        alpha=0.3,
        label="S&P 500 drawdown",
    )
    ax_bot.axhline(-0.15, color="white", alpha=0.4, linewidth=0.8, linestyle="--", label="-15% threshold")
    ax_bot.set_ylabel("Drawdown")
    ax_bot.set_xlabel("Date")
    ax_bot.set_ylim(min(hist_df["SPX_Drawdown"].min() * 1.1, -1.0), 0.05)
    ax_bot.legend(loc="lower left", frameon=False)
    ax_bot.grid(alpha=0.15, linestyle="--")

    for ax in (ax_top, ax_mid):
        ax.tick_params(labelbottom=False)

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08, hspace=0.05)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_dir / "bubbly_backtest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Real-time comparison quick chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_df.index, hist_df["Composite"], label="Composite (z)", color="#1f77b4")
    ax.plot(
        hist_df.index,
        hist_df["Composite_real_time"],
        label="Composite pseudo real-time",
        color="#ff7f0e",
        linestyle="--",
    )
    ax.axhline(1, color="white", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(2, color="white", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("Composite (z)")
    ax.set_xlabel("Date")
    ax.set_title("Composite (full vs. pseudo real-time)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.15, linestyle="--")

    ax2 = ax.twinx()
    ax2.fill_between(
        hist_df.index,
        hist_df["SPX_Drawdown"],
        0,
        color="#d62828",
        alpha=0.2,
    )
    ax2.set_ylabel("S&P 500 drawdown")
    ax2.set_ylim(min(hist_df["SPX_Drawdown"].min() * 1.1, -1.0), 0.05)
    fig.tight_layout()
    plt.savefig(out_dir / "bubbly_realtime_backtest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
