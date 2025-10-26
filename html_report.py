"""Utilities for creating the stylized Bubbly HTML report."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
from numbers import Number

import pandas as pd


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
