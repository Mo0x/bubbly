# 🍾 Bubbly Market Monitor

**Bubbly** is a macroeconomic composite model designed to detect unsustainable market regimes ("bubbles") by combining valuation pressure with liquidity and volatility triggers.

It generates:
- 📊 An interactive dashboard (`output/bubbly_dashboard.html`)
- 📄 A stylized HTML report (`output/bubbly_report.html`)
- 📈 Static backtest charts (`output/bubbly_backtest.png`)
- 💾 A historical dataset (`output/bubbly_history.csv`)

---

## 🧠 Methodology

### Index Composition
The Bubbly Index is a weighted composite of **10 indicators**, categorized into two drivers: **Valuation Pressure** (long-term structural setup) and **Liquidity/Volatility Triggers** (short-term catalysts).

#### 1. Valuation Pressure (Structural)
High values indicate an expensive, overextended market.
- **Buffett Ratio (30%)**: Total US Stock Market Cap / GDP. The ultimate measure of valuations relative to the economy.
- **Shiller CAPE (30%)**: Cyclically Adjusted P/E Ratio. Valuations normalized for the business cycle.
- **C&I Loans YoY (22%)**: Commercial & Industrial Loans growth. A proxy for corporate leverage and "animal spirits."

#### 2. Liquidity & Volatility Triggers (Catalysts)
High values indicate tightening liquidity or rising stress. (Note: Inverted indicators mean *lower* values add to the bubble score).
- **M2 Money Supply YoY (-25%)**: *Inverted*. Slower money growth = tighter liquidity = higher stress.
- **Fed Balance Sheet YoY (-20%)**: *Inverted*. QT (shrinking balance sheet) adds to the score.
- **Reverse Repo (RRP) YoY (15%)**: Rising RRP drains liquidity from the system.
- **High Yield Spread (24%)**: Rising spreads indicate credit stress.
- **Investment Grade Spread (12%)**: Rising spreads indicate higher quality credit stress.
- **Vol Term Structure (24%)**: VIX / VXV ratio. An inverted curve (high short-term vol) signals fear.
- **VIX (-18%)**: *Inverted*. Low volatility (complacency) adds to the bubble score during the buildup phase.

### Calculation Logic
1.  **Normalization**: Each raw indicator is converted into an **expanding Z-score** (standard score).
    *   *Why expanding?* To prevent look-ahead bias. The Z-score at time $t$ is calculated using only data available up to $t$.
    *   *Min History*: Requires 24 months of data before generating a score.
    *   *Clipping*: Z-scores are clipped at $\pm 4.0$ to contain outliers.
2.  **Aggregation**: The Z-scores are weighted (using the weights above) and summed to produce the **Composite Score**.

### Regime Definitions
| Composite Score | Regime | Description |
| :--- | :--- | :--- |
| **< 1.0** | 🟢 **Expansion** | Healthy market behavior. Valuations and liquidity are supportive. |
| **1.0 – 2.0** | 🟡 **Euphoria** | Warning zone. Valuation pressure is building or liquidity is fading. |
| **≥ 2.0** | 🔴 **Instability** | Danger zone. Highly stretched valuations meeting distinct liquidity/credit stress. |

---

## 🚀 How to Run

### 1. Setup
Create an `apikeys.env` file in the root directory (optional, but recommended for full data access):
```env
FRED_API_KEY=your_key_here
QUANDL_API_KEY=your_key_here
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute
Run the full pipeline wrapper:
```bash
./run.sh
```
This script handles:
-   Dependency checks.
-   Data fetching (FRED, Yahoo, Shiller).
-   Manual M2 cache fallback (resilience against FRED API outages).
-   Report generation.

### 3. View Results
Open the generated artifacts in the `output/` directory:
-   `bubbly_dashboard.html`: Interactive view.
-   `bubbly_report.html`: Summary report.

---

## 🛠 Resilience Notes
**M2 Money Supply**: We source `WM2NS` from FRED. If the FRED API fails (common for this specific series), the script automatically falls back to `data/m2_manual.csv`. If you encounter data gaps, manually update this CSV with the latest values.
