# commodity-risk-analytics

A market risk analytics dashboard for energy commodities — WTI Crude Oil, Natural Gas, Coal, and Baltic Dry Index (freight).

Built as a practical demonstration of quantitative risk management methods applied to energy markets.

## Features

- **Historical VaR & CVaR** at 90%, 95%, 99% confidence levels
- **Rolling annualised volatility** with configurable window
- **Drawdown analysis** per commodity
- **Correlation matrix** across commodities
- **Portfolio-level VaR** with adjustable weights and diversification benefit calculation
- **Normalised price performance** chart
- **Live data mode** via EIA Open Data API (free key required)
- **Demo mode** using Geometric Brownian Motion simulation (no API key needed)

## Tech Stack

- Python 3.10+
- `streamlit` — dashboard UI
- `plotly` — interactive charts
- `pandas` / `numpy` — data manipulation
- `scipy` — parametric VaR
- EIA Open Data API — live commodity prices

## Quick Start

```bash
# Clone
git clone https://github.com/your-username/commodity-risk-analytics.git
cd commodity-risk-analytics

# Install dependencies
pip install -r requirements.txt

# Run dashboard (demo mode — no API key needed)
streamlit run app.py
```

## Live Data (Optional)

Get a free API key at [eia.gov/opendata](https://www.eia.gov/opendata/), then select **EIA API (Live)** in the sidebar and enter your key.

## Project Structure

```
commodity-risk-analytics/
├── app.py              # Streamlit dashboard
├── data_loader.py      # EIA API + GBM synthetic data
├── risk_metrics.py     # VaR, CVaR, vol, drawdown, correlation
├── requirements.txt
└── README.md
```

## Risk Methodology

| Metric | Method |
|--------|--------|
| VaR | Historical (non-parametric) |
| CVaR / Expected Shortfall | Average loss beyond VaR threshold |
| Volatility | Rolling standard deviation, annualised (×√252) |
| Drawdown | Peak-to-trough from rolling maximum |
| Portfolio VaR | Historical simulation with user-defined weights |

## Background

This project reflects hands-on experience monitoring commodity price risk in an industrial setting (coal, natural gas, freight/navlun) and translates that domain knowledge into a quantitative risk framework consistent with industry standards (Basel III, FRTB).

---
*Prices in demo mode are synthetic and generated using Geometric Brownian Motion calibrated to historical volatility ranges. Not financial advice.*
