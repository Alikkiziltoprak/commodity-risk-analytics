"""
data_loader.py
--------------
Commodity price data loader for commodity-risk-analytics.

Live mode  : fetches from EIA API (requires free API key from eia.gov)
Demo mode  : generates realistic synthetic prices using geometric Brownian motion
             so the dashboard runs without any API key during development.

Usage:
    from data_loader import load_commodity_data
    df = load_commodity_data(source="demo")          # synthetic
    df = load_commodity_data(source="eia", api_key="YOUR_KEY")  # live
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import urllib.request
import json


# ── Commodity parameters (calibrated to historical ranges) ──────────────────
COMMODITY_PARAMS = {
    "WTI Crude Oil ($/bbl)": {
        "start_price": 78.0,
        "annual_vol": 0.35,
        "annual_drift": 0.02,
        "ticker_eia": "PET.RWTC.D",
    },
    "Natural Gas ($/MMBtu)": {
        "start_price": 2.80,
        "annual_vol": 0.55,
        "annual_drift": 0.01,
        "ticker_eia": "NG.RNGWHHD.D",
    },
    "Coal ($/ton)": {
        "start_price": 130.0,
        "annual_vol": 0.28,
        "annual_drift": -0.01,
        "ticker_eia": None,
    },
    "Baltic Dry Index": {
        "start_price": 1450.0,
        "annual_vol": 0.60,
        "annual_drift": 0.00,
        "ticker_eia": None,
    },
}


def _generate_gbm_prices(
    start_price: float,
    annual_vol: float,
    annual_drift: float,
    n_days: int,
    seed: int = 42,
) -> np.ndarray:
    """Geometric Brownian Motion price simulation."""
    np.random.seed(seed)
    dt = 1 / 252
    daily_returns = np.exp(
        (annual_drift - 0.5 * annual_vol**2) * dt
        + annual_vol * np.sqrt(dt) * np.random.randn(n_days)
    )
    prices = start_price * np.cumprod(daily_returns)
    return prices


def load_demo_data(n_days: int = 504) -> pd.DataFrame:
    """
    Generate 2 years of realistic synthetic commodity prices.
    Returns a DataFrame with DatetimeIndex and one column per commodity.
    """
    end_date = datetime.today()
    # Business days only
    dates = pd.bdate_range(end=end_date, periods=n_days)

    data = {}
    for i, (name, params) in enumerate(COMMODITY_PARAMS.items()):
        prices = _generate_gbm_prices(
            start_price=params["start_price"],
            annual_vol=params["annual_vol"],
            annual_drift=params["annual_drift"],
            n_days=n_days,
            seed=42 + i,
        )
        data[name] = prices

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def load_eia_data(api_key: str, n_days: int = 504) -> pd.DataFrame:
    """
    Fetch WTI Crude and Natural Gas from EIA API.
    Get a free key at: https://www.eia.gov/opendata/
    Coal and BDI fall back to demo data (not available on EIA).
    """
    results = {}
    series_map = {
        "WTI Crude Oil ($/bbl)": "PET.RWTC.D",
        "Natural Gas ($/MMBtu)": "NG.RNGWHHD.D",
    }

    for name, series_id in series_map.items():
        url = (
            f"https://api.eia.gov/v2/seriesid/{series_id}"
            f"?api_key={api_key}&frequency=daily"
            f"&data[0]=value&sort[0][column]=period"
            f"&sort[0][direction]=desc&length={n_days}"
        )
        try:
            req = urllib.request.urlopen(url, timeout=10)
            raw = json.loads(req.read())
            records = raw["response"]["data"]
            s = pd.Series(
                {r["period"]: float(r["value"]) for r in records if r["value"] is not None}
            )
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            results[name] = s
        except Exception as e:
            print(f"[WARNING] EIA fetch failed for {name}: {e}. Using demo data.")
            demo = load_demo_data(n_days)
            results[name] = demo[name]

    # Coal and BDI: use demo (no free EIA series)
    demo_df = load_demo_data(n_days)
    results["Coal ($/ton)"] = demo_df["Coal ($/ton)"]
    results["Baltic Dry Index"] = demo_df["Baltic Dry Index"]

    df = pd.DataFrame(results)
    df.index.name = "Date"
    df = df.dropna(how="all").sort_index()
    return df


def load_commodity_data(source: str = "demo", api_key: str = None, n_days: int = 504) -> pd.DataFrame:
    """
    Main entry point.

    Parameters
    ----------
    source  : "demo" | "eia"
    api_key : required when source="eia"
    n_days  : number of trading days to load (default 504 ≈ 2 years)
    """
    if source == "eia":
        if not api_key:
            raise ValueError("api_key is required for source='eia'. Get one free at eia.gov/opendata")
        return load_eia_data(api_key=api_key, n_days=n_days)
    return load_demo_data(n_days=n_days)


if __name__ == "__main__":
    df = load_commodity_data(source="demo")
    print(df.tail())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
