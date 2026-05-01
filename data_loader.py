"""
data_loader.py
--------------
Commodity price data loader for commodity-risk-analytics.

Live mode  : fetches from EIA API (requires free API key from eia.gov)
Demo mode  : generates realistic synthetic prices using geometric Brownian motion

Usage:
    from data_loader import load_commodity_data
    df = load_commodity_data(source="demo")
    df = load_commodity_data(source="eia")  # reads key from .env
"""

import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()

COMMODITY_PARAMS = {
    "WTI Crude Oil ($/bbl)": {
        "start_price": 78.0,
        "annual_vol": 0.35,
        "annual_drift": 0.02,
    },
    "Natural Gas ($/MMBtu)": {
        "start_price": 2.80,
        "annual_vol": 0.55,
        "annual_drift": 0.01,
    },
    "Coal ($/ton)": {
        "start_price": 130.0,
        "annual_vol": 0.28,
        "annual_drift": -0.01,
    },
    "Baltic Dry Index": {
        "start_price": 1450.0,
        "annual_vol": 0.60,
        "annual_drift": 0.00,
    },
}


def _generate_gbm_prices(start_price, annual_vol, annual_drift, n_days, seed=42):
    np.random.seed(seed)
    dt = 1 / 252
    daily_returns = np.exp(
        (annual_drift - 0.5 * annual_vol**2) * dt
        + annual_vol * np.sqrt(dt) * np.random.randn(n_days)
    )
    return start_price * np.cumprod(daily_returns)


def load_demo_data(n_days=504):
    end_date = datetime.today()
    dates = pd.bdate_range(end=end_date, periods=n_days)
    data = {}
    for i, (name, params) in enumerate(COMMODITY_PARAMS.items()):
        data[name] = _generate_gbm_prices(
            params["start_price"], params["annual_vol"],
            params["annual_drift"], n_days, seed=42 + i
        )
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def _fetch_eia_series(series_url, n_days):
    req = urllib.request.urlopen(series_url, timeout=10)
    raw = json.loads(req.read())
    records = raw["response"]["data"]
    s = pd.Series(
        {r["period"]: float(r["value"]) for r in records if r["value"] is not None}
    )
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_eia_data(n_days=504):
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY not found. Add it to your .env file.")

    demo_df = load_demo_data(n_days)
    results = {}

    # WTI Crude
    try:
        url = (
            f"https://api.eia.gov/v2/petroleum/pri/spt/data/"
            f"?api_key={api_key}&frequency=daily&data[0]=value"
            f"&facets[series][]=RWTC"
            f"&sort[0][column]=period&sort[0][direction]=desc&length={n_days}"
        )
        results["WTI Crude Oil ($/bbl)"] = _fetch_eia_series(url, n_days)
        print("WTI Crude: OK")
    except Exception as e:
        print(f"WTI Crude fetch failed: {e} — using demo data")
        results["WTI Crude Oil ($/bbl)"] = demo_df["WTI Crude Oil ($/bbl)"]

    # Natural Gas
    try:
        url = (
            f"https://api.eia.gov/v2/natural-gas/pri/fut/data/"
            f"?api_key={api_key}&frequency=daily&data[0]=value"
            f"&facets[series][]=RNGWHHD"
            f"&sort[0][column]=period&sort[0][direction]=desc&length={n_days}"
        )
        results["Natural Gas ($/MMBtu)"] = _fetch_eia_series(url, n_days)
        print("Natural Gas: OK")
    except Exception as e:
        print(f"Natural Gas fetch failed: {e} — using demo data")
        results["Natural Gas ($/MMBtu)"] = demo_df["Natural Gas ($/MMBtu)"]

    # Coal and BDI — demo (not available on EIA)
    results["Coal ($/ton)"] = demo_df["Coal ($/ton)"]
    results["Baltic Dry Index"] = demo_df["Baltic Dry Index"]

    df = pd.DataFrame(results).dropna(how="all").sort_index()
    df = df.ffill()
    df.index.name = "Date"
    return df


def load_commodity_data(source="demo", api_key=None, n_days=504):
    if source == "eia":
        return load_eia_data(n_days=n_days)
    return load_demo_data(n_days=n_days)


if __name__ == "__main__":
    print("=== Demo Data ===")
    df = load_commodity_data(source="demo")
    print(df.tail(3))

    print("\n=== EIA Live Data ===")
    df_live = load_commodity_data(source="eia")
    print(df_live.tail(3))