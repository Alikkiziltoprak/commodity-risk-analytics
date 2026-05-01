"""
risk_metrics.py
---------------
Market risk calculations for commodity portfolios.

Covers the core metrics expected in energy trading risk roles:
- Daily / Log returns
- Historical VaR (Value at Risk) at multiple confidence levels
- CVaR / Expected Shortfall
- Volatility (rolling & annualised)
- Correlation matrix
- Max Drawdown
- Sharpe-style return/risk ratio
"""

import numpy as np
import pandas as pd
from typing import Dict


# ── Returns ──────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from price series.

    Parameters
    ----------
    prices : DataFrame of prices (rows = dates, cols = commodities)
    method : "log" (default) or "simple"
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ── VaR ──────────────────────────────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (non-parametric) Value at Risk.
    Returns the loss (positive number) at the given confidence level.

    Example: VaR(0.95) = 0.032 means 95% of days losses are below 3.2%.
    """
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric VaR assuming normally distributed returns.
    """
    from scipy import stats
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) — average loss beyond VaR.
    More conservative than VaR; preferred under Basel III / FRTB.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns < -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def var_summary(returns: pd.DataFrame, confidence_levels: list = [0.90, 0.95, 0.99]) -> pd.DataFrame:
    """
    VaR and CVaR table for all commodities at multiple confidence levels.
    Returns a DataFrame (commodities × metrics).
    """
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        row = {"Commodity": col}
        for cl in confidence_levels:
            label = f"{int(cl*100)}%"
            row[f"VaR ({label})"] = historical_var(r, cl)
            row[f"CVaR ({label})"] = cvar(r, cl)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Commodity")


# ── Volatility ────────────────────────────────────────────────────────────────

def annualised_vol(returns: pd.Series) -> float:
    """Annualised volatility (252 trading days)."""
    return float(returns.std() * np.sqrt(252))


def rolling_vol(returns: pd.Series, window: int = 30) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of returns."""
    return returns.corr()


# ── Drawdown ──────────────────────────────────────────────────────────────────

def max_drawdown(prices: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.
    Returns a positive number (e.g. 0.32 = 32% drawdown).
    """
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(-drawdown.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Full drawdown time series for charting."""
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


# ── Portfolio ─────────────────────────────────────────────────────────────────

def portfolio_var(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.95,
) -> float:
    """
    Historical portfolio VaR given asset weights.

    Parameters
    ----------
    weights : dict mapping column name → weight (must sum to 1)
    """
    w = pd.Series(weights)
    w = w / w.sum()  # normalise
    port_returns = returns[w.index].dot(w)
    return historical_var(port_returns, confidence)


# ── Summary table ─────────────────────────────────────────────────────────────

def risk_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """
    One-stop risk summary table for all commodities.
    Columns: Ann. Vol, VaR 95%, CVaR 95%, Max Drawdown, Last Price, YTD Return
    """
    returns = compute_returns(prices)
    rows = []
    for col in prices.columns:
        r = returns[col].dropna()
        p = prices[col].dropna()
        ytd_start = p[p.index.year == p.index[-1].year].iloc[0] if len(p) > 0 else p.iloc[0]
        ytd_return = (p.iloc[-1] / ytd_start - 1)
        rows.append({
            "Commodity": col,
            "Last Price": round(p.iloc[-1], 2),
            "YTD Return": f"{ytd_return:+.1%}",
            "Ann. Volatility": f"{annualised_vol(r):.1%}",
            "VaR 95% (1-day)": f"{historical_var(r, 0.95):.2%}",
            "CVaR 95% (1-day)": f"{cvar(r, 0.95):.2%}",
            "Max Drawdown": f"{max_drawdown(p):.1%}",
        })
    return pd.DataFrame(rows).set_index("Commodity")


if __name__ == "__main__":
    from data_loader import load_commodity_data
    prices = load_commodity_data()
    returns = compute_returns(prices)

    print("=== Risk Summary ===")
    print(risk_summary(prices).to_string())

    print("\n=== VaR / CVaR Table ===")
    print(var_summary(returns).round(4).to_string())

    print("\n=== Correlation Matrix ===")
    print(correlation_matrix(returns).round(3).to_string())
