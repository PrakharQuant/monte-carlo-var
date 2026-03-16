"""
src/var_engine.py
-----------------
Core simulation and risk functions for the Monte Carlo VaR project.
Pure Python — no Streamlit imports. Can be used standalone or imported.
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for NSE tickers via Yahoo Finance.
    Returns a DataFrame: index=dates, columns=tickers.
    """
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    data = raw["Close"]
    # Flatten multi-level column index produced by yfinance for multi-tickers
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in data.columns
        ]
    return data.dropna(how="all")


def compute_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns from price data."""
    return price_data.pct_change().dropna()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float,
    mc_sims: int = 10000,
    T: int = 100,
    seed: int = None,
) -> np.ndarray:
    """
    Monte Carlo simulation of portfolio value paths.

    Parameters
    ----------
    mean_returns        : mean daily return per asset
    cov_matrix          : covariance matrix of daily returns
    weights             : portfolio weights (will be normalised internally)
    initial_investment  : starting portfolio value (₹)
    mc_sims             : number of simulation paths
    T                   : time horizon in trading days
    seed                : optional random seed

    Returns
    -------
    portfolio_sims : ndarray of shape (T, mc_sims)
    """
    if seed is not None:
        np.random.seed(seed)

    weights = np.array(weights)
    weights = weights / weights.sum()          # guarantee sum = 1
    mean_arr = mean_returns.values             # numpy array for speed
    cov_arr  = cov_matrix.values

    portfolio_sims = np.empty((T, mc_sims))

    for m in range(mc_sims):
        Z = np.random.multivariate_normal(mean_arr, cov_arr, T)
        daily_ret = Z @ weights                # dot product
        portfolio_sims[:, m] = initial_investment * np.cumprod(1.0 + daily_ret)

    return portfolio_sims


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

def calculate_var(
    portfolio_sims: np.ndarray,
    initial_investment: float,
    confidence_levels: list = [0.90, 0.95, 0.99],
) -> dict:
    """
    VaR at multiple confidence levels from simulation ending values.

    Returns dict keyed by confidence level, each value contains:
        ending_value  : portfolio value at that percentile
        var_amount    : loss = initial_investment - ending_value
        var_pct       : var_amount as % of initial_investment
    """
    ending = portfolio_sims[-1, :]
    results = {}
    for cl in confidence_levels:
        lower_pct   = (1.0 - cl) * 100.0
        end_val     = np.percentile(ending, lower_pct)
        var_amount  = initial_investment - end_val
        results[cl] = {
            "ending_pct":   lower_pct,
            "ending_value": end_val,
            "var_amount":   var_amount,
            "var_pct":      var_amount / initial_investment * 100.0,
        }
    return results


def calculate_cvar(
    portfolio_sims: np.ndarray,
    initial_investment: float,
    confidence_level: float = 0.95,
) -> dict:
    """
    Conditional VaR (CVaR / Expected Shortfall).
    Average loss BEYOND the VaR threshold. Basel III preferred metric.
    """
    ending       = portfolio_sims[-1, :]
    lower_pct    = (1.0 - confidence_level) * 100.0
    var_threshold = np.percentile(ending, lower_pct)
    tail         = ending[ending <= var_threshold]
    cvar_value   = float(np.mean(tail))
    cvar_amount  = initial_investment - cvar_value
    return {
        "cvar_value":  cvar_value,
        "cvar_amount": cvar_amount,
        "cvar_pct":    cvar_amount / initial_investment * 100.0,
        "tail_count":  int(len(tail)),
    }


def calculate_historical_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float,
    confidence_levels: list = [0.90, 0.95, 0.99],
) -> dict:
    """
    Non-parametric Historical VaR from actual observed returns.
    Single-day figure — scale by √T for multi-day comparison.
    """
    weights = np.array(weights) / np.sum(weights)
    port_ret = returns.values @ weights
    results  = {}
    for cl in confidence_levels:
        lower_pct  = (1.0 - cl) * 100.0
        var_return = float(np.percentile(port_ret, lower_pct))
        var_amount = initial_investment * abs(var_return)
        results[cl] = {
            "var_return": var_return,
            "var_amount": var_amount,
            "var_pct":    abs(var_return) * 100.0,
        }
    return results


def stress_test(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float,
    shock_annual: float = -0.30,
    mc_sims: int = 5000,
    T: int = 100,
    seed: int = 99,
) -> np.ndarray:
    """
    Stress scenario: shock mean returns by shock_annual / 252 per day.
    E.g. shock_annual = -0.30 simulates a -30% annualised bear market.
    """
    daily_shock    = shock_annual / 252.0
    stressed_means = mean_returns + daily_shock
    return run_monte_carlo(
        stressed_means, cov_matrix, weights,
        initial_investment, mc_sims, T, seed
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def build_var_table(
    var_results: dict,
    initial_investment: float,
) -> pd.DataFrame:
    """Clean summary DataFrame of VaR across confidence levels."""
    rows = []
    for cl, v in var_results.items():
        rows.append({
            "Confidence":      f"{int(cl * 100)}%",
            "VaR (₹)":         f"₹{v['var_amount']:,.0f}",
            "VaR (% Capital)": f"{v['var_pct']:.2f}%",
            "Portfolio Floor": f"₹{v['ending_value']:,.0f}",
        })
    return pd.DataFrame(rows)

