# Monte Carlo VaR — Indian Equity Portfolio Risk Simulator

# Monte Carlo VaR — Indian Equity Portfolio Risk Simulator

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://montecarlovar.streamlit.app)
![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.26+-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.1+-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-48BB78?style=flat)
![Status](https://img.shields.io/badge/status-live-48BB78?style=flat)

An interactive risk analytics platform for NSE-listed equity portfolios built with Python and Streamlit.
The project is structured in two layers: a mathematical foundations module that builds intuition
interactively, and a finance module that applies those concepts to real market data and industry-standard risk metrics.



---

## Project Structure

```
monte-carlo-var/
│
├── app.py                        # Home page & navigation
│
├── pages/
│   ├── 1_Math_Concepts.py        # Interactive mathematical foundations (6 modules)
│   ├── 2_Portfolio_VaR.py        # Monte Carlo simulation & VaR
│   └── 3_Advanced_Risk.py        # CVaR, stress testing, rolling VaR
│
├── src/
│   ├── __init__.py
│   └── var_engine.py             # Core simulation & risk functions (no Streamlit)
│
├── requirements.txt
└── README.md
```

---

## Pages

### 📐 Page 1 — Math Concepts

Six interactive modules. Every slider updates charts in real time.

| Module | What it covers |
|---|---|
| 1 · Normal Distribution | Adjust μ and σ; watch the VaR cutoff move on the PDF |
| 2 · Covariance & Correlation | Build a 2-asset covariance matrix; see diversification benefit change with ρ |
| 3 · Random Walk (GBM) | Simulate Geometric Brownian Motion paths with the Itô correction |
| 4 · Multivariate Normal | See how 5 correlated assets are drawn simultaneously; inspect correlation heatmap |
| 5 · Cholesky Decomposition | Visualise how Σ = LLᵀ transforms independent normals into correlated returns |
| 6 · Monte Carlo Method | Step-by-step algorithm, live histogram that builds as simulations increase, VaR convergence chart |

### 📈 Page 2 — Portfolio VaR

- Select from 12 major NSE stocks with custom weights and investment size
- Downloads live historical data via Yahoo Finance
- Runs up to 10,000 Monte Carlo simulation paths
- Simulation fan chart with 5th, 50th, 95th percentile bands
- Ending value histogram with VaR markers at 90%, 95%, 99%
- Historical correlation heatmap from downloaded data
- Full portfolio composition table with per-stock mean return and volatility

### ⚠️ Page 3 — Advanced Risk

- **CVaR / Expected Shortfall** — average loss beyond the VaR threshold; the Basel III standard metric
- **Historical VaR vs Monte Carlo VaR** — non-parametric comparison to measure the cost of the normality assumption
- **Stress Testing** — adjustable annual return shock to simulate bear market scenarios; side-by-side distribution comparison
- **Rolling VaR** — 60-day rolling 1-day Historical VaR across the full history, revealing how portfolio risk evolved over time

---

## Mathematics

### Monte Carlo Simulation

At each timestep $t$ in simulation $m$, draw correlated returns:

$$\mathbf{r}_t^{(m)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

Portfolio return for that day:

$$R_t^{(m)} = \mathbf{w}^\top \mathbf{r}_t^{(m)}$$

Compound into a portfolio value path:

$$V_T^{(m)} = V_0 \cdot \prod_{t=1}^{T} \left(1 + R_t^{(m)}\right)$$

### Value at Risk

$$\text{VaR}_\alpha = V_0 - Q_{1-\alpha}\!\left(\left\{V_T^{(m)}\right\}_{m=1}^{M}\right)$$

where $Q_{1-\alpha}$ is the $(1-\alpha)$-th quantile of simulated ending values.

### Conditional VaR (Expected Shortfall)

$$\text{CVaR}_\alpha = V_0 - \mathbb{E}\!\left[V_T^{(m)} \mid V_T^{(m)} \leq Q_{1-\alpha}\right]$$

CVaR averages all losses beyond the VaR threshold. It is more sensitive to tail severity and is the measure required by Basel III / FRTB for internal risk models.

### Cholesky Decomposition

To generate correlated returns from independent standard normals $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top \quad \Rightarrow \quad \mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}, \quad \mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

This transformation is performed internally by `np.random.multivariate_normal` on every call in the simulation loop.

### Covariance Matrix Regularisation

To prevent `SVD did not converge` errors on near-singular matrices:

$$\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \varepsilon \mathbf{I}, \quad \varepsilon = 10^{-8}$$

This nudges all eigenvalues slightly positive without meaningfully affecting risk estimates, since typical daily return variances are on the order of $10^{-4}$.

---

## Assumptions & Limitations

| Assumption | Implication |
|---|---|
| Returns are multivariate normal | Underestimates fat tails; real equity returns are leptokurtic |
| Historical mean and covariance are stationary | Parameters shift across market regimes; calm-period estimates understate crisis risk |
| Correlations are static | In stress periods, NSE large-cap correlations spike toward +1, reducing diversification |
| Portfolio weights are constant | No rebalancing is modelled across the simulation horizon |
| No transaction costs | Real portfolios incur slippage, impact, and brokerage fees |

---

## Setup & Deployment

### Run Locally

```bash
git clone https://github.com/your-username/monte-carlo-var.git
cd monte-carlo-var
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
3. Click **New app** → select this repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

Streamlit reads `requirements.txt` automatically. Any commit to `main` triggers an automatic redeploy.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web app and multipage routing |
| `yfinance` | NSE historical price data via Yahoo Finance |
| `numpy` | Simulation engine — multivariate normal draws, matrix ops |
| `pandas` | Return computation, rolling statistics, data wrangling |
| `matplotlib` | All charts — fan plots, histograms, heatmaps |
| `scipy` | Normal distribution functions for single-asset VaR |

---

## Author

Built by **Prakhar** as part of a quantitative finance project portfolio.
Exploring Monte Carlo simulation, risk analytics, and applied statistics
in the context of Indian equity markets.
