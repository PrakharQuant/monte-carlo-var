---
title: Monte Carlo VaR — Indian Equity Risk Simulator
emoji: 📊
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---

# Monte Carlo VaR — Indian Equity Portfolio Risk Simulator

An interactive risk analytics platform for NSE-listed equity portfolios.
Built with Python and Streamlit. Covers the mathematics of uncertainty
through to production-style risk metrics used in quantitative finance.

## Pages

| Page | Description |
|---|---|
| 📐 Math Concepts | 5 interactive modules — Normal distribution, Covariance, GBM, Multivariate Normal, Cholesky |
| 📈 Portfolio VaR | Configure NSE portfolio, run 10,000 Monte Carlo paths, view VaR at 90/95/99% |
| ⚠️ Advanced Risk | CVaR / Expected Shortfall, Historical vs MC VaR, Stress Testing, Rolling VaR |

## Run Locally

```bash
git clone https://huggingface.co/spaces/your-username/monte-carlo-var
cd monte-carlo-var
pip install -r requirements.txt
streamlit run app.py
```

## Mathematics

### Monte Carlo Simulation

$$\mathbf{r}_t^{(m)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$V_T^{(m)} = V_0 \cdot \prod_{t=1}^{T} (1 + \mathbf{w}^\top \mathbf{r}_t^{(m)})$$

### Value at Risk

$$\text{VaR}_\alpha = V_0 - Q_\alpha\left(\{V_T^{(m)}\}_{m=1}^{M}\right)$$

### Conditional VaR (Expected Shortfall)

$$\text{CVaR}_\alpha = V_0 - \mathbb{E}\left[V_T^{(m)} \mid V_T^{(m)} \leq Q_\alpha\right]$$

### Cholesky Decomposition

$$\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top \implies \mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}, \quad \mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

## Assumptions & Limitations

| Assumption | Implication |
|---|---|
| Returns are multivariate normal | Underestimates fat tails; real returns are leptokurtic |
| Historical parameters are stable | Non-stationarity: mean/covariance shift across regimes |
| Static correlation | In crises, correlations spike toward 1 |
| Constant weights | No rebalancing modelled across the horizon |
| No transaction costs | Real portfolios incur slippage and fees |

## Tech Stack

`Python` · `Streamlit` · `yfinance` · `NumPy` · `Pandas` · `Matplotlib` · `SciPy`

---

Built by **Prakhar** · Quantitative Finance · NSE Indian Equities
