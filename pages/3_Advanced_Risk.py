"""
pages/3_Advanced_Risk.py
Streamlit multipage — HuggingFace Spaces compatible.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.var_engine import (
    download_data, compute_returns, run_monte_carlo,
    calculate_var, calculate_cvar, calculate_historical_var, stress_test,
)

st.set_page_config(page_title="Advanced Risk", page_icon="⚠️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #4a5568;
    margin: 1.8rem 0 0.6rem 0; display: block;
}
.callout {
    background: #111128; border-left: 3px solid #e94560;
    padding: 0.9rem 1.3rem; border-radius: 0 8px 8px 0;
    color: #8899aa; font-size: 0.87rem; line-height: 1.7;
}
.callout b { color: #a9cce3; }
.insight {
    background: #0c2240; border: 1px solid #1a4060; border-radius: 8px;
    padding: 0.8rem 1.1rem; color: #90b8d8;
    font-size: 0.86rem; line-height: 1.65; margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

BG = "#0a0a0f"; ACCENT = "#e94560"; BLUE = "#4a9eda"
GREEN = "#48bb78"; ORANGE = "#ed8936"; GREY = "#1e1e38"

def dark_fig(figsize=(10, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors="#4a5568")
    ax.xaxis.label.set_color("#4a5568"); ax.yaxis.label.set_color("#4a5568")
    ax.title.set_color("#cbd5e0")
    for sp in ax.spines.values(): sp.set_edgecolor(GREY)
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

NSE_UNIVERSE = {
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS", "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS", "Bharti Airtel": "BHARTIARTL.NS",
}

selected = st.sidebar.multiselect(
    "Stocks", list(NSE_UNIVERSE.keys()),
    default=["Reliance", "TCS", "HDFC Bank", "ICICI Bank", "Infosys"],
)
if len(selected) < 2:
    st.warning("Select at least 2 stocks."); st.stop()

tickers = [NSE_UNIVERSE[s] for s in selected]
n = len(tickers)
weights = np.array([1.0 / n] * n)          # equal weight for this page

initial_investment = st.sidebar.number_input("Investment (₹)", 100_000, 100_000_000, 1_000_000, 100_000)
T   = int(st.sidebar.number_input("Horizon (days)", 10, 252, 100, 10))
mc_sims = st.sidebar.selectbox("Simulations", [1000, 5000, 10000], index=1)
start_date = st.sidebar.date_input("Start", pd.to_datetime("2020-01-01"))
end_date   = st.sidebar.date_input("End",   pd.to_datetime("2025-01-01"))
run_btn = st.sidebar.button("▶ Run Analysis", type="primary", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────
st.title("⚠️ Advanced Risk Analytics")

if not run_btn:
    st.info("Configure in the sidebar and click **▶ Run Analysis**.")
    for title, desc in [
        ("CVaR / Expected Shortfall",
         "The average loss in the worst scenarios beyond VaR. Required by Basel III. "
         "More tail-sensitive than plain VaR."),
        ("Historical VaR vs Monte Carlo VaR",
         "Compares parametric (simulation-based) VaR against non-parametric historical VaR. "
         "Divergence reveals the cost of the normality assumption."),
        ("Stress Testing",
         "Shocks mean returns downward to simulate a sustained bear market. "
         "Shows how VaR changes under extreme but plausible conditions."),
        ("Rolling VaR",
         "60-day rolling 1-day Historical VaR across the full history. "
         "Reveals how portfolio risk evolved over time — spikes mark crisis periods."),
    ]:
        st.markdown(f"**{title}** — {desc}")
    st.stop()

# ── Load & compute ────────────────────────────────────────────────────────
with st.spinner("📡 Downloading data..."):
    try:
        prices  = download_data(tickers, str(start_date), str(end_date))
        returns = compute_returns(prices)
    except Exception as e:
        st.error(f"Download failed: {e}"); st.stop()

valid = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 50]
if len(valid) < 2:
    st.error("Not enough valid data."); st.stop()
if len(valid) < len(tickers):
    st.warning(f"Dropped: {set(tickers)-set(valid)}")
    idx_v   = [tickers.index(t) for t in valid]
    tickers = valid
    selected= [selected[i] for i in idx_v]
    weights = weights[idx_v]; weights /= weights.sum()
    prices  = prices[tickers]; returns = returns[tickers]

mean_returns = returns.mean()
cov_matrix   = returns.cov()

with st.spinner("⚙️ Running simulations..."):
    base_sims = run_monte_carlo(
        mean_returns, cov_matrix, weights,
        initial_investment, mc_sims, T, seed=42
    )

var_mc   = calculate_var(base_sims, initial_investment)
cvar_res = calculate_cvar(base_sims, initial_investment, 0.95)
var_hist = calculate_historical_var(returns, weights, initial_investment)
ending   = base_sims[-1, :]


# ==========================================================================
# SECTION 1 — CVaR vs VaR
# ==========================================================================
st.markdown('<span class="section-label">1 · CVaR vs VaR — Tail Risk</span>', unsafe_allow_html=True)

st.markdown("""
**CVaR (Conditional VaR / Expected Shortfall)** is the *average* loss given that the loss
exceeds the VaR threshold. It captures *severity*, not just *probability* of the tail.
This is the risk measure now required by Basel III / FRTB for internal models.
""")

col_plot, col_stats = st.columns([2, 1], gap="large")

with col_plot:
    fig, ax = dark_fig((9, 4.5))

    ax.hist(ending, bins=120, color=BLUE, edgecolor="none", alpha=0.6)

    var_v  = var_mc[0.95]["ending_value"]
    cvar_v = cvar_res["cvar_value"]

    tail_vals = ending[ending <= var_v]
    ax.hist(tail_vals, bins=40, color=ACCENT, edgecolor="none", alpha=0.55,
            label="Tail scenarios (worst 5%)")

    ax.axvline(var_v,  color=ACCENT,  linewidth=2.0, linestyle="--",
               label=f"VaR 95%  → loss ₹{var_mc[0.95]['var_amount']/1e5:.2f}L")
    ax.axvline(cvar_v, color=ORANGE, linewidth=2.0, linestyle="-",
               label=f"CVaR 95% → loss ₹{cvar_res['cvar_amount']/1e5:.2f}L")
    ax.axvline(initial_investment, color=GREEN, linewidth=1.5, linestyle=":",
               label="Initial investment")

    ax.set_title("Portfolio Ending Values — VaR vs CVaR")
    ax.set_xlabel("Portfolio Value (₹)"); ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
    ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col_stats:
    ratio = cvar_res["cvar_amount"] / var_mc[0.95]["var_amount"]
    st.markdown(f"""
    <div class="callout">
    <b>VaR 95%</b><br>
    ₹{var_mc[0.95]['var_amount']:,.0f}<br>
    ({var_mc[0.95]['var_pct']:.2f}% of capital)<br><br>
    <b>CVaR 95%</b><br>
    ₹{cvar_res['cvar_amount']:,.0f}<br>
    ({cvar_res['cvar_pct']:.2f}% of capital)<br><br>
    <b>CVaR / VaR ratio:</b> {ratio:.2f}x<br>
    <small>Ratio &gt; 1.2 → meaningful tail risk beyond VaR</small><br><br>
    <b>Tail paths:</b> {cvar_res['tail_count']:,} of {mc_sims:,}
    </div>
    <div class="insight" style="margin-top:0.8rem">
    💡 VaR only marks the threshold.<br>
    CVaR averages everything <i>beyond</i> it —
    it tells you how bad things actually get
    in your worst-case scenarios.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================================
# SECTION 2 — Historical VaR vs MC VaR
# ==========================================================================
st.markdown('<span class="section-label">2 · Historical VaR vs Monte Carlo VaR — Model Risk</span>', unsafe_allow_html=True)

st.markdown("""
**Historical VaR** uses actual observed past returns with no distribution assumption — fat tails
are captured as they happened. **Monte Carlo VaR** assumes multivariate normality, which can
underestimate tail risk. The gap between them is called **model risk**.
""")

conf_levels = [0.90, 0.95, 0.99]
rows = []
for cl in conf_levels:
    mc_v   = var_mc[cl]["var_amount"]
    hist_v = var_hist[cl]["var_amount"] * np.sqrt(T)  # scale 1-day → T-day via √T
    diff   = mc_v - hist_v
    rows.append({
        "Confidence":         f"{int(cl*100)}%",
        "MC VaR (₹)":         f"₹{mc_v:,.0f}",
        "Hist VaR √T (₹)":    f"₹{hist_v:,.0f}",
        "Gap (₹)":            f"₹{abs(diff):,.0f}",
        "MC conservative?":   "✅ Yes" if diff > 0 else "⚠️ No",
    })

col_tbl, col_bar = st.columns([1, 1], gap="large")

with col_tbl:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Historical VaR scaled to T-day horizon via the square-root-of-time rule: VaR(T) ≈ VaR(1) × √T. "
        "A positive gap means the MC model is more conservative (safer). "
        "A negative gap means the normal distribution is underestimating tail risk."
    )

with col_bar:
    fig, ax = dark_fig((7, 4))
    x  = np.arange(3)
    bw = 0.34
    mc_vals   = [var_mc[cl]["var_amount"] / 1e5   for cl in conf_levels]
    hist_vals = [var_hist[cl]["var_amount"] * np.sqrt(T) / 1e5 for cl in conf_levels]

    ax.bar(x - bw/2, mc_vals,   width=bw, label="Monte Carlo VaR", color=BLUE,   alpha=0.85)
    ax.bar(x + bw/2, hist_vals, width=bw, label="Historical VaR (√T)", color=ORANGE, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["90%", "95%", "99%"])
    ax.set_ylabel("VaR (₹ Lakhs)")
    ax.set_title("MC VaR vs Historical VaR")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.1f}L"))
    ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ==========================================================================
# SECTION 3 — Stress Testing
# ==========================================================================
st.markdown('<span class="section-label">3 · Stress Testing — Bear Market Scenario</span>', unsafe_allow_html=True)

col_ctrl, col_stress = st.columns([1, 3], gap="large")

with col_ctrl:
    shock_pct = st.slider("Annual return shock (%)", -70, -5, -30, 5, key="stress_shock")
    shock = shock_pct / 100.0
    st.markdown(f"""
    <div class="callout">
    <b>Annual shock:</b> {shock_pct}%<br>
    <b>Daily shock:</b> {shock/252*100:.4f}%<br><br>
    Simulates sustained bear market where all assets
    earn {abs(shock_pct)}% less per year than their
    historical average — applied additively to
    the estimated mean returns.
    </div>
    """, unsafe_allow_html=True)

with col_stress:
    with st.spinner("Running stress scenario..."):
        stressed_sims = stress_test(
            mean_returns, cov_matrix, weights,
            initial_investment, shock, mc_sims, T, seed=99
        )
    var_stressed  = calculate_var(stressed_sims, initial_investment)
    stressed_end  = stressed_sims[-1, :]

    fig, ax = dark_fig((9, 4.2))
    ax.hist(ending,       bins=100, color=BLUE,   alpha=0.50, edgecolor="none", label="Base scenario")
    ax.hist(stressed_end, bins=100, color=ACCENT, alpha=0.50, edgecolor="none", label=f"Stressed ({shock_pct}%/yr)")
    ax.axvline(initial_investment,                color=GREEN,  linewidth=1.6, linestyle=":",
               label="Initial investment")
    ax.axvline(var_mc[0.95]["ending_value"],      color=BLUE,   linewidth=1.6, linestyle="--",
               label=f"Base VaR95: ₹{var_mc[0.95]['var_amount']/1e5:.2f}L")
    ax.axvline(var_stressed[0.95]["ending_value"], color=ACCENT, linewidth=1.6, linestyle="--",
               label=f"Stress VaR95: ₹{var_stressed[0.95]['var_amount']/1e5:.2f}L")
    ax.set_title(f"Base vs Stressed Distribution — {T} Days")
    ax.set_xlabel("Portfolio Value (₹)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
    ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# Stress comparison table
st.markdown(f"""
| Metric | Base Scenario | Stressed Scenario | Δ Change |
|---|---|---|---|
| VaR 95%          | ₹{var_mc[0.95]['var_amount']:,.0f}    | ₹{var_stressed[0.95]['var_amount']:,.0f}    | +₹{var_stressed[0.95]['var_amount'] - var_mc[0.95]['var_amount']:,.0f}  |
| VaR 99%          | ₹{var_mc[0.99]['var_amount']:,.0f}    | ₹{var_stressed[0.99]['var_amount']:,.0f}    | +₹{var_stressed[0.99]['var_amount'] - var_mc[0.99]['var_amount']:,.0f}  |
| Expected Value   | ₹{np.mean(ending):,.0f} | ₹{np.mean(stressed_end):,.0f} | ₹{np.mean(stressed_end) - np.mean(ending):,.0f} |
| % Paths Below Initial | {(ending < initial_investment).mean()*100:.1f}% | {(stressed_end < initial_investment).mean()*100:.1f}% | +{((stressed_end < initial_investment).mean() - (ending < initial_investment).mean())*100:.1f}pp |
""")


# ==========================================================================
# SECTION 4 — Rolling Historical VaR
# ==========================================================================
st.markdown('<span class="section-label">4 · Rolling 60-Day Historical VaR</span>', unsafe_allow_html=True)

st.markdown("""
This tracks how **1-day portfolio risk evolved over time** using a 60-day rolling window
of historical returns. Spikes correspond to periods of heightened market volatility.
""")

port_daily_returns = returns.values @ weights
port_series = pd.Series(port_daily_returns, index=returns.index)

window = 60
roll_var_95 = port_series.rolling(window).quantile(0.05).abs() * initial_investment
roll_var_99 = port_series.rolling(window).quantile(0.01).abs() * initial_investment

fig2, ax2 = dark_fig((12, 4))
ax2.plot(roll_var_95.index, roll_var_95 / 1e5, color=BLUE,   linewidth=1.1, alpha=0.9,
         label="Rolling VaR 95% (1-day)")
ax2.plot(roll_var_99.index, roll_var_99 / 1e5, color=ACCENT, linewidth=1.1, alpha=0.9,
         label="Rolling VaR 99% (1-day)")
ax2.fill_between(roll_var_95.index, roll_var_95 / 1e5, color=BLUE, alpha=0.08)

ax2.set_title("Rolling 60-Day Historical VaR Over Time (1-Day Horizon, Equal-Weight Portfolio)")
ax2.set_xlabel("Date"); ax2.set_ylabel("1-Day VaR (₹ Lakhs)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.2f}L"))
ax2.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=9)
plt.tight_layout(); st.pyplot(fig2); plt.close()

st.markdown("""
<div class="insight">
💡 The spike around early 2020 is the COVID-19 crash.
The elevated band through 2022 reflects the Russia-Ukraine war and Fed rate hike cycle.
Rolling VaR is how risk desks at banks monitor real-time portfolio risk — if it is rising
steadily, the book's risk is increasing even if the point-in-time Monte Carlo VaR
(computed from the full 5-year window) has not been recalculated.
</div>
""", unsafe_allow_html=True)

