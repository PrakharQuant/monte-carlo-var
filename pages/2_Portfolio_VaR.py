"""
pages/2_Portfolio_VaR.py
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
    calculate_var, calculate_cvar, build_var_table,
)

st.set_page_config(page_title="Portfolio VaR", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #4a5568;
    margin: 1.8rem 0 0.6rem 0; display: block;
}
.metric-card {
    background: #111128; border: 1px solid #1e1e38;
    border-radius: 10px; padding: 1.1rem 1rem; text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem; font-weight: 600; color: #e94560; line-height: 1.2;
}
.metric-label {
    color: #4a5568; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

BG = "#0a0a0f"; ACCENT = "#e94560"; BLUE = "#4a9eda"
GREEN = "#48bb78"; GREY = "#1e1e38"

def dark_fig(figsize=(10, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.tick_params(colors="#4a5568")
    ax.xaxis.label.set_color("#4a5568"); ax.yaxis.label.set_color("#4a5568")
    ax.title.set_color("#cbd5e0")
    for sp in ax.spines.values(): sp.set_edgecolor(GREY)
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Portfolio Configuration")

NSE_UNIVERSE = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "ICICI Bank":          "ICICIBANK.NS",
    "Infosys":             "INFY.NS",
    "Wipro":               "WIPRO.NS",
    "SBI":                 "SBIN.NS",
    "Bharti Airtel":       "BHARTIARTL.NS",
    "ITC":                 "ITC.NS",
    "Maruti Suzuki":       "MARUTI.NS",
    "HUL":                 "HINDUNILVR.NS",
    "L&T":                 "LT.NS",
}

selected_names = st.sidebar.multiselect(
    "Select stocks (3–6 recommended)",
    list(NSE_UNIVERSE.keys()),
    default=["Reliance Industries", "TCS", "HDFC Bank", "ICICI Bank", "Infosys"],
)

if len(selected_names) < 2:
    st.warning("Please select at least 2 stocks."); st.stop()

tickers = [NSE_UNIVERSE[n] for n in selected_names]
n_stocks = len(tickers)

st.sidebar.markdown("**Weights (%) — will be normalised to 100%**")
raw_w = []
for name in selected_names:
    w = st.sidebar.number_input(
        name, min_value=1, max_value=99,
        value=int(100 // n_stocks), step=1, key=f"w_{name}"
    )
    raw_w.append(w)

total_w = sum(raw_w)
weights = np.array(raw_w, dtype=float) / total_w
if abs(total_w - 100) > 1:
    st.sidebar.caption(f"⚠️ Weights sum to {total_w}% — normalised to 100%.")

st.sidebar.markdown("---")
initial_investment = st.sidebar.number_input(
    "Initial Investment (₹)", 100_000, 100_000_000, 1_000_000, 100_000
)

col_a, col_b = st.sidebar.columns(2)
mc_sims = col_a.selectbox("Simulations", [1000, 5000, 10000], index=2)
T       = int(col_b.number_input("Horizon (days)", 10, 252, 100, 10))

start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date   = st.sidebar.date_input("End date",   pd.to_datetime("2025-01-01"))
seed       = int(st.sidebar.number_input("Random seed", 0, 9999, 42))

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────
st.title("📈 Portfolio Monte Carlo VaR")

if not run_btn:
    st.info("Configure your portfolio in the sidebar, then click **▶ Run Simulation**.")
    st.markdown("""
    This page will:
    - Download historical NSE price data from Yahoo Finance
    - Estimate mean daily returns and the full covariance matrix
    - Run Monte Carlo simulation paths over your chosen horizon
    - Calculate VaR at 90%, 95%, and 99% confidence levels
    - Display the simulation fan chart, ending value histogram, and correlation heatmap
    """)
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────
with st.spinner("📡 Downloading price data from Yahoo Finance..."):
    try:
        prices  = download_data(tickers, str(start_date), str(end_date))
        returns = compute_returns(prices)
    except Exception as e:
        st.error(f"Download failed: {e}"); st.stop()

# Drop any tickers that failed to download
valid_tickers = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 50]
if len(valid_tickers) < 2:
    st.error("Not enough valid data downloaded. Try different tickers or date range."); st.stop()

if len(valid_tickers) < len(tickers):
    dropped = set(tickers) - set(valid_tickers)
    st.warning(f"Dropped due to insufficient data: {dropped}")

idx_valid   = [tickers.index(t) for t in valid_tickers]
tickers     = valid_tickers
selected_names = [selected_names[i] for i in idx_valid]
weights     = weights[idx_valid]
weights     = weights / weights.sum()
prices      = prices[tickers]
returns     = returns[tickers]

mean_returns = returns.mean()
cov_matrix   = returns.cov()

# ── Simulation ────────────────────────────────────────────────────────────
with st.spinner(f"⚙️ Running {mc_sims:,} Monte Carlo simulations..."):
    port_sims = run_monte_carlo(
        mean_returns, cov_matrix, weights,
        initial_investment, mc_sims, T, seed
    )

var_res      = calculate_var(port_sims, initial_investment)
cvar_res     = calculate_cvar(port_sims, initial_investment, 0.95)
ending       = port_sims[-1, :]

# ── Metrics ───────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Risk Summary</span>', unsafe_allow_html=True)

cols = st.columns(5)
metrics = [
    (f"₹{initial_investment/1e5:.1f}L",              "Initial Investment"),
    (f"₹{np.mean(ending)/1e5:.2f}L",                 "Expected Value"),
    (f"₹{var_res[0.95]['var_amount']/1e5:.2f}L",     "95% VaR"),
    (f"₹{cvar_res['cvar_amount']/1e5:.2f}L",         "CVaR / ES"),
    (f"{var_res[0.95]['var_pct']:.1f}%",              "VaR as % Capital"),
]
for col, (val, label) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Fan Chart ─────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Simulation Paths</span>', unsafe_allow_html=True)

fig, ax = dark_fig((12, 5))
draw_n = min(mc_sims, 2000)
ax.plot(port_sims[:, :draw_n], color=BLUE, alpha=0.04, linewidth=0.55)

p05 = np.percentile(port_sims, 5,  axis=1)
p50 = np.percentile(port_sims, 50, axis=1)
p95 = np.percentile(port_sims, 95, axis=1)

ax.fill_between(range(T), p05, p95, color=BLUE, alpha=0.07)
ax.plot(p95, color=GREEN,  linewidth=1.6, linestyle="--", label="95th percentile")
ax.plot(p50, color="white", linewidth=1.6,                label="Median")
ax.plot(p05, color=ACCENT, linewidth=1.6, linestyle="--", label="5th percentile (VaR)")
ax.axhline(initial_investment, color=ACCENT, linewidth=1.2, linestyle=":",
           alpha=0.7, label="Initial Investment")

ax.set_title(f"Monte Carlo Simulation — {mc_sims:,} Paths, {T}-Day Horizon")
ax.set_xlabel("Days"); ax.set_ylabel("Portfolio Value (₹)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=9, loc="upper left")
plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Distribution ──────────────────────────────────────────────────────────
st.markdown(f'<span class="section-label">Distribution of Outcomes — Day {T}</span>', unsafe_allow_html=True)

fig2, ax2 = dark_fig((12, 4.5))
ax2.hist(ending, bins=120, color=BLUE, edgecolor="none", alpha=0.7)

styles = {0.90: (":", 1.4), 0.95: ("--", 1.9), 0.99: ("-", 1.9)}
for cl, v in var_res.items():
    ls, lw = styles[cl]
    ax2.axvline(v["ending_value"], color=ACCENT, linestyle=ls, linewidth=lw,
                label=f"VaR {int(cl*100)}%  ₹{v['var_amount']/1e5:.2f}L")

ax2.axvline(cvar_res["cvar_value"], color="orange", linestyle="-.", linewidth=1.6,
            label=f"CVaR 95%  ₹{cvar_res['cvar_amount']/1e5:.2f}L")
ax2.axvline(initial_investment, color=GREEN, linewidth=1.6,
            label=f"Initial  ₹{initial_investment/1e5:.1f}L")

ax2.set_title(f"Distribution of Portfolio Value After {T} Days")
ax2.set_xlabel("Portfolio Value (₹)"); ax2.set_ylabel("Frequency")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
ax2.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=9)
plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── Tables ────────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">VaR Summary Table</span>', unsafe_allow_html=True)

col_tbl, col_comp = st.columns([1, 1], gap="large")

with col_tbl:
    st.dataframe(build_var_table(var_res, initial_investment),
                 use_container_width=True, hide_index=True)
    st.caption(
        f"CVaR (Expected Shortfall) at 95%: **₹{cvar_res['cvar_amount']:,.0f}** "
        f"({cvar_res['cvar_pct']:.2f}% of capital) — "
        f"average loss in the worst {cvar_res['tail_count']} scenarios."
    )

with col_comp:
    comp_df = pd.DataFrame({
        "Stock":         selected_names,
        "Ticker":        tickers,
        "Weight":        [f"{w*100:.1f}%" for w in weights],
        "μ (daily)":     [f"{mean_returns[t]*100:.3f}%" for t in tickers],
        "σ (daily)":     [f"{np.sqrt(cov_matrix.loc[t,t])*100:.3f}%" for t in tickers],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── Correlation heatmap ───────────────────────────────────────────────────
st.markdown('<span class="section-label">Historical Correlation Matrix</span>', unsafe_allow_html=True)

corr = returns.corr()
fig3, ax3 = plt.subplots(figsize=(min(7, n_stocks*1.3 + 1), min(5, n_stocks*0.9 + 1)))
fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG)

im = ax3.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_xticks(range(len(tickers))); ax3.set_yticks(range(len(tickers)))
ax3.set_xticklabels(selected_names, rotation=40, ha="right", color="#4a5568", fontsize=8)
ax3.set_yticklabels(selected_names, color="#4a5568", fontsize=8)
ax3.set_title("Asset Correlation (Historical)", color="#cbd5e0")
for i in range(len(tickers)):
    for j in range(len(tickers)):
        ax3.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                 color="black", fontsize=8, fontweight="bold")

plt.tight_layout()
col_hm, _ = st.columns([1, 1])
with col_hm:
    st.pyplot(fig3); plt.close()

