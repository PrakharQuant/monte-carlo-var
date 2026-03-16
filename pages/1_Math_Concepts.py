"""
pages/1_Math_Concepts.py
Streamlit multipage — HuggingFace Spaces compatible.
"""

import sys, pathlib
# Make `src/` importable regardless of working directory on HF Spaces
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Math Concepts", page_icon="📐", layout="wide")

# Re-inject shared CSS (each page must do this — HF Spaces doesn't persist CSS across pages)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.concept-header {
    font-family: 'IBM Plex Mono', monospace; color: #e94560;
    font-size: 0.68rem; letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.math-box {
    background: #111128; border-left: 3px solid #e94560;
    padding: 1rem 1.4rem; border-radius: 0 8px 8px 0;
    font-family: 'IBM Plex Mono', monospace; color: #cbd5e0;
    font-size: 0.86rem; line-height: 1.7; margin: 0.8rem 0;
}
.insight {
    background: #0c2240; border: 1px solid #1a4060;
    border-radius: 8px; padding: 0.8rem 1.1rem;
    color: #90b8d8; font-size: 0.86rem; line-height: 1.65; margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plot theme ────────────────────────────────────────────────────────────
BG     = "#0a0a0f"
ACCENT = "#e94560"
BLUE   = "#4a9eda"
GREEN  = "#48bb78"
GREY   = "#1e1e38"

def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#4a5568")
    ax.xaxis.label.set_color("#4a5568")
    ax.yaxis.label.set_color("#4a5568")
    ax.title.set_color("#cbd5e0")
    for sp in ax.spines.values():
        sp.set_edgecolor(GREY)
    return fig, ax


# ── Header ────────────────────────────────────────────────────────────────
st.title("📐 Math Concepts")
st.caption("Interactive building blocks — adjust sliders to see the mathematics move in real time.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Normal Distribution",
    "2 · Covariance & Correlation",
    "3 · Random Walk (GBM)",
    "4 · Multivariate Normal",
    "5 · Cholesky Decomposition",
])


# ==========================================================================
# TAB 1 — Normal Distribution & VaR
# ==========================================================================
with tab1:
    st.markdown('<p class="concept-header">Concept 01 — The Normal Distribution</p>', unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        mu    = st.slider("Mean (μ) — avg daily return", -0.020, 0.020, 0.001, 0.001, format="%.3f", key="t1_mu")
        sigma = st.slider("Std Dev (σ) — daily volatility", 0.005, 0.050, 0.015, 0.001, format="%.3f", key="t1_s")
        conf  = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="t1_c")

        alpha   = 1.0 - conf
        var_ret = stats.norm.ppf(alpha, mu, sigma)

        st.markdown(f"""
        <div class="math-box">
        μ &nbsp;= {mu:.4f} &nbsp;({mu*100:.2f}% / day)<br>
        σ &nbsp;= {sigma:.4f} ({sigma*100:.2f}% / day)<br>
        VaR return = {var_ret:.4f}<br>
        ({conf*100:.0f}% CI → {alpha*100:.0f}th percentile)
        </div>
        <div class="insight">
        💡 At {conf*100:.0f}% confidence there is only a {alpha*100:.0f}% chance
        the daily return falls below <b>{var_ret*100:.2f}%</b>.
        Multiply by portfolio value to get the ₹ VaR.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        fig, ax = dark_fig((9, 4.2))
        x = np.linspace(mu - 4.2*sigma, mu + 4.2*sigma, 500)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, color=BLUE, linewidth=2)
        ax.fill_between(x, y, where=(x <= var_ret), color=ACCENT, alpha=0.45,
                        label=f"Loss tail ({alpha*100:.0f}%)")
        ax.fill_between(x, y, where=(x >  var_ret), color=BLUE,  alpha=0.12)
        ax.axvline(var_ret, color=ACCENT, linewidth=1.8, linestyle="--",
                   label=f"VaR @ {conf*100:.0f}%")
        ax.axvline(mu,      color=GREEN,  linewidth=1.2, linestyle=":",
                   label="Mean (μ)")

        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Probability Density")
        ax.set_title("Normal Distribution of Daily Returns")
        ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Why this matters for Monte Carlo:** Each simulated day draws a return from this distribution.
    The mean drives the long-run direction; the standard deviation determines how wide the simulation
    fan spreads. For 5 correlated assets we need the multivariate generalisation — covered in Tab 4.
    """)


# ==========================================================================
# TAB 2 — Covariance & Correlation
# ==========================================================================
with tab2:
    st.markdown('<p class="concept-header">Concept 02 — Covariance & Correlation</p>', unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        s1  = st.slider("σ₁ Asset A volatility", 0.005, 0.040, 0.015, 0.001, key="t2_s1")
        s2  = st.slider("σ₂ Asset B volatility", 0.005, 0.040, 0.020, 0.001, key="t2_s2")
        rho = st.slider("ρ  Correlation", -1.0, 1.0, 0.45, 0.05, key="t2_rho")

        cov12 = rho * s1 * s2
        port_var = 0.25*s1**2 + 0.25*s2**2 + 2*0.5*0.5*cov12
        port_sig = np.sqrt(port_var)
        uncorr_sig = np.sqrt(0.25*s1**2 + 0.25*s2**2)

        st.markdown(f"""
        <div class="math-box">
        Σ = [ σ₁²&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ρσ₁σ₂ ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;[ ρσ₁σ₂  σ₂²&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ]<br><br>
        Σ = [ {s1**2:.5f}  {cov12:.5f} ]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;[ {cov12:.5f}  {s2**2:.5f} ]
        </div>
        <div class="insight">
        Equal-weight portfolio σ = {port_sig*100:.3f}%/day<br>
        Uncorrelated baseline &nbsp;&nbsp;= {uncorr_sig*100:.3f}%/day<br>
        Diversification gain &nbsp;&nbsp;&nbsp;= {(uncorr_sig - port_sig)*100:.3f}%<br><br>
        💡 At ρ = -1, portfolio variance → 0 (perfect hedge).
        At ρ = +1, no diversification benefit exists.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        Sigma2 = np.array([[s1**2, cov12], [cov12, s2**2]])
        samples = np.random.multivariate_normal([0, 0], Sigma2, 600)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax in axes:
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(BG)
            ax.tick_params(colors="#4a5568")
            for sp in ax.spines.values():
                sp.set_edgecolor(GREY)

        axes[0].scatter(samples[:,0], samples[:,1], s=8, alpha=0.35, color=BLUE)
        axes[0].set_title(f"Correlated Returns (ρ = {rho:.2f})", color="#cbd5e0")
        axes[0].set_xlabel("Asset A return", color="#4a5568")
        axes[0].set_ylabel("Asset B return", color="#4a5568")

        hm = np.array([[s1**2, cov12], [cov12, s2**2]])
        axes[1].imshow(hm, cmap="RdBu_r", aspect="auto")
        axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
        axes[1].set_xticklabels(["Asset A","Asset B"], color="#4a5568", fontsize=9)
        axes[1].set_yticklabels(["Asset A","Asset B"], color="#4a5568", fontsize=9)
        axes[1].set_title("Covariance Matrix", color="#cbd5e0")
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{hm[i,j]:.5f}", ha="center", va="center",
                             color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Key insight:** The covariance matrix is the heart of multi-asset simulation.
    When markets crash, correlations between stocks typically spike toward +1,
    which means Monte Carlo VaR built on calm-period correlations can
    underestimate tail risk during stress events — a fundamental model limitation.
    """)


# ==========================================================================
# TAB 3 — Random Walk & GBM
# ==========================================================================
with tab3:
    st.markdown('<p class="concept-header">Concept 03 — Geometric Brownian Motion</p>', unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        mu_a  = st.slider("Annual drift (μ)",       -0.20, 0.50, 0.12, 0.01, format="%.2f", key="t3_mu")
        sig_a = st.slider("Annual volatility (σ)",   0.05, 0.70, 0.25, 0.01, format="%.2f", key="t3_sig")
        n_p   = st.slider("Number of paths",         10, 300, 80, 10, key="t3_np")
        T_rw  = st.slider("Horizon (days)",          20, 252, 100, 10, key="t3_T")
        seed  = st.number_input("Random seed", 0, 999, 42, key="t3_seed")

        mu_d  = mu_a  / 252.0
        sig_d = sig_a / np.sqrt(252.0)
        ito   = -0.5 * sig_d**2

        st.markdown(f"""
        <div class="math-box">
        GBM: S(t) = S₀ exp((μ–σ²/2)t + σW(t))<br><br>
        Daily μ &nbsp;&nbsp;= {mu_d:.5f}<br>
        Daily σ &nbsp;&nbsp;= {sig_d:.5f}<br>
        Itô term&nbsp; = {ito:.6f}<br><br>
        E[S(T)] = S₀ · exp(μ·T)
        </div>
        <div class="insight">
        💡 The –σ²/2 Itô correction prevents an upward bias.
        Without it, arithmetic compounding overestimates the
        expected price because log-normal means exceed medians.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        np.random.seed(int(seed))
        S0 = 100.0
        paths = np.zeros((T_rw + 1, n_p))
        paths[0] = S0
        for t in range(1, T_rw + 1):
            Z = np.random.standard_normal(n_p)
            paths[t] = paths[t-1] * np.exp((mu_d + ito) + sig_d * Z)

        fig, ax = dark_fig((9, 4.5))
        ax.plot(paths, color=BLUE, alpha=0.12, linewidth=0.7)
        ax.plot(paths[:, 0], color=ACCENT, linewidth=1.6, label="Sample path")
        ax.plot(np.median(paths, axis=1), color="white", linewidth=1.5,
                linestyle=":", label="Median path")
        ax.axhline(S0, color=GREEN, linestyle="--", linewidth=1, label="S₀ = 100")

        ax.set_title(f"Geometric Brownian Motion — {n_p} Paths")
        ax.set_xlabel("Days"); ax.set_ylabel("Price")
        ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Connection to your VaR project:** Each simulation path in the Monte Carlo model is one GBM
    trajectory for the portfolio. Instead of simulating individual stock prices, we simulate the
    portfolio's combined daily return from a multivariate normal — then compound it via `cumprod`.
    The fan chart you produced is exactly what you see here, aggregated across 10,000 paths.
    """)


# ==========================================================================
# TAB 4 — Multivariate Normal
# ==========================================================================
with tab4:
    st.markdown('<p class="concept-header">Concept 04 — Multivariate Normal Distribution</p>', unsafe_allow_html=True)

    st.markdown("""
    The multivariate normal generalises the bell curve to **multiple correlated random variables
    drawn simultaneously.** In `np.random.multivariate_normal(mean_returns, cov_matrix, T)`, one
    call generates T rows where each row contains correlated daily returns for all 5 stocks at once.
    """)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        names = ["RELIANCE", "TCS", "HDFC", "ICICI", "INFY"]
        defaults = [0.08, 0.06, 0.05, 0.07, 0.06]
        means_pct = [
            st.slider(nm, -0.5, 0.5, d, 0.01, format="%.2f%%", key=f"t4_{nm}")
            for nm, d in zip(names, defaults)
        ]
        means_arr = np.array(means_pct) / 100.0

        st.markdown(f"""
        <div class="math-box">
        X ~ N(μ, Σ)<br><br>
        μ = [<br>
        {"<br>".join([f"&nbsp;&nbsp;{m:.4f} ({m*100:.2f}%/day)" for m in means_arr])}
        <br>]<br><br>
        Σ = 5×5 covariance matrix<br>
        (estimated from historical data)
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        vols = np.array([0.015, 0.014, 0.013, 0.014, 0.016])
        base_rho = 0.45
        corr_m = np.full((5,5), base_rho); np.fill_diagonal(corr_m, 1.0)
        corr_m[0,3] = corr_m[3,0] = 0.60
        corr_m[1,4] = corr_m[4,1] = 0.65
        cov_demo = np.outer(vols, vols) * corr_m

        samples = np.random.multivariate_normal(means_arr, cov_demo, 1000)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax in axes:
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            ax.tick_params(colors="#4a5568", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(GREY)

        im = axes[0].imshow(corr_m, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5))
        axes[0].set_xticklabels(names, color="#4a5568", fontsize=8)
        axes[0].set_yticklabels(names, color="#4a5568", fontsize=8)
        axes[0].set_title("Correlation Structure", color="#cbd5e0")
        for i in range(5):
            for j in range(5):
                axes[0].text(j, i, f"{corr_m[i,j]:.2f}", ha="center", va="center",
                             color="black", fontsize=7, fontweight="bold")

        axes[1].scatter(samples[:,0], samples[:,1], s=5, alpha=0.25, color=BLUE)
        axes[1].set_title(f"RELIANCE vs TCS (ρ={corr_m[0,1]:.2f})", color="#cbd5e0")
        axes[1].set_xlabel("RELIANCE return", color="#4a5568")
        axes[1].set_ylabel("TCS return", color="#4a5568")

        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("""
    **Why this matters:** Because all 5 returns are drawn together in a single call,
    if HDFC Bank has a bad day in the simulation, ICICI Bank is also more likely to have
    a bad day — reflecting their historical co-movement. The simulation captures correlation,
    not just individual volatility.
    """)


# ==========================================================================
# TAB 5 — Cholesky Decomposition
# ==========================================================================
with tab5:
    st.markdown('<p class="concept-header">Concept 05 — Cholesky Decomposition</p>', unsafe_allow_html=True)

    st.markdown("""
    `np.random.multivariate_normal` internally uses Cholesky decomposition to convert
    independent standard normals into correlated ones. Understanding it demystifies the
    simulation engine completely.
    """)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("""
        **The algorithm:**

        1. Start with **Z** ~ N(0, I) — independent
        2. Decompose **Σ = LLᵀ** (Cholesky)
        3. Set **X = μ + LZ**

        Then Cov(**X**) = **L** · Cov(**Z**) · **Lᵀ** = **LILᵀ = Σ** ✓
        """)

        rho_ch = st.slider("ρ Correlation",  -0.95, 0.95, 0.65, 0.05, key="t5_rho")
        sa     = st.slider("σ_A",             0.005, 0.030, 0.015, 0.001, key="t5_sa")
        sb     = st.slider("σ_B",             0.005, 0.030, 0.018, 0.001, key="t5_sb")

        Sc = np.array([[sa**2, rho_ch*sa*sb], [rho_ch*sa*sb, sb**2]])

        try:
            L = np.linalg.cholesky(Sc)
            st.markdown(f"""
            <div class="math-box">
            Σ = LLᵀ where L =<br><br>
            [ {L[0,0]:.5f} &nbsp; 0.00000 ]<br>
            [ {L[1,0]:.5f} &nbsp; {L[1,1]:.5f} ]<br><br>
            Verify:<br>
            L[0,0]²&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {L[0,0]**2:.6f} = σ_A² ✓<br>
            L[1,0]·L[0,0] = {L[1,0]*L[0,0]:.6f} = ρσ_Aσ_B ✓
            </div>
            """, unsafe_allow_html=True)
        except np.linalg.LinAlgError:
            st.error("Matrix not positive definite — adjust parameters.")

    with col_plot:
        try:
            L  = np.linalg.cholesky(Sc)
            Z2 = np.random.default_rng(42).standard_normal((2, 2500))
            X2 = L @ Z2

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            for ax in axes:
                fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
                ax.tick_params(colors="#4a5568")
                for sp in ax.spines.values(): sp.set_edgecolor(GREY)

            axes[0].scatter(Z2[0], Z2[1], s=4, alpha=0.18, color="#718096")
            axes[0].set_title("Independent Z ~ N(0, I)", color="#cbd5e0")
            axes[0].set_xlabel("Z₁", color="#4a5568"); axes[0].set_ylabel("Z₂", color="#4a5568")
            axes[0].set_aspect("equal")

            axes[1].scatter(X2[0], X2[1], s=4, alpha=0.18, color=BLUE)
            axes[1].set_title(f"Correlated X = LZ  (ρ = {rho_ch:.2f})", color="#cbd5e0")
            axes[1].set_xlabel("X₁", color="#4a5568"); axes[1].set_ylabel("X₂", color="#4a5568")

            plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception:
            st.warning("Adjust parameters to see the plot.")

    st.markdown("""
    **Connection to your project:** When NumPy calls `multivariate_normal(mean_returns, cov_matrix, T)`,
    it factorises the covariance matrix via Cholesky internally, generates T rows of independent normals,
    and transforms them. You don't see this step — but it is happening every single call inside
    your 10,000-iteration Monte Carlo loop.
    """)

