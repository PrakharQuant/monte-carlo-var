"""
pages/1_Math_Concepts.py
Streamlit multipage — Streamlit Community Cloud compatible.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Math Concepts", page_icon="📐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    text-align: justify;
}

/* Justify all paragraph text */
p, li, div.stMarkdown p {
    text-align: justify !important;
}

/* Allow tab labels to wrap onto two lines */
button[data-baseweb="tab"] {
    white-space: normal !important;
    word-break: break-word !important;
    text-align: center !important;
    line-height: 1.3 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    height: auto !important;
    min-height: 48px !important;
    max-width: 120px !important;
}

button[data-baseweb="tab"] p {
    text-align: center !important;
}
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
.step-box {
    background: #0f1a0f; border: 1px solid #1a3a1a;
    border-radius: 8px; padding: 0.9rem 1.2rem;
    color: #90c890; font-size: 0.86rem; line-height: 1.8; margin: 0.4rem 0;
}
.step-box b { color: #48bb78; }
</style>
""", unsafe_allow_html=True)

BG     = "#0a0a0f"
ACCENT = "#e94560"
BLUE   = "#4a9eda"
GREEN  = "#48bb78"
GREY   = "#1e1e38"
ORANGE = "#ed8936"

def dark_fig(figsize=(9, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#a0aec0")
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#e2e8f0")
    for sp in ax.spines.values():
        sp.set_edgecolor(GREY)
    return fig, ax

def style_axes(ax):
    """Apply dark theme to axes created outside dark_fig."""
    ax.set_facecolor(BG)
    ax.tick_params(colors="#a0aec0")
    ax.xaxis.label.set_color("#a0aec0")
    ax.yaxis.label.set_color("#a0aec0")
    ax.title.set_color("#e2e8f0")
    for sp in ax.spines.values():
        sp.set_edgecolor(GREY)


st.title("📐 Math Concepts")
st.caption("Interactive building blocks — adjust sliders to see the mathematics respond in real time.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Normal Distribution",
    "2 · Covariance & Correlation",
    "3 · Random Walk (GBM)",
    "4 · Multivariate Normal",
    "5 · Cholesky Decomposition",
    "6 · Monte Carlo Method",
])


# ==========================================================================
# TAB 1 — Normal Distribution
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
        ax.axvline(mu, color=GREEN, linewidth=1.2, linestyle=":", label="Mean (μ)")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Probability Density")
        ax.set_title("Normal Distribution of Daily Returns")
        ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Implementation note:** Each simulated day in the Monte Carlo loop draws from this distribution.
    Mean drives the long-run drift; standard deviation controls fan width. For 5 correlated assets
    this generalises to the multivariate normal — covered in Tab 4.
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

        cov12      = rho * s1 * s2
        port_var   = 0.25*s1**2 + 0.25*s2**2 + 2*0.5*0.5*cov12
        port_sig   = np.sqrt(port_var)
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
        Sigma2  = np.array([[s1**2, cov12], [cov12, s2**2]])
        samples = np.random.multivariate_normal([0, 0], Sigma2, 600)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        fig.patch.set_facecolor(BG)
        for ax in axes:
            style_axes(ax)

        axes[0].scatter(samples[:,0], samples[:,1], s=8, alpha=0.35, color=BLUE)
        axes[0].set_title(f"Correlated Returns (ρ = {rho:.2f})")
        axes[0].set_xlabel("Asset A return")
        axes[0].set_ylabel("Asset B return")

        hm = np.array([[s1**2, cov12], [cov12, s2**2]])
        axes[1].imshow(hm, cmap="RdBu_r", aspect="auto")
        axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Asset A", "Asset B"], color="#a0aec0", fontsize=9)
        axes[1].set_yticklabels(["Asset A", "Asset B"], color="#a0aec0", fontsize=9)
        axes[1].set_title("Covariance Matrix")
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f"{hm[i,j]:.5f}", ha="center", va="center",
                             color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""
    **Design note:** The covariance matrix encodes both individual volatility and inter-asset
    co-movement. In crisis periods, correlations between NSE large-caps historically spike
    toward +1 — meaning the simulation's calm-period covariance matrix will underestimate
    tail risk. This is a known and intentional limitation of the parametric Monte Carlo approach.
    """)


# ==========================================================================
# TAB 3 — Random Walk & GBM
# ==========================================================================
with tab3:
    st.markdown('<p class="concept-header">Concept 03 — Geometric Brownian Motion</p>', unsafe_allow_html=True)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        mu_a  = st.slider("Annual drift (μ)",      -0.20, 0.50, 0.12, 0.01, format="%.2f", key="t3_mu")
        sig_a = st.slider("Annual volatility (σ)",  0.05, 0.70, 0.25, 0.01, format="%.2f", key="t3_sig")
        n_p   = st.slider("Number of paths",        10, 300, 80, 10, key="t3_np")
        T_rw  = st.slider("Horizon (days)",         20, 252, 100, 10, key="t3_T")
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
        S0    = 100.0
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
    **Design note:** Each of the 10,000 paths in this project is one GBM trajectory for the portfolio.
    Rather than simulating individual stock prices, the model simulates the combined daily portfolio
    return from a multivariate normal draw, then compounds it via `cumprod`. The fan chart is the
    aggregate of all paths.
    """)


# ==========================================================================
# TAB 4 — Multivariate Normal
# ==========================================================================
with tab4:
    st.markdown('<p class="concept-header">Concept 04 — Multivariate Normal Distribution</p>', unsafe_allow_html=True)

    st.markdown("""
    The multivariate normal generalises the bell curve to **multiple correlated random variables
    drawn simultaneously.** In `np.random.multivariate_normal(mean_returns, cov_matrix, T)`, one
    call generates T rows — each row containing correlated daily returns for all 5 stocks at once.
    """)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        names    = ["RELIANCE", "TCS", "HDFC", "ICICI", "INFY"]
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
        vols   = np.array([0.015, 0.014, 0.013, 0.014, 0.016])
        corr_m = np.full((5, 5), 0.45); np.fill_diagonal(corr_m, 1.0)
        corr_m[0,3] = corr_m[3,0] = 0.60
        corr_m[1,4] = corr_m[4,1] = 0.65
        cov_demo = np.outer(vols, vols) * corr_m
        samples  = np.random.multivariate_normal(means_arr, cov_demo, 1000)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.patch.set_facecolor(BG)
        for ax in axes:
            style_axes(ax)
            ax.tick_params(labelsize=7)

        im = axes[0].imshow(corr_m, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5))
        axes[0].set_xticklabels(names, color="#a0aec0", fontsize=8)
        axes[0].set_yticklabels(names, color="#a0aec0", fontsize=8)
        axes[0].set_title("Correlation Structure")
        for i in range(5):
            for j in range(5):
                axes[0].text(j, i, f"{corr_m[i,j]:.2f}", ha="center", va="center",
                             color="black", fontsize=7, fontweight="bold")

        axes[1].scatter(samples[:,0], samples[:,1], s=5, alpha=0.25, color=BLUE)
        axes[1].set_title(f"RELIANCE vs TCS (ρ={corr_m[0,1]:.2f})")
        axes[1].set_xlabel("RELIANCE return")
        axes[1].set_ylabel("TCS return")

        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("""
    **Design note:** Drawing all 5 returns in a single `multivariate_normal` call ensures co-movement
    is preserved — if HDFC Bank has a bad simulated day, ICICI Bank is statistically more likely to
    as well, reflecting their historical correlation structure.
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

        rho_ch = st.slider("ρ Correlation", -0.95, 0.95, 0.65, 0.05, key="t5_rho")
        sa     = st.slider("σ_A", 0.005, 0.030, 0.015, 0.001, key="t5_sa")
        sb     = st.slider("σ_B", 0.005, 0.030, 0.018, 0.001, key="t5_sb")

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
            fig.patch.set_facecolor(BG)
            for ax in axes:
                style_axes(ax)

            axes[0].scatter(Z2[0], Z2[1], s=4, alpha=0.18, color="#718096")
            axes[0].set_title("Independent Z ~ N(0, I)")
            axes[0].set_xlabel("Z₁"); axes[0].set_ylabel("Z₂")
            axes[0].set_aspect("equal")

            axes[1].scatter(X2[0], X2[1], s=4, alpha=0.18, color=BLUE)
            axes[1].set_title(f"Correlated X = LZ  (ρ = {rho_ch:.2f})")
            axes[1].set_xlabel("X₁"); axes[1].set_ylabel("X₂")

            plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception:
            st.warning("Adjust parameters to see the plot.")

    st.markdown("""
    **Implementation note:** `np.random.multivariate_normal(mean_returns, cov_matrix, T)`
    performs this Cholesky transformation internally on every call inside the Monte Carlo loop —
    the correlated daily returns it produces are X = μ + LZ, never the raw independent Z directly.
    """)


# ==========================================================================
# TAB 6 — Monte Carlo Method
# ==========================================================================
with tab6:
    st.markdown('<p class="concept-header">Concept 06 — The Monte Carlo Method</p>', unsafe_allow_html=True)

    st.markdown("""
    Monte Carlo is a computational technique that estimates an unknown quantity by running
    a large number of random experiments and observing the distribution of outcomes.
    In finance, the unknown quantity is the future portfolio value — and each random experiment
    is one simulated sequence of daily returns over the chosen time horizon.
    """)

    # ── Step-by-step algorithm ──────────────────────────────────────────────
    st.markdown("#### The Algorithm — Step by Step")

    st.markdown("""
    <div class="step-box">
    <b>Step 1 — Estimate parameters from history</b><br>
    From 5 years of NSE price data, compute the mean daily return vector <b>μ</b> (5×1)
    and the covariance matrix <b>Σ</b> (5×5). These are the statistical fingerprints of
    the portfolio extracted from observed market behaviour.
    </div>

    <div class="step-box">
    <b>Step 2 — Draw one day of correlated returns</b><br>
    Sample a single row from the multivariate normal: <b>r</b> ~ N(<b>μ</b>, <b>Σ</b>).
    This gives one hypothetical daily return for each of the 5 stocks simultaneously,
    preserving their correlation structure via Cholesky decomposition internally.
    </div>

    <div class="step-box">
    <b>Step 3 — Compute the portfolio return for that day</b><br>
    R = <b>w</b>ᵀ <b>r</b> — a weighted dot product of the 5 individual stock returns.
    This collapses the 5-dimensional draw into a single portfolio-level return for the day.
    </div>

    <div class="step-box">
    <b>Step 4 — Repeat for T days and compound</b><br>
    Repeat Steps 2–3 for T consecutive days. Compound daily into a portfolio value path:
    V(t) = V₀ · ∏(1 + Rₜ). This is one complete simulation path — one possible future.
    </div>

    <div class="step-box">
    <b>Step 5 — Repeat M = 10,000 times independently</b><br>
    Run Steps 2–4 independently M times. Each run is a statistically independent future
    scenario. Together they form the empirical distribution of possible portfolio outcomes.
    </div>

    <div class="step-box">
    <b>Step 6 — Read off VaR from the distribution</b><br>
    Sort all 10,000 ending values. The 5th percentile is the value below which only 5%
    of scenarios fall. VaR = V₀ − 5th percentile. No closed-form formula is needed —
    the simulated distribution itself is the answer.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Live distribution build-up ──────────────────────────────────────────
    st.markdown("#### Watch the Distribution Form — Live")
    st.markdown("""
    Increase the number of simulations and observe the histogram converge toward a
    stable shape. The VaR estimate in the math box updates accordingly.
    """)

    col_ctrl, col_plot = st.columns([1, 2], gap="large")

    with col_ctrl:
        mc_n   = st.select_slider(
            "Number of simulations",
            options=[10, 50, 100, 500, 1000, 2000, 5000, 10000],
            value=1000, key="t6_n"
        )
        T_mc   = st.slider("Horizon (days)", 10, 252, 100, 10, key="t6_T")
        mu_mc  = st.slider("Daily mean return (μ)", -0.002, 0.003, 0.0008, 0.0001,
                           format="%.4f", key="t6_mu")
        sig_mc = st.slider("Daily volatility (σ)",  0.005, 0.030, 0.013, 0.001,
                           format="%.3f", key="t6_sig")
        seed_mc = st.number_input("Random seed", 0, 999, 42, key="t6_seed")
        V0 = 1_000_000

        np.random.seed(int(seed_mc))
        sim_rets    = np.random.normal(mu_mc, sig_mc, (T_mc, mc_n))
        ending_vals = V0 * np.prod(1 + sim_rets, axis=0)

        var_5   = np.percentile(ending_vals, 5)
        var_1   = np.percentile(ending_vals, 1)
        mean_v  = np.mean(ending_vals)
        median_v = np.median(ending_vals)
        var_amt = V0 - var_5

        st.markdown(f"""
        <div class="math-box">
        Simulations : {mc_n:,}<br>
        Horizon &nbsp;&nbsp;&nbsp;&nbsp;: {T_mc} days<br>
        μ (daily) &nbsp;&nbsp;: {mu_mc:.4f}<br>
        σ (daily) &nbsp;&nbsp;: {sig_mc:.3f}<br><br>
        Median &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: ₹{median_v/1e5:.2f}L<br>
        Mean &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: ₹{mean_v/1e5:.2f}L<br>
        5th pct &nbsp;&nbsp;&nbsp;&nbsp;: ₹{var_5/1e5:.2f}L<br>
        1st pct &nbsp;&nbsp;&nbsp;&nbsp;: ₹{var_1/1e5:.2f}L<br><br>
        VaR 95% &nbsp;&nbsp;&nbsp;&nbsp;: ₹{var_amt/1e5:.2f}L
        </div>
        <div class="insight">
        💡 With only 10–50 simulations the histogram is jagged
        and the VaR estimate jumps around. By 5,000+ simulations
        it stabilises — this is the Law of Large Numbers at work.
        </div>
        """, unsafe_allow_html=True)

    with col_plot:
        fig, ax = dark_fig((9, 5))
        n_bins = min(80, max(10, mc_n // 10))
        ax.hist(ending_vals, bins=n_bins, color=BLUE, edgecolor="none", alpha=0.72)
        ax.axvline(var_5,   color=ACCENT,  linewidth=2.0, linestyle="--",
                   label=f"VaR 95% → ₹{var_amt/1e5:.2f}L")
        ax.axvline(var_1,   color=ORANGE,  linewidth=1.6, linestyle="--",
                   label=f"VaR 99% → ₹{(V0-var_1)/1e5:.2f}L")
        ax.axvline(V0,      color=GREEN,   linewidth=1.6, linestyle=":",
                   label="Initial ₹10L")
        ax.axvline(mean_v,  color="white", linewidth=1.2, linestyle=":",
                   label=f"Mean ₹{mean_v/1e5:.2f}L")
        ax.set_title(f"Distribution of Portfolio Values After {T_mc} Days  ({mc_n:,} simulations)")
        ax.set_xlabel("Ending Portfolio Value (₹)")
        ax.set_ylabel("Frequency")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L"))
        ax.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── VaR convergence ─────────────────────────────────────────────────────
    st.markdown("#### VaR Estimate Convergence")
    st.markdown("""
    This tracks how the 95% VaR estimate evolves as more simulation paths are added.
    It stabilises well before 10,000 — demonstrating why that is the standard
    minimum threshold in practice.
    """)

    np.random.seed(0)
    all_rets    = np.random.normal(mu_mc, sig_mc, (T_mc, 10000))
    all_endings = V0 * np.prod(1 + all_rets, axis=0)
    checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    var_conv    = [V0 - np.percentile(all_endings[:n], 5) for n in checkpoints]

    fig2, ax2 = dark_fig((10, 3.5))
    ax2.plot(checkpoints, [v/1e5 for v in var_conv],
             color=ACCENT, linewidth=2, marker="o", markersize=5, markerfacecolor=BG)
    ax2.axhline(var_conv[-1]/1e5, color=GREEN, linewidth=1.2, linestyle="--",
                label=f"Converged VaR ≈ ₹{var_conv[-1]/1e5:.2f}L")
    ax2.set_xscale("log")
    ax2.set_title("VaR Estimate Convergence vs Number of Simulations")
    ax2.set_xlabel("Number of Simulations (log scale)")
    ax2.set_ylabel("95% VaR (₹ Lakhs)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.2f}L"))
    ax2.legend(facecolor=GREY, edgecolor="none", labelcolor="#cbd5e0", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("""
    **Design note:** 10,000 simulations is the default in this project — it balances
    statistical stability against compute time on Streamlit Community Cloud.
    CVaR and 99% VaR depend on a smaller tail sample (roughly 100 and 500 paths
    respectively at 10,000 simulations), so higher counts improve their accuracy
    more than they improve the 95% VaR estimate.
    """)
