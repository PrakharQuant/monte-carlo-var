"""
app.py — Home page and entry point for HuggingFace Spaces.
HF Spaces requires app_file: app.py in README frontmatter.
Streamlit multipage routing works identically on Spaces.
"""

import streamlit as st

st.set_page_config(
    page_title="Monte Carlo VaR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS (loaded once on home, inherited by all pages via st.markdown) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif; text-align: justify;
}
/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #111128 55%, #0c2a4a 100%);
    padding: 2.8rem 2.5rem;
    border-radius: 14px;
    border: 1px solid #e9456044;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #e94560;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #8899aa;
    font-size: 1rem;
    font-weight: 300;
    max-width: 680px;
    line-height: 1.6;
    margin: 0 0 1.2rem 0;
}
/* ── Badges ── */
.badge {
    display: inline-block;
    background: #e9456014;
    color: #e94560;
    border: 1px solid #e9456040;
    border-radius: 4px;
    padding: 3px 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin: 2px 2px;
}
/* ── Page cards ── */
.card {
    background: #111128;
    border: 1px solid #1e1e38;
    border-radius: 12px;
    padding: 1.6rem 1.5rem;
    height: 100%;
    transition: border-color 0.25s ease, transform 0.15s ease;
}
.card:hover {
    border-color: #e94560;
    transform: translateY(-2px);
}
.card-icon  { font-size: 1.6rem; margin-bottom: 0.6rem; }
.card-title {
    font-family: 'IBM Plex Mono', monospace;
    color: #e2e8f0;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.card-body  { color: #718096; font-size: 0.87rem; line-height: 1.65; }
/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    color: #4a5568;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 2rem 0 0.8rem 0;
    display: block;
}
/* ── Info panel ── */
.info-panel {
    background: #0c1a2e;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    color: #7fb3d3;
    font-size: 0.88rem;
    line-height: 1.7;
}
.info-panel b { color: #a9cce3; }
/* ── Metric cards (used across pages) ── */
.metric-card {
    background: #111128;
    border: 1px solid #1e1e38;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem;
    font-weight: 600;
    color: #e94560;
    line-height: 1.2;
}
.metric-label {
    color: #4a5568;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 5px;
}
/* ── Math / callout boxes ── */
.math-box {
    background: #111128;
    border-left: 3px solid #e94560;
    padding: 1rem 1.4rem;
    border-radius: 0 8px 8px 0;
    font-family: 'IBM Plex Mono', monospace;
    color: #cbd5e0;
    font-size: 0.86rem;
    line-height: 1.7;
    margin: 0.8rem 0;
}
.insight {
    background: #0c2240;
    border: 1px solid #1a4060;
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    color: #90b8d8;
    font-size: 0.86rem;
    line-height: 1.65;
    margin-top: 0.6rem;
}
.callout {
    background: #111128;
    border-left: 3px solid #e94560;
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    color: #90a0b8;
    font-size: 0.87rem;
    line-height: 1.7;
}
/* ── Footer ── */
.footer {
    color: #2d3748;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1a1a2e;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Monte Carlo VaR</p>
    <p class="hero-sub">
        An interactive risk analytics platform for Indian equity portfolios.
        Explore the mathematics of uncertainty, simulate thousands of market
        scenarios, and quantify portfolio risk using industry-standard techniques.
    </p>
    <span class="badge">Python</span>
    <span class="badge">Streamlit</span>
    <span class="badge">NumPy</span>
    <span class="badge">yfinance</span>
    <span class="badge">NSE Equities</span>
    <span class="badge">Monte Carlo</span>
    <span class="badge">VaR / CVaR</span>
    <span class="badge">Stress Testing</span>
</div>
""", unsafe_allow_html=True)


# ── Page cards ────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">App Pages</span>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📐</div>
        <div class="card-title">01 — Math Concepts</div>
        <div class="card-body">
            Five interactive modules covering the mathematical foundations:
            Normal distributions, covariance & correlation, Geometric Brownian Motion,
            the Multivariate Normal, and the Cholesky decomposition — all with
            live sliders and real-time plots.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📈</div>
        <div class="card-title">02 — Portfolio VaR</div>
        <div class="card-body">
            Pick from 12 major NSE stocks, set custom weights and investment size,
            then run up to 10,000 Monte Carlo simulation paths. View the simulation
            fan chart, ending value distribution, and a VaR summary table at
            90%, 95%, and 99% confidence levels.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">⚠️</div>
        <div class="card-title">03 — Advanced Risk</div>
        <div class="card-body">
            Go beyond VaR: Conditional VaR (Expected Shortfall) as required by
            Basel III, Historical VaR vs Monte Carlo VaR divergence to measure
            model risk, bear-market stress testing with an adjustable return shock,
            and rolling 60-day VaR to track risk evolution over time.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── What is VaR ──────────────────────────────────────────────────────────
st.markdown('<span class="section-label">What is Value at Risk?</span>', unsafe_allow_html=True)

col_text, col_info = st.columns([3, 2], gap="large")

with col_text:
    st.markdown("""
    **Value at Risk (VaR)** answers a fundamental question in risk management:

    > *"What is the maximum I can lose over a given time horizon, with a given level of confidence?"*

    A **95% VaR of ₹1,20,000 over 100 days** means there is only a 5% probability
    your portfolio loses more than ₹1.20 lakhs in the next 100 trading days.

    The **Monte Carlo approach** estimates this by simulating thousands of possible
    future return scenarios drawn from the historical joint distribution of your assets —
    rather than relying on a single closed-form formula. This makes it flexible:
    you can swap in fat-tailed distributions, apply stress scenarios, and compute
    any quantile of the resulting distribution.
    """)

with col_info:
    st.markdown("""
    <div class="info-panel">
        <b>Key Concepts Covered</b><br><br>
        • Multivariate Normal Distribution<br>
        • Covariance &amp; Correlation Matrices<br>
        • Geometric Brownian Motion<br>
        • Itô Correction (–σ²/2 term)<br>
        • Cholesky Decomposition<br>
        • Compounding &amp; Path Simulation<br>
        • VaR at Multiple Confidence Levels<br>
        • CVaR / Expected Shortfall<br>
        • Stress Testing<br>
        • Rolling Historical VaR
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by Prakhar &nbsp;·&nbsp; Monte Carlo VaR &nbsp;·&nbsp;
    NSE Indian Equities &nbsp;·&nbsp; Python + Streamlit
</div>
""", unsafe_allow_html=True)
