import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Deep Boltzmann Machine Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a0a00; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #b45309; }
.etf-card    { background:linear-gradient(135deg,#1a0a00 0%,#92400e 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#0f0500 0%,#78350f 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Deep Boltzmann Machine Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Salakhutdinov & Hinton (2009) · '
    'Two-layer binary latent DBM · Contrastive Divergence CD-1 · '
    'Mean-field free energy scoring · Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## ⚡ DBM Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Architecture:** v={config.N_LAGS}+macro+4 → h1={config.DBM_H1} → h2={config.DBM_H2}")
st.sidebar.markdown(
    f"**Training:** CD-{config.DBM_CD_K} · {config.DBM_EPOCHS} epochs · "
    f"lr={config.DBM_LR} · momentum={config.DBM_MOMENTUM}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "dbm_engine_2")
tab2_path = find_latest(files, "dbm_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — DBM Free Energy Signal")

    with st.expander("📖 Deep Boltzmann Machine Methodology", expanded=True):
        st.markdown("""
A **Deep Boltzmann Machine** defines a joint distribution over visible features
v and two layers of binary latent variables h¹, h²:

```
P(v, h¹, h²) ∝ exp(v^T W¹ h¹  +  h¹^T W² h²  +  bv·v  +  bh1·h¹  +  bh2·h²)
```

**Training:** Contrastive Divergence CD-1 with mean-field positive phase refinement:
1. Clamp v to data → sample h¹ (given v, h²=0) → sample h² → refine h¹ (given v, h²)
2. Run 1 Gibbs step: h² → h¹ → v̂ → h¹ → h²
3. Update: ΔW ∝ E_data[v h¹ᵀ] − E_model[v̂ h¹ᵀ]

**Free energy scoring (mean-field approximation):**
```
F(v) = −bv·v  −  Σⱼ log(1 + exp(W¹ⱼ·v  +  Ãⱼ  +  bh1ⱼ))
```
Where Ã = μh2 @ W2ᵀ is the mean-field h2 contribution.

**Signal:** score = −(F(v_today) − μ_FE) / σ_FE

| Free energy vs rolling baseline | Interpretation | Signal |
|--------------------------------|----------------|--------|
| Much higher than usual | Anomalous / out-of-distribution | Negative |
| Near baseline | Familiar regime | Neutral |
| Slightly below baseline | Well-modelled, potentially oversold | Positive |

**Distinct from VAE (in suite):** DBM uses binary latents + energy-based joint
distribution (no encoder/decoder). The free energy is a natural anomaly score
without reconstruction loss.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">DBM score = {etf['dbm_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "DBM Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("DBM Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Salakhutdinov & Hinton (2009) DBM · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore DBM Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — short-term regime model; DBM captures recent correlation structure
- **126d** — 6-month model; balances regime stationarity with recency
- **252d** — 1-year model; most stable latent structure; recommended for FI
- **504d** — 2-year model; captures full rate cycle in latent space
        """)

    st.markdown(f"### DBM Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">DBM score = {etf['dbm_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "DBM Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
