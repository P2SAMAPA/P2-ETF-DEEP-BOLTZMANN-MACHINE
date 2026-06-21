# ⚡ P2-ETF-DEEP-BOLTZMANN-MACHINE

**Deep Boltzmann Machine Engine — Salakhutdinov & Hinton (2009)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine trains a **Deep Boltzmann Machine (DBM)** on the joint distribution
of ETF return features and macro signals, then uses the learned model's
**free energy** as an anomaly score for ETF ranking.

ETFs with low free energy are "familiar" to the model — they are in a regime
the DBM has seen before. ETFs with anomalously high free energy are in an
out-of-distribution state — the model flags them as regime-shifted.

### Distinction from VAE (in suite)

| | VAE | **DBM (this engine)** |
|---|---|---|
| Latent type | Continuous Gaussian | **Binary {0,1}** |
| Architecture | Encoder + Decoder | **Energy-based joint distribution** |
| Training | ELBO lower bound | **Contrastive Divergence** |
| Anomaly score | Reconstruction loss | **Free energy F(v)** |
| Layers | 1 stochastic layer | **2 stochastic hidden layers** |
| Theory | Variational Bayes | **Boltzmann statistics** |

---

## Theory

### Energy Function

```
E(v, h¹, h²) = −v^T W¹ h¹  −  h¹^T W² h²
               −bv·v  −  bh1·h¹  −  bh2·h²
```

| Symbol | Dim | Description |
|--------|-----|-------------|
| v | nv | Visible units (binarised ETF features) |
| h¹ | 64 | First hidden layer (latent market microstructure) |
| h² | 32 | Second hidden layer (latent macro regime) |
| W¹ | nv×64 | Visible-to-h1 weights |
| W² | 64×32 | h1-to-h2 weights |

### Conditional Distributions

```
P(h¹ⱼ=1 | v, h²) = σ(W¹ⱼ · v  +  W²ⱼ · h²  +  bh1ⱼ)
P(vᵢ=1  | h¹)    = σ(W¹ᵢᵀ · h¹  +  bvᵢ)
P(h²ₖ=1 | h¹)    = σ(W²ₖᵀ · h¹  +  bh2ₖ)
```

### Training: Contrastive Divergence (CD-1)

```
Positive phase: v_data → h¹ (mean-field with h²) → h²
Negative phase: h² → h¹ → v̂ → h¹ → h²   (1 Gibbs step)

ΔW¹ ∝ E_data[v h¹ᵀ] − E_model[v̂ h¹ᵀ]
ΔW² ∝ E_data[h¹ h²ᵀ] − E_model[h¹ h²ᵀ]
```

With momentum (0.9) and L2 weight decay (1e-4).

### Free Energy Scoring

```
F(v) = −bv · v  −  Σⱼ softplus(W¹ⱼ·v  +  Ãⱼ  +  bh1ⱼ)
```

Where Ã = μh2 @ W2ᵀ (mean-field h2 contribution).

**Signal:**
```
score = −(F(v_today) − μ_FE) / σ_FE
```

| F(v_today) vs baseline | Regime | Signal |
|------------------------|--------|--------|
| Much higher | Anomalous, out-of-distribution | Negative |
| Near baseline | Familiar regime | Neutral |
| Below baseline | Well-modelled, oversold vs model | Positive |

---

## Visible Feature Vector

For each ETF at time t:

```
v_t = [r_{t-1}, ..., r_{t-21},    ← 21 lagged returns
       VIX_t, DXY_t, T10Y2Y_t,    ← macro levels (normalised)
       IG_SPREAD_t, HY_SPREAD_t,   ← credit signals
       |r_t|,                       ← vol proxy
       Σr_{t-5:t},                  ← 5d momentum
       skew(r_{t-21:t}),            ← rolling skewness
       kurt(r_{t-21:t})]            ← rolling kurtosis
```

Binarised per feature via median threshold before training.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-DEEP-BOLTZMANN-MACHINE/
├── config.py          # Universes, DBM architecture, CD hyperparameters
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── dbm_engine.py      # Core: DBM class, CD-1 training, free energy scoring
├── trainer.py         # Orchestrator: load → train → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (CD-1 is fast — no parallel needed)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DEEP-BOLTZMANN-MACHINE
cd P2-ETF-DEEP-BOLTZMANN-MACHINE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-dbm-results`

---

## References

- Salakhutdinov, R. & Hinton, G. (2009). Deep Boltzmann Machines.
  *AISTATS 2009*.
- Hinton, G.E. (2002). Training products of experts by minimizing contrastive
  divergence. *Neural Computation*, 14(8), 1771–1800.
- Hinton, G.E., Osindero, S. & Teh, Y.W. (2006). A fast learning algorithm
  for deep belief nets. *Neural Computation*, 18(7), 1527–1554.
- Salakhutdinov, R. (2015). Learning deep generative models.
  *Annual Review of Statistics*, 2, 361–385.
