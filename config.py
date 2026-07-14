import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-dbm-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Visible unit construction ─────────────────────────────────────────────────
# Visible vector v for each ETF at each time t:
#   [lag_returns(N_LAGS),  macro_levels(normalised),  vol_proxy, momentum, skew, kurt]
N_LAGS = 21          # lagged return features

# ── DBM architecture ──────────────────────────────────────────────────────────
# Layer sizes: visible → hidden1 → hidden2
# Visible dim = N_LAGS + n_macro_cols + 4 (vol, momentum, skew, kurt)
# Set dynamically in engine based on available macro cols
DBM_H1 = 64          # first hidden layer size
DBM_H2 = 32          # second hidden layer size

# ── Training (Contrastive Divergence) ────────────────────────────────────────
DBM_CD_K    = 1      # CD-k steps (CD-1 is standard)
DBM_EPOCHS  = 80
DBM_LR      = 0.01
DBM_BATCH   = 64
DBM_MOMENTUM = 0.9   # momentum for weight updates
DBM_L2      = 1e-4   # L2 weight decay

# ── Free energy score ─────────────────────────────────────────────────────────
# Signal = free energy of current visible state under learned DBM
# F(v) = -bv·v - Σ_j log(1 + exp(W1_j·v + bh1_j)) [RBM approx for DBM]
# Low free energy → in-distribution (familiar regime) → positive signal
# High free energy → anomalous / out-of-distribution → negative signal
#
# We use the TWO-LAYER free energy approximation (mean-field):
#   F(v) ≈ -bv·v - Σ_j softplus(W1_j·v + bh1_j + W2_j·μh2 + bh2_j)
# where μh2 is the mean-field fixed point for h2 given h1.

# Score = -(free_energy - rolling_mean_FE) / rolling_std_FE
# Negative FE anomaly → regime shift → avoid
# Positive FE anomaly → oversold vs model → buy
FE_LOOKBACK = 63     # bars for rolling FE normalisation

TOP_N = 3
