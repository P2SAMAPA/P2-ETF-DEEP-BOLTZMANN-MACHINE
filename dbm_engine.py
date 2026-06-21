"""
dbm_engine.py — Deep Boltzmann Machine Engine
==============================================

Theory
------
A **Deep Boltzmann Machine (DBM)** is an energy-based generative model with
multiple layers of latent binary variables. Unlike a VAE (which uses an
encoder-decoder with continuous latents) or a normalising flow (which requires
invertibility), the DBM defines a joint distribution over visible and hidden
units via an energy function.

**Joint distribution:**

    P(v, h¹, h²) = (1/Z) · exp(−E(v, h¹, h²))

**Energy function (2-layer DBM):**

    E(v, h¹, h²) = −v^T W¹ h¹  −  h¹^T W² h²
                   −b_v^T v  −  b_{h1}^T h¹  −  b_{h2}^T h²

Where:
    v   ∈ {0,1}^{n_v}   — visible units (binarised ETF features)
    h¹  ∈ {0,1}^{n_h1}  — first hidden layer
    h²  ∈ {0,1}^{n_h2}  — second hidden layer
    W¹  ∈ ℝ^{n_v × n_h1}  — visible-to-h1 weights
    W²  ∈ ℝ^{n_h1 × n_h2} — h1-to-h2 weights

**Conditional distributions (all sigmoid):**

    P(h¹_j=1 | v, h²) = σ(W¹_j · v  +  W²_j · h²  +  b_{h1,j})
    P(v_i=1  | h¹)    = σ(W¹_i^T · h¹  +  b_{v,i})
    P(h²_k=1 | h¹)    = σ(W²_k^T · h¹  +  b_{h2,k})

**Training: Contrastive Divergence (CD-k)**

CD-k approximates the gradient of log-likelihood by running k steps of
alternating Gibbs sampling from the current data point:

    1. Clamp v to data, sample h¹, h², ... (positive phase)
    2. Run k Gibbs steps: h² → h¹ → v → h¹ → h² (negative phase)
    3. ΔW¹ ∝ E_data[v h¹^T] − E_model[v h¹^T]
    4. ΔW² ∝ E_data[h¹ h²^T] − E_model[h¹ h²^T]

**Key distinction from VAE and ESN (in your suite):**
    - VAE: continuous latents, ELBO lower bound, encoder-decoder
    - DBM: binary latents, energy-based, multi-layer joint distribution
    - ESN: reservoir computing, fixed random weights, no generative model

**Free Energy Score**

After training, the DBM assigns a **free energy** F(v) to any visible vector:

    F(v) = −b_v · v  −  Σ_j softplus(W¹_j · v  +  Ã_j  +  b_{h1,j})

Where Ã_j is the mean-field contribution from h²:

    Ã_j = Σ_k W²_{jk} · μ_{h2,k}

and μ_{h2} is the mean-field fixed point: μ_{h2,k} = σ(W²_k · μ_{h1} + b_{h2,k})

**Signal construction:**

For each ETF per window:
1. Train DBM on rolling window of binarised feature vectors
2. Compute free energy F(v_t) for all t in window → get rolling FE series
3. Score = −(F(v_today) − μ_FE) / σ_FE

Interpretation:
    - Low FE (below rolling mean) → current state is in-distribution → neutral
    - Very high FE → anomalous state → model has never seen this → negative signal
    - Moderate negative FE anomaly → oversold vs. model prior → positive signal

References
----------
- Salakhutdinov, R. & Hinton, G. (2009). Deep Boltzmann Machines.
  AISTATS 2009.
- Hinton, G.E. (2002). Training products of experts by minimizing contrastive
  divergence. Neural Computation, 14(8), 1771–1800.
- Hinton, G.E., Osindero, S. & Teh, Y.W. (2006). A fast learning algorithm
  for deep belief nets. Neural Computation, 18(7), 1527–1554.
- Salakhutdinov, R. (2015). Learning deep generative models.
  Annual Review of Statistics, 2, 361–385.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -30, 30)))


def _sample_bernoulli(p: np.ndarray) -> np.ndarray:
    return (np.random.rand(*p.shape) < p).astype(np.float64)


def _binarise(X: np.ndarray, method: str = "median") -> np.ndarray:
    """
    Convert continuous features to binary {0,1}.
    method='median': 1 if above median across samples, 0 otherwise.
    method='sign':   1 if positive, 0 otherwise.
    """
    if method == "sign":
        return (X > 0).astype(np.float64)
    # Median per feature
    med = np.median(X, axis=0, keepdims=True)
    return (X > med).astype(np.float64)


# ── Feature construction ───────────────────────────────────────────────────────

def _build_visible_vectors(
    log_ret:    np.ndarray,   # (T,) log returns
    macro_norm: np.ndarray,   # (T, M) normalised macro signals
    window:     int,
) -> Tuple[np.ndarray, int]:
    """
    Build visible vector matrix V: (N_samples, n_v).

    Visible features at time t (using N_LAGS lags):
        [r_{t-1}, ..., r_{t-N_LAGS},          ← lagged returns
         macro_1_t, ..., macro_M_t,             ← macro levels
         |r_t| (vol proxy),                     ← realised vol proxy
         Σ r_{t-5:t} (5d momentum),             ← short momentum
         skew(r_{t-21:t}),                      ← rolling skewness
         kurt(r_{t-21:t})]                      ← rolling kurtosis
    """
    T       = len(log_ret)
    M       = macro_norm.shape[1]
    N_LAGS  = config.N_LAGS
    n_v     = N_LAGS + M + 4

    rows = []
    for t in range(max(N_LAGS, 21), T):
        lags     = log_ret[t-N_LAGS:t][::-1]         # most recent first
        mac      = macro_norm[t]
        vol      = abs(log_ret[t])
        mom5     = log_ret[max(0,t-5):t].sum()
        window21 = log_ret[max(0,t-21):t]
        skew     = float(pd.Series(window21).skew()) if len(window21) > 3 else 0.0
        kurt     = float(pd.Series(window21).kurt()) if len(window21) > 3 else 0.0
        # Clip extremes
        skew = np.clip(skew, -5, 5)
        kurt = np.clip(kurt, -5, 10)
        row  = np.concatenate([lags, mac, [vol, mom5, skew, kurt]])
        rows.append(row)

    if not rows:
        return np.empty((0, n_v)), n_v

    V = np.array(rows, dtype=np.float64)
    # NaN guard
    V = np.nan_to_num(V, nan=0.0, posinf=3.0, neginf=-3.0)
    return V, n_v


# ── Deep Boltzmann Machine ────────────────────────────────────────────────────

class DBM:
    """
    Two-layer Deep Boltzmann Machine.
    Units: visible v ∈ {0,1}^nv, hidden h1 ∈ {0,1}^nh1, h2 ∈ {0,1}^nh2.
    """
    def __init__(self, n_v: int, n_h1: int, n_h2: int,
                 rng: np.random.Generator):
        scale1 = np.sqrt(2.0 / (n_v + n_h1))
        scale2 = np.sqrt(2.0 / (n_h1 + n_h2))
        self.W1  = rng.normal(0, scale1, (n_v, n_h1))
        self.W2  = rng.normal(0, scale2, (n_h1, n_h2))
        self.bv  = np.zeros(n_v)
        self.bh1 = np.zeros(n_h1)
        self.bh2 = np.zeros(n_h2)
        # Momentum buffers
        self.vW1  = np.zeros_like(self.W1)
        self.vW2  = np.zeros_like(self.W2)
        self.vbv  = np.zeros_like(self.bv)
        self.vbh1 = np.zeros_like(self.bh1)
        self.vbh2 = np.zeros_like(self.bh2)

    # ── Conditionals ──────────────────────────────────────────────────────────

    def ph1_given_v_h2(self, V: np.ndarray,
                       H2: np.ndarray) -> np.ndarray:
        """P(h1=1 | v, h2): (B, nh1)"""
        return _sigmoid(V @ self.W1 + H2 @ self.W2.T + self.bh1)

    def pv_given_h1(self, H1: np.ndarray) -> np.ndarray:
        """P(v=1 | h1): (B, nv)"""
        return _sigmoid(H1 @ self.W1.T + self.bv)

    def ph2_given_h1(self, H1: np.ndarray) -> np.ndarray:
        """P(h2=1 | h1): (B, nh2)"""
        return _sigmoid(H1 @ self.W2 + self.bh2)

    # ── CD-k training step ────────────────────────────────────────────────────

    def cd_step(self, V_data: np.ndarray, k: int = 1,
                lr: float = 0.01, mom: float = 0.9,
                l2: float = 1e-4) -> float:
        """
        One CD-k update on a batch of visible vectors.
        V_data: (B, nv) — already binarised.
        Returns mean reconstruction error.
        """
        B = len(V_data)

        # ── Positive phase: data → h1 → h2 ──────────────────────────────────
        # DBM positive phase uses mean-field for h1 (conditioned on both v and h2=0 initially)
        ph1_pos = _sigmoid(V_data @ self.W1 + self.bh1)   # init h2=0
        h1_pos  = _sample_bernoulli(ph1_pos)
        ph2_pos = self.ph2_given_h1(h1_pos)
        h2_pos  = _sample_bernoulli(ph2_pos)
        # Refine h1 with h2 (one mean-field iteration)
        ph1_pos = self.ph1_given_v_h2(V_data, h2_pos)
        h1_pos  = _sample_bernoulli(ph1_pos)

        # Positive statistics
        pos_vh1 = V_data.T @ ph1_pos / B       # (nv, nh1)
        pos_h1h2 = h1_pos.T @ ph2_pos / B      # (nh1, nh2)

        # ── Negative phase: k steps of Gibbs ─────────────────────────────────
        v_neg  = V_data.copy()
        h1_neg = h1_pos.copy()
        h2_neg = h2_pos.copy()

        for _ in range(k):
            # Sample h1 from v, h2
            ph1_neg = self.ph1_given_v_h2(v_neg, h2_neg)
            h1_neg  = _sample_bernoulli(ph1_neg)
            # Sample h2 from h1
            ph2_neg = self.ph2_given_h1(h1_neg)
            h2_neg  = _sample_bernoulli(ph2_neg)
            # Sample v from h1
            pv_neg  = self.pv_given_h1(h1_neg)
            v_neg   = _sample_bernoulli(pv_neg)

        # Final mean-field for cleaner statistics
        ph1_neg2 = self.ph1_given_v_h2(v_neg, h2_neg)

        # Negative statistics
        neg_vh1  = v_neg.T @ ph1_neg2 / B
        neg_h1h2 = ph1_neg2.T @ ph2_neg / B

        # ── Weight updates with momentum + L2 ────────────────────────────────
        dW1  = pos_vh1  - neg_vh1
        dW2  = pos_h1h2 - neg_h1h2
        dbv  = (V_data - v_neg).mean(axis=0)
        dbh1 = (ph1_pos - ph1_neg2).mean(axis=0)
        dbh2 = (ph2_pos - ph2_neg).mean(axis=0)

        self.vW1  = mom * self.vW1  + lr * (dW1  - l2 * self.W1)
        self.vW2  = mom * self.vW2  + lr * (dW2  - l2 * self.W2)
        self.vbv  = mom * self.vbv  + lr * dbv
        self.vbh1 = mom * self.vbh1 + lr * dbh1
        self.vbh2 = mom * self.vbh2 + lr * dbh2

        self.W1  += self.vW1
        self.W2  += self.vW2
        self.bv  += self.vbv
        self.bh1 += self.vbh1
        self.bh2 += self.vbh2

        # Reconstruction error
        recon_err = float(np.mean((V_data - pv_neg) ** 2))
        return recon_err

    # ── Free energy ───────────────────────────────────────────────────────────

    def free_energy_batch(self, V: np.ndarray) -> np.ndarray:
        """
        Approximate free energy F(v) for a batch of visible vectors.
        V: (B, nv) — continuous or binary.

        Two-layer free energy (mean-field approximation):
            F(v) = -bv·v - Σ_j softplus(W1_j·v + Ã_j + bh1_j)

        Where Ã_j = Σ_k W2_{jk} · μh2_k is the mean-field h2 contribution.

        We compute μh2 via one mean-field iteration:
            μh1 = σ(v @ W1 + bh1)
            μh2 = σ(μh1 @ W2 + bh2)
            μh1 = σ(v @ W1 + μh2 @ W2.T + bh1)   (refine)
        """
        # Mean-field iterations
        mu_h1  = _sigmoid(V @ self.W1 + self.bh1)
        mu_h2  = _sigmoid(mu_h1 @ self.W2 + self.bh2)
        mu_h1  = _sigmoid(V @ self.W1 + mu_h2 @ self.W2.T + self.bh1)

        # Ã: mean-field h2 contribution to h1 pre-activation
        A_tilde = mu_h2 @ self.W2.T                     # (B, nh1)

        # Free energy
        fe = (
            -(V @ self.bv)                              # (B,)
            - _softplus(V @ self.W1 + A_tilde + self.bh1).sum(axis=1)  # (B,)
        )
        return fe


# ── Training wrapper ──────────────────────────────────────────────────────────

def _train_dbm(V_bin: np.ndarray, n_v: int,
               rng: np.random.Generator) -> DBM:
    """Train DBM with CD-1 on binarised visible data."""
    dbm    = DBM(n_v, config.DBM_H1, config.DBM_H2, rng)
    N      = len(V_bin)
    B      = min(config.DBM_BATCH, N)

    for epoch in range(config.DBM_EPOCHS):
        idx  = rng.permutation(N)
        errs = []
        for i in range(0, N, B):
            bi    = idx[i:i+B]
            if len(bi) < 2:
                continue
            err = dbm.cd_step(
                V_bin[bi], k=config.DBM_CD_K,
                lr=config.DBM_LR, mom=config.DBM_MOMENTUM,
                l2=config.DBM_L2,
            )
            errs.append(err)

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1}/{config.DBM_EPOCHS}  "
                  f"recon_err={np.mean(errs):.5f}")

    return dbm


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_dbm_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Train a DBM per ETF and return free-energy-based cross-sectional z-scores.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite DBM z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.N_LAGS + 21 + 10
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu       = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std      = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = macro_vals

    rng        = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        mac     = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        # Build visible vectors over the full history
        V_all, n_v = _build_visible_vectors(log_ret, mac, window)
        if len(V_all) < config.DBM_BATCH:
            print(f"    {ticker}: insufficient samples ({len(V_all)}), skipping")
            continue

        # Training window: last `window` visible vectors
        V_train = V_all[-window:] if len(V_all) > window else V_all

        # Normalise to [0,1] per feature then binarise
        V_norm  = (V_train - V_train.mean(axis=0, keepdims=True))
        V_norm /= (V_train.std(axis=0, keepdims=True) + 1e-8)
        V_bin   = _binarise(V_norm, method="median")

        print(f"    Training DBM for {ticker} "
              f"(N={len(V_bin)}, n_v={n_v}, h1={config.DBM_H1}, h2={config.DBM_H2})")

        try:
            dbm = _train_dbm(V_bin, n_v, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        # ── Free energy scoring ───────────────────────────────────────────────
        # Compute FE for all training samples to get rolling baseline
        V_cont = V_norm  # use continuous (not binary) for smoother FE
        fe_all = dbm.free_energy_batch(V_cont)    # (N,)

        if len(fe_all) < 5:
            continue

        # Rolling mean and std for z-scoring
        fe_mu  = fe_all[-config.FE_LOOKBACK:].mean()
        fe_std = fe_all[-config.FE_LOOKBACK:].std() + 1e-8

        # Score today = -(FE_today - mu) / std
        # Low FE today vs rolling → familiar regime → positive
        # High FE today vs rolling → anomalous → negative
        fe_today = float(fe_all[-1])
        score    = -(fe_today - fe_mu) / fe_std

        # Clip extremes
        score = float(np.clip(score, -5, 5))
        raw_scores[ticker] = score

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
