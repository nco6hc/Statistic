"""
Targeted Improvements: M1+ · M2+ · M3+
========================================

M1+  (Hybrid LSTM+Markov)
  Two new feature blocks added to the 220-dim enriched feature set → 330-dim:
    ① 30-day frequency (55-dim) — separates long-term baseline from short-term burst
    ② Co-occurrence neighbour (55-dim) — 1-step GNN-style message passing baked in:
       cooc_nbr[t, s] = Σ_{j ∈ selected(t-1)} cumulative_cooc[s,j] / days_so_far
       (causal: only uses co-occurrence history strictly before day t)
  Same BiLSTM+Attention architecture and hyperparams as M1, retrained from scratch.

M2+  (TD-HBB)
  Per-student adaptive λ_s  (vs single global λ=0.99):
    variance_s = std of 14-day rolling selection rate on training data
    λ_s = 0.995 − 0.045 × (variance_s − min_var) / (max_var − min_var)
    range: λ_s ∈ [0.950, 0.995]
    Interpretation: volatile/bursty students → lower λ (shorter memory)
                    stable/regular students  → higher λ (longer memory)
  Fit and update are fully vectorized; same n_tier=3 call-order grouping.

M3+  (Markov)
  Bayesian Model Averaging over orders {2, 3, 4}  (vs single fixed order-3):
    • Drop order-1 (noisy, underfits temporal patterns)
    • Add order-4: 2^4=16 contexts per student, ~73 obs/context on 1177 days
    • BMA weights = softmax of val-split log-likelihoods (τ=1.0)
    • Sub-models refit on FULL training set after weight estimation
    Smoothing α separately tuned: grid over {0.3, 0.5, 1.0}, pick best on val-LL.

Usage:
    cd dl_pipeline
    python improvements/run_improvements.py
"""

from __future__ import annotations
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time, warnings
warnings.filterwarnings('ignore')

from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain    import FactoredMarkovChain

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55
K_SEL       = 6
TRAIN_RATIO = 0.9
SIM_DAYS    = 100
SEQ_LEN     = 14
SEED        = 42

# M1 / M1+ training hyperparams (optimal from optimize_hybrid.py)
H_ALPHA    = 0.9
H_NG       = 10
H_TEMP     = 2.0
H_ITERS    = 5
H_REPLAY   = 10
H_LR       = 5e-4
H_EPOCHS   = 50
H_PATIENCE = 15

# M2+ per-student λ range
LAM_MIN  = 0.950
LAM_MAX  = 0.995
LAM_BASE = 0.990          # M2 baseline
LAM_WINDOW = 14           # rolling window for variance estimation

# M3+ smoothing grid
M3P_ALPHA_GRID  = [0.3, 0.5, 1.0]
M3P_ORDERS      = [2, 3, 4]
M3P_VAL_FRAC    = 0.15

BENCHMARKS = {
    'M1  Hybrid LSTM+Markov (α=0.9)': 32.17,
    'M2  TD-HBB λ=0.99             ': 32.17,
    'M3  Order-3 Markov             ': 31.67,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]
    T       = len(df)
    binary  = np.zeros((T, N_S), dtype=np.float32)
    avg_pos = np.zeros(N_S, dtype=np.float64)
    pos_cnt = np.zeros(N_S, dtype=np.float64)
    for row_idx, row in df.iterrows():
        for pi, col in enumerate(id_cols):
            sid = int(row[col]) - 1
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
                avg_pos[sid] += pi + 1
                pos_cnt[sid] += 1.0
    avg_pos = np.where(pos_cnt > 0, avg_pos / pos_cnt, 3.5)
    return binary, avg_pos


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _freq(binary, window):
    """Cumulative-sum sliding window frequency, causal."""
    n, s   = binary.shape
    feat   = np.zeros((n, s), dtype=np.float32)
    cumsum = np.vstack([np.zeros((1, s), dtype=np.float32),
                        np.cumsum(binary, axis=0)])
    for t in range(1, n):
        start  = max(0, t - window)
        feat[t] = (cumsum[t] - cumsum[start]) / (t - start)
    return feat


def _recency(binary):
    """1 / (days_since_last + 1), causal."""
    n, s    = binary.shape
    feat    = np.zeros((n, s), dtype=np.float32)
    last    = -np.ones(s, dtype=np.float64)
    for t in range(n):
        mask = last >= 0
        if mask.any():
            feat[t, mask] = 1.0 / (t - last[mask] + 1)
        last[binary[t] == 1] = t
    return feat


def _cooc_nbr(binary):
    """
    Co-occurrence neighbour feature (causal, no leakage).

    cooc_nbr[t, s] = Σ_{j ∈ selected(t-1)} cumulative_cooc[s, j]  /  (t-1)

    Algorithm:
      At step t, compute feature using cooc accumulated from days 0..t-2,
      applied to yesterday's selections binary[t-1].
      Then update cooc with binary[t-1] (so next step's cooc includes t-1).

    Vectorised update: cooc += outer(y, y); zero diagonal.
    """
    n, ns  = binary.shape
    feat   = np.zeros((n, ns), dtype=np.float32)
    cooc   = np.zeros((ns, ns), dtype=np.float64)
    days   = 0   # days of history in cooc

    for t in range(n):
        # ① Compute feature for day t (use cooc from 0..t-2)
        if t >= 2 and days > 0:
            y_prev = binary[t - 1].astype(np.float64)
            feat[t] = (cooc @ y_prev / days).astype(np.float32)

        # ② Update cooc with binary[t-1] after computing feature for t
        if t >= 1:
            y = binary[t - 1].astype(np.float64)
            upd = np.outer(y, y)
            np.fill_diagonal(upd, 0.0)
            cooc += upd
            days += 1

    return feat


def build_features_v1(binary):
    """220-dim: raw + freq7 + freq14 + recency  (M1 baseline)."""
    print("  [v1] raw(55) + freq7(55) + freq14(55) + recency(55) = 220-dim")
    return np.concatenate([
        binary,
        _freq(binary, 7),
        _freq(binary, 14),
        _recency(binary),
    ], axis=1).astype(np.float32)


def build_features_v2(binary):
    """330-dim: v1 + freq30(55) + cooc_nbr(55)  (M1+ improved)."""
    print("  [v2] raw(55) + freq7(55) + freq14(55) + recency(55) "
          "+ freq30(55) + cooc_nbr(55) = 330-dim")
    print("       Computing co-occurrence neighbour features (causal)...", end='', flush=True)
    t0   = time.perf_counter()
    cooc = _cooc_nbr(binary)
    print(f" done in {time.perf_counter()-t0:.1f}s")
    return np.concatenate([
        binary,
        _freq(binary, 7),
        _freq(binary, 14),
        _recency(binary),
        _freq(binary, 30),
        cooc,
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quantile_groups(values, n):
    cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)


def _estimate_hyperpriors(S, N, grp):
    G     = int(grp.max()) + 1
    mu_g  = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2:
            mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r  = rates[m]; mu = float(r.mean()); var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) \
                  if var > 1e-10 else 1000.0
    return mu_g, ka_g


def _posterior_ab(S, N, grp, mu_g, kappa_g):
    a = mu_g[grp] * kappa_g[grp] + S
    b = (1.0 - mu_g[grp]) * kappa_g[grp] + np.maximum(N - S, 0.0)
    return a, b


# ─────────────────────────────────────────────────────────────────────────────
# M2 baseline — fixed-λ TD-HBB
# ─────────────────────────────────────────────────────────────────────────────

class TDHBBFixed:
    """TD-HBB with single global λ (M2 baseline, λ=0.99)."""

    def __init__(self, decay=LAM_BASE, n_tier=3, refresh_every=10):
        self.lam = decay; self.n_tier = n_tier; self.refresh = refresh_every
        self.S_eff = np.zeros(N_S); self.N_eff = np.zeros(N_S)
        self.grp = np.zeros(N_S, dtype=np.int32)
        self.mu_g = np.array([K_SEL / N_S]); self.kappa_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n   = len(binary); lam = self.lam
        w   = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self.S_eff = (binary * w[:, None]).sum(axis=0).astype(np.float64)
        self.N_eff = np.full(N_S, w.sum(), dtype=np.float64)
        self.grp   = _quantile_groups(avg_pos, self.n_tier)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)

    def predict_probs(self):
        a, b  = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                               self.mu_g, self.kappa_g)
        p     = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs):
        self.S_eff  = self.lam * self.S_eff + obs.astype(np.float64)
        self.N_eff  = self.lam * self.N_eff + 1.0
        self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)


# ─────────────────────────────────────────────────────────────────────────────
# M2+ — per-student adaptive-λ TD-HBB
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_student_lambda(binary_train: np.ndarray,
                                lam_min: float = LAM_MIN,
                                lam_max: float = LAM_MAX,
                                window: int    = LAM_WINDOW) -> np.ndarray:
    """
    Map each student's rolling selection variance to a decay rate.
      high variance → low λ  (short memory, tracks bursts)
      low  variance → high λ (long memory, exploits stability)
    """
    n, ns    = binary_train.shape
    roll_var = np.zeros(ns, dtype=np.float64)
    cumsum   = np.vstack([np.zeros((1, ns)), np.cumsum(binary_train, axis=0)])
    for s in range(ns):
        means = []
        for t in range(window, n):
            means.append((cumsum[t, s] - cumsum[t - window, s]) / window)
        roll_var[s] = float(np.var(means)) if means else 0.0

    v_min, v_max = roll_var.min(), roll_var.max()
    if v_max - v_min < 1e-12:
        return np.full(ns, (lam_min + lam_max) / 2)
    norm  = (roll_var - v_min) / (v_max - v_min)          # 0 = stable, 1 = volatile
    lam_s = lam_max - (lam_max - lam_min) * norm           # volatile → lower λ
    return lam_s.astype(np.float64)


class TDHBBAdaptive:
    """TD-HBB with per-student λ_s  (M2+)."""

    def __init__(self, n_tier=3, refresh_every=10):
        self.n_tier = n_tier; self.refresh = refresh_every
        self.lam_s  = np.full(N_S, LAM_BASE)   # set in fit()
        self.S_eff  = np.zeros(N_S); self.N_eff = np.zeros(N_S)
        self.grp    = np.zeros(N_S, dtype=np.int32)
        self.mu_g   = np.array([K_SEL / N_S]); self.kappa_g = np.array([2.0])
        self.n_obs  = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary)
        self.lam_s = compute_per_student_lambda(binary)
        # Per-student exponential weighting
        for s in range(N_S):
            w          = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S_eff[s] = float((binary[:, s] * w).sum())
            self.N_eff[s] = float(w.sum())
        self.grp = _quantile_groups(avg_pos, self.n_tier)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)
        lam_arr = self.lam_s
        print(f"    λ range: [{lam_arr.min():.4f}, {lam_arr.max():.4f}]  "
              f"mean={lam_arr.mean():.4f}  std={lam_arr.std():.4f}")

    def predict_probs(self):
        a, b = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                              self.mu_g, self.kappa_g)
        p    = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs):
        # Vectorised per-student decay
        self.S_eff  = self.lam_s * self.S_eff + obs.astype(np.float64)
        self.N_eff  = self.lam_s * self.N_eff + 1.0
        self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)


# ─────────────────────────────────────────────────────────────────────────────
# Variable-Order Markov (inlined from variable_markov/)
# ─────────────────────────────────────────────────────────────────────────────

class VOMCModel:
    def __init__(self, order, n_students=N_S, smoothing=1.0):
        self.order = order; self.N = n_students; self.alpha = smoothing
        n_ctx         = 2 ** order
        self.counts   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.totals   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.base_cnt = np.zeros(n_students, dtype=np.float64)
        self.n_days   = 0.0

    @staticmethod
    def _encode(history, order):
        ctx = np.zeros(history.shape[1], dtype=np.int32)
        for lag in range(order):
            ctx += history[-(lag + 1)].astype(np.int32) * (2 ** lag)
        return ctx

    def fit(self, binary):
        n = len(binary)
        self.n_days = float(n)
        self.base_cnt = binary.sum(axis=0).astype(np.float64)
        self.counts[:] = 0.0; self.totals[:] = 0.0
        for t in range(self.order, n):
            ctx   = self._encode(binary[t - self.order: t], self.order)
            today = binary[t]
            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if today[s] == 1:
                    self.counts[s, c] += 1.0

    def predict_probs(self, window):
        if len(window) < self.order:
            base = (self.base_cnt + self.alpha) / (self.n_days + 2 * self.alpha)
            return (base / base.sum()).astype(np.float32)
        ctx   = self._encode(window[-self.order:], self.order)
        probs = np.zeros(self.N, dtype=np.float64)
        for s in range(self.N):
            c = ctx[s]
            tot = self.totals[s, c]
            if tot > 0:
                probs[s] = (self.counts[s, c] + self.alpha) / (tot + 2 * self.alpha)
            else:
                probs[s] = (self.base_cnt[s] + self.alpha) / (self.n_days + 2 * self.alpha)
        probs = np.clip(probs, 1e-9, None)
        return (probs / probs.sum()).astype(np.float32)

    def log_likelihood(self, binary):
        ll = 0.0
        for t in range(self.order, len(binary)):
            ctx   = self._encode(binary[t - self.order: t], self.order)
            today = binary[t]
            for s in range(self.N):
                c   = ctx[s]
                tot = self.totals[s, c]
                if tot > 0:
                    p = (self.counts[s, c] + self.alpha) / (tot + 2 * self.alpha)
                else:
                    p = (self.base_cnt[s] + self.alpha) / (self.n_days + 2 * self.alpha)
                p   = float(np.clip(p, 1e-9, 1 - 1e-9))
                obs = int(today[s])
                ll += obs * np.log(p) + (1 - obs) * np.log(1 - p)
        return ll

    def update(self, obs, window):
        if len(window) >= self.order:
            ctx = self._encode(window[-self.order:], self.order)
            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if obs[s] == 1:
                    self.counts[s, c] += 1.0
        self.base_cnt += obs.astype(np.float64)
        self.n_days   += 1.0


# ─────────────────────────────────────────────────────────────────────────────
# M3 baseline — pure order-3 VOMC
# ─────────────────────────────────────────────────────────────────────────────

class M3Baseline:
    def __init__(self):
        self.model   = VOMCModel(order=3, smoothing=1.0)
        self.history = None

    def fit(self, binary):
        self.model.fit(binary); self.history = binary.copy()

    def predict_probs(self):
        if self.history is None or len(self.history) < 3:
            return np.ones(N_S, dtype=np.float32) / N_S
        return self.model.predict_probs(self.history[-14:])

    def update(self, obs):
        if self.history is not None and len(self.history) >= 3:
            self.model.update(obs, self.history[-3:])
        self.history = (obs[np.newaxis, :] if self.history is None
                        else np.vstack([self.history, obs[np.newaxis, :]]))


# ─────────────────────────────────────────────────────────────────────────────
# M3+ — BMA of orders {2, 3, 4} with smoothing tuning
# ─────────────────────────────────────────────────────────────────────────────

class BMAMarkovPlus:
    """
    BMA over orders {2, 3, 4}.

    1. For each smoothing α in grid {0.3, 0.5, 1.0}:
       a. Fit each order on binary[:val_cut]
       b. Compute val log-likelihood for each order
    2. Pick α that maximises total val log-likelihood
    3. Refit all sub-models on FULL training data with chosen α
    4. Set BMA weights = softmax(val log-likelihoods) with chosen α
    """

    def __init__(self, orders=None, alpha_grid=None, val_frac=M3P_VAL_FRAC):
        self.orders     = orders or M3P_ORDERS
        self.alpha_grid = alpha_grid or M3P_ALPHA_GRID
        self.val_frac   = val_frac
        self.models     = {}       # {order: VOMCModel}
        self.weights    = None     # np.ndarray (len(orders),)
        self.best_alpha = 1.0
        self.history    = None

    def fit(self, binary):
        n       = len(binary)
        val_cut = int(n * (1 - self.val_frac))

        # ── Grid over smoothing α ─────────────────────────────────────────
        best_total_ll = -np.inf
        best_alpha    = 1.0
        best_weights  = np.ones(len(self.orders)) / len(self.orders)

        for alpha in self.alpha_grid:
            mods = {k: VOMCModel(order=k, smoothing=alpha) for k in self.orders}
            for m in mods.values():
                m.fit(binary[:val_cut])
            log_lls = np.array([mods[k].log_likelihood(binary[val_cut:])
                                 for k in self.orders], dtype=np.float64)
            total_ll = log_lls.sum()
            if total_ll > best_total_ll:
                best_total_ll = total_ll
                best_alpha    = alpha
                # BMA weights via softmax
                ll_norm = log_lls - log_lls.max()
                w       = np.exp(ll_norm); w /= w.sum()
                best_weights = w

        self.best_alpha = best_alpha
        self.weights    = best_weights
        ll_str = "  ".join(f"order-{k}:{best_weights[i]:.3f}"
                            for i, k in enumerate(self.orders))
        print(f"    Best α={best_alpha}  BMA weights: {ll_str}")

        # ── Refit sub-models on FULL training data with best α ────────────
        self.models = {k: VOMCModel(order=k, smoothing=best_alpha)
                       for k in self.orders}
        for m in self.models.values():
            m.fit(binary)

        self.history = binary.copy()

    def predict_probs(self):
        if self.history is None or len(self.history) < max(self.orders):
            return np.ones(N_S, dtype=np.float32) / N_S
        window = self.history[-14:]
        p      = np.zeros(N_S, dtype=np.float64)
        for i, k in enumerate(self.orders):
            p += self.weights[i] * self.models[k].predict_probs(window).astype(np.float64)
        p = np.clip(p, 1e-9, None)
        return (p / p.sum()).astype(np.float32)

    def update(self, obs):
        win = (self.history[-max(self.orders):]
               if self.history is not None else np.zeros((1, N_S)))
        for k, m in self.models.items():
            m.update(obs, win[-k:] if len(win) >= k else win)
        self.history = (obs[np.newaxis, :] if self.history is None
                        else np.vstack([self.history, obs[np.newaxis, :]]))


# ─────────────────────────────────────────────────────────────────────────────
# LSTM training (shared for M1 and M1+)
# ─────────────────────────────────────────────────────────────────────────────

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, enriched, binary, seq_len=SEQ_LEN):
        self.E = enriched; self.B = binary; self.sl = seq_len

    def __len__(self):  return len(self.E) - self.sl

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.E[idx: idx + self.sl]),
                torch.FloatTensor(self.B[idx + self.sl]))


def train_lstm(binary_train, enriched_train, device, tag=''):
    nf        = enriched_train.shape[1]
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model     = EnhancedLSTMPredictor(
        n_features=nf, n_students=N_S,
        hidden_size=128, num_layers=2, dropout=0.3).to(device)
    split_v   = int(len(binary_train) * 0.9)
    tr_ld = DataLoader(SeqDataset(enriched_train[:split_v], binary_train[:split_v]),
                       batch_size=32, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(SeqDataset(enriched_train[split_v:], binary_train[split_v:]),
                       batch_size=32, shuffle=False, drop_last=False)
    if len(tr_ld) == 0 or len(vl_ld) == 0:
        return model
    opt   = optim.Adam(model.parameters(), lr=H_LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)
    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()
    for epoch in range(H_EPOCHS):
        model.train()
        for X, y in tr_ld:
            opt.zero_grad()
            loss = criterion(model(X.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for X, y in vl_ld:
                vl += criterion(model(X.to(device)), y.to(device)).item()
        vl /= len(vl_ld); sched.step(vl)
        if vl < best_loss:
            best_loss, best_st, no_imp = vl, model.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= H_PATIENCE:
                print(f"    {tag} Early stop @ epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_loss={vl:.5f}")
    print(f"    {tag} Training done in {time.perf_counter()-t0:.1f}s")
    model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Oracle group sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_groups(probs, n_groups=H_NG, temperature=H_TEMP):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups = []
    for _ in range(n_groups * 6):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SEL:].tolist()))
    return groups[:n_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def simulate_hybrid(lstm_model, markov_model, binary, enriched, split,
                    device, alpha=H_ALPHA, tag='') -> np.ndarray:
    """M1 / M1+ simulation with online LSTM fine-tuning."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    n_sim     = min(SIM_DAYS, len(binary) - split)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt       = optim.Adam(lstm_model.parameters(), lr=0.0003)
    replay: list = []
    accs         = []
    t0           = time.perf_counter()

    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN:
            accs.append(0.0); continue
        esq = enriched[idx - SEQ_LEN: idx]
        bsq = binary  [idx - SEQ_LEN: idx]

        # LSTM probs
        lstm_model.eval()
        with torch.no_grad():
            lp = lstm_model(torch.FloatTensor(esq).unsqueeze(0).to(device)
                            ).cpu().numpy()[0]
        lp /= (lp.sum() + 1e-9)

        # Markov probs
        mp = markov_model.predict_probabilities(bsq)

        # Blend
        hyb = alpha * lp + (1.0 - alpha) * mp
        hyb /= (hyb.sum() + 1e-9)

        # Oracle evaluation
        groups = sample_groups(hyb)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)

        # Online update
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            recent  = replay[-H_REPLAY:]
            Xb = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            lstm_model.train()
            for _ in range(H_ITERS):
                opt.zero_grad()
                loss = criterion(lstm_model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                opt.step()
        if len(bsq) >= 2:
            markov_model.update(binary[idx], bsq[-1], bsq[-2])

    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter()-t0:.1f}s")
    return arr


def simulate_tdhbb(model, binary, split, tag='') -> np.ndarray:
    """M2 / M2+ simulation."""
    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()
    for i in range(n_sim):
        idx    = split + i
        probs  = model.predict_probs()
        groups = sample_groups(probs)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        model.update(binary[idx])
    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter()-t0:.1f}s")
    return arr


def simulate_markov(model, binary, split, tag='') -> np.ndarray:
    """M3 / M3+ simulation."""
    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()
    for i in range(n_sim):
        idx    = split + i
        probs  = model.predict_probs()
        groups = sample_groups(probs)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        model.update(binary[idx])
    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter()-t0:.1f}s")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Result formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_result(name, arr, delta=None):
    dist  = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
    nz    = {k: v for k, v in dist.items() if v > 0}
    delta_str = f'  (Δ={delta:+.2f}%)' if delta is not None else ''
    print(f"\n  ── {name} ──{delta_str}")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("TARGETED IMPROVEMENTS: M1+ · M2+ · M3+".center(76))
    print("=" * 76)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data...")
    binary, avg_pos = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test\n")

    # ── Build feature sets ────────────────────────────────────────────────
    print("  Building feature sets...")
    enriched_v1 = build_features_v1(binary)   # 220-dim (M1 baseline)
    enriched_v2 = build_features_v2(binary)   # 330-dim (M1+)

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # M1 BASELINE vs M1+
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  M1 BASELINE  vs  M1+  (Hybrid LSTM+Markov)")
    print("=" * 76)

    # ── M1 baseline ───────────────────────────────────────────────────────
    print("\n  ─ M1 baseline (220-dim features) ─")
    torch.manual_seed(SEED); np.random.seed(SEED)
    lstm_m1  = train_lstm(binary[:split], enriched_v1[:split], device, tag='[M1]')
    markov_m1 = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov_m1.fit(binary[:split])
    arr_m1 = simulate_hybrid(lstm_m1, markov_m1, binary, enriched_v1, split, device,
                              tag='M1')
    print_result('M1 baseline', arr_m1)
    results['M1 baseline (220-dim)'] = arr_m1

    # ── M1+ improved ──────────────────────────────────────────────────────
    print("\n  ─ M1+ improved (330-dim: +freq30 +cooc_nbr) ─")
    torch.manual_seed(SEED); np.random.seed(SEED)
    lstm_m1p  = train_lstm(binary[:split], enriched_v2[:split], device, tag='[M1+]')
    markov_m1p = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov_m1p.fit(binary[:split])
    arr_m1p = simulate_hybrid(lstm_m1p, markov_m1p, binary, enriched_v2, split, device,
                               tag='M1+')
    print_result('M1+ (330-dim)', arr_m1p, delta=arr_m1p.mean() - arr_m1.mean())
    results['M1+ (330-dim: +freq30+cooc)'] = arr_m1p

    # ══════════════════════════════════════════════════════════════════════
    # M2 BASELINE vs M2+
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  M2 BASELINE  vs  M2+  (TD-HBB)")
    print("=" * 76)

    # ── M2 baseline ───────────────────────────────────────────────────────
    print("\n  ─ M2 baseline (global λ=0.99) ─")
    m2_base = TDHBBFixed(decay=LAM_BASE, n_tier=3)
    m2_base.fit(binary[:split], avg_pos)
    arr_m2 = simulate_tdhbb(m2_base, binary, split, tag='M2')
    print_result('M2 baseline (λ=0.99)', arr_m2)
    results['M2 baseline (λ=0.99 fixed)'] = arr_m2

    # ── M2+ adaptive ──────────────────────────────────────────────────────
    print("\n  ─ M2+ improved (per-student adaptive λ) ─")
    m2_plus = TDHBBAdaptive(n_tier=3)
    m2_plus.fit(binary[:split], avg_pos)
    arr_m2p = simulate_tdhbb(m2_plus, binary, split, tag='M2+')
    print_result('M2+ (adaptive λ)', arr_m2p, delta=arr_m2p.mean() - arr_m2.mean())
    results['M2+ (per-student adaptive λ)'] = arr_m2p

    # ══════════════════════════════════════════════════════════════════════
    # M3 BASELINE vs M3+
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  M3 BASELINE  vs  M3+  (Markov)")
    print("=" * 76)

    # ── M3 baseline ───────────────────────────────────────────────────────
    print("\n  ─ M3 baseline (Order-3 VOMC, α=1.0) ─")
    m3_base = M3Baseline()
    m3_base.fit(binary[:split])
    arr_m3 = simulate_markov(m3_base, binary, split, tag='M3')
    print_result('M3 baseline (order-3)', arr_m3)
    results['M3 baseline (order-3, α=1.0)'] = arr_m3

    # ── M3+ BMA ───────────────────────────────────────────────────────────
    print("\n  ─ M3+ improved (BMA orders {2,3,4}, α tuned) ─")
    m3_plus = BMAMarkovPlus()
    m3_plus.fit(binary[:split])
    arr_m3p = simulate_markov(m3_plus, binary, split, tag='M3+')
    print_result('M3+ (BMA 2+3+4)', arr_m3p, delta=arr_m3p.mean() - arr_m3.mean())
    results['M3+ (BMA orders {2,3,4})'] = arr_m3p

    # ══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    prev_best = max(BENCHMARKS.values())   # 32.17
    print("\n" + "─" * 76)
    print("  IMPROVEMENT SUMMARY")
    print("─" * 76)

    pairs = [
        ('M1', arr_m1, 'M1+', arr_m1p, 'co-occurrence nbr + 30d freq'),
        ('M2', arr_m2, 'M2+', arr_m2p, 'per-student adaptive λ'),
        ('M3', arr_m3, 'M3+', arr_m3p, 'BMA orders {2,3,4}, tuned α'),
    ]
    print(f"\n  {'Model':<5}  {'Baseline':>8}  {'Improved':>8}  {'Δ':>7}  "
          f"{'Change':<35}  Result")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*35}  {'─'*10}")
    for base_name, base_arr, plus_name, plus_arr, desc in pairs:
        delta = plus_arr.mean() - base_arr.mean()
        vs_rec = plus_arr.mean() - prev_best
        result = ('★ NEW BEST' if plus_arr.mean() > prev_best
                  else (f'Δ_record={vs_rec:+.2f}%'))
        print(f"  {base_name:<5}  {base_arr.mean():>8.2f}%  "
              f"{plus_arr.mean():>8.2f}%  {delta:>+7.2f}%  "
              f"{desc:<35}  {result}")

    # ══════════════════════════════════════════════════════════════════════
    # FULL LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  FULL LEADERBOARD")
    print("=" * 76)
    print(f"\n  {'Model':<42}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  {'Std':>7}")
    print(f"  {'─'*42}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<42}  {v:>7.2f}%  {'—':>7}  {'—':>7}  {'—':>7}  [ref]")
    print(f"  {'─'*42}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for n, a in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        flag = '  ★ NEW BEST' if a.mean() > prev_best else ''
        print(f"  {n:<42}  {a.mean():>7.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    # ── Notes ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 76)
    print("  INTERPRETATION NOTES")
    print("─" * 76)
    print("""
  M1+  co-occurrence neighbour feature
    • Encodes: "given yesterday's group, how often is student s historically
      co-selected with those students?"
    • This is the same signal that the GNN (32.00%) exploited via message
      passing — here baked directly into the LSTM input.
    • freq30 separates stable long-term base rate from short-term bursts.

  M2+  per-student adaptive λ
    • High-variance / bursty students get shorter memory (λ ≈ 0.95).
    • Stable / regular students get longer memory (λ ≈ 0.995).
    • If M2+ matches or beats M2: variance is a useful differentiator.
    • If M2+ is worse: the homogeneous λ=0.99 is already near-optimal
      and per-student adaptation introduces estimation noise.

  M3+  BMA orders {2,3,4}
    • Drops order-1 (too low resolution, dominated by higher orders).
    • Adds order-4 (16 contexts/student, ~73 obs each — well above sparse
      threshold of ~20, so Laplace smoothing handles unseen contexts).
    • BMA weights determined by held-out log-likelihood → automatically
      down-weights the order that overfits the validation window.
""")
    best_new = max(results, key=lambda x: results[x].mean())
    best_val = results[best_new].mean()
    print("=" * 76)
    print(f"  BEST NEW MODEL: {best_new}  →  {best_val:.2f}%")
    if best_val > prev_best:
        print(f"  ★ NEW OVERALL RECORD  (was {prev_best:.2f}%,  Δ={best_val-prev_best:+.2f}%)")
    else:
        print(f"  Gap to record: {best_val - prev_best:+.2f}%  (record = {prev_best:.2f}%)")
    print("=" * 76)


if __name__ == '__main__':
    main()
