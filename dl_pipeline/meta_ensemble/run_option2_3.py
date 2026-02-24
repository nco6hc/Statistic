"""
Meta-Learning Ensemble — Options 2 & 3
========================================

Option 2 : Per-Student Best Picker (PSBP)
  Static routing: for each student s, assign it to whichever base model had
  the highest recall on meta-training days.  Routing is learned once and fixed.

Option 3 : Temporal Gate
  Dynamic routing: track per-model rolling oracle-hit accuracy over a W=14-day
  sliding window.  Two flavours:
    TGH (Hard) — hard-switch to the single model with best rolling accuracy.
    TGS (Soft) — blend models with weights proportional to rolling accuracy.

Option 2+3 Combined : Per-Student Temporal (PST)
  Per-student dynamic routing.  For every (student s, model k) pair, maintain a
  rolling Brier-score window (last W=14 days).  Each student is routed to the
  model with the lowest recent Brier score, updating every test step.

Reference: Uniform Average (1/3 each) included as the floor benchmark.

Usage:
    cd dl_pipeline
    python meta_ensemble/run_option2_3.py
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
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings
warnings.filterwarnings('ignore')

from enhanced_features import build_enriched_features
from enhanced_models    import EnhancedLSTMPredictor, FocalLoss
from markov_chain       import FactoredMarkovChain

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55
K_SEL       = 6
TRAIN_RATIO = 0.9
META_FRAC   = 0.75      # fraction of train used to fit base models in phase 1
SIM_DAYS    = 100
SEQ_LEN     = 14
SEED        = 42

# Hybrid (M1) best hyperparams
H_ALPHA   = 0.9
H_NG      = 10
H_TEMP    = 2.0
H_ITERS   = 5
H_REPLAY  = 10
H_LR      = 0.0005
H_EPOCHS  = 50
H_PATIENCE = 15

# Meta group-sampling params (same as Hybrid oracle)
META_NG   = 10
META_TEMP = 2.0

# Temporal-gate rolling window
GATE_WINDOW = 14

BENCHMARKS = {
    'M1  Hybrid LSTM+Markov (α=0.9)': 32.17,
    'M2  TD-HBB λ=0.99             ': 32.17,
    'M3  Order-3 Markov             ': 31.67,
    'BMA (Order 1+2+3)              ': 31.50,
    'DPM optimised                  ': 30.83,
    'Meta: Uniform Average          ': 31.33,   # previous run
    'Meta: Linear Stack             ': 31.33,   # previous run
}

# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB helpers (inlined from hbb_advanced)
# ─────────────────────────────────────────────────────────────────────────────

def _quantile_groups(values: np.ndarray, n: int) -> np.ndarray:
    cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)


def _estimate_hyperpriors(S: np.ndarray, N: np.ndarray, grp: np.ndarray):
    G = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2:
            mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r = rates[m]; mu = float(r.mean()); var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) \
                  if var > 1e-10 else 1000.0
    return mu_g, ka_g


def _posterior_ab(S, N, grp, mu_g, kappa_g):
    alpha = mu_g[grp] * kappa_g[grp] + S
    beta_ = (1.0 - mu_g[grp]) * kappa_g[grp] + np.maximum(N - S, 0.0)
    return alpha, beta_


# ─────────────────────────────────────────────────────────────────────────────
# Variable-Order Markov (Order-3) — inlined
# ─────────────────────────────────────────────────────────────────────────────

class VOMCModel:
    def __init__(self, order: int, n_students: int = N_S, smoothing: float = 1.0):
        self.order  = order
        self.N      = n_students
        self.alpha  = smoothing
        n_ctx       = 2 ** order
        self.counts   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.totals   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.base_cnt = np.zeros(n_students, dtype=np.float64)
        self.n_days   = 0.0

    @staticmethod
    def _encode(history: np.ndarray, order: int) -> np.ndarray:
        ctx = np.zeros(history.shape[1], dtype=np.int32)
        for lag in range(order):
            ctx += history[-(lag + 1)].astype(np.int32) * (2 ** lag)
        return ctx

    def fit(self, binary: np.ndarray) -> None:
        n = len(binary)
        self.n_days   = float(n)
        self.base_cnt = binary.sum(axis=0).astype(np.float64)
        self.counts[:] = 0.0; self.totals[:] = 0.0
        for t in range(self.order, n):
            history = binary[t - self.order: t]
            ctx     = self._encode(history, self.order)
            today   = binary[t]
            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if today[s] == 1:
                    self.counts[s, c] += 1.0

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        if len(window) < self.order:
            base = (self.base_cnt + self.alpha) / (self.n_days + 2 * self.alpha)
            return (base / base.sum()).astype(np.float32)
        history = window[-self.order:]
        ctx     = self._encode(history, self.order)
        probs   = np.zeros(self.N, dtype=np.float64)
        for s in range(self.N):
            c   = ctx[s]
            cnt = self.counts[s, c]; tot = self.totals[s, c]
            if tot > 0:
                probs[s] = (cnt + self.alpha) / (tot + 2 * self.alpha)
            else:
                probs[s] = (self.base_cnt[s] + self.alpha) / (self.n_days + 2 * self.alpha)
        probs = np.clip(probs, 1e-9, None)
        return (probs / probs.sum()).astype(np.float32)

    def update(self, obs: np.ndarray, window: np.ndarray) -> None:
        if len(window) >= self.order:
            history = window[-self.order:]
            ctx     = self._encode(history, self.order)
            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if obs[s] == 1: self.counts[s, c] += 1.0
        self.base_cnt += obs.astype(np.float64)
        self.n_days   += 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Group sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_groups(probs: np.ndarray, n_groups: int = META_NG,
                  temperature: float = META_TEMP) -> list[list[int]]:
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups: list[list[int]] = []
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
# LSTM training utility
# ─────────────────────────────────────────────────────────────────────────────

class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, enriched, binary, seq_len=SEQ_LEN):
        self.enriched = enriched; self.binary = binary; self.seq_len = seq_len

    def __len__(self):
        return len(self.enriched) - self.seq_len

    def __getitem__(self, idx):
        X = self.enriched[idx: idx + self.seq_len]
        y = self.binary  [idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


def train_lstm_model(binary_train, enriched_train, n_features, device, tag=''):
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=N_S,
        hidden_size=128, num_layers=2, dropout=0.3).to(device)
    split_v = int(len(binary_train) * 0.9)
    tr_ds = HybridDataset(enriched_train[:split_v], binary_train[:split_v])
    vl_ds = HybridDataset(enriched_train[split_v:], binary_train[split_v:])
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False, drop_last=False)
    if len(tr_ld) == 0 or len(vl_ld) == 0:
        return model, model.state_dict()
    optimizer = optim.Adam(model.parameters(), lr=H_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=5)
    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()
    for epoch in range(H_EPOCHS):
        model.train()
        for X, y in tr_ld:
            optimizer.zero_grad()
            loss = criterion(model(X.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in vl_ld:
                vl += criterion(model(X.to(device)), y.to(device)).item()
        vl /= len(vl_ld)
        scheduler.step(vl)
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
    return model, best_st


# ─────────────────────────────────────────────────────────────────────────────
# Base model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class M1_HybridWrapper:
    def __init__(self, n_features, device):
        self.device = device; self.n_features = n_features
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
        self.model = None; self.markov = None
        self.replay: list = []; self._opt = None

    def fit(self, binary, enriched, tag=''):
        torch.manual_seed(SEED); np.random.seed(SEED)
        self.model, _ = train_lstm_model(binary, enriched, self.n_features, self.device, tag)
        self.markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
        self.markov.fit(binary)
        self._opt = optim.Adam(self.model.parameters(), lr=0.0003)
        self.replay = []

    def predict_probs(self, enriched_seq, binary_seq):
        self.model.eval()
        with torch.no_grad():
            X_t    = torch.FloatTensor(enriched_seq).unsqueeze(0).to(self.device)
            lstm_p = self.model(X_t).cpu().numpy()[0]
        lstm_n   = lstm_p / (lstm_p.sum() + 1e-9)
        markov_p = self.markov.predict_probabilities(binary_seq)
        hyb      = H_ALPHA * lstm_n + (1.0 - H_ALPHA) * markov_p
        return (hyb / (hyb.sum() + 1e-9)).astype(np.float32)

    def update(self, binary_t, enriched_seq, binary_seq):
        self.replay.append((enriched_seq.copy(), binary_t.copy()))
        if len(self.replay) >= 5:
            recent  = self.replay[-H_REPLAY:]
            X_batch = torch.FloatTensor(np.array([r[0] for r in recent])).to(self.device)
            y_batch = torch.FloatTensor(np.array([r[1] for r in recent])).to(self.device)
            self.model.train()
            for _ in range(H_ITERS):
                self._opt.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._opt.step()
        n = len(binary_seq)
        if n >= 2:
            self.markov.update(binary_t, binary_seq[-1], binary_seq[-2])


class M2_TDHBBWrapper:
    def __init__(self, decay=0.99, n_tier=3, refresh_every=10):
        self.decay = decay; self.n_tier = n_tier; self.refresh_every = refresh_every
        self.S_eff = np.zeros(N_S, dtype=np.float64)
        self.N_eff = np.zeros(N_S, dtype=np.float64)
        self.grp   = np.zeros(N_S, dtype=np.int32)
        self.mu_g  = np.array([K_SEL / N_S])
        self.kappa_g = np.array([2.0]); self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n   = len(binary); lam = self.decay
        w   = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self.S_eff = (binary * w[:, None]).sum(axis=0).astype(np.float64)
        self.N_eff = np.full(N_S, w.sum(), dtype=np.float64)
        self.grp   = _quantile_groups(avg_pos, self.n_tier)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)

    def predict_probs(self):
        a, b  = _posterior_ab(self.S_eff, self.N_eff, self.grp, self.mu_g, self.kappa_g)
        probs = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return probs / probs.sum()

    def update(self, obs):
        self.S_eff  = self.decay * self.S_eff + obs.astype(np.float64)
        self.N_eff  = self.decay * self.N_eff + 1.0
        self.n_obs += 1.0
        if self.refresh_every > 0 and int(self.n_obs) % self.refresh_every == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(self.S_eff, self.N_eff, self.grp)


class M3_VOMCWrapper:
    def __init__(self, order=3):
        self.vomc    = VOMCModel(order=order)
        self.history = None

    def fit(self, binary):
        self.vomc.fit(binary); self.history = binary.copy()

    def predict_probs(self):
        if self.history is None or len(self.history) < self.vomc.order:
            return np.ones(N_S, dtype=np.float32) / N_S
        return self.vomc.predict_probs(self.history[-14:])

    def update(self, obs):
        if self.history is not None and len(self.history) >= self.vomc.order:
            self.vomc.update(obs, self.history[-self.vomc.order:])
        self.history = obs[np.newaxis, :] if self.history is None \
                       else np.vstack([self.history, obs[np.newaxis, :]])


# ─────────────────────────────────────────────────────────────────────────────
# NEW Meta-learner implementations
# ─────────────────────────────────────────────────────────────────────────────

class MetaUniform:
    """Reference baseline: equal-weight average."""
    name = 'Uniform Average (ref)'

    def fit(self, X_meta, y_meta): pass

    def predict(self, p_list):
        p = np.mean(p_list, axis=0)
        return p / (p.sum() + 1e-9)

    def update_online(self, *args): pass


class MetaPerStudentBestPicker:
    """
    Option 2 — Per-Student Best Picker (PSBP).

    For each student s, compute the per-model recall on meta-training days:
      recall_k_s = (days where y[t,s]==1  AND  s ∈ top-6(p_k_t))
                   ─────────────────────────────────────────────────
                          (days where y[t,s]==1)

    Assign student s to argmax_k recall_k_s.  Routing is static — fitted once,
    used unchanged throughout the 100-day simulation.
    """
    name = 'Per-Student Best Picker'

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        T, n_models, NS = X_meta.shape
        recall  = np.zeros((n_models, NS), dtype=np.float64)
        n_sel   = np.zeros(NS, dtype=np.float64)

        for t in range(T):
            for s in range(NS):
                if y_meta[t, s] == 1.0:
                    n_sel[s] += 1
                    for k in range(n_models):
                        top6 = set(np.argsort(X_meta[t, k, :])[-6:].tolist())
                        if s in top6:
                            recall[k, s] += 1.0

        for s in range(NS):
            if n_sel[s] > 0:
                recall[:, s] /= n_sel[s]   # normalise to [0,1]

        self.best_model = np.argmax(recall, axis=0)   # (N_S,)
        self.recall     = recall

        counts = [int((self.best_model == k).sum()) for k in range(n_models)]
        avg_r  = [recall[k].mean() for k in range(n_models)]
        print(f"    PSBP assignments  : M1={counts[0]}  M2={counts[1]}  M3={counts[2]}")
        print(f"    Mean recall       : M1={avg_r[0]:.3f}  M2={avg_r[1]:.3f}  M3={avg_r[2]:.3f}")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        NS     = len(p_list[0])
        hybrid = np.array([float(p_list[self.best_model[s]][s]) for s in range(NS)],
                          dtype=np.float32)
        hybrid = np.clip(hybrid, 1e-9, None)
        return hybrid / hybrid.sum()

    def update_online(self, *args): pass


class MetaTemporalGateHard:
    """
    Option 3a — Temporal Gate Hard (TGH).

    Maintain a rolling W-day window of per-model oracle hit rate
    ( hits-in-top6 / 6 ) for each day.  At every step, route ALL students
    to the single model with the highest rolling average.  Switch is immediate.

    Pre-warmed on the last W days of meta-training data.
    """
    name = 'Temporal Gate (Hard)'

    def __init__(self, window: int = GATE_WINDOW):
        self.window = window
        self.rolls  = [[] for _ in range(3)]   # list of recent hit rates per model
        self.active = 0

    def _rolling_means(self):
        return [float(np.mean(r)) if r else 0.0 for r in self.rolls]

    def _update_window(self, p_list: list[np.ndarray], actual_set: set):
        for k, p in enumerate(p_list):
            top6 = set(np.argsort(p)[-6:].tolist())
            hits = len(top6 & actual_set) / K_SEL
            self.rolls[k].append(hits)
            if len(self.rolls[k]) > self.window:
                self.rolls[k].pop(0)
        self.active = int(np.argmax(self._rolling_means()))

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        T, n_models, NS = X_meta.shape
        start = max(0, T - self.window)
        for t in range(start, T):
            p_list     = [X_meta[t, k] for k in range(n_models)]
            actual_set = set(np.where(y_meta[t] == 1)[0].tolist())
            self._update_window(p_list, actual_set)
        means = self._rolling_means()
        self.active = int(np.argmax(means))
        print(f"    TGH after pre-warm: active=M{self.active+1}  "
              f"(M1={means[0]:.3f}  M2={means[1]:.3f}  M3={means[2]:.3f})")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        p = p_list[self.active].copy().astype(np.float32)
        return p / (p.sum() + 1e-9)

    def update_online(self, p_list: list[np.ndarray], actual_set: set):
        self._update_window(p_list, actual_set)


class MetaTemporalGateSoft:
    """
    Option 3b — Temporal Gate Soft (TGS).

    Same rolling W-day oracle hit-rate window as TGH, but instead of a hard
    switch the three models are blended with weights proportional to their
    rolling average hit rate:

      w_k  =  mean_roll_k / Σ_j mean_roll_j

    Pre-warmed on the last W days of meta-training data.
    """
    name = 'Temporal Gate (Soft)'

    def __init__(self, window: int = GATE_WINDOW):
        self.window = window
        self.rolls  = [[] for _ in range(3)]

    def _weights(self) -> np.ndarray:
        means = np.array([float(np.mean(r)) if r else 1/3 for r in self.rolls])
        s = means.sum()
        return means / s if s > 1e-9 else np.ones(3) / 3

    def _update_window(self, p_list: list[np.ndarray], actual_set: set):
        for k, p in enumerate(p_list):
            top6 = set(np.argsort(p)[-6:].tolist())
            hits = len(top6 & actual_set) / K_SEL
            self.rolls[k].append(hits)
            if len(self.rolls[k]) > self.window:
                self.rolls[k].pop(0)

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        T, n_models, NS = X_meta.shape
        start = max(0, T - self.window)
        for t in range(start, T):
            p_list     = [X_meta[t, k] for k in range(n_models)]
            actual_set = set(np.where(y_meta[t] == 1)[0].tolist())
            self._update_window(p_list, actual_set)
        w = self._weights()
        print(f"    TGS after pre-warm: M1={w[0]:.3f}  M2={w[1]:.3f}  M3={w[2]:.3f}")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        w = self._weights()
        p = sum(w[k] * p_list[k] for k in range(3)).astype(np.float32)
        return p / (p.sum() + 1e-9)

    def update_online(self, p_list: list[np.ndarray], actual_set: set):
        self._update_window(p_list, actual_set)


class MetaPerStudentTemporal:
    """
    Combined Option 2+3 — Per-Student Temporal (PST).

    For every (student s, model k) pair, maintain a rolling W-day Brier-score
    window that is updated on EVERY test day (not just when s is selected):

      brier(k, s, t)  =  ( p_k_s_t  −  y_s_t )²

    At prediction time, route each student to:

      k*(s)  =  argmin_k  mean_brier(k, s, last W days)

    Fallback to static recall-based assignment (from meta-train) if the
    rolling window is not yet filled for all three models.

    This is the richest variant: per-student routing that adapts in real time
    to drift without requiring the student to have been recently selected.
    """
    name = 'Per-Student Temporal'

    def __init__(self, window: int = GATE_WINDOW):
        self.window      = window
        # brier_rolls[k][s] = list of recent Brier scores (length ≤ window)
        self.brier_rolls = [[[] for _ in range(N_S)] for _ in range(3)]
        self.static_best = np.zeros(N_S, dtype=int)    # fallback

    # ── offline fit ─────────────────────────────────────────────────────────

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        T, n_models, NS = X_meta.shape

        # ① compute static recall as fallback
        recall = np.zeros((n_models, NS), dtype=np.float64)
        n_sel  = np.zeros(NS, dtype=np.float64)
        for t in range(T):
            for s in range(NS):
                if y_meta[t, s] == 1.0:
                    n_sel[s] += 1
                    for k in range(n_models):
                        top6 = set(np.argsort(X_meta[t, k, :])[-6:].tolist())
                        if s in top6:
                            recall[k, s] += 1.0
        for s in range(NS):
            if n_sel[s] > 0:
                recall[:, s] /= n_sel[s]
        self.static_best = np.argmax(recall, axis=0)

        # ② pre-warm rolling Brier windows on last `window` meta-train days
        start = max(0, T - self.window)
        for t in range(start, T):
            for k in range(n_models):
                p_k = X_meta[t, k]           # (N_S,)
                for s in range(NS):
                    brier = (float(p_k[s]) - float(y_meta[t, s])) ** 2
                    self.brier_rolls[k][s].append(brier)
                    if len(self.brier_rolls[k][s]) > self.window:
                        self.brier_rolls[k][s].pop(0)

        # Summarise initial assignments
        assignments = self._compute_assignments()
        counts = [int((assignments == k).sum()) for k in range(n_models)]
        # Average Brier per model (across students)
        avg_b  = [np.mean([np.mean(self.brier_rolls[k][s])
                           for s in range(NS)
                           if self.brier_rolls[k][s]])
                  for k in range(n_models)]
        print(f"    PST assignments   : M1={counts[0]}  M2={counts[1]}  M3={counts[2]}")
        print(f"    Avg Brier (lower=better): "
              f"M1={avg_b[0]:.4f}  M2={avg_b[1]:.4f}  M3={avg_b[2]:.4f}")

    def _compute_assignments(self) -> np.ndarray:
        assignments = np.empty(N_S, dtype=int)
        for s in range(N_S):
            means  = [np.mean(self.brier_rolls[k][s])
                      if self.brier_rolls[k][s] else None
                      for k in range(3)]
            # All three models have data → use rolling Brier (lower is better)
            if all(m is not None for m in means):
                assignments[s] = int(np.argmin(means))
            else:
                assignments[s] = int(self.static_best[s])
        return assignments

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        assignments = self._compute_assignments()
        hybrid = np.array([float(p_list[assignments[s]][s]) for s in range(N_S)],
                          dtype=np.float32)
        hybrid = np.clip(hybrid, 1e-9, None)
        return hybrid / hybrid.sum()

    def update_online(self, p_list: list[np.ndarray], actual_set: set):
        """Update Brier windows for ALL students (not just selected ones)."""
        y = np.zeros(N_S, dtype=np.float32)
        for s in actual_set:
            y[s] = 1.0
        for k, p in enumerate(p_list):
            for s in range(N_S):
                brier = (float(p[s]) - float(y[s])) ** 2
                self.brier_rolls[k][s].append(brier)
                if len(self.brier_rolls[k][s]) > self.window:
                    self.brier_rolls[k][s].pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: collect meta-training features
# ─────────────────────────────────────────────────────────────────────────────

def collect_meta_features(m1, m2, m3, binary, enriched, meta_split, split):
    T_meta = split - meta_split
    X_meta = np.zeros((T_meta, 3, N_S), dtype=np.float32)
    y_meta = np.zeros((T_meta, N_S),    dtype=np.float32)
    print(f"\n  Collecting {T_meta} meta-training samples "
          f"(days {meta_split}–{split-1})...")
    t0 = time.perf_counter()
    for i, actual_idx in enumerate(range(meta_split, split)):
        if actual_idx < SEQ_LEN:
            continue
        enriched_seq = enriched[actual_idx - SEQ_LEN: actual_idx]
        binary_seq   = binary  [actual_idx - SEQ_LEN: actual_idx]
        p1 = m1.predict_probs(enriched_seq, binary_seq)
        p2 = m2.predict_probs()
        p3 = m3.predict_probs()
        X_meta[i, 0] = p1; X_meta[i, 1] = p2; X_meta[i, 2] = p3
        y_meta[i]    = binary[actual_idx]
        m1.update(binary[actual_idx], enriched_seq, binary_seq)
        m2.update(binary[actual_idx])
        m3.update(binary[actual_idx])
    print(f"  Meta-feature collection done in {time.perf_counter()-t0:.1f}s")
    return X_meta, y_meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(m1, m2, m3, meta_learner, binary, enriched, split):
    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()
    for i in range(n_sim):
        actual_idx = split + i
        if actual_idx < SEQ_LEN:
            accs.append(0.0); continue
        enriched_seq = enriched[actual_idx - SEQ_LEN: actual_idx]
        binary_seq   = binary  [actual_idx - SEQ_LEN: actual_idx]
        p1 = m1.predict_probs(enriched_seq, binary_seq)
        p2 = m2.predict_probs()
        p3 = m3.predict_probs()
        p_list  = [p1, p2, p3]
        p_meta  = meta_learner.predict(p_list)
        groups  = sample_groups(p_meta, META_NG, META_TEMP)
        actual  = set(np.where(binary[actual_idx] == 1)[0].tolist())
        best    = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        m1.update(binary[actual_idx], enriched_seq, binary_seq)
        m2.update(binary[actual_idx])
        m3.update(binary[actual_idx])
        meta_learner.update_online(p_list, actual)
    return np.array(accs) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("META-ENSEMBLE  —  OPTIONS 2 & 3  (Routing + Temporal Gating)".center(76))
    print("M1: Hybrid  |  M2: TD-HBB λ=0.99  |  M3: Order-3 Markov".center(76))
    print("=" * 76)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}  |  Gate window W={GATE_WINDOW} days")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data...")
    binary, avg_pos = load_data()
    print("  Computing enriched features...")
    enriched   = build_enriched_features(binary)
    n_features = enriched.shape[1]
    split      = int(len(binary) * TRAIN_RATIO)
    meta_split = int(split * META_FRAC)
    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test")
    print(f"  meta_split={meta_split}  |  meta-train: {split-meta_split} days\n")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 — fit base models on binary[0:meta_split]
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 76)
    print("  PHASE 1 — Fit base models on first 75% of train")
    print("=" * 76)

    print("\n  ─ M1: Hybrid LSTM+Markov ─")
    m1_meta = M1_HybridWrapper(n_features, device)
    m1_meta.fit(binary[:meta_split], enriched[:meta_split], tag='[M1-meta]')

    print("\n  ─ M2: TD-HBB λ=0.99 ─")
    m2_meta = M2_TDHBBWrapper(decay=0.99, n_tier=3)
    m2_meta.fit(binary[:meta_split], avg_pos)
    print(f"    Fitted TD-HBB on {meta_split} days")

    print("\n  ─ M3: Order-3 Markov ─")
    m3_meta = M3_VOMCWrapper(order=3)
    m3_meta.fit(binary[:meta_split])
    print(f"    Fitted VOMC(order=3) on {meta_split} days")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2 — collect meta-training (p1, p2, p3, y) on days [meta_split:split]
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 2 — Collect meta-training features")
    print("=" * 76)
    X_meta, y_meta = collect_meta_features(
        m1_meta, m2_meta, m3_meta, binary, enriched, meta_split, split)
    valid  = y_meta.sum(axis=1) > 0
    X_meta = X_meta[valid]; y_meta = y_meta[valid]
    print(f"  Meta-training samples after filtering: {len(X_meta)}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3 — train meta-learners
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 3 — Train meta-learners")
    print("=" * 76)

    meta_learners = [
        MetaUniform(),
        MetaPerStudentBestPicker(),
        MetaTemporalGateHard(window=GATE_WINDOW),
        MetaTemporalGateSoft(window=GATE_WINDOW),
        MetaPerStudentTemporal(window=GATE_WINDOW),
    ]
    for ml in meta_learners:
        print(f"\n  ─ {ml.name} ─")
        t0 = time.perf_counter()
        ml.fit(X_meta, y_meta)
        print(f"    Training done in {time.perf_counter()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4 — refit base models on full training [0:split]
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 4 — Refit base models on full training set")
    print("=" * 76)

    print("\n  ─ M1: Hybrid LSTM+Markov ─")
    m1 = M1_HybridWrapper(n_features, device)
    m1.fit(binary[:split], enriched[:split], tag='[M1-full]')

    print("\n  ─ M2: TD-HBB λ=0.99 ─")
    m2 = M2_TDHBBWrapper(decay=0.99, n_tier=3)
    m2.fit(binary[:split], avg_pos)
    print(f"    Fitted TD-HBB on {split} days")

    print("\n  ─ M3: Order-3 Markov ─")
    m3 = M3_VOMCWrapper(order=3)
    m3.fit(binary[:split])
    print(f"    Fitted VOMC(order=3) on {split} days")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5 — simulate 100 test days per meta-learner
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 5 — Simulation (100 test days)")
    print("=" * 76)

    import copy as _copy
    base_states = {
        'm1_sd'  : m1.model.state_dict(),
        'm1_mk'  : m1.markov,
        'm2_S'   : m2.S_eff.copy(), 'm2_N': m2.N_eff.copy(),
        'm2_grp' : m2.grp.copy(), 'm2_mu': m2.mu_g.copy(), 'm2_ka': m2.kappa_g.copy(),
        'm3_vomc': _copy.deepcopy(m3.vomc),
        'm3_hist': m3.history.copy(),
    }

    results = {}
    for ml in meta_learners:
        print(f"\n  ─ {ml.name} ─")
        # Restore base models to identical post-phase-4 state
        m1_r = M1_HybridWrapper(n_features, device)
        m1_r.model = EnhancedLSTMPredictor(
            n_features=n_features, n_students=N_S,
            hidden_size=128, num_layers=2, dropout=0.3).to(device)
        m1_r.model.load_state_dict(base_states['m1_sd'])
        m1_r.markov    = base_states['m1_mk']
        m1_r._opt      = optim.Adam(m1_r.model.parameters(), lr=0.0003)
        m1_r.replay    = []
        m1_r.criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

        m2_r = M2_TDHBBWrapper(decay=0.99, n_tier=3)
        m2_r.S_eff   = base_states['m2_S'].copy()
        m2_r.N_eff   = base_states['m2_N'].copy()
        m2_r.grp     = base_states['m2_grp'].copy()
        m2_r.mu_g    = base_states['m2_mu'].copy()
        m2_r.kappa_g = base_states['m2_ka'].copy()
        m2_r.n_obs   = float(split)

        m3_r = M3_VOMCWrapper(order=3)
        m3_r.vomc    = _copy.deepcopy(base_states['m3_vomc'])
        m3_r.history = base_states['m3_hist'].copy()

        t_sim   = time.perf_counter()
        arr     = run_simulation(m1_r, m2_r, m3_r, ml, binary, enriched, split)
        elapsed = time.perf_counter() - t_sim
        results[ml.name] = arr

        dist = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
        nz   = {k: v for k, v in dist.items() if v > 0}
        print(f"\n  ── {ml.name} ──")
        print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
              f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
        print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
        print(f"  Sim time: {elapsed:.2f}s")

    # ══════════════════════════════════════════════════════════════════════
    # LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════
    prev_best  = 32.17
    best_prev  = max(BENCHMARKS.values())

    print("\n" + "=" * 76)
    print("  FULL LEADERBOARD")
    print("=" * 76)
    print(f"\n  {'Model':<44}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  {'Std':>7}")
    print(f"  {'─'*44}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<44}  {v:>7.2f}%  {'—':>7}  {'—':>7}  {'—':>7}  [prev]")
    print(f"  {'─'*44}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    sorted_res = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    for n, a in sorted_res:
        flag = '  ★ NEW BEST' if a.mean() > prev_best else ''
        print(f"  {'Opt2/3: ' + n:<44}  {a.mean():>7.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    # ── Comparison table ────────────────────────────────────────────────
    print("\n" + "─" * 76)
    print("  OPTIONS 2 & 3 COMPARISON  (Δ vs Uniform Average ref)")
    print("─" * 76)
    base_avg = results.get('Uniform Average (ref)', np.zeros(1)).mean()
    print(f"\n  {'Learner':<32}  {'Avg':>8}  {'ΔAvg':>7}  {'Std':>7}  "
          f"{'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*32}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    for n, a in sorted_res:
        delta = a.mean() - base_avg
        print(f"  {n:<32}  {a.mean():>8.2f}%  {delta:>+7.2f}  "
              f"{a.std():>7.4f}  {a.max():>8.2f}%  {a.min():>8.2f}%")

    # ── Notes ────────────────────────────────────────────────────────────
    print("\n" + "─" * 76)
    print("  DESIGN NOTES")
    print("─" * 76)
    print(f"""
  Per-Student Best Picker (PSBP)  [Option 2]
    • Static per-student model assignment, learned on 295 meta-train days.
    • Recall-based: student s → model argmax_k(recall_k_s).
    • Benefits when different models are systematically better for different
      subsets of students (e.g. M2 for high-frequency, M3 for bursty).
    • Limitation: static — cannot adapt if student behaviour drifts.

  Temporal Gate Hard (TGH)  [Option 3a]
    • W={GATE_WINDOW}-day rolling oracle-hit window; hard-switch to the best model.
    • Responds to regime shifts (e.g. TD-HBB dominates stable periods,
      LSTM+Markov better during structural change).
    • One model at a time — simple and interpretable.

  Temporal Gate Soft (TGS)  [Option 3b]
    • Same W={GATE_WINDOW}-day rolling window; blends models with accuracy-proportional
      weights.  No abrupt switching — smoother behaviour at regime boundaries.
    • Acts like a dynamic Linear Stack that refreshes every day.

  Per-Student Temporal (PST)  [Options 2+3 combined]
    • Per-student, per-model rolling Brier-score window (W={GATE_WINDOW} days).
    • Updated every day for ALL students (not just selected ones).
    • Richest signal: adapts to both student-specific drift AND global regime
      changes.  Requires W days of burn-in before full routing kicks in.
""")
    best_name = sorted_res[0][0]
    best_val  = sorted_res[0][1].mean()
    print("=" * 76)
    print(f"  BEST (Options 2/3): {best_name}  →  {best_val:.2f}%")
    if best_val > prev_best:
        print(f"  ★ NEW OVERALL BEST  (was {prev_best:.2f}%)")
    print("=" * 76)


if __name__ == '__main__':
    main()
