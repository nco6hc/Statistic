"""
Meta-Learning Ensemble of Top-3 Models
=======================================

Base models (best 3 from leaderboard @ 100-day oracle evaluation):
  M1  Hybrid LSTM+Markov  (FocalLoss, α=0.9, ng=10, T=2.0, iters=5, replay=10)  → 32.17%
  M2  TD-HBB λ=0.99       (call-order grouping, n_tier=3)                        → 32.17%
  M3  Order-3 Markov      (VOMCModel order=3, Laplace α=1.0)                     → 31.67%

Meta-training protocol (time-series safe — no look-ahead):
  Split:      90% train / 10% test  →  1177 train days | 131 test days
  meta_split: 75% of train          →   882 fit days   | 295 meta-training days

  PHASE 1  fit M1, M2, M3 on binary[0:882]
  PHASE 2  stream-predict binary[882:1177]  →  collect (p1_t, p2_t, p3_t, y_t)
           (each model updates online after its prediction, no look-ahead)
  PHASE 3  train 5 meta-learners on the 295×55 meta-training set
  PHASE 4  refit all base models on full train binary[0:1177]
  PHASE 5  simulate 100 test days:
           p1, p2, p3 → meta-learner → combined prob → oracle groups

Meta-learner variants:
  A  Uniform Average    equal weights [1/3, 1/3, 1/3]           (no-learning baseline)
  B  Linear Stack       learned w ∈ Δ²  minimise focal BCE       (scipy optimize)
  C  Per-Student LR     55 × LogisticRegression on [p1_s,p2_s,p3_s]
  D  MLP Stacker        165→128→64→55  FocalLoss  50 epochs  PyTorch
  E  Online Hedge       exponential-weight bandit, updated per test day

Usage:
    cd dl_pipeline
    python -m meta_ensemble.run_meta_top3
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

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55
K_SEL       = 6
TRAIN_RATIO = 0.9
META_FRAC   = 0.75      # fraction of train used to fit base models in phase 1
SIM_DAYS    = 100
SEQ_LEN     = 14
SEED        = 42

# Hybrid (M1) best hyperparams
H_ALPHA     = 0.9
H_NG        = 10
H_TEMP      = 2.0
H_ITERS     = 5
H_REPLAY    = 10
H_LR        = 0.0005
H_EPOCHS    = 50
H_PATIENCE  = 15

# Meta group-sampling params (same as Hybrid oracle)
META_NG     = 10
META_TEMP   = 2.0

# Online Hedge learning rate
HEDGE_ETA   = 0.15

# MLP stacker training
MLP_EPOCHS  = 50
MLP_LR      = 0.001
MLP_BS      = 32

BENCHMARKS = {
    'M1  Hybrid LSTM+Markov (α=0.9)': 32.17,
    'M2  TD-HBB λ=0.99             ': 32.17,
    'M3  Order-3 Markov             ': 31.67,
    'BMA (Order 1+2+3)              ': 31.50,
    'DPM optimised                  ': 30.83,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers copied from hbb_advanced for TD-HBB
# ──────────────────────────────────────────────────────────────────────────────

def _quantile_groups(values: np.ndarray, n: int) -> np.ndarray:
    cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)


def _estimate_hyperpriors(S: np.ndarray, N: np.ndarray,
                           grp: np.ndarray):
    G = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2:
            mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r   = rates[m]; mu = float(r.mean()); var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) \
                  if var > 1e-10 else 1000.0
    return mu_g, ka_g


def _posterior_ab(S, N, grp, mu_g, kappa_g):
    alpha = mu_g[grp] * kappa_g[grp] + S
    beta_ = (1.0 - mu_g[grp]) * kappa_g[grp] + np.maximum(N - S, 0.0)
    return alpha, beta_


# ──────────────────────────────────────────────────────────────────────────────
# Variable-order Markov (Order-3) — copied inline from variable_markov/
# ──────────────────────────────────────────────────────────────────────────────

class VOMCModel:
    """Per-student Variable-order Markov Chain of fixed order k."""

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
        self.counts[:] = 0.0
        self.totals[:] = 0.0
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
            cnt = self.counts[s, c]
            tot = self.totals[s, c]
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
                if obs[s] == 1:
                    self.counts[s, c] += 1.0
        self.base_cnt += obs.astype(np.float64)
        self.n_days   += 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Group sampling helper
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# LSTM training utility
# ──────────────────────────────────────────────────────────────────────────────

class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, enriched, binary, seq_len=SEQ_LEN):
        self.enriched = enriched
        self.binary   = binary
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.enriched) - self.seq_len

    def __getitem__(self, idx):
        X = self.enriched[idx: idx + self.seq_len]
        y = self.binary  [idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


def train_lstm_model(binary_train: np.ndarray, enriched_train: np.ndarray,
                     n_features: int, device: str,
                     tag: str = '') -> tuple:
    """Train EnhancedLSTMPredictor on the given data. Returns (model, state_dict)."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=N_S,
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    split_v = int(len(binary_train) * 0.9)
    tr_ds = HybridDataset(enriched_train[:split_v], binary_train[:split_v])
    vl_ds = HybridDataset(enriched_train[split_v:], binary_train[split_v:])
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False, drop_last=False)

    if len(tr_ld) == 0 or len(vl_ld) == 0:
        return model, model.state_dict()

    optimizer  = optim.Adam(model.parameters(), lr=H_LR, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=5)
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


# ──────────────────────────────────────────────────────────────────────────────
# Base model wrappers — unified predict_probs / update interface
# ──────────────────────────────────────────────────────────────────────────────

class M1_HybridWrapper:
    """LSTM + FactoredMarkovChain with online fine-tuning."""

    def __init__(self, n_features: int, device: str):
        self.device     = device
        self.n_features = n_features
        self.criterion  = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
        self.model      = None
        self.markov     = None
        self.replay: list = []
        self._opt       = None

    def fit(self, binary: np.ndarray, enriched: np.ndarray, tag: str = ''):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.model, state = train_lstm_model(binary, enriched, self.n_features,
                                             self.device, tag)
        self.markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
        self.markov.fit(binary)
        self._opt   = optim.Adam(self.model.parameters(), lr=0.0003)
        self.replay = []

    def predict_probs(self, enriched_seq: np.ndarray,
                      binary_seq: np.ndarray) -> np.ndarray:
        """Return blended (α=0.9) LSTM+Markov probability vector."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(enriched_seq).unsqueeze(0).to(self.device)
            lstm_p = self.model(X_t).cpu().numpy()[0]
        lstm_n = lstm_p / (lstm_p.sum() + 1e-9)
        markov_p = self.markov.predict_probabilities(binary_seq)
        hyb = H_ALPHA * lstm_n + (1.0 - H_ALPHA) * markov_p
        hyb = hyb / (hyb.sum() + 1e-9)
        return hyb.astype(np.float32)

    def update(self, binary_t: np.ndarray, enriched_seq: np.ndarray,
               binary_seq: np.ndarray):
        """Online fine-tune LSTM + update Markov."""
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
        # Markov update needs 3 days context
        n = len(binary_seq)
        if n >= 2:
            self.markov.update(binary_t, binary_seq[-1], binary_seq[-2])


class M2_TDHBBWrapper:
    """Time-Decay Beta-Binomial (λ=0.99, call-order grouping)."""

    def __init__(self, decay: float = 0.99, n_tier: int = 3,
                 refresh_every: int = 10):
        self.decay         = decay
        self.n_tier        = n_tier
        self.refresh_every = refresh_every
        self.S_eff  = np.zeros(N_S, dtype=np.float64)
        self.N_eff  = np.zeros(N_S, dtype=np.float64)
        self.grp    = np.zeros(N_S, dtype=np.int32)
        self.mu_g   = np.array([K_SEL / N_S])
        self.kappa_g = np.array([2.0])
        self.n_obs  = 0.0

    def fit(self, binary: np.ndarray, avg_pos: np.ndarray):
        n   = len(binary)
        lam = self.decay
        w   = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self.S_eff = (binary * w[:, None]).sum(axis=0).astype(np.float64)
        self.N_eff = np.full(N_S, w.sum(), dtype=np.float64)
        self.grp   = _quantile_groups(avg_pos, self.n_tier)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)

    def predict_probs(self) -> np.ndarray:
        a, b   = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                                self.mu_g, self.kappa_g)
        probs  = (a / (a + b)).astype(np.float32)
        probs  = np.clip(probs, 1e-9, None)
        return probs / probs.sum()

    def update(self, obs: np.ndarray):
        self.S_eff  = self.decay * self.S_eff + obs.astype(np.float64)
        self.N_eff  = self.decay * self.N_eff + 1.0
        self.n_obs += 1.0
        if self.refresh_every > 0 and int(self.n_obs) % self.refresh_every == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)


class M3_VOMCWrapper:
    """Order-3 Variable-Order Markov Chain."""

    def __init__(self, order: int = 3):
        self.vomc   = VOMCModel(order=order)
        self.history: np.ndarray | None = None

    def fit(self, binary: np.ndarray):
        self.vomc.fit(binary)
        self.history = binary.copy()

    def predict_probs(self) -> np.ndarray:
        if self.history is None or len(self.history) < self.vomc.order:
            return np.ones(N_S, dtype=np.float32) / N_S
        return self.vomc.predict_probs(self.history[-14:])

    def update(self, obs: np.ndarray):
        if self.history is not None and len(self.history) >= self.vomc.order:
            self.vomc.update(obs, self.history[-self.vomc.order:])
        if self.history is None:
            self.history = obs[np.newaxis, :]
        else:
            self.history = np.vstack([self.history, obs[np.newaxis, :]])


# ──────────────────────────────────────────────────────────────────────────────
# MLP Stacker architecture
# ──────────────────────────────────────────────────────────────────────────────

class MLPStacker(nn.Module):
    def __init__(self, n_base: int = 3, n_students: int = N_S):
        super().__init__()
        inp = n_base * n_students
        self.net = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_students), nn.Sigmoid()
        )

    def forward(self, x):  # (B, 3*N_S)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Meta-learner implementations
# ──────────────────────────────────────────────────────────────────────────────

class MetaUniform:
    """Baseline: equal-weight average."""
    name = 'Uniform Average'

    def fit(self, X_meta, y_meta): pass

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        p = np.mean(p_list, axis=0)
        return p / (p.sum() + 1e-9)

    def update_online(self, *args): pass


class MetaLinearStack:
    """
    Learned scalar weights w1, w2, w3 ∈ Δ² (simplex).
    Minimise: sum_t sum_s FocalBCE(Σk w_k * p_k_ts, y_ts)
    """
    name = 'Linear Stack'

    def __init__(self):
        self.w = np.array([1/3, 1/3, 1/3], dtype=np.float64)

    @staticmethod
    def _focal_bce(p, y, gamma=2.0, eps=1e-7):
        p = np.clip(p, eps, 1 - eps)
        ce = -y * np.log(p) - (1 - y) * np.log(1 - p)
        pt = np.where(y == 1, p, 1 - p)
        return (((1 - pt) ** gamma) * ce).mean()

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        # X_meta: (T, 3, N_S)   y_meta: (T, N_S)
        from scipy.optimize import minimize

        def objective(params):
            w = np.exp(params); w /= w.sum()
            p_combined = (w[0] * X_meta[:, 0] + w[1] * X_meta[:, 1] +
                          w[2] * X_meta[:, 2])  # (T, N_S)
            return self._focal_bce(p_combined, y_meta)

        x0  = np.zeros(3)
        res = minimize(objective, x0, method='BFGS',
                       options={'maxiter': 200, 'disp': False})
        raw = np.exp(res.x); raw /= raw.sum()
        self.w = raw
        print(f"    Linear Stack weights: M1={raw[0]:.3f}  "
              f"M2={raw[1]:.3f}  M3={raw[2]:.3f}")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        p = sum(self.w[i] * p_list[i] for i in range(3))
        return p / (p.sum() + 1e-9)

    def update_online(self, *args): pass


class MetaPerStudentLR:
    """
    55 independent logistic regressions.
    For student s: inputs=[p1_s, p2_s, p3_s]  label=y_s
    """
    name = 'Per-Student LR'

    def __init__(self):
        self.models = []

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        # X_meta: (T, 3, N_S)  →  for student s: X_s = X_meta[:, :, s] (T, 3)
        T = X_meta.shape[0]
        self.models = []
        pos_sum = y_meta.sum()
        for s in range(N_S):
            X_s = X_meta[:, :, s]   # (T, 3)
            y_s = y_meta[:, s]      # (T,)
            # At least one positive and one negative example needed
            if y_s.sum() == 0 or y_s.sum() == T:
                self.models.append(None)
                continue
            lr = LogisticRegression(C=5.0, max_iter=200, solver='lbfgs')
            lr.fit(X_s, y_s)
            self.models.append(lr)
        fitted = sum(1 for m in self.models if m is not None)
        print(f"    Per-Student LR: {fitted}/{N_S} students fitted")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        X = np.column_stack(p_list)   # (3, N_S) → (N_S, 3)
        probs = np.zeros(N_S, dtype=np.float32)
        for s in range(N_S):
            if self.models[s] is not None:
                try:
                    probs[s] = float(self.models[s].predict_proba(
                        X[s:s+1])[0, 1])
                except Exception:
                    probs[s] = float(np.mean([p[s] for p in p_list]))
            else:
                probs[s] = float(np.mean([p[s] for p in p_list]))
        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum()

    def update_online(self, *args): pass


class MetaMLPStacker:
    """Small PyTorch MLP trained with FocalLoss on meta-features."""
    name = 'MLP Stacker'

    def __init__(self, device: str = 'cpu'):
        self.device  = device
        self.mlp     = None
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        # X_meta: (T, 3, N_S)  →  flatten to (T, 165)
        T   = X_meta.shape[0]
        X_f = X_meta.reshape(T, -1).astype(np.float32)
        y_f = y_meta.astype(np.float32)

        self.mlp = MLPStacker(n_base=3, n_students=N_S).to(self.device)
        opt      = optim.Adam(self.mlp.parameters(), lr=MLP_LR, weight_decay=1e-4)
        sched    = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)

        ds  = TensorDataset(torch.FloatTensor(X_f), torch.FloatTensor(y_f))
        ld  = DataLoader(ds, batch_size=min(MLP_BS, T), shuffle=True, drop_last=False)

        best_loss, best_st, no_imp = float('inf'), None, 0
        for ep in range(MLP_EPOCHS):
            self.mlp.train()
            for Xb, yb in ld:
                opt.zero_grad()
                loss = self.criterion(self.mlp(Xb.to(self.device)),
                                      yb.to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.mlp.parameters(), 1.0)
                opt.step()
            # val loss on full meta set
            self.mlp.eval()
            with torch.no_grad():
                vl = self.criterion(
                    self.mlp(torch.FloatTensor(X_f).to(self.device)),
                    torch.FloatTensor(y_f).to(self.device)
                ).item()
            sched.step(vl)
            if vl < best_loss:
                best_loss, best_st, no_imp = vl, self.mlp.state_dict(), 0
            else:
                no_imp += 1
                if no_imp >= 10:
                    print(f"    MLP early stop @ epoch {ep+1}")
                    break
            if (ep + 1) % 10 == 0:
                print(f"    MLP Epoch {ep+1:3d}  val_loss={vl:.5f}")
        self.mlp.load_state_dict(best_st)

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        X = np.concatenate(p_list).astype(np.float32)[np.newaxis, :]
        self.mlp.eval()
        with torch.no_grad():
            p = self.mlp(torch.FloatTensor(X).to(self.device)).cpu().numpy()[0]
        p = np.clip(p, 1e-9, None)
        return p / p.sum()

    def update_online(self, *args): pass


class MetaOnlineHedge:
    """
    Multiplicative weights (Hedge) algorithm.
    After each test day, reward each model by its oracle accuracy,
    then update: w_k ← w_k * exp(η * reward_k) / Z.
    """
    name = 'Online Hedge'

    def __init__(self, n_models: int = 3, eta: float = HEDGE_ETA):
        self.w   = np.ones(n_models) / n_models
        self.eta = eta

    def fit(self, X_meta: np.ndarray, y_meta: np.ndarray):
        # Pre-warm weights on meta-training data day by day
        T = X_meta.shape[0]
        for t in range(T):
            p_list   = [X_meta[t, k] for k in range(3)]
            y_t      = y_meta[t]
            actual_s = set(np.where(y_t == 1)[0].tolist())
            rewards  = np.zeros(3)
            for k in range(3):
                groups = sample_groups(p_list[k])
                rewards[k] = max(len(set(g) & actual_s) for g in groups) / K_SEL
            self._hedge_update(rewards)
        print(f"    Hedge weights after pre-warm: M1={self.w[0]:.3f}  "
              f"M2={self.w[1]:.3f}  M3={self.w[2]:.3f}")

    def predict(self, p_list: list[np.ndarray]) -> np.ndarray:
        p = sum(self.w[i] * p_list[i] for i in range(len(p_list)))
        return p / (p.sum() + 1e-9)

    def update_online(self, p_list: list[np.ndarray], actual_set: set):
        rewards = np.zeros(len(p_list))
        for k, p in enumerate(p_list):
            groups      = sample_groups(p)
            rewards[k]  = max(len(set(g) & actual_set) for g in groups) / K_SEL
        self._hedge_update(rewards)

    def _hedge_update(self, rewards: np.ndarray):
        self.w = self.w * np.exp(self.eta * rewards)
        self.w /= self.w.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: collect meta-training features
# ──────────────────────────────────────────────────────────────────────────────

def collect_meta_features(m1: M1_HybridWrapper, m2: M2_TDHBBWrapper,
                           m3: M3_VOMCWrapper,
                           binary: np.ndarray, enriched: np.ndarray,
                           meta_split: int, split: int) -> tuple:
    """
    Stream base models through [meta_split:split].
    Returns:
        X_meta  (T_meta, 3, N_S)  — base model probability vectors
        y_meta  (T_meta, N_S)     — actual binary labels
    """
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
        # Predictions from each base model
        p1 = m1.predict_probs(enriched_seq, binary_seq)
        p2 = m2.predict_probs()
        p3 = m3.predict_probs()
        X_meta[i, 0] = p1
        X_meta[i, 1] = p2
        X_meta[i, 2] = p3
        y_meta[i]    = binary[actual_idx]
        # Online updates
        m1.update(binary[actual_idx], enriched_seq, binary_seq)
        m2.update(binary[actual_idx])
        m3.update(binary[actual_idx])
    print(f"  Meta-feature collection done in {time.perf_counter()-t0:.1f}s")
    return X_meta, y_meta


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(m1: M1_HybridWrapper, m2: M2_TDHBBWrapper,
                   m3: M3_VOMCWrapper, meta_learner,
                   binary: np.ndarray, enriched: np.ndarray,
                   split: int) -> np.ndarray:
    """
    Simulate SIM_DAYS test days using meta-learner.
    Returns accuracy array (SIM_DAYS,).
    """
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
        # Base model probabilities
        p1 = m1.predict_probs(enriched_seq, binary_seq)
        p2 = m2.predict_probs()
        p3 = m3.predict_probs()
        p_list = [p1, p2, p3]
        # Meta-learner combination
        p_meta  = meta_learner.predict(p_list)
        # Oracle evaluation
        groups  = sample_groups(p_meta, META_NG, META_TEMP)
        actual  = set(np.where(binary[actual_idx] == 1)[0].tolist())
        best    = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        # Online updates
        m1.update(binary[actual_idx], enriched_seq, binary_seq)
        m2.update(binary[actual_idx])
        m3.update(binary[actual_idx])
        meta_learner.update_online(p_list, actual)

    arr = np.array(accs) * 100.0
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 76)
    print("META-LEARNING ENSEMBLE  —  TOP-3 MODELS".center(76))
    print("M1: Hybrid  |  M2: TD-HBB λ=0.99  |  M3: Order-3 Markov".center(76))
    print("=" * 76)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data...")
    binary, avg_pos = load_data()
    print("  Computing enriched features...")
    enriched   = build_enriched_features(binary)
    n_features = enriched.shape[1]
    split      = int(len(binary) * TRAIN_RATIO)
    meta_split = int(split * META_FRAC)
    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test")
    print(f"  meta_split={meta_split}  (base models fit on first {meta_split} days)")
    print(f"  meta-train set: {split - meta_split} days  "
          f"(days {meta_split}–{split-1})\n")

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
    # Keep only non-zero rows (skip early SEQ_LEN warmup days)
    valid  = y_meta.sum(axis=1) > 0
    X_meta = X_meta[valid]
    y_meta = y_meta[valid]
    print(f"  Meta-training samples after filtering: {len(X_meta)}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3 — train meta-learners
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  PHASE 3 — Train meta-learners")
    print("=" * 76)

    meta_learners = [
        MetaUniform(),
        MetaLinearStack(),
        MetaPerStudentLR(),
        MetaMLPStacker(device=device),
        MetaOnlineHedge(n_models=3, eta=HEDGE_ETA),
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
    results = {}
    base_states = {
        'm1': (m1.model.state_dict(), m1.markov),
        'm2': (m2.S_eff.copy(), m2.N_eff.copy(), m2.grp.copy(),
               m2.mu_g.copy(), m2.kappa_g.copy()),
        'm3': (m3.vomc, m3.history.copy()),
    }

    for ml in meta_learners:
        print(f"\n  ─ {ml.name} ─")
        # Restore base models to post-training state (fresh for each meta-learner)
        # M1
        m1_r = M1_HybridWrapper(n_features, device)
        m1_r.model = EnhancedLSTMPredictor(
            n_features=n_features, n_students=N_S,
            hidden_size=128, num_layers=2, dropout=0.3
        ).to(device)
        m1_r.model.load_state_dict(base_states['m1'][0])
        m1_r.markov  = base_states['m1'][1]
        m1_r._opt    = optim.Adam(m1_r.model.parameters(), lr=0.0003)
        m1_r.replay  = []
        m1_r.criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
        # M2
        m2_r = M2_TDHBBWrapper(decay=0.99, n_tier=3)
        m2_r.S_eff, m2_r.N_eff = base_states['m2'][0].copy(), base_states['m2'][1].copy()
        m2_r.grp, m2_r.mu_g, m2_r.kappa_g = (base_states['m2'][2].copy(),
            base_states['m2'][3].copy(), base_states['m2'][4].copy())
        m2_r.n_obs = float(split)
        # M3
        import copy as _copy
        m3_r = M3_VOMCWrapper(order=3)
        m3_r.vomc    = _copy.deepcopy(base_states['m3'][0])
        m3_r.history = base_states['m3'][1].copy()

        t_sim = time.perf_counter()
        arr   = run_simulation(m1_r, m2_r, m3_r, ml, binary, enriched, split)
        elapsed = time.perf_counter() - t_sim
        results[ml.name] = arr
        dist   = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
        nz     = {k: v for k, v in dist.items() if v > 0}
        print(f"\n  ── {ml.name} ──")
        print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
              f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
        print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
        print(f"  Sim time: {elapsed:.2f}s")

    # ══════════════════════════════════════════════════════════════════════
    # LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════
    prev_best = max(BENCHMARKS.values())
    print("\n" + "=" * 76)
    print("  FULL LEADERBOARD")
    print("=" * 76)
    print(f"\n  {'Model':<40}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  {'Std':>7}")
    print(f"  {'─'*40}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<40}  {v:>7.2f}%  {'—':>7}  {'—':>7}  {'—':>7}  [prev]")
    print(f"  {'─'*40}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    for n, a in sorted_results:
        flag = '  ★ NEW BEST' if a.mean() > prev_best else ''
        print(f"  {'Meta: ' + n:<40}  {a.mean():>7.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    # ── Meta-learner comparison ─────────────────────────────────────────
    print("\n" + "─" * 76)
    print("  META-LEARNER COMPARISON  (Δ vs Uniform Average)")
    print("─" * 76)
    base_avg = results.get('Uniform Average', np.zeros(1)).mean()
    print(f"\n  {'Learner':<24}  {'Avg':>8}  {'ΔAvg':>7}  {'Std':>7}  "
          f"{'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*24}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    for n, a in sorted_results:
        delta = a.mean() - base_avg
        print(f"  {n:<24}  {a.mean():>8.2f}%  {delta:>+7.2f}  "
              f"{a.std():>7.4f}  {a.max():>8.2f}%  {a.min():>8.2f}%")

    # ── Design notes ────────────────────────────────────────────────────
    best_name = sorted_results[0][0]
    best_val  = sorted_results[0][1].mean()
    print("\n" + "─" * 76)
    print("  ENSEMBLE DESIGN NOTES")
    print("─" * 76)
    print("""
  Uniform Average
    • Equal 1/3 weight on each model. Strong baseline when models are
      similarly calibrated. Sets the floor for meta-learning benefit.

  Linear Stack
    • Learned scalar weights w ∈ Δ² via focal-BCE minimisation on meta-train.
    • Optimal when one model is consistently stronger across all students.
    • May over-fit if meta-train is small (<200 days).

  Per-Student LR
    • 55 independent logistic regressions on [p1_s, p2_s, p3_s].
    • Allows each student to be handled differently — e.g. M2 better
      for high-frequency students, M3 better for bursty students.
    • Risk: sparse positive labels (~10.9%) means some LRs are poorly calibrated.

  MLP Stacker
    • Non-linear combiner: sees interactions between all 3×55=165 inputs.
    • Can capture group-level signals (e.g. "if p1 and p2 both high for
      student A, but p3 is low, use p1+p2").
    • Most expressive but needs adequate meta-training data (~295 days).

  Online Hedge
    • Exponential weights; reward = oracle accuracy from each standalone model.
    • No offline fitting — adapts in real time to regime changes.
    • Weights reflect RECENT performance, not long-run average.
    • η=0.15: decays slowly (~6-day half-life for dominance shifts).
""")
    print("=" * 76)
    print(f"  BEST META-LEARNER: {best_name}  →  {best_val:.2f}%")
    print("=" * 76)


if __name__ == '__main__':
    main()
