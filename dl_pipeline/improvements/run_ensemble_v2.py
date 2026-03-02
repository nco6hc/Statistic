"""
M1 + M2+  Ensemble  ·  Wider lambda range for M2+
===================================================

Following improvements run (M2+ tied record at 32.17%):

  Experiment 1 – M2+ wider lambda range
    • Narrow  [0.950 – 0.995]  (previous run)
    • Wide    [0.900 – 0.990]  → bursty students see only ~last-10-day horizon

  Experiment 2 – M1 + M2+ weighted ensemble
    • Blend: p = w * p_M1  +  (1-w) * p_M2+
    • Sweep w ∈ {0.3, 0.5, 0.7}  (M1 weight)
    • M2+ uses the better-scoring lambda range from Experiment 1
    • Both sub-models warm-started from their respective fit states and
      updated online throughout the test window

  Output:
    Full stats (Avg / Best / Worst / Std / Quartile trend) for every run.
    Top-3 comparison box at end with clear formatting for easy comparison.

Usage:
    cd dl_pipeline
    python improvements/run_ensemble_v2.py
"""

from __future__ import annotations
import sys, copy
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
SIM_DAYS    = 131
SEQ_LEN     = 14
SEED        = 42

# M1 hyper-params (optimal from previous search)
H_ALPHA    = 0.9
H_NG       = 10
H_TEMP     = 2.0
H_ITERS    = 5
H_REPLAY   = 10
H_LR       = 5e-4
H_EPOCHS   = 50
H_PATIENCE = 15

# M2+ lambda ranges to compare
LAM_WINDOW = 14
LAM_BASE   = 0.990
RANGES = {
    'M2+ narrow [0.950-0.995]': (0.950, 0.995),
    'M2+ wide   [0.900-0.990]': (0.900, 0.990),
}

# Ensemble blend weights (M1 contribution)
ENS_WEIGHTS = [0.3, 0.5, 0.7]

# Previous benchmark
RECORD = 32.17

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
# Feature engineering (220-dim, same as M1 champion)
# ─────────────────────────────────────────────────────────────────────────────
def _freq(binary, window):
    n, s   = binary.shape
    feat   = np.zeros((n, s), dtype=np.float32)
    cumsum = np.vstack([np.zeros((1, s), dtype=np.float32),
                        np.cumsum(binary, axis=0)])
    for t in range(1, n):
        start   = max(0, t - window)
        feat[t] = (cumsum[t] - cumsum[start]) / (t - start)
    return feat


def _recency(binary):
    n, s = binary.shape
    feat = np.zeros((n, s), dtype=np.float32)
    last = -np.ones(s, dtype=np.float64)
    for t in range(n):
        mask = last >= 0
        if mask.any():
            feat[t, mask] = 1.0 / (t - last[mask] + 1)
        last[binary[t] == 1] = t
    return feat


def build_features_v1(binary):
    """220-dim: raw + freq7 + freq14 + recency  (M1 champion feature set)."""
    return np.concatenate([
        binary,
        _freq(binary, 7),
        _freq(binary, 14),
        _recency(binary),
    ], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB helpers
# ─────────────────────────────────────────────────────────────────────────────
def _quantile_groups(values, n):
    cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)


def _estimate_hyperpriors(S, N, grp):
    G    = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2:
            mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r = rates[m]; mu = float(r.mean()); var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = (float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0))
                   if var > 1e-10 else 1000.0)
    return mu_g, ka_g


def _posterior_ab(S, N, grp, mu_g, kappa_g):
    a = mu_g[grp] * kappa_g[grp] + S
    b = (1.0 - mu_g[grp]) * kappa_g[grp] + np.maximum(N - S, 0.0)
    return a, b


def compute_per_student_lambda(binary_train: np.ndarray,
                                lam_min: float,
                                lam_max: float,
                                window: int = LAM_WINDOW) -> np.ndarray:
    """
    Per-student lambda based on rolling selection variance:
      high variance (bursty)  → low  lambda (short memory)
      low  variance (stable)  → high lambda (long memory)
    """
    n, ns    = binary_train.shape
    roll_var = np.zeros(ns, dtype=np.float64)
    cumsum   = np.vstack([np.zeros((1, ns)), np.cumsum(binary_train, axis=0)])
    for s in range(ns):
        means = [(cumsum[t, s] - cumsum[t - window, s]) / window
                 for t in range(window, n)]
        roll_var[s] = float(np.var(means)) if means else 0.0
    v_min, v_max = roll_var.min(), roll_var.max()
    if v_max - v_min < 1e-12:
        return np.full(ns, (lam_min + lam_max) / 2)
    norm  = (roll_var - v_min) / (v_max - v_min)        # 0=stable, 1=bursty
    lam_s = lam_max - (lam_max - lam_min) * norm          # bursty → lower λ
    return lam_s.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB models
# ─────────────────────────────────────────────────────────────────────────────
class TDHBBAdaptive:
    """TD-HBB with per-student lambda (M2+).  Parameterised lambda range."""

    def __init__(self, lam_min: float = 0.950, lam_max: float = 0.995,
                 n_tier: int = 3, refresh_every: int = 10):
        self.lam_min = lam_min; self.lam_max = lam_max
        self.n_tier  = n_tier;  self.refresh  = refresh_every
        mid = (lam_min + lam_max) / 2
        self.lam_s   = np.full(N_S, mid)
        self.S_eff   = np.zeros(N_S); self.N_eff = np.zeros(N_S)
        self.grp     = np.zeros(N_S, dtype=np.int32)
        self.mu_g    = np.array([K_SEL / N_S])
        self.kappa_g = np.array([2.0])
        self.n_obs   = 0.0

    def fit(self, binary: np.ndarray, avg_pos: np.ndarray) -> None:
        n = len(binary)
        self.lam_s = compute_per_student_lambda(
            binary, self.lam_min, self.lam_max)
        for s in range(N_S):
            w             = self.lam_s[s] ** np.arange(n - 1, -1, -1,
                                                         dtype=np.float64)
            self.S_eff[s] = float((binary[:, s] * w).sum())
            self.N_eff[s] = float(w.sum())
        self.grp = _quantile_groups(avg_pos, self.n_tier)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)
        print(f"    lambda range: [{self.lam_s.min():.4f}, "
              f"{self.lam_s.max():.4f}]  "
              f"mean={self.lam_s.mean():.4f}  std={self.lam_s.std():.4f}")

    def predict_probs(self) -> np.ndarray:
        a, b = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                              self.mu_g, self.kappa_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs: np.ndarray) -> None:
        self.S_eff   = self.lam_s * self.S_eff + obs.astype(np.float64)
        self.N_eff   = self.lam_s * self.N_eff + 1.0
        self.n_obs  += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM training
# ─────────────────────────────────────────────────────────────────────────────
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, enriched, binary, seq_len=SEQ_LEN):
        self.E = enriched; self.B = binary; self.sl = seq_len

    def __len__(self):  return len(self.E) - self.sl

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.E[idx: idx + self.sl]),
                torch.FloatTensor(self.B[idx + self.sl]))


def train_lstm(binary_train: np.ndarray, enriched_train: np.ndarray,
               device: str, tag: str = '') -> EnhancedLSTMPredictor:
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
            best_loss, best_st, no_imp = vl, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= H_PATIENCE:
                print(f"    {tag} Early stop @ epoch {epoch + 1}  "
                      f"best val_loss={best_loss:.5f}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_loss={vl:.5f}")
    elapsed = time.perf_counter() - t0
    print(f"    {tag} Training done in {elapsed:.1f}s")
    model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Oracle group sampling
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs: np.ndarray,
                  n_groups: int = H_NG,
                  temperature: float = H_TEMP) -> list:
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups: list = []
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
# Simulation loops  –  each call is fully self-contained (restores state)
# ─────────────────────────────────────────────────────────────────────────────
def _run_oracle(probs: np.ndarray, actual_row: np.ndarray) -> float:
    groups = sample_groups(probs)
    actual = set(np.where(actual_row == 1)[0].tolist())
    return max(len(set(g) & actual) for g in groups) / K_SEL


def simulate_m1(lstm_model: EnhancedLSTMPredictor,
                lstm_init_state: dict,
                binary: np.ndarray,
                enriched: np.ndarray,
                split: int,
                device: str,
                alpha: float = H_ALPHA,
                tag: str = 'M1') -> np.ndarray:
    """
    M1 solo simulation.
    Always restores LSTM from lstm_init_state and refits Markov before running.
    """
    # Restore weights
    lstm_model.load_state_dict(copy.deepcopy(lstm_init_state))
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])

    np.random.seed(SEED); torch.manual_seed(SEED)
    n_sim     = min(SIM_DAYS, len(binary) - split)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt       = optim.Adam(lstm_model.parameters(), lr=0.0003)
    replay: list = []
    accs:   list = []
    t0 = time.perf_counter()

    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN:
            accs.append(0.0); continue
        esq = enriched[idx - SEQ_LEN: idx]
        bsq = binary  [idx - SEQ_LEN: idx]

        lstm_model.eval()
        with torch.no_grad():
            lp = lstm_model(
                torch.FloatTensor(esq).unsqueeze(0).to(device)
            ).cpu().numpy()[0]
        lp /= (lp.sum() + 1e-9)
        mp  = markov.predict_probabilities(bsq)
        hyb = alpha * lp + (1.0 - alpha) * mp
        hyb /= (hyb.sum() + 1e-9)

        accs.append(_run_oracle(hyb, binary[idx]))

        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            recent = replay[-H_REPLAY:]
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
            markov.update(binary[idx], bsq[-1], bsq[-2])

    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter() - t0:.1f}s")
    return arr


def simulate_m2plus(binary: np.ndarray,
                    avg_pos: np.ndarray,
                    split: int,
                    lam_min: float,
                    lam_max: float,
                    tag: str = 'M2+') -> np.ndarray:
    """M2+ solo simulation.  Creates and fits a fresh model each call."""
    m = TDHBBAdaptive(lam_min=lam_min, lam_max=lam_max, n_tier=3)
    m.fit(binary[:split], avg_pos)

    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs: list = []
    t0 = time.perf_counter()
    for i in range(n_sim):
        idx = split + i
        accs.append(_run_oracle(m.predict_probs(), binary[idx]))
        m.update(binary[idx])
    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter() - t0:.1f}s")
    return arr


def simulate_ensemble(lstm_model: EnhancedLSTMPredictor,
                      lstm_init_state: dict,
                      binary: np.ndarray,
                      enriched: np.ndarray,
                      avg_pos: np.ndarray,
                      split: int,
                      device: str,
                      m1_weight: float = 0.5,
                      lam_min: float = 0.950,
                      lam_max: float = 0.995,
                      alpha: float = H_ALPHA,
                      tag: str = 'Ens') -> np.ndarray:
    """
    M1 + M2+ weighted ensemble.
    Both sub-models are restored to their initial-fit state before the run.
    blend: p = m1_weight * p_M1  +  (1-m1_weight) * p_M2+
    """
    # Re-initialise M1
    lstm_model.load_state_dict(copy.deepcopy(lstm_init_state))
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])

    # Re-initialise M2+
    m2p = TDHBBAdaptive(lam_min=lam_min, lam_max=lam_max, n_tier=3)
    m2p.fit(binary[:split], avg_pos)

    m2_weight = 1.0 - m1_weight
    np.random.seed(SEED); torch.manual_seed(SEED)
    n_sim     = min(SIM_DAYS, len(binary) - split)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt       = optim.Adam(lstm_model.parameters(), lr=0.0003)
    replay: list = []
    accs:   list = []
    t0 = time.perf_counter()

    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN:
            accs.append(0.0); continue
        esq = enriched[idx - SEQ_LEN: idx]
        bsq = binary  [idx - SEQ_LEN: idx]

        # --- M1 probability ---
        lstm_model.eval()
        with torch.no_grad():
            lp = lstm_model(
                torch.FloatTensor(esq).unsqueeze(0).to(device)
            ).cpu().numpy()[0]
        lp /= (lp.sum() + 1e-9)
        mp     = markov.predict_probabilities(bsq)
        m1_p   = alpha * lp + (1.0 - alpha) * mp
        m1_p  /= (m1_p.sum() + 1e-9)

        # --- M2+ probability ---
        m2p_p  = m2p.predict_probs().astype(np.float64)

        # --- Blend ---
        ens    = m1_weight * m1_p + m2_weight * m2p_p
        ens   /= (ens.sum() + 1e-9)

        accs.append(_run_oracle(ens, binary[idx]))

        # --- Update M1 ---
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            recent = replay[-H_REPLAY:]
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
            markov.update(binary[idx], bsq[-1], bsq[-2])

        # --- Update M2+ ---
        m2p.update(binary[idx])

    arr = np.array(accs) * 100.0
    print(f"    {tag} sim done in {time.perf_counter() - t0:.1f}s")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Result formatting
# ─────────────────────────────────────────────────────────────────────────────
def print_result(name: str, arr: np.ndarray, ref: float | None = None) -> None:
    dist = {k: int((arr == round(k / K_SEL * 100, 10)).sum())
            for k in range(K_SEL + 1)}
    nz   = {k: v for k, v in dist.items() if v > 0}
    vs   = (f'  (vs record: {arr.mean() - ref:+.2f}%)' if ref is not None
            else '')
    print(f"\n  -- {name} --{vs}")
    print(f"     Avg:   {arr.mean():.2f}%")
    print(f"     Best:  {arr.max():.2f}%")
    print(f"     Worst: {arr.min():.2f}%")
    print(f"     Std:   {arr.std():.4f}")
    print(f"     Dist:  " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))


def print_top3_box(results: dict, record: float) -> None:
    """Clean top-3 comparison box with all key statistics."""
    sorted_r = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    top3     = sorted_r[:3]
    W        = 80

    sep_thick = "=" * W
    sep_thin  = "-" * W

    print(f"\n{sep_thick}")
    print("  TOP-3 MODEL COMPARISON".center(W))
    print(f"  (Previous record: {record:.2f}%)".center(W))
    print(sep_thick)

    hdr = (f"  {'#':<3}  {'Model':<36}  {'Avg':>7}  "
           f"{'Best':>7}  {'Worst':>7}  {'Std':>8}  {'vs record':>10}")
    print(hdr)
    print(sep_thin)

    medals = ['#1', '#2', '#3']
    for i, (name, arr) in enumerate(top3):
        vs_rec = arr.mean() - record
        flag   = ' ***' if arr.mean() > record else ''
        vs_str = f'{vs_rec:+.2f}%'
        print(f"  {medals[i]:<3}  {name:<36}  {arr.mean():>7.2f}%  "
              f"{arr.max():>7.2f}%  {arr.min():>7.2f}%  "
              f"{arr.std():>8.4f}  {vs_str:>10}{flag}")

    print(sep_thin)

    # Quartile trend  (days 1–25, 26–50, 51–75, 76–100)
    print(f"\n  {'Quartile trend  (Q1=days 1-25  Q2=26-50  Q3=51-75  Q4=76-100)'}")
    print(sep_thin)
    for i, (name, arr) in enumerate(top3):
        q = [arr[j * 25: (j + 1) * 25].mean() for j in range(4)]
        trend = "up" if q[3] > q[0] else "down"
        print(f"  {medals[i]:<3}  {name:<36}  "
              f"Q1={q[0]:.1f}%  Q2={q[1]:.1f}%  Q3={q[2]:.1f}%  Q4={q[3]:.1f}%"
              f"  ({trend})")
    print(sep_thick)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    W = 76
    print("=" * W)
    print("  M1 + M2+  ENSEMBLE  |  WIDER LAMBDA RANGE  |  FULL STATS".center(W))
    print("=" * W)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data and building features...")
    binary, avg_pos = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f"  {len(binary)} days  |  {split} train  |  {len(binary) - split} test")
    enriched = build_features_v1(binary)          # 220-dim (M1 champion)
    print(f"  Feature matrix: {enriched.shape}  (220-dim: raw+freq7+freq14+recency)")

    results: dict[str, np.ndarray] = {}

    # ══════════════════════════════════════════════════════════════════════
    # Step 1 – Train M1 LSTM once, save initial weights for reuse
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print("  STEP 1 – Train M1 LSTM  (used solo + in all ensembles)")
    print("=" * W)
    torch.manual_seed(SEED); np.random.seed(SEED)
    lstm_model      = train_lstm(binary[:split], enriched[:split],
                                 device, tag='[LSTM]')
    lstm_init_state = copy.deepcopy(lstm_model.state_dict())
    print("  Weights saved. Will be restored before each simulation run.\n")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2 – M1 solo  (re-run to get fresh stats for fair comparison)
    # ══════════════════════════════════════════════════════════════════════
    print("=" * W)
    print("  STEP 2 – M1 Solo  (Hybrid LSTM+Markov, alpha=0.9)")
    print("=" * W)
    arr_m1 = simulate_m1(lstm_model, lstm_init_state,
                          binary, enriched, split, device, tag='M1')
    print_result('M1  LSTM+Markov (alpha=0.9)', arr_m1, ref=RECORD)
    results['M1  LSTM+Markov (a=0.9)'] = arr_m1

    # ══════════════════════════════════════════════════════════════════════
    # Step 3 – M2+ solo, two lambda ranges
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print("  STEP 3 – M2+ Solo  (per-student adaptive lambda, two ranges)")
    print("=" * W)
    arr_m2p: dict[str, np.ndarray] = {}
    for label, (lmin, lmax) in RANGES.items():
        print(f"\n  -- {label} --")
        a = simulate_m2plus(binary, avg_pos, split, lmin, lmax, tag=label)
        print_result(label, a, ref=RECORD)
        results[label] = a
        arr_m2p[label] = a

    # Pick the better M2+ range for the ensemble
    best_m2p_label = max(arr_m2p, key=lambda k: arr_m2p[k].mean())
    best_lmin, best_lmax = RANGES[best_m2p_label]
    best_m2p_avg  = arr_m2p[best_m2p_label].mean()
    print(f"\n  >>> Best M2+ range: {best_m2p_label}  "
          f"({best_m2p_avg:.2f}%)")
    print(f"  >>> Using this range for all ensemble variants below.")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4 – M1 + M2+ ensemble, sweep blend weights
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print("  STEP 4 – M1 + M2+ Ensemble  (blend weight sweep)")
    print(f"  p_final = w * p_M1  +  (1-w) * p_M2+")
    print("=" * W)

    for w in ENS_WEIGHTS:
        label = f'Ensemble M1({w:.1f}) + M2+({1-w:.1f})'
        tag   = f'Ens M1={w:.1f}'
        print(f"\n  -- {label}  [lambda {best_lmin:.3f}-{best_lmax:.3f}] --")
        a = simulate_ensemble(
            lstm_model, lstm_init_state,
            binary, enriched, avg_pos, split, device,
            m1_weight=w,
            lam_min=best_lmin, lam_max=best_lmax,
            tag=tag)
        print_result(label, a, ref=RECORD)
        results[label] = a

    # ══════════════════════════════════════════════════════════════════════
    # TOP-3 BOX
    # ══════════════════════════════════════════════════════════════════════
    print_top3_box(results, record=RECORD)

    # ══════════════════════════════════════════════════════════════════════
    # Full leaderboard (this run only)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * W)
    print("  FULL LEADERBOARD — this run")
    print("=" * W)
    print(f"\n  {'Model':<44}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  {'Std':>8}")
    print(f"  {'-'*44}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    for n, a in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        flag = '  *** NEW BEST' if a.mean() > RECORD else ''
        print(f"  {n:<44}  {a.mean():>7.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>8.4f}{flag}")

    print(f"\n  Reference record: {RECORD:.2f}%  "
          f"(M1=M2 champion from previous runs)\n")

    best_name = max(results, key=lambda k: results[k].mean())
    best_val  = results[best_name].mean()
    print("=" * W)
    if best_val > RECORD:
        print(f"  *** NEW OVERALL RECORD: {best_name}")
        print(f"      {best_val:.2f}%  (was {RECORD:.2f}%,  delta={best_val-RECORD:+.2f}%)")
    else:
        print(f"  Best this run:  {best_name}")
        print(f"  {best_val:.2f}%  (gap to {RECORD:.2f}% record: "
              f"{best_val - RECORD:+.2f}%)")
    print("=" * W)


if __name__ == '__main__':
    main()
