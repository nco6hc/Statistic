"""
Long-Run Comparison: M2+ Ultra-Wide vs Ensemble M1(0.3)+M2+(0.7)
==================================================================
Tests both models across:
  1. Rolling 30-day windows on the 131 test days  (both models)
  2. Walk-forward cross-validation on 7 historical segments  (M2+ only – fast)
     to reveal consistency across the entire 1308-day history.

Verdict: which model is safer to use in the long run?
"""
from __future__ import annotations
import sys, copy, time, warnings
warnings.filterwarnings('ignore')
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

from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain    import FactoredMarkovChain

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55
K_SEL       = 6
TRAIN_RATIO = 0.9
SEQ_LEN     = 14
SEED        = 42
WINDOW      = 30          # rolling window size (days)

# M1 hyper-params
H_ALPHA    = 0.9
H_NG       = 10
H_TEMP     = 2.0
H_ITERS    = 5
H_REPLAY   = 10
H_LR       = 5e-4
H_EPOCHS   = 50
H_PATIENCE = 15
LAM_WINDOW = 14

# Best models from previous benchmarks
M2_UW_LAM  = (0.850, 0.995)   # M2+ Ultra-Wide
ENS_W_M1   = 0.3               # Ensemble M1 weight
ENS_LAM    = (0.900, 0.990)    # M2+ Wide range used inside ensemble


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
            val = row[col]
            if pd.notna(val):
                sid = int(val) - 1
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

def build_features(binary):
    return np.concatenate([binary, _freq(binary, 7),
                           _freq(binary, 14), _recency(binary)],
                          axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB helpers
# ─────────────────────────────────────────────────────────────────────────────
def _qgroups(vals, n):
    cuts = np.percentile(vals, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(vals, cuts).astype(np.int32)

def _hyperpriors(S, N, grp):
    G    = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2: mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r = rates[m]; mu = float(r.mean()); var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) if var > 1e-10 else 1000.0
    return mu_g, ka_g

def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp] * ka_g[grp] + S
    b = (1.0 - mu_g[grp]) * ka_g[grp] + np.maximum(N - S, 0.0)
    return a, b

def _per_student_lambda(binary_train, lam_min, lam_max):
    n, ns = binary_train.shape
    roll_var = np.zeros(ns, dtype=np.float64)
    cumsum   = np.vstack([np.zeros((1, ns)), np.cumsum(binary_train, axis=0)])
    for s in range(ns):
        means = [(cumsum[t, s] - cumsum[t - LAM_WINDOW, s]) / LAM_WINDOW
                 for t in range(LAM_WINDOW, n)]
        roll_var[s] = float(np.var(means)) if means else 0.0
    vmin, vmax = roll_var.min(), roll_var.max()
    if vmax - vmin < 1e-12:
        return np.full(ns, (lam_min + lam_max) / 2)
    norm = (roll_var - vmin) / (vmax - vmin)
    return (lam_max - (lam_max - lam_min) * norm).astype(np.float64)


class M2UltraWide:
    """M2+ Ultra-Wide (λ=0.850-0.995, T=2.0) — best standalone model."""
    def __init__(self, lam_min=0.850, lam_max=0.995, n_tier=3, refresh=10, temp=2.0):
        self.lam_min = lam_min; self.lam_max = lam_max
        self.n_tier  = n_tier;  self.refresh  = refresh; self.temp = temp
        mid = (lam_min + lam_max) / 2
        self.lam_s = np.full(N_S, mid)
        self.S = np.zeros(N_S); self.N = np.zeros(N_S)
        self.grp = np.zeros(N_S, dtype=np.int32)
        self.mu_g = np.array([K_SEL / N_S]); self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary)
        stds = binary.std(axis=0); smin, smax = stds.min(), stds.max()
        t = (stds - smin) / (smax - smin) if smax > smin else np.zeros(N_S)
        self.lam_s = self.lam_max - t * (self.lam_max - self.lam_min)
        for s in range(N_S):
            w = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S[s] = float((binary[:, s] * w).sum()); self.N[s] = float(w.sum())
        self.grp = _qgroups(avg_pos, self.n_tier)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None); return p / p.sum()

    def update(self, obs):
        self.S = self.lam_s * self.S + obs.astype(np.float64)
        self.N = self.lam_s * self.N + 1.0; self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self): return copy.deepcopy(self.__dict__)
    def load(self, s):
        for k, v in s.items(): setattr(self, k, copy.deepcopy(v))


# ─────────────────────────────────────────────────────────────────────────────
# Group sampling & scoring
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs, n_groups=H_NG, temperature=H_TEMP):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature); p /= p.sum()
    groups: list = []
    for _ in range(n_groups * 6):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups: groups.append(g)
        if len(groups) >= n_groups: break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SEL:].tolist()))
    return groups[:n_groups]

def best_score(actual_row, groups):
    actual = set(np.where(actual_row == 1)[0].tolist())
    return max(len(set(g) & actual) for g in groups) / K_SEL * 100.0


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

def train_lstm(binary_train, enriched_train, device):
    nf   = enriched_train.shape[1]
    crit = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model = EnhancedLSTMPredictor(n_features=nf, n_students=N_S,
                                   hidden_size=128, num_layers=2, dropout=0.3).to(device)
    sp = int(len(binary_train) * 0.9)
    tr = DataLoader(SeqDataset(enriched_train[:sp], binary_train[:sp]),
                    batch_size=32, shuffle=True, drop_last=True)
    vl = DataLoader(SeqDataset(enriched_train[sp:], binary_train[sp:]),
                    batch_size=32, shuffle=False, drop_last=False)
    if len(tr) == 0 or len(vl) == 0: return model
    opt   = optim.Adam(model.parameters(), lr=H_LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)
    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()
    for epoch in range(H_EPOCHS):
        model.train()
        for X, y in tr:
            opt.zero_grad()
            loss = crit(model(X.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for X, y in vl:
                vloss += crit(model(X.to(device)), y.to(device)).item()
        vloss /= len(vl); sched.step(vloss)
        if vloss < best_loss:
            best_loss, best_st, no_imp = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= H_PATIENCE:
                print(f"    LSTM early stop @ epoch {epoch+1}  val={best_loss:.5f}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    LSTM epoch {epoch+1:3d}  val={vloss:.5f}")
    print(f"    LSTM trained in {time.perf_counter()-t0:.1f}s")
    model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────
def run_m2uw(binary, avg_pos, split, n_days):
    """Run M2+ Ultra-Wide on binary[split : split+n_days]."""
    m = M2UltraWide(); m.fit(binary[:split], avg_pos)
    scores = []
    for i in range(n_days):
        idx = split + i
        if idx >= len(binary): break
        truth = np.where(binary[idx] > 0)[0].tolist()
        if len(truth) != K_SEL: m.update(binary[idx]); continue
        scores.append(best_score(binary[idx], sample_groups(m.probs())))
        m.update(binary[idx])
    return np.array(scores)


def run_ensemble(lstm_model, lstm_init, binary, enriched, avg_pos, split, n_days, device):
    """Run Ensemble M1(0.3)+M2+(0.7) on binary[split : split+n_days]."""
    lstm_model.load_state_dict(copy.deepcopy(lstm_init))
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])

    m2p = M2UltraWide(lam_min=ENS_LAM[0], lam_max=ENS_LAM[1])
    m2p.fit(binary[:split], avg_pos)

    crit   = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt    = optim.Adam(lstm_model.parameters(), lr=0.0003)
    replay: list = []; scores: list = []
    np.random.seed(SEED); torch.manual_seed(SEED)

    for i in range(n_days):
        idx = split + i
        if idx >= len(binary): break
        truth = np.where(binary[idx] > 0)[0].tolist()
        if len(truth) != K_SEL:
            m2p.update(binary[idx]); continue
        if idx < SEQ_LEN: scores.append(0.0); m2p.update(binary[idx]); continue

        esq = enriched[idx - SEQ_LEN: idx]
        bsq = binary  [idx - SEQ_LEN: idx]

        lstm_model.eval()
        with torch.no_grad():
            lp = lstm_model(torch.FloatTensor(esq).unsqueeze(0).to(device)).cpu().numpy()[0]
        lp /= (lp.sum() + 1e-9)
        mp  = markov.predict_probabilities(bsq)
        hyb = H_ALPHA * lp + (1.0 - H_ALPHA) * mp; hyb /= hyb.sum() + 1e-9

        m2_p = m2p.probs().astype(np.float64)
        p_blend = ENS_W_M1 * hyb + (1.0 - ENS_W_M1) * m2_p
        p_blend /= p_blend.sum()

        scores.append(best_score(binary[idx], sample_groups(p_blend)))

        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            recent = replay[-H_REPLAY:]
            Xb = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            lstm_model.train()
            for _ in range(H_ITERS):
                opt.zero_grad()
                loss = crit(lstm_model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                opt.step()
        if len(bsq) >= 2:
            markov.update(binary[idx], bsq[-1], bsq[-2])
        m2p.update(binary[idx])

    return np.array(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def stats(arr):
    if len(arr) == 0: return {'avg': 0, 'best': 0, 'worst': 0, 'std': 0}
    return {'avg': np.mean(arr), 'best': np.max(arr),
            'worst': np.min(arr), 'std': np.std(arr)}

def bar(v, low=27.0, high=34.0, w=20):
    pct = max(0.0, min(1.0, (v - low) / (high - low)))
    filled = int(round(pct * w))
    return '█' * filled + '░' * (w - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 72)
    print('  LONG-RUN COMPARISON: M2+ Ultra-Wide  vs  Ensemble M1(0.3)+M2+(0.7)')
    print('=' * 72)

    print('\n  Loading data...')
    binary, avg_pos = load_data()
    T = len(binary); split = int(TRAIN_RATIO * T)
    test_n = T - split
    enriched = build_features(binary)
    print(f'  Total: {T} days  |  Train: {split}  |  Test: {test_n}')

    # ── PART 1: Walk-forward on ALL data (M2+ only — fast) ────────────────
    print('\n' + '=' * 72)
    print('  PART 1 — Walk-Forward (M2+ Ultra-Wide): 7 historical segments')
    print('  Each segment: 160 days train-from-scratch + 130 days test')
    print('  Purpose: is M2+ stable across different eras of the data?')
    print('=' * 72)

    seg_size  = 130
    min_train = 160
    segments  = []
    start = min_train
    while start + seg_size <= T:
        segments.append((start, min(start + seg_size, T)))
        start += seg_size
    if len(segments) > 7: segments = segments[-7:]  # keep last 7

    print(f'\n  {"Segment":<22}  {"Train":>6}  {"Test":>5}  {"Avg":>7}  {"Std":>6}  Chart (27─34%)')
    print(f'  {"─"*22}  {"─"*6}  {"─"*5}  {"─"*7}  {"─"*6}  {"─"*20}')
    m2uw_history = []
    for (ts, te) in segments:
        sc = run_m2uw(binary, avg_pos, ts, te - ts)
        s  = stats(sc)
        lbl = f'days {ts+1}–{te}'
        print(f'  {lbl:<22}  {ts:>6}  {te-ts:>5}  {s["avg"]:>6.2f}%  {s["std"]:>5.2f}  {bar(s["avg"])}')
        m2uw_history.append(s['avg'])

    trend = '↑ improving' if m2uw_history[-1] > m2uw_history[0] else \
            '↓ declining' if m2uw_history[-1] < m2uw_history[0] else '→ flat'
    print(f'\n  M2+ Ultra-Wide trend across history: {trend}')
    print(f'  Min segment: {min(m2uw_history):.2f}%   Max segment: {max(m2uw_history):.2f}%   '
          f'Range: {max(m2uw_history)-min(m2uw_history):.2f}%')

    # ── PART 2: Rolling windows on 131 test days (both models) ────────────
    print('\n' + '=' * 72)
    print(f'  PART 2 — Rolling {WINDOW}-Day Windows on {test_n} Test Days')
    print('  Both M2+ Ultra-Wide and Ensemble trained on same 1177 days')
    print('=' * 72)

    print(f'\n  Training LSTM for Ensemble (one-time, ~30s)...')
    lstm_model  = train_lstm(binary[:split], enriched[:split], device)
    lstm_init   = copy.deepcopy(lstm_model.state_dict())

    windows = []
    w = WINDOW
    for start in range(0, test_n, w):
        end = min(start + w, test_n)
        if end - start < 5: break
        windows.append((split + start, split + end, start + 1, end))

    print(f'\n  {"Window":<20}  {"Days":>6}  {"M2+UW":>7}  {"Ensemble":>9}  '
          f'{"Leader":>10}  {"Gap":>6}')
    print(f'  {"─"*20}  {"─"*6}  {"─"*7}  {"─"*9}  {"─"*10}  {"─"*6}')

    m2_avgs, ens_avgs = [], []
    for (ws, we, d1, d2) in windows:
        n = we - ws
        sc_m2  = run_m2uw(binary, avg_pos, ws, n)
        sc_ens = run_ensemble(lstm_model, lstm_init, binary, enriched, avg_pos, ws, n, device)

        avg_m2  = np.mean(sc_m2)  if len(sc_m2)  > 0 else 0.0
        avg_ens = np.mean(sc_ens) if len(sc_ens) > 0 else 0.0
        m2_avgs.append(avg_m2); ens_avgs.append(avg_ens)

        leader = 'M2+ UW' if avg_m2 > avg_ens else 'Ensemble'
        gap    = abs(avg_m2 - avg_ens)
        lbl    = f'days {d1}–{d2}'
        print(f'  {lbl:<20}  {n:>6}  {avg_m2:>6.2f}%  {avg_ens:>8.2f}%  '
              f'{leader:>10}  {gap:>5.2f}%')

    m2_wins  = sum(m > e for m, e in zip(m2_avgs, ens_avgs))
    ens_wins = sum(e > m for m, e in zip(m2_avgs, ens_avgs))
    m2_mean  = np.mean(m2_avgs); ens_mean = np.mean(ens_avgs)
    m2_std   = np.std(m2_avgs);  ens_std  = np.std(ens_avgs)

    # ── PART 3: Final verdict ───────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  FINAL VERDICT — Long-Run Performance Summary')
    print('=' * 72)

    print(f'\n  Rolling-window breakdown  ({len(windows)} windows × {WINDOW} days each):')
    print(f'  {"Model":<30}  {"Wins":>5}  {"Avg of Avgs":>12}  {"Std of Avgs":>12}  {"Consistency"}')
    print(f'  {"─"*30}  {"─"*5}  {"─"*12}  {"─"*12}  {"─"*15}')
    for lbl, wins, mean, std in [
        ('M2+ Ultra-Wide (λ=0.850-0.995)', m2_wins,  m2_mean,  m2_std),
        ('Ensemble M1(0.3)+M2+(0.7)',       ens_wins, ens_mean, ens_std),
    ]:
        consistency = 'High' if std < 1.5 else 'Medium' if std < 2.5 else 'Low'
        print(f'  {lbl:<30}  {wins:>5}  {mean:>11.2f}%  {std:>11.2f}%  {consistency}')

    print(f'\n  M2+ Ultra-Wide walk-forward (7 eras):')
    print(f'    Range: {min(m2uw_history):.2f}% – {max(m2uw_history):.2f}%'
          f'   Std across eras: {np.std(m2uw_history):.2f}%')
    print(f'    Consistent across all eras: {"YES ✅" if np.std(m2uw_history) < 2.0 else "NO ⚠️"}')

    print()
    if m2_mean >= ens_mean and np.std(m2_avgs) <= np.std(ens_avgs):
        winner = 'M2+ Ultra-Wide'
        reason = 'higher average AND lower variance — more reliable day-to-day'
    elif ens_mean > m2_mean and np.std(ens_avgs) <= np.std(m2_avgs):
        winner = 'Ensemble M1(0.3)+M2+(0.7)'
        reason = 'higher average AND lower variance — worth the extra complexity'
    elif m2_mean >= ens_mean:
        winner = 'M2+ Ultra-Wide'
        reason = 'higher average — Ensemble variance does not justify added complexity'
    else:
        winner = 'Ensemble M1(0.3)+M2+(0.7)'
        reason = 'higher average over the rolling test windows'

    print(f'  🏆 LONG-RUN WINNER: {winner}')
    print(f'     Reason: {reason}')
    print()
    print('  Practical note:')
    print('  • M2+ Ultra-Wide  : ~0.1s per day, no GPU, self-updating')
    print('  • Ensemble        : ~30s training + ~10s sim, needs retraining periodically')
    print('=' * 72)


if __name__ == '__main__':
    main()
