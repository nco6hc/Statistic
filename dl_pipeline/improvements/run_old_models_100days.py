"""
Fair Re-Run: Old Models at SIM_DAYS=100, n_groups=5
=====================================================
Re-runs the old-generation models under identical conditions as the modern
benchmark (n_groups=5, SIM_DAYS=100, every-day evaluation) so all results
are directly comparable.

Old models re-tested:
  • Enhanced LSTM        (BiLSTM+Attention, 220-dim, FocalLoss)
  • LSTM+Markov Hybrid   (Enhanced LSTM + FactoredMarkov, α=0.9)
  • TD-HBB λ=0.99        (Bayesian BB with time-decay, call-order grouping)

Modern benchmark (already measured, shown for comparison):
  • M2+ narrow T=1.0     → 26.33%  ★ current best
  • M2+ narrow T=2.0     → 26.00%
  • M2 global λ=0.99     → 25.50%
  • M1 LSTM+Markov       → 24.83%

Usage:
    cd dl_pipeline
    python improvements/run_old_models_100days.py
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
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

from enhanced_features import build_enriched_features
from enhanced_models    import EnhancedLSTMPredictor, FocalLoss
from markov_chain       import FactoredMarkovChain

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH   = str(_DL_DIR.parent / 'Database.xlsx')
N_S          = 55
K_SEL        = 6
TRAIN_RATIO  = 0.9
SIM_DAYS     = 100
SEQ_LEN      = 14
SEED         = 42
N_GROUPS     = 5      # your tool constraint

# LSTM hyper-params (same as run_best_at_5.py)
H_LR         = 5e-4
H_EPOCHS     = 50
H_PATIENCE   = 15
H_ITERS      = 5
H_REPLAY     = 10
H_ALPHA      = 0.9    # LSTM weight in hybrid blend

# Temperature sweep
TEMP_SWEEP   = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# Known modern results for comparison column (n_groups=5)
MODERN = {
    'M2+ narrow T=1.0 ★': 26.33,
    'M2+ narrow T=2.0':   26.00,
    'M2 global λ=0.99':   25.50,
    'M1 LSTM+Markov':     24.83,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    T       = len(df)
    binary  = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64)
    pos_cnt = np.zeros(N_S, dtype=np.float64)

    for row_idx, row in df.iterrows():
        sel = []
        for pos_idx, col in enumerate(id_cols):
            sid = int(row[col]) - 1
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
                pos_sum[sid] += pos_idx + 1
                pos_cnt[sid] += 1.0
                sel.append(sid)

    avg_pos = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)
    return binary, avg_pos


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB (Bayesian BB with time-decay)
# ─────────────────────────────────────────────────────────────────────────────
def _qgroups(vals, n):
    cuts = np.percentile(vals, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(vals, cuts).astype(np.int32)

def _hyperpriors(S, N, grp):
    G     = int(grp.max()) + 1
    mu_g  = np.zeros(G)
    ka_g  = np.ones(G) * 10.0
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

def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp] * ka_g[grp] + S
    b = (1.0 - mu_g[grp]) * ka_g[grp] + np.maximum(N - S, 0.0)
    return a, b

class TimeDecayBB:
    def __init__(self, lam=0.99, n_tier=3, refresh=10):
        self.lam = lam; self.n_tier = n_tier; self.refresh = refresh
        self.S = np.zeros(N_S); self.N = np.zeros(N_S)
        self.grp = np.zeros(N_S, dtype=np.int32)
        self.mu_g = np.array([K_SEL / N_S]); self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n   = len(binary)
        w   = self.lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self.S   = (binary * w[:, None]).sum(axis=0).astype(np.float64)
        self.N   = np.full(N_S, w.sum(), dtype=np.float64)
        self.grp = _qgroups(avg_pos, self.n_tier)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs):
        self.S   = self.lam * self.S + obs.astype(np.float64)
        self.N   = self.lam * self.N + 1.0
        self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self):  return {k: copy.deepcopy(v) for k, v in self.__dict__.items()}
    def load(self, s): [setattr(self, k, copy.deepcopy(v)) for k, v in s.items()]


# ─────────────────────────────────────────────────────────────────────────────
# LSTM training
# ─────────────────────────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, E, B, sl=SEQ_LEN):
        self.E = E; self.B = B; self.sl = sl
    def __len__(self): return len(self.E) - self.sl
    def __getitem__(self, i):
        return (torch.FloatTensor(self.E[i:i + self.sl]),
                torch.FloatTensor(self.B[i + self.sl]))

def train_lstm(bin_tr, enr_tr, device):
    nf    = enr_tr.shape[1]
    crit  = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model = EnhancedLSTMPredictor(n_features=nf, n_students=N_S,
                                   hidden_size=128, num_layers=2,
                                   dropout=0.3).to(device)
    sv = int(len(bin_tr) * 0.9)
    tl = DataLoader(SeqDS(enr_tr[:sv], bin_tr[:sv]),
                    batch_size=32, shuffle=True, drop_last=True)
    vl = DataLoader(SeqDS(enr_tr[sv:], bin_tr[sv:]),
                    batch_size=32, shuffle=False)
    opt = optim.Adam(model.parameters(), lr=H_LR, weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)
    best, bst, ni = float('inf'), None, 0
    for ep in range(H_EPOCHS):
        model.train()
        for X, y in tl:
            opt.zero_grad()
            loss = crit(model(X.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        model.eval(); v = 0.
        with torch.no_grad():
            for X, y in vl:
                v += crit(model(X.to(device)), y.to(device)).item()
        v /= len(vl); sch.step(v)
        if v < best: best, bst, ni = v, copy.deepcopy(model.state_dict()), 0
        else:
            ni += 1
            if ni >= H_PATIENCE:
                print(f'    [LSTM] Early stop @ epoch {ep+1}  best={best:.5f}')
                break
        if (ep + 1) % 10 == 0:
            print(f'    [LSTM] Epoch {ep+1:3d}  val={v:.5f}')
    print(f'    [LSTM] Done  (best val={best:.5f})')
    model.load_state_dict(bst)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Group sampling & scoring
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs, n_groups, temperature):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1. / temperature)
    p  /= p.sum()
    gs: list = []
    for _ in range(max(n_groups * 10, 200)):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in gs:
            gs.append(g)
        if len(gs) >= n_groups:
            break
    if len(gs) < n_groups:
        ranked = np.argsort(probs)[::-1].tolist()
        for start in range(len(ranked) - K_SEL + 1):
            g = sorted(ranked[start:start + K_SEL])
            if g not in gs:
                gs.append(g)
            if len(gs) >= n_groups:
                break
    return gs[:n_groups]

def best_score(probs, actual_row, n_groups, temperature):
    groups = sample_groups(probs, n_groups, temperature)
    actual = set(np.where(actual_row == 1)[0].tolist())
    return max(len(set(g) & actual) for g in groups) / K_SEL


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────
def run_tdhbb(model, snap, binary, split, n_groups, temperature):
    """Run TD-HBB simulation (no LSTM)."""
    model.load(snap)
    np.random.seed(SEED)
    accs = []
    for i in range(min(SIM_DAYS, len(binary) - split)):
        idx = split + i
        accs.append(best_score(model.probs(), binary[idx], n_groups, temperature))
        model.update(binary[idx])
    return np.array(accs) * 100.

def run_enhanced_lstm(lstm, init_state, binary, enriched, split, device,
                      n_groups, temperature):
    """Run Enhanced LSTM only (no Markov blend)."""
    lstm.load_state_dict(copy.deepcopy(init_state))
    np.random.seed(SEED); torch.manual_seed(SEED)
    crit   = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt    = optim.Adam(lstm.parameters(), lr=0.0003)
    replay: list = []; accs: list = []
    for i in range(min(SIM_DAYS, len(binary) - split)):
        idx = split + i
        if idx < SEQ_LEN:
            accs.append(0.); continue
        esq = enriched[idx - SEQ_LEN:idx]
        lstm.eval()
        with torch.no_grad():
            lp = lstm(torch.FloatTensor(esq).unsqueeze(0).to(device)).cpu().numpy()[0]
        lp = np.clip(lp, 1e-9, None); lp /= lp.sum()
        accs.append(best_score(lp, binary[idx], n_groups, temperature))
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            rec = replay[-H_REPLAY:]
            Xb = torch.FloatTensor(np.array([r[0] for r in rec])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in rec])).to(device)
            lstm.train()
            for _ in range(H_ITERS):
                opt.zero_grad()
                loss = crit(lstm(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), 1.)
                opt.step()
    return np.array(accs) * 100.

def run_hybrid(lstm, init_state, binary, enriched, split, device,
               n_groups, temperature):
    """Run LSTM+Markov Hybrid (α=0.9)."""
    lstm.load_state_dict(copy.deepcopy(init_state))
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])
    np.random.seed(SEED); torch.manual_seed(SEED)
    crit   = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt    = optim.Adam(lstm.parameters(), lr=0.0003)
    replay: list = []; accs: list = []
    for i in range(min(SIM_DAYS, len(binary) - split)):
        idx = split + i
        if idx < SEQ_LEN:
            accs.append(0.); continue
        esq = enriched[idx - SEQ_LEN:idx]
        bsq = binary[idx - SEQ_LEN:idx]
        lstm.eval()
        with torch.no_grad():
            lp = lstm(torch.FloatTensor(esq).unsqueeze(0).to(device)).cpu().numpy()[0]
        lp  = np.clip(lp, 1e-9, None); lp /= lp.sum()
        mp  = markov.predict_probabilities(bsq)
        hyb = H_ALPHA * lp + (1 - H_ALPHA) * mp
        hyb = np.clip(hyb, 1e-9, None); hyb /= hyb.sum()
        accs.append(best_score(hyb, binary[idx], n_groups, temperature))
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay) >= 5:
            rec = replay[-H_REPLAY:]
            Xb = torch.FloatTensor(np.array([r[0] for r in rec])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in rec])).to(device)
            lstm.train()
            for _ in range(H_ITERS):
                opt.zero_grad()
                loss = crit(lstm(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), 1.)
                opt.step()
        if len(bsq) >= 2:
            markov.update(binary[idx], bsq[-1], bsq[-2])
    return np.array(accs) * 100.


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_arr(arr):
    dist = {}
    for k in range(K_SEL + 1):
        target = round(k / K_SEL * 100, 2)
        dist[k] = int(np.sum(np.abs(arr - target) < 0.01))
    dist_str = "  ".join(f"{k}/6={dist[k]}" for k in range(K_SEL + 1) if dist[k] > 0)
    return (f"Avg={arr.mean():.2f}%  Best={arr.max():.2f}%  "
            f"Worst={arr.min():.2f}%  Std={arr.std():.4f}  [{dist_str}]")

def print_row(label, arr, ref=None, flag=''):
    vs = f'  (vs modern best: {arr.mean()-ref:+.2f}%)' if ref is not None else ''
    print(f'  {label:<38}  {fmt_arr(arr)}{vs}{flag}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    W = 88
    print('=' * W)
    print('  FAIR RE-RUN: Old Models at SIM_DAYS=100, n_groups=5'.center(W))
    print('  (identical conditions to modern benchmark)'.center(W))
    print('=' * W)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n  Device: {device}')

    # ── Load & prepare data ───────────────────────────────────────────────
    print('\n  Loading data...')
    binary, avg_pos = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f'  {len(binary)} days  |  {split} train  |  {len(binary)-split} test')
    print('  Building 220-dim enriched features...')
    enriched = build_enriched_features(binary)
    print(f'  Feature shape: {enriched.shape}')

    # ── Fit TD-HBB ───────────────────────────────────────────────────────
    print('\n  Fitting TD-HBB λ=0.99...')
    tdhbb = TimeDecayBB(lam=0.99, n_tier=3, refresh=10)
    tdhbb.fit(binary[:split], avg_pos)
    snap_tdhbb = tdhbb.snap()
    print('    Done.')

    # ── Train LSTM ───────────────────────────────────────────────────────
    print('\n  Training Enhanced LSTM...')
    lstm       = train_lstm(binary[:split], enriched[:split], device)
    init_state = copy.deepcopy(lstm.state_dict())

    print('\n' + '=' * W)
    print('  PART 1 — All old models at n_groups=5, temperature=1.5'.center(W))
    print('  (T=1.5 was the original default for these models)'.center(W))
    print('=' * W)

    results = {}
    MODERN_BEST = 26.33   # M2+ narrow T=1.0

    T_DEFAULT = 1.5

    print(f'\n  {"Model":<38}  Avg      Best     Worst    Std       Dist')
    print(f'  {"-"*38}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*30}')

    # TD-HBB
    arr = run_tdhbb(tdhbb, snap_tdhbb, binary, split, N_GROUPS, T_DEFAULT)
    print_row('TD-HBB λ=0.99', arr, ref=MODERN_BEST)
    results['TD-HBB λ=0.99'] = arr

    # Enhanced LSTM only
    arr = run_enhanced_lstm(lstm, init_state, binary, enriched, split,
                            device, N_GROUPS, T_DEFAULT)
    print_row('Enhanced LSTM (no Markov)', arr, ref=MODERN_BEST)
    results['Enhanced LSTM'] = arr

    # LSTM+Markov Hybrid (α=0.9)
    arr = run_hybrid(lstm, init_state, binary, enriched, split,
                     device, N_GROUPS, T_DEFAULT)
    print_row(f'LSTM+Markov Hybrid (α={H_ALPHA})', arr, ref=MODERN_BEST)
    results[f'LSTM+Markov Hybrid (α={H_ALPHA})'] = arr

    # ── Temperature sweep on best old model ──────────────────────────────
    best_old_name = max(results, key=lambda k: results[k].mean())
    best_old_arr  = results[best_old_name]

    print(f'\n  >>> Best old model at T=1.5:  {best_old_name}  '
          f'({best_old_arr.mean():.2f}%)')

    print(f'\n{"=" * W}')
    print(f'  PART 2 — Temperature sweep on best old model: {best_old_name}'.center(W))
    print(f'  (n_groups=5 fixed)'.center(W))
    print('=' * W)
    print(f'  {"Temp":<8}  Avg      Best     Worst    Std       vs T=1.5')
    print(f'  {"-"*8}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*10}')

    def run_best_old(temp):
        if 'Hybrid' in best_old_name:
            return run_hybrid(lstm, init_state, binary, enriched, split,
                              device, N_GROUPS, temp)
        elif 'Enhanced LSTM' in best_old_name:
            return run_enhanced_lstm(lstm, init_state, binary, enriched, split,
                                     device, N_GROUPS, temp)
        else:
            return run_tdhbb(tdhbb, snap_tdhbb, binary, split, N_GROUPS, temp)

    temp_results = {}
    base_sc = best_old_arr.mean()
    for temp in TEMP_SWEEP:
        arr    = run_best_old(temp)
        vs     = arr.mean() - base_sc
        flag   = '  ***' if arr.mean() > base_sc else ''
        print(f'  {temp:<8}  {arr.mean():>7.2f}%  {arr.max():>7.2f}%  '
              f'{arr.min():>7.2f}%  {arr.std():>8.4f}  {vs:>+.2f}%{flag}')
        temp_results[temp] = arr

    best_temp     = max(temp_results, key=lambda t: temp_results[t].mean())
    best_temp_arr = temp_results[best_temp]

    # ── Final leaderboard ────────────────────────────────────────────────
    print(f'\n{"=" * W}')
    print('  FINAL LEADERBOARD  (n_groups=5, SIM_DAYS=100, all models)'.center(W))
    print('=' * W)
    print(f'\n  {"Rank":<5} {"Model":<40} {"Avg":>8}  {"Best":>8}  {"Worst":>8}  {"Std":>8}  Era')
    print(f'  {"-"*5} {"-"*40} {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*7}')

    # Combine old (best temp) + modern known scores
    all_scores: list[tuple] = []

    # Add old models — winner uses best-temp result, others use T=1.5 result
    for name in results:
        use_arr = temp_results[best_temp] if name == best_old_name else results[name]
        all_scores.append((name, use_arr.mean(), use_arr.max(),
                           use_arr.min(), use_arr.std(), 'Old→100d'))

    # Add modern known scores (no raw array, so use placeholders)
    for name, avg in MODERN.items():
        all_scores.append((name, avg, 50.0, 0.0, None, 'Modern'))

    all_scores.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, avg, best, worst, std, era) in enumerate(all_scores, 1):
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f' {rank} ')
        std_s = f'{std:>8.4f}' if std is not None else '       —'
        star  = '  ← BEST EVER' if rank == 1 else ''
        print(f'  {medal:<5} {name:<40} {avg:>8.2f}%  {best:>7.2f}%  '
              f'{worst:>7.2f}%  {std_s}  {era}{star}')

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"=" * W}')
    print('  SUMMARY'.center(W))
    print('=' * W)
    winner = all_scores[0]
    print(f"""
  Condition:    n_groups=5  (5 group predictions per day)
                SIM_DAYS=100  (100 consecutive test days, every-day eval)

  Overall best: {winner[0]}
                Avg={winner[1]:.2f}%  Best={winner[2]:.2f}%  Worst={winner[3]:.2f}%

  Old model champion (100d):  {best_old_name} @ T={best_temp}
                               Avg={best_temp_arr.mean():.2f}%

  Modern champion (100d):     M2+ narrow T=1.0
                               Avg=26.33%

  Key finding:
  ─────────────────────────────────────────────────────────────────────
  The 100-day re-run gives a definitive answer on whether the old
  models' advantage (measured at 50 days) was real or luck-based.
  ─────────────────────────────────────────────────────────────────────
""")
    print('=' * W)


if __name__ == '__main__':
    main()
