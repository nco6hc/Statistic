"""
Re-test Old Champions at n_groups=10 on Updated Database
==========================================================
Original champions (100 sim days on old DB):
  M1 Hybrid LSTM+Markov (α=0.9)  → 32.17%
  M2 TD-HBB (λ=0.99)              → 32.17%
  M3 Order-3 Markov               → 31.67%

Testing on updated 1308-day database with 100 sim days.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain

EXCEL_PATH = str(_DL_DIR.parent / 'Database.xlsx')
N_S = 55; K_SEL = 6; N_GROUPS = 10; TEMP = 2.0; SEED = 42; SIM_DAYS = 131; SEQ_LEN = 14
H_ALPHA = 0.9; H_NG = 10; H_TEMP = 2.0; H_ITERS = 5; H_REPLAY = 10; H_LR = 5e-4
H_EPOCHS = 50; H_PATIENCE = 15


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_excel(EXCEL_PATH); df = df.sort_values('Days').reset_index(drop=True)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]; T = len(df)
    binary = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64); pos_cnt = np.zeros(N_S, dtype=np.float64)
    for row_idx, row in df.iterrows():
        for pi, col in enumerate(id_cols):
            val = row[col]
            if pd.notna(val):
                sid = int(val) - 1
                if 0 <= sid < N_S:
                    binary[row_idx, sid] = 1.0; pos_sum[sid] += pi + 1; pos_cnt[sid] += 1.0
    avg_pos = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)
    enriched = build_enriched_features(binary)
    return binary, avg_pos, enriched


# ══════════════════════════════════════════════════════════════════════════════
# M2 helpers
# ══════════════════════════════════════════════════════════════════════════════
def _qgroups(vals, n):
    cuts = np.percentile(vals, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(vals, cuts).astype(np.int32)

def _hyperpriors(S, N, grp):
    G = int(grp.max()) + 1; mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2: mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r = rates[m]; mu = float(r.mean()); var = float(r.var()); mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) if var > 1e-10 else 1000.0
    return mu_g, ka_g

def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp] * ka_g[grp] + S
    b = (1.0 - mu_g[grp]) * ka_g[grp] + np.maximum(N - S, 0.0)
    return a, b


# ══════════════════════════════════════════════════════════════════════════════
# M2 Fixed (λ=0.99)
# ══════════════════════════════════════════════════════════════════════════════
class M2Fixed:
    def __init__(self, lam=0.99, n_tier=3, refresh=10):
        self.lam = lam; self.n_tier = n_tier; self.refresh = refresh
        self.S = np.zeros(N_S, dtype=np.float64); self.N = np.zeros(N_S, dtype=np.float64)
        self.grp = np.zeros(N_S, dtype=np.int32); self.mu_g = np.array([K_SEL / N_S]); self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary); w = self.lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        for s in range(N_S):
            self.S[s] = float((binary[:, s] * w).sum()); self.N[s] = float(w.sum())
        self.grp = _qgroups(avg_pos, self.n_tier); self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None); return p / p.sum()

    def update(self, obs):
        self.S = self.lam * self.S + obs.astype(np.float64); self.N = self.lam * self.N + 1.0; self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self): return copy.deepcopy(self.__dict__)
    def load(self, s):
        for k, v in s.items(): setattr(self, k, copy.deepcopy(v))


# ══════════════════════════════════════════════════════════════════════════════
# M2 Adaptive (Narrow & Wide)
# ══════════════════════════════════════════════════════════════════════════════
class M2Adaptive:
    def __init__(self, lam_min=0.950, lam_max=0.995, n_tier=3, refresh=10):
        self.lam_min = lam_min; self.lam_max = lam_max; self.n_tier = n_tier; self.refresh = refresh
        self.lam_s = np.full(N_S, (lam_min + lam_max) / 2.)
        self.S = np.zeros(N_S, dtype=np.float64); self.N = np.zeros(N_S, dtype=np.float64)
        self.grp = np.zeros(N_S, dtype=np.int32); self.mu_g = np.array([K_SEL / N_S]); self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary); stds = binary.std(axis=0); smin, smax = stds.min(), stds.max()
        t = (stds - smin) / (smax - smin) if smax > smin else np.zeros(N_S)
        self.lam_s = self.lam_max - t * (self.lam_max - self.lam_min)
        for s in range(N_S):
            w = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S[s] = float((binary[:, s] * w).sum()); self.N[s] = float(w.sum())
        self.grp = _qgroups(avg_pos, self.n_tier); self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None); return p / p.sum()

    def update(self, obs):
        self.S = self.lam_s * self.S + obs.astype(np.float64); self.N = self.lam_s * self.N + 1.0; self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self): return copy.deepcopy(self.__dict__)
    def load(self, s):
        for k, v in s.items(): setattr(self, k, copy.deepcopy(v))


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════
def sample_groups(probs, n_groups=N_GROUPS, temperature=TEMP):
    rng = np.random.default_rng(SEED); p = np.clip(probs, 1e-9, None) ** (1.0 / temperature); p /= p.sum(); gs = []
    for _ in range(max(n_groups * 10, 200)):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in gs: gs.append(g)
        if len(gs) >= n_groups: break
    if len(gs) < n_groups:
        ranked = np.argsort(probs)[::-1].tolist()
        for start in range(len(ranked) - K_SEL + 1):
            g = sorted(ranked[start:start + K_SEL])
            if g not in gs: gs.append(g)
            if len(gs) >= n_groups: break
    return gs[:n_groups]

def best_score(truth, groups):
    gt_set = set(truth); return max(len(set(g) & gt_set) for g in groups) / K_SEL * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# M1 Hybrid runner
# ══════════════════════════════════════════════════════════════════════════════
def run_m1_hybrid(binary, enriched, split, device):
    train_bin = binary[:split]; train_enr = enriched[:split]; test_bin = binary[split:]
    lstm = EnhancedLSTMPredictor(n_features=enriched.shape[1], n_students=N_S).to(device)
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0); markov.fit(train_bin)
    torch.manual_seed(SEED); np.random.seed(SEED)
    crit = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05); opt = optim.Adam(lstm.parameters(), lr=H_LR)
    
    # Keep LSTM in eval mode always to avoid BatchNorm batch_size=1 error
    lstm.eval()
    
    replay = []; best_loss = float('inf'); wait = 0

    for ep in range(H_EPOCHS):
        epoch_loss = 0.0; batches = 0
        for i in range(SEQ_LEN, len(train_bin)):
            seq = torch.from_numpy(train_enr[i - SEQ_LEN:i]).unsqueeze(0).to(device)
            tgt = torch.from_numpy(train_bin[i]).unsqueeze(0).to(device)
            out = lstm(seq).squeeze(0); loss = crit(out, tgt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0); opt.step()
            epoch_loss += loss.item(); batches += 1
        epoch_loss /= batches
        if epoch_loss < best_loss: best_loss = epoch_loss; wait = 0
        else:
            wait += 1
            if wait >= H_PATIENCE: break

    init_state = copy.deepcopy(lstm.state_dict()); markov_snap = copy.deepcopy(markov.__dict__)
    lstm.eval(); scores = []
    for rep in range(H_ITERS):
        lstm.load_state_dict(copy.deepcopy(init_state)); markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
        for k, v in markov_snap.items(): setattr(markov, k, copy.deepcopy(v))
        torch.manual_seed(SEED + rep); np.random.seed(SEED + rep)
        for i in range(min(SIM_DAYS, len(test_bin))):
            idx = split + i; truth = np.where(test_bin[i] > 0)[0].tolist()
            if len(truth) != K_SEL: continue
            with torch.no_grad():
                seq = torch.from_numpy(enriched[idx - SEQ_LEN:idx]).unsqueeze(0).to(device)
                lstm_p = torch.sigmoid(lstm(seq)).squeeze(0).cpu().numpy()
            
            # Markov needs recent history (at least 2 days)
            recent = binary[max(0, idx-2):idx] if idx >= 2 else binary[:idx]
            if len(recent) < 2:
                recent = np.vstack([np.zeros((1, N_S)), recent])
            markov_p = markov.predict_probabilities(recent)
            
            p_hybrid = H_ALPHA * lstm_p + (1 - H_ALPHA) * markov_p; p_hybrid /= p_hybrid.sum()
            groups = sample_groups(p_hybrid, n_groups=H_NG, temperature=H_TEMP)
            scores.append(best_score(truth, groups))
            
            # Update Markov with proper history
            yesterday = binary[idx-1] if idx >= 1 else np.zeros(N_S)
            day_before = binary[idx-2] if idx >= 2 else np.zeros(N_S)
            markov.update(test_bin[i], yesterday, day_before)
            
            if len(replay) < H_REPLAY: replay.append((enriched[idx - SEQ_LEN:idx], test_bin[i]))
            if i % 5 == 0 and replay:
                for seq_r, tgt_r in replay:
                    seq_t = torch.from_numpy(seq_r).unsqueeze(0).to(device)
                    tgt_t = torch.from_numpy(tgt_r).unsqueeze(0).to(device)
                    out_r = lstm(seq_t).squeeze(0)
                    loss = crit(out_r, tgt_t)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0); opt.step()
    
    print(f'      M1 completed {H_ITERS} iterations on {len(scores)} test days')
    if len(scores) >= 4:
        q = np.percentile(scores, [25, 50, 75])
        print(f'      Quartiles: Q1={q[0]:.1f}%  Q2={q[1]:.1f}%  Q3={q[2]:.1f}%')
    
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# M2 runner
# ══════════════════════════════════════════════════════════════════════════════
def run_m2(binary, avg_pos, split, lam=0.99):
    m = M2Fixed(lam=lam); m.fit(binary[:split], avg_pos); scores = []
    for i in range(min(SIM_DAYS, len(binary) - split)):
        idx = split + i; truth = np.where(binary[idx] > 0)[0].tolist()
        if len(truth) != K_SEL: continue
        probs = m.probs(); groups = sample_groups(probs, n_groups=N_GROUPS, temperature=TEMP)
        scores.append(best_score(truth, groups)); m.update(binary[idx])
    return scores


class M2Fixed:
    def __init__(self, lam=0.99, n_tier=3, refresh=10):
        self.lam = lam; self.n_tier = n_tier; self.refresh = refresh
        self.S = np.zeros(N_S, dtype=np.float64); self.N = np.zeros(N_S, dtype=np.float64)
        self.grp = np.zeros(N_S, dtype=np.int32); self.mu_g = np.array([K_SEL / N_S]); self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary); w = self.lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        for s in range(N_S):
            self.S[s] = float((binary[:, s] * w).sum()); self.N[s] = float(w.sum())
        self.grp = _qgroups(avg_pos, self.n_tier); self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None); return p / p.sum()

    def update(self, obs):
        self.S = self.lam * self.S + obs.astype(np.float64); self.N = self.lam * self.N + 1.0; self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*70)
    print('  Re-test Old Champions at n_groups=10 (Updated Database)')
    print('='*70)

    binary, avg_pos, enriched = load_data()
    T = len(binary); split = int(0.9 * T); test_n = min(SIM_DAYS, T - split)
    print(f'\n  Total days : {T}')
    print(f'  Train      : {split}')
    print(f'  Test (used): {test_n}')
    print(f'  Device     : {device}')
    print(f'  n_groups   : {N_GROUPS}')
    print(f'  Temperature: {TEMP}\n')

    results = []

    # M2 Fixed λ=0.99
    print('  [1/3] Running M2 Fixed (λ=0.99, T=2.0)...')
    s2 = run_m2(binary, avg_pos, split, lam=0.99)
    r2 = {'name': 'M2 TD-HBB (λ=0.99)', 'avg': np.mean(s2), 'best': np.max(s2),
          'worst': np.min(s2), 'std': np.std(s2)}
    results.append(r2)
    print(f'    Avg={r2["avg"]:5.2f}%  Best={r2["best"]:5.2f}%  Worst={r2["worst"]:5.2f}%  Std={r2["std"]:6.2f}')

    # M2+ Narrow
    print('\n  [2/3] Running M2+ Narrow (λ=0.950-0.995, T=1.0)...')
    m_narrow = M2Adaptive(lam_min=0.950, lam_max=0.995); m_narrow.fit(binary[:split], avg_pos)
    s_narrow = []
    for i in range(test_n):
        idx = split + i; truth = np.where(binary[idx] > 0)[0].tolist()
        if len(truth) != K_SEL: continue
        probs = m_narrow.probs(); groups = sample_groups(probs, n_groups=N_GROUPS, temperature=1.0)
        s_narrow.append(best_score(truth, groups)); m_narrow.update(binary[idx])
    r_narrow = {'name': 'M2+ Narrow (λ=0.950-0.995)', 'avg': np.mean(s_narrow), 'best': np.max(s_narrow),
                'worst': np.min(s_narrow), 'std': np.std(s_narrow)}
    results.append(r_narrow)
    print(f'    Avg={r_narrow["avg"]:5.2f}%  Best={r_narrow["best"]:5.2f}%  Worst={r_narrow["worst"]:5.2f}%  Std={r_narrow["std"]:6.2f}')

    # M1 Hybrid LSTM+Markov
    print('\n  [3/3] Running M1 Hybrid (α=0.9, ng=10, T=2.0)...')
    print('    Training LSTM (this takes ~30 seconds)...')
    s1 = run_m1_hybrid(binary, enriched, split, device)
    r1 = {'name': 'M1 Hybrid LSTM+Markov (α=0.9)', 'avg': np.mean(s1), 'best': np.max(s1),
          'worst': np.min(s1), 'std': np.std(s1)}
    results.append(r1)
    print(f'    Avg={r1["avg"]:5.2f}%  Best={r1["best"]:5.2f}%  Worst={r1["worst"]:5.2f}%  Std={r1["std"]:6.2f}')

    # Rank
    results.sort(key=lambda x: x['avg'], reverse=True)
    print('\n' + '='*70)
    print(f'  FINAL RANKING (n_groups={N_GROUPS}, Updated 1308-Day Database)')
    print('='*70)
    print(f'  {"Rank":<5} {"Model":<40} {"Avg":>7} {"Best":>7} {"Worst":>7} {"Std":>7}')
    print('  ' + '-'*68)
    for rank, r in enumerate(results, 1):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
        print(f'  {medal} {rank:<3} {r["name"]:<40} {r["avg"]:6.2f}% {r["best"]:6.2f}% {r["worst"]:6.2f}% {r["std"]:7.2f}')
    print('='*70)
    print(f'\n  Note: Old benchmark showed 32.17% for M1/M2 on smaller database.')
    print(f'  Current results reflect the updated {T}-day dataset.')
    print('='*70)


if __name__ == '__main__': main()
