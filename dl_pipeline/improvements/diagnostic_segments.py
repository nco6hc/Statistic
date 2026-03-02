"""
Diagnostic: Test M2+ Narrow on different data segments
========================================================
Check if recent days are harder to predict than older days.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXCEL_PATH = ROOT / 'Database.xlsx'
N_S = 55; K_SEL = 6; N_GROUPS = 10; TEMP = 1.0; SEED = 42


def _qgroups(vals, n):
    cuts = np.percentile(vals, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(vals, cuts).astype(np.int32)

def _hyperpriors(S, N, grp):
    G = int(grp.max()) + 1; mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
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

def run_segment(binary, avg_pos, train_end, test_start, test_end, segment_name):
    m = M2Adaptive(lam_min=0.950, lam_max=0.995)
    m.fit(binary[:train_end], avg_pos)
    scores = []
    for t in range(test_start, test_end):
        truth = np.where(binary[t] > 0)[0].tolist()
        if len(truth) != K_SEL: continue
        probs = m.probs(); groups = sample_groups(probs); sc = best_score(truth, groups)
        scores.append(sc); m.update(binary[t])
    return {'name': segment_name, 'avg': np.mean(scores), 'best': np.max(scores),
            'worst': np.min(scores), 'std': np.std(scores), 'n': len(scores)}

def main():
    df = pd.read_excel(EXCEL_PATH); df = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    T = len(df); binary = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64); pos_cnt = np.zeros(N_S, dtype=np.float64)
    for row_idx, row in df.iterrows():
        for pos_idx, col in enumerate(id_cols):
            val = row[col]
            if pd.notna(val):
                sid = int(val) - 1
                if 0 <= sid < N_S:
                    binary[row_idx, sid] = 1.0; pos_sum[sid] += pos_idx + 1; pos_cnt[sid] += 1.0
    avg_pos = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)

    print('='*70)
    print('  M2+ Narrow Diagnostic: Different Data Segments')
    print('='*70)
    print(f'\n  Total days in database: {T}\n')

    # Standard 90/10 split
    split_90 = int(0.9 * T)
    test_start = split_90; test_end = min(test_start + 100, T)
    print(f'  [Current] Test on newest 100 days ({test_start+1}-{test_end})...')
    r1 = run_segment(binary, avg_pos, split_90, test_start, test_end, 'Newest 100')
    print(f'    Avg={r1["avg"]:.2f}%  Best={r1["best"]:.2f}%  Worst={r1["worst"]:.2f}%  Std={r1["std"]:.2f}')

    # Middle segment (old benchmark likely tested here)
    old_total = 1200; old_split = int(0.9 * old_total)
    test_start2 = old_split; test_end2 = min(test_start2 + 100, T)
    print(f'\n  [Historical] Test on days {test_start2+1}-{test_end2} (if DB was {old_total} days)...')
    r2 = run_segment(binary, avg_pos, old_split, test_start2, test_end2, 'Middle 100 (old era)')
    print(f'    Avg={r2["avg"]:.2f}%  Best={r2["best"]:.2f}%  Worst={r2["worst"]:.2f}%  Std={r2["std"]:.2f}')

    # All 131 test days
    test_start3 = split_90; test_end3 = T
    print(f'\n  [All test] All {test_end3 - test_start3} test days ({test_start3+1}-{test_end3})...')
    r3 = run_segment(binary, avg_pos, split_90, test_start3, test_end3, 'All 131 test days')
    print(f'    Avg={r3["avg"]:.2f}%  Best={r3["best"]:.2f}%  Worst={r3["worst"]:.2f}%  Std={r3["std"]:.2f}')

    print('\n' + '='*70)
    print('  Summary')
    print('='*70)
    for r in [r1, r2, r3]:
        print(f'  {r["name"]:<25}  Avg={r["avg"]:5.2f}%  (n={r["n"]})')
    print('='*70)

if __name__ == '__main__': main()
