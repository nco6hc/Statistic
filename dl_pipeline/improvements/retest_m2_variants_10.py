"""
Re-test M2 Variants at n_groups=10 on Updated Database
========================================================
Old champion: M2 Fixed λ=0.99 → 32.17% (on older database)
Testing: M2 Fixed + M2+ Narrow + M2+ Wide on updated 1308-day DB.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXCEL_PATH = ROOT / 'Database.xlsx'
N_S = 55; K_SEL = 6; N_GROUPS = 10; TEMP = 2.0; SEED = 42; SIM_DAYS = 131


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


def run_m2(model, binary, avg_pos, split):
    print(f'      Training on {split} days...')
    model.fit(binary[:split], avg_pos)
    
    # Diagnostic: show model state after fit
    if hasattr(model, 'lam_s'):
        print(f'      λ range: [{model.lam_s.min():.3f}, {model.lam_s.max():.3f}]')
        print(f'      λ mean: {model.lam_s.mean():.3f}')
    else:
        print(f'      λ fixed: {model.lam:.3f}')
    
    eff_n = model.N.mean()
    print(f'      Effective history: {eff_n:.1f} days')
    print(f'      Group tiers: {len(np.unique(model.grp))} groups')
    
    probs_init = model.probs()
    top5 = np.argsort(probs_init)[::-1][:5]
    print(f'      Top-5 students: {[int(s)+1 for s in top5]}')
    print(f'      Top-5 probs: {[f"{probs_init[s]*100:.2f}%" for s in top5]}')
    
    scores = []
    print(f'      Simulating {min(SIM_DAYS, len(binary) - split)} test days...')
    for i in range(min(SIM_DAYS, len(binary) - split)):
        idx = split + i; truth = np.where(binary[idx] > 0)[0].tolist()
        if len(truth) != K_SEL: continue
        probs = model.probs(); groups = sample_groups(probs); scores.append(best_score(truth, groups))
        model.update(binary[idx])
    
    # Quartile analysis
    if len(scores) >= 4:
        q = np.percentile(scores, [25, 50, 75])
        print(f'      Quartiles: Q1={q[0]:.1f}%  Q2={q[1]:.1f}%  Q3={q[2]:.1f}%')
    
    return scores


def main():
    df = pd.read_excel(EXCEL_PATH); df = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')]); T = len(df)
    binary = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64); pos_cnt = np.zeros(N_S, dtype=np.float64)
    for row_idx, row in df.iterrows():
        for pos_idx, col in enumerate(id_cols):
            val = row[col]
            if pd.notna(val):
                sid = int(val) - 1
                if 0 <= sid < N_S:
                    binary[row_idx, sid] = 1.0; pos_sum[sid] += pos_idx + 1; pos_cnt[sid] += 1.0
    avg_pos = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)

    split = int(0.9 * T); test_n = min(SIM_DAYS, T - split)
    print('='*70)
    print('  M2 Variants Re-test at n_groups=10 (Updated Database)')
    print('='*70)
    print(f'\n  Total days : {T}')
    print(f'  Train      : {split}')
    print(f'  Test (used): {test_n}')
    print(f'  n_groups   : {N_GROUPS}')
    print(f'  Temperature: {TEMP}\n')

    configs = [
        ('M2 Fixed (λ=0.99, T=2.0) [OLD CHAMPION]', M2Fixed, {'lam': 0.99}),
        ('M2+ Narrow (λ=0.950-0.995, T=1.0)', M2Adaptive, {'lam_min': 0.950, 'lam_max': 0.995}),
        ('M2+ Wide (λ=0.900-0.990, T=2.0)', M2Adaptive, {'lam_min': 0.900, 'lam_max': 0.990}),
        ('M2+ Ultra-Wide (λ=0.850-0.995, T=2.0)', M2Adaptive, {'lam_min': 0.850, 'lam_max': 0.995}),
    ]

    results = []
    for idx, (name, cls, kw) in enumerate(configs, 1):
        print(f'\n  [{idx}/{len(configs)}] {name}')
        print('  ' + '-'*68)
        m = cls(**kw); scores = run_m2(m, binary, avg_pos, split)
        r = {'name': name, 'avg': np.mean(scores), 'best': np.max(scores),
             'worst': np.min(scores), 'std': np.std(scores), 'n': len(scores)}
        results.append(r)
        print(f'      ✅ Result: Avg={r["avg"]:5.2f}%  Best={r["best"]:5.2f}%  Worst={r["worst"]:5.2f}%  Std={r["std"]:6.2f}  (n={r["n"]} days)')

    results.sort(key=lambda x: x['avg'], reverse=True)
    print('\n' + '='*70)
    print(f'  FINAL RANKING (n_groups={N_GROUPS}, Updated {T}-Day Database)')
    print('='*70)
    print(f'  {"Rank":<5} {"Model":<45} {"Avg":>7} {"Best":>7} {"Worst":>7} {"Std":>7}')
    print('  ' + '-'*68)
    for rank, r in enumerate(results, 1):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
        print(f'  {medal} {rank:<3} {r["name"]:<45} {r["avg"]:6.2f}% {r["best"]:6.2f}% {r["worst"]:6.2f}% {r["std"]:7.2f}')
    print('='*70)
    print(f'\n  📝 Note: Old benchmark (32.17%) was on smaller database.')
    print(f'     Current results on {T} days show updated performance.')
    print('='*70)


if __name__ == '__main__': main()
