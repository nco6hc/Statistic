"""
M2+ Wide vs M2+ Narrow at n_groups = 10
=========================================
Benchmark both M2 variants with 10 predictions per day.
Uses first 100 test days to match previous benchmark methodology.
"""
import sys, copy
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXCEL_PATH = ROOT / 'Database.xlsx'
N_S        = 55
K_SEL      = 6
N_GROUPS   = 10
TEMP       = 1.0
SEED       = 42
SIM_DAYS   = 131  # limit to first 131 test days for consistent comparison


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    T       = len(df)
    binary  = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64)
    pos_cnt = np.zeros(N_S, dtype=np.float64)

    for row_idx, row in df.iterrows():
        for pos_idx, col in enumerate(id_cols):
            val = row[col]
            if pd.notna(val):
                sid = int(val) - 1
                if 0 <= sid < N_S:
                    binary[row_idx, sid] = 1.0
                    pos_sum[sid] += pos_idx + 1
                    pos_cnt[sid] += 1.0

    avg_pos = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)
    return binary, avg_pos


# ══════════════════════════════════════════════════════════════════════════════
# M2 helpers
# ══════════════════════════════════════════════════════════════════════════════
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
        r   = rates[m]
        mu  = float(r.mean())
        var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) \
                  if var > 1e-10 else 1000.0
    return mu_g, ka_g

def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp] * ka_g[grp] + S
    b = (1.0 - mu_g[grp]) * ka_g[grp] + np.maximum(N - S, 0.0)
    return a, b


# ══════════════════════════════════════════════════════════════════════════════
# M2 Fixed (Wide)
# ══════════════════════════════════════════════════════════════════════════════
class M2Fixed:
    def __init__(self, lam=0.98, n_tier=3, refresh=10):
        self.lam = lam
        self.n_tier = n_tier
        self.refresh = refresh
        self.S   = np.zeros(N_S, dtype=np.float64)
        self.N   = np.zeros(N_S, dtype=np.float64)
        self.grp = np.zeros(N_S, dtype=np.int32)
        self.mu_g = np.array([K_SEL / N_S])
        self.ka_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary)
        w = self.lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        for s in range(N_S):
            self.S[s] = float((binary[:, s] * w).sum())
            self.N[s] = float(w.sum())
        self.grp = _qgroups(avg_pos, self.n_tier)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p    = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs):
        self.S     = self.lam * self.S + obs.astype(np.float64)
        self.N     = self.lam * self.N + 1.0
        self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self):
        return copy.deepcopy(self.__dict__)

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, copy.deepcopy(v))


# ══════════════════════════════════════════════════════════════════════════════
# M2 Adaptive (Narrow)
# ══════════════════════════════════════════════════════════════════════════════
class M2Adaptive:
    def __init__(self, lam_min=0.950, lam_max=0.995, n_tier=3, refresh=10):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_tier  = n_tier
        self.refresh = refresh
        self.lam_s   = np.full(N_S, (lam_min + lam_max) / 2.)
        self.S       = np.zeros(N_S, dtype=np.float64)
        self.N       = np.zeros(N_S, dtype=np.float64)
        self.grp     = np.zeros(N_S, dtype=np.int32)
        self.mu_g    = np.array([K_SEL / N_S])
        self.ka_g    = np.array([2.0])
        self.n_obs   = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary)
        stds    = binary.std(axis=0)
        smin, smax = stds.min(), stds.max()
        t = (stds - smin) / (smax - smin) if smax > smin else np.zeros(N_S)
        self.lam_s = self.lam_max - t * (self.lam_max - self.lam_min)
        for s in range(N_S):
            w = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S[s] = float((binary[:, s] * w).sum())
            self.N[s] = float(w.sum())
        self.grp   = _qgroups(avg_pos, self.n_tier)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p    = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs):
        self.S     = self.lam_s * self.S + obs.astype(np.float64)
        self.N     = self.lam_s * self.N + 1.0
        self.n_obs += 1.0
        if self.refresh > 0 and int(self.n_obs) % self.refresh == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    def snap(self):
        return copy.deepcopy(self.__dict__)

    def load(self, s):
        for k, v in s.items():
            setattr(self, k, copy.deepcopy(v))


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════
def sample_groups(probs, n_groups=N_GROUPS, temperature=TEMP):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    gs  = []
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


def best_score(ground_truth, candidate_groups):
    gt_set  = set(ground_truth)
    best_ov = max(len(set(g) & gt_set) for g in candidate_groups)
    return best_ov / K_SEL * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════
def run_m2(model_class, name, binary, avg_pos, split_idx, **kwargs):
    train_bin, test_bin = binary[:split_idx], binary[split_idx:]
    m = model_class(**kwargs)
    m.fit(train_bin, avg_pos)

    scores = []
    n_test = min(SIM_DAYS, len(test_bin))  # limit to first SIM_DAYS
    for t in range(n_test):
        truth = np.where(test_bin[t] > 0)[0].tolist()
        if len(truth) != K_SEL:
            continue
        probs  = m.probs()
        groups = sample_groups(probs, n_groups=N_GROUPS, temperature=TEMP)
        sc     = best_score(truth, groups)
        scores.append(sc)
        m.update(test_bin[t])

    return {
        'name'  : name,
        'avg'   : np.mean(scores),
        'best'  : np.max(scores),
        'worst' : np.min(scores),
        'std'   : np.std(scores),
        'n_days': len(scores),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print('  M2+ Wide vs M2+ Narrow at n_groups = 10')
    print('=' * 70)

    binary, avg_pos = load_data()
    total_days = len(binary)
    split_idx  = int(0.9 * total_days)
    test_days  = total_days - split_idx

    print(f'\n  Total days    : {total_days}')
    print(f'  Train         : {split_idx}')
    print(f'  Test (avail)  : {test_days}')
    print(f'  Test (used)   : {min(SIM_DAYS, test_days)}')
    print(f'  Temperature   : {TEMP}')
    print(f'  n_groups      : {N_GROUPS}')
    print(f'  Seed          : {SEED}\n')

    configs = [
        ('M2+ Wide (λ=0.98)', M2Fixed, {'lam': 0.98}),
        ('M2+ Narrow (λ=0.950-0.995)', M2Adaptive, {'lam_min': 0.950, 'lam_max': 0.995}),
    ]

    results = []
    for name, cls, kw in configs:
        print(f'  Running {name}...')
        r = run_m2(cls, name, binary, avg_pos, split_idx, **kw)
        results.append(r)
        print(f'    Avg={r["avg"]:5.2f}%  Best={r["best"]:5.2f}%  Worst={r["worst"]:5.2f}%  Std={r["std"]:6.2f}')

    # Rank
    results.sort(key=lambda x: x['avg'], reverse=True)
    print('\n' + '=' * 70)
    print(f'  FINAL RANKING (n_groups={N_GROUPS}, T={TEMP})')
    print('=' * 70)
    print(f'  {"Rank":<5} {"Model":<35} {"Avg":>7} {"Best":>7} {"Worst":>7} {"Std":>7}')
    print('  ' + '-' * 68)
    for rank, r in enumerate(results, 1):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '
        print(f'  {medal} {rank:<3} {r["name"]:<35} {r["avg"]:6.2f}% {r["best"]:6.2f}% '
              f'{r["worst"]:6.2f}% {r["std"]:7.2f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
