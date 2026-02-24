"""
n_groups Sweep  —  Finding the Performance Ceiling
====================================================

The oracle's job: given a probability distribution over 55 students,
propose n_groups candidate groups of size 6, then score the BEST one.

With n_groups=10 we are barely sampling the space.
Choose(55,6) = 341,055,180 possible groups.

This sweep tests n_groups in [10, 20, 30, 50, 75, 100, 150, 200]
on the current champion model: M2+ wide lambda [0.90-0.99].

Model is fit once, then each n_groups value re-runs the 100-day simulation
from the same starting state (deterministic seed per step).

Also tests temperature in [1.0, 1.5, 2.0, 3.0] at the best n_groups.

Usage:
    cd dl_pipeline
    python improvements/run_ngroups_sweep.py
"""

from __future__ import annotations
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import time, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH   = str(_DL_DIR.parent / 'Database.xlsx')
N_S          = 55
K_SEL        = 6
TRAIN_RATIO  = 0.9
SIM_DAYS     = 100
SEED         = 42

# Champion M2+ settings
LAM_MIN      = 0.900
LAM_MAX      = 0.990
LAM_WINDOW   = 14
N_TIER       = 3
REFRESH_EVERY = 10

# Sweep ranges
NGROUPS_SWEEP  = [10, 20, 30, 50, 75, 100, 150, 200]
TEMP_SWEEP     = [1.0, 1.5, 2.0, 3.0]
TEMP_BASELINE  = 2.0   # used during n_groups sweep
PREV_RECORD    = 32.17

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
                                lam_min: float = LAM_MIN,
                                lam_max: float = LAM_MAX,
                                window: int    = LAM_WINDOW) -> np.ndarray:
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
    norm  = (roll_var - v_min) / (v_max - v_min)
    lam_s = lam_max - (lam_max - lam_min) * norm
    return lam_s.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# M2+ model  (per-student adaptive lambda)
# ─────────────────────────────────────────────────────────────────────────────
class TDHBBAdaptive:
    def __init__(self):
        self.lam_s   = np.full(N_S, (LAM_MIN + LAM_MAX) / 2)
        self.S_eff   = np.zeros(N_S)
        self.N_eff   = np.zeros(N_S)
        self.grp     = np.zeros(N_S, dtype=np.int32)
        self.mu_g    = np.array([K_SEL / N_S])
        self.kappa_g = np.array([2.0])
        self.n_obs   = 0.0

    def fit(self, binary: np.ndarray, avg_pos: np.ndarray) -> None:
        n = len(binary)
        self.lam_s = compute_per_student_lambda(binary)
        for s in range(N_S):
            w             = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S_eff[s] = float((binary[:, s] * w).sum())
            self.N_eff[s] = float(w.sum())
        self.grp = _quantile_groups(avg_pos, N_TIER)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)

    def get_probs(self) -> np.ndarray:
        a, b = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                              self.mu_g, self.kappa_g)
        p = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs: np.ndarray) -> None:
        self.S_eff   = self.lam_s * self.S_eff + obs.astype(np.float64)
        self.N_eff   = self.lam_s * self.N_eff + 1.0
        self.n_obs  += 1.0
        if REFRESH_EVERY > 0 and int(self.n_obs) % REFRESH_EVERY == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)

    def snapshot(self) -> dict:
        """Return a copy of all mutable state for fast restoration."""
        return {
            'lam_s':   self.lam_s.copy(),
            'S_eff':   self.S_eff.copy(),
            'N_eff':   self.N_eff.copy(),
            'grp':     self.grp.copy(),
            'mu_g':    self.mu_g.copy(),
            'kappa_g': self.kappa_g.copy(),
            'n_obs':   self.n_obs,
        }

    def restore(self, snap: dict) -> None:
        self.lam_s   = snap['lam_s'].copy()
        self.S_eff   = snap['S_eff'].copy()
        self.N_eff   = snap['N_eff'].copy()
        self.grp     = snap['grp'].copy()
        self.mu_g    = snap['mu_g'].copy()
        self.kappa_g = snap['kappa_g'].copy()
        self.n_obs   = snap['n_obs']


# ─────────────────────────────────────────────────────────────────────────────
# Oracle group sampler  (parameterised)
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs: np.ndarray,
                  n_groups: int,
                  temperature: float) -> list:
    """
    Sample n_groups unique candidate groups of size K_SEL.

    Strategy:
      - Tempered softmax: p ∝ probs^(1/T)
      - Draw without replacement until n_groups unique groups found
      - Fall back to deterministic top-k if sampling stalls
    """
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()

    groups: list = []
    # More generous attempt budget for larger n_groups
    budget = max(n_groups * 10, 500)
    for _ in range(budget):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break

    # Pad with deterministic top-ranked if we still need more
    if len(groups) < n_groups:
        ranked = np.argsort(probs)[::-1].tolist()
        # Generate top-k deterministic groups by sliding window
        for start in range(len(ranked) - K_SEL + 1):
            g = sorted(ranked[start: start + K_SEL])
            if g not in groups:
                groups.append(g)
            if len(groups) >= n_groups:
                break

    return groups[:n_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation  (one full 100-day run per (n_groups, temperature) combo)
# ─────────────────────────────────────────────────────────────────────────────
def simulate(model: TDHBBAdaptive,
             snap: dict,
             binary: np.ndarray,
             split: int,
             n_groups: int,
             temperature: float) -> np.ndarray:
    """Restore model from snap, run 100-day simulation, return per-day scores."""
    model.restore(snap)
    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs: list = []

    for i in range(n_sim):
        idx    = split + i
        probs  = model.get_probs()
        groups = sample_groups(probs, n_groups, temperature)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        model.update(binary[idx])

    return np.array(accs) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_arr(arr: np.ndarray) -> str:
    return (f"Avg={arr.mean():.2f}%  Best={arr.max():.2f}%  "
            f"Worst={arr.min():.2f}%  Std={arr.std():.4f}")


def quartiles(arr: np.ndarray) -> str:
    q = [arr[j * 25: (j + 1) * 25].mean() for j in range(4)]
    arrow = '↑' if q[3] > q[0] else '↓'
    return (f"Q1={q[0]:.1f}%  Q2={q[1]:.1f}%  "
            f"Q3={q[2]:.1f}%  Q4={q[3]:.1f}%  {arrow}")


def print_sweep_table(rows: list, title: str, col1: str,
                      record: float) -> None:
    W  = 82
    print(f"\n{'='*W}")
    print(f"  {title}".center(W))
    print(f"{'='*W}")
    hdr = (f"  {col1:<10}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  "
           f"{'Std':>8}  {'vs rec':>8}  {'Quartiles'}")
    print(hdr)
    print(f"  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*36}")
    for label, arr in rows:
        vs  = arr.mean() - record
        flg = ' ***' if arr.mean() > record else ''
        print(f"  {label:<10}  {arr.mean():>7.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>8.4f}  "
              f"{vs:>+8.2f}%  {quartiles(arr)}{flg}")
    print(f"{'='*W}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    W = 82
    print('=' * W)
    print('  n_groups + TEMPERATURE SWEEP  —  Finding the Performance Ceiling'.center(W))
    print(f'  Base model: M2+ wide lambda [{LAM_MIN:.2f}–{LAM_MAX:.3f}]  '
          f'Current record: {PREV_RECORD:.2f}%'.center(W))
    print('=' * W)

    # ── Load & fit ────────────────────────────────────────────────────────
    print('\n  Loading data...')
    binary, avg_pos = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f'  {len(binary)} days  |  {split} train  |  {len(binary) - split} test\n')

    print('  Fitting M2+ champion model on training data...')
    model = TDHBBAdaptive()
    model.fit(binary[:split], avg_pos)
    print(f'  lambda range: [{model.lam_s.min():.4f}, {model.lam_s.max():.4f}]  '
          f'mean={model.lam_s.mean():.4f}')
    # Save snapshot of the post-fit state — reused for every simulation
    snap = model.snapshot()
    print('  Snapshot saved. Each simulation restores from this point.\n')

    # ══════════════════════════════════════════════════════════════════════
    # PART 1 — n_groups sweep  (temperature fixed at 2.0)
    # ══════════════════════════════════════════════════════════════════════
    print('=' * W)
    print(f'  PART 1 — n_groups sweep  (temperature = {TEMP_BASELINE})'.center(W))
    print('=' * W)
    print(f'  {"n_groups":<10}  Avg      Best     Worst    Std        vs rec   Quartiles')
    print(f'  {"-"*10}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*8}  {"-"*36}')

    ng_rows: list = []
    best_ng_val   = -np.inf
    best_ng       = NGROUPS_SWEEP[0]

    for ng in NGROUPS_SWEEP:
        t0  = time.perf_counter()
        arr = simulate(model, snap, binary, split,
                       n_groups=ng, temperature=TEMP_BASELINE)
        elapsed = time.perf_counter() - t0
        ng_rows.append((str(ng), arr))

        vs  = arr.mean() - PREV_RECORD
        flg = ' ***' if arr.mean() > PREV_RECORD else ''
        print(f'  {ng:<10}  {arr.mean():>7.2f}%  {arr.max():>7.2f}%  '
              f'{arr.min():>7.2f}%  {arr.std():>8.4f}  '
              f'{vs:>+8.2f}%  {quartiles(arr)}  [{elapsed:.1f}s]{flg}')

        if arr.mean() > best_ng_val:
            best_ng_val = arr.mean()
            best_ng     = ng

    print(f'\n  >>> Best n_groups = {best_ng}  ({best_ng_val:.2f}%)')

    # ══════════════════════════════════════════════════════════════════════
    # PART 2 — temperature sweep  (n_groups = best_ng)
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*W}')
    print(f'  PART 2 — temperature sweep  (n_groups = {best_ng})'.center(W))
    print('=' * W)
    print(f'  {"temp":<10}  Avg      Best     Worst    Std        vs rec   Quartiles')
    print(f'  {"-"*10}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*8}  {"-"*36}')

    temp_rows: list = []
    best_temp_val   = -np.inf
    best_temp       = TEMP_SWEEP[0]

    for temp in TEMP_SWEEP:
        t0  = time.perf_counter()
        arr = simulate(model, snap, binary, split,
                       n_groups=best_ng, temperature=temp)
        elapsed = time.perf_counter() - t0
        temp_rows.append((str(temp), arr))

        vs  = arr.mean() - PREV_RECORD
        flg = ' ***' if arr.mean() > PREV_RECORD else ''
        print(f'  {temp:<10}  {arr.mean():>7.2f}%  {arr.max():>7.2f}%  '
              f'{arr.min():>7.2f}%  {arr.std():>8.4f}  '
              f'{vs:>+8.2f}%  {quartiles(arr)}  [{elapsed:.1f}s]{flg}')

        if arr.mean() > best_temp_val:
            best_temp_val = arr.mean()
            best_temp     = temp

    print(f'\n  >>> Best temperature = {best_temp}  ({best_temp_val:.2f}%)')

    # ══════════════════════════════════════════════════════════════════════
    # PART 3 — combined best: best_ng × best_temp
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*W}')
    print(f'  PART 3 — Combined best  (n_groups={best_ng}, temp={best_temp})'.center(W))
    print('=' * W)
    arr_best = simulate(model, snap, binary, split,
                        n_groups=best_ng, temperature=best_temp)
    print(f'\n  {fmt_arr(arr_best)}')
    print(f'  {quartiles(arr_best)}')
    dist = {k: int((arr_best == round(k / K_SEL * 100, 10)).sum())
            for k in range(K_SEL + 1)}
    nz   = {k: v for k, v in dist.items() if v > 0}
    print(f'  Dist: ' + '  '.join(f'{k}/6={v}' for k, v in nz.items()))
    vs   = arr_best.mean() - PREV_RECORD
    print(f'  vs previous record ({PREV_RECORD:.2f}%):  {vs:+.2f}%'
          + ('  *** NEW BEST' if arr_best.mean() > PREV_RECORD else ''))

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE — plateau analysis
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*W}')
    print('  SUMMARY — n_groups vs Avg score  (plateau detection)'.center(W))
    print('=' * W)
    print(f'\n  {"n_groups":<12}  {"Avg":>7}  {"Δ vs prev":>10}  '
          f'{"Plateau?":>10}')
    print(f'  {"-"*12}  {"-"*7}  {"-"*10}  {"-"*10}')
    prev_val = None
    for label, arr in ng_rows:
        delta    = (arr.mean() - prev_val) if prev_val is not None else float('nan')
        plateau  = '  ← plateau' if (prev_val is not None and abs(delta) < 0.20) else ''
        delta_s  = f'{delta:+.2f}%' if not np.isnan(delta) else '   —'
        print(f'  {label:<12}  {arr.mean():>7.2f}%  {delta_s:>10}  {plateau}')
        prev_val = arr.mean()

    # Estimate true ceiling
    ng_vals  = [(int(r[0]), r[1].mean()) for r in ng_rows]
    ceiling  = max(v for _, v in ng_vals)
    ceiling_ng = [ng for ng, v in ng_vals if v == ceiling][0]
    print(f'\n  Estimated ceiling (n_groups sweep):  {ceiling:.2f}%  '
          f'@ n_groups={ceiling_ng}')
    print(f'  Model record (n_groups=10):          {PREV_RECORD:.2f}%')
    print(f'  Gain from oracle improvement:        '
          f'{ceiling - PREV_RECORD:+.2f}%')

    new_record = max(arr_best.mean(), ceiling)
    print(f'\n{"="*W}')
    if new_record > PREV_RECORD:
        print(f'  *** NEW RECORD:  {new_record:.2f}%  '
              f'(n_groups={best_ng}, temp={best_temp})')
        print(f'      Δ = {new_record - PREV_RECORD:+.2f}%  over previous {PREV_RECORD:.2f}%')
    else:
        print(f'  No improvement beyond {PREV_RECORD:.2f}% —'
              f' n_groups=10 is already near-saturated for this model.')
    print('=' * W)


if __name__ == '__main__':
    main()
