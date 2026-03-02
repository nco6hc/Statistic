"""
Prediction API — M2+ Ultra-Wide (λ=0.850-0.995, T=2.0)
========================================================
Best model: 31.17% avg accuracy, 10 predictions per day
Completely self-contained — no GPU / PyTorch required.

Usage:
  python prediction_api.py predict 1309
  python prediction_api.py correct 1309 5,10,15,20,25,30
  python prediction_api.py status
  python prediction_api.py update
"""

import sys
import json
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
EXCEL_PATH  = ROOT / 'Database.xlsx'
PRED_DIR    = ROOT / 'predictions'
STATE_FILE  = PRED_DIR / 'm2uw_state.pkl'
LOG_FILE    = PRED_DIR / 'm2uw_log.json'

N_S         = 55       # total students
K_SEL       = 6        # selected per day
N_GROUPS    = 10       # predictions per day
TEMPERATURE = 2.0      # sampling temperature
LAM_MIN     = 0.850    # per-student λ lower bound (bursty students)
LAM_MAX     = 0.995    # per-student λ upper bound (stable students)
N_TIER      = 3        # hierarchical prior groups
REFRESH     = 10       # re-estimate hyperpriors every N observations
SEED        = 42


# ─────────────────────────────────────────────────────────────────────────────
# M2+ Ultra-Wide model
# ─────────────────────────────────────────────────────────────────────────────
def _qgroups(vals: np.ndarray, n: int) -> np.ndarray:
    cuts = np.percentile(vals, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(vals, cuts).astype(np.int32)


def _hyperpriors(S: np.ndarray, N: np.ndarray, grp: np.ndarray):
    G    = int(grp.max()) + 1
    mu_g = np.zeros(G)
    ka_g = np.ones(G) * 10.0
    rates = np.where(N > 0, S / N, K_SEL / N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2:
            mu_g[g] = rates.mean(); ka_g[g] = 10.0; continue
        r   = rates[m]
        mu  = float(r.mean())
        var = float(r.var())
        mu_g[g] = mu
        ka_g[g] = float(np.clip(mu * (1 - mu) / var - 1, 2.0, 1000.0)) if var > 1e-10 else 1000.0
    return mu_g, ka_g


def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp] * ka_g[grp] + S
    b = (1.0 - mu_g[grp]) * ka_g[grp] + np.maximum(N - S, 0.0)
    return a, b


class M2UltraWide:
    """
    M2+ Ultra-Wide: per-student adaptive exponential decay (λ=0.850–0.995)
    with hierarchical Bayesian priors.

    High-variance / bursty students → low  λ (short memory, reacts fast)
    Stable / regular students       → high λ (long memory, smooth estimates)
    """

    def __init__(self):
        self.lam_s = np.full(N_S, (LAM_MIN + LAM_MAX) / 2.0)
        self.S     = np.zeros(N_S, dtype=np.float64)   # weighted selection sum
        self.N     = np.zeros(N_S, dtype=np.float64)   # effective observations
        self.grp   = np.zeros(N_S, dtype=np.int32)
        self.mu_g  = np.array([K_SEL / N_S])
        self.ka_g  = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary: np.ndarray, avg_pos: np.ndarray) -> None:
        """Train from scratch on binary history (shape T×55)."""
        n = len(binary)
        stds  = binary.std(axis=0)
        smin, smax = stds.min(), stds.max()
        t = (stds - smin) / (smax - smin) if smax > smin else np.zeros(N_S)
        self.lam_s = LAM_MAX - t * (LAM_MAX - LAM_MIN)
        for s in range(N_S):
            w         = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S[s] = float((binary[:, s] * w).sum())
            self.N[s] = float(w.sum())
        self.grp          = _qgroups(avg_pos, N_TIER)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs        = float(n)

    def probs(self) -> np.ndarray:
        """Return normalised selection probability for each of the 55 students."""
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p    = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs: np.ndarray) -> None:
        """Update model with one day's observation (binary vector length 55)."""
        self.S     = self.lam_s * self.S + obs.astype(np.float64)
        self.N     = self.lam_s * self.N + 1.0
        self.n_obs += 1.0
        if REFRESH > 0 and int(self.n_obs) % REFRESH == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)


# ─────────────────────────────────────────────────────────────────────────────
# Group sampling
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs: np.ndarray, n_groups: int = N_GROUPS,
                  temperature: float = TEMPERATURE) -> list:
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups = []
    for _ in range(max(n_groups * 20, 400)):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if len(groups) < n_groups:
        ranked = np.argsort(probs)[::-1].tolist()
        for start in range(len(ranked) - K_SEL + 1):
            g = sorted(ranked[start: start + K_SEL])
            if g not in groups:
                groups.append(g)
            if len(groups) >= n_groups:
                break
    return groups[:n_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_database():
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
    avg_pos  = np.where(pos_cnt > 0, pos_sum / pos_cnt, 3.5).astype(np.float64)
    day_nums = df['Days'].tolist()
    return binary, avg_pos, day_nums


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_state(model: M2UltraWide) -> None:
    PRED_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, 'wb') as f:
        pickle.dump({k: copy.deepcopy(v) for k, v in model.__dict__.items()}, f)


def load_state():
    if not STATE_FILE.exists():
        return None
    model = M2UltraWide()
    with open(STATE_FILE, 'rb') as f:
        saved = pickle.load(f)
    for k, v in saved.items():
        setattr(model, k, copy.deepcopy(v))
    return model


def load_log() -> dict:
    PRED_DIR.mkdir(exist_ok=True)
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {'predictions': {}, 'corrections': {}}


def save_log(log: dict) -> None:
    PRED_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Core commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_predict(day_number: int) -> None:
    print('=' * 70)
    print(f'  PREDICTION — Day {day_number}')
    print(f'  Model: M2+ Ultra-Wide  (λ={LAM_MIN}–{LAM_MAX}, T={TEMPERATURE}, {N_GROUPS} groups)')
    print('=' * 70)

    log = load_log()

    # Already predicted?
    if str(day_number) in log['predictions']:
        entry  = log['predictions'][str(day_number)]
        status = 'corrected ✅' if str(day_number) in log['corrections'] else 'pending ⏳'
        print(f'\n  ⚠  Day {day_number} already predicted  [{status}]')
        print(f'  Predicted at: {entry["timestamp"]}')
        _print_groups(entry['groups'])
        return

    # Load or build model state
    model = load_state()
    if model is None:
        print('\n  No saved state — building from full database...')
        binary, avg_pos, day_nums = load_database()
        model = M2UltraWide()
        model.fit(binary, avg_pos)
        save_state(model)
        print(f'  ✓ Model trained on {len(binary)} days')
        print(f'  ✓ λ range: [{model.lam_s.min():.4f}, {model.lam_s.max():.4f}]'
              f'  mean={model.lam_s.mean():.4f}')
    else:
        print(f'\n  ✓ Loaded saved state  (observations seen: {int(model.n_obs)})')

    # Generate predictions
    probs  = model.probs()
    groups = sample_groups(probs)

    # Top-10 most likely students
    top10_idx  = np.argsort(probs)[::-1][:10]
    top10_prob = probs[top10_idx]
    print(f'\n  Top-10 most likely students:')
    print(f'  {"Rank":<5} {"Student ID":>10} {"Probability":>12}')
    print(f'  {"─"*5} {"─"*10} {"─"*12}')
    for rank, (sid, prob) in enumerate(zip(top10_idx, top10_prob), 1):
        print(f'  {rank:<5} {sid+1:>10} {prob:>11.2%}')

    print(f'\n  {N_GROUPS} group predictions for Day {day_number}:')
    _print_groups(groups)

    # Save to log (model state is NOT updated until correction is received)
    log['predictions'][str(day_number)] = {
        'day':       day_number,
        'timestamp': datetime.now().isoformat(),
        'groups':    groups,
        'probs':     probs.tolist(),
    }
    save_log(log)
    print(f'\n  💾 Saved. After class, submit the result with:')
    print(f'     python prediction_api.py correct {day_number} 1,2,3,4,5,6')


def cmd_correct(day_number: int, actual_students: list) -> None:
    print('=' * 70)
    print(f'  CORRECTION — Day {day_number}')
    print('=' * 70)

    if len(actual_students) != K_SEL:
        print(f'  ❌ Expected {K_SEL} students, got {len(actual_students)}')
        sys.exit(1)
    if not all(1 <= s <= N_S for s in actual_students):
        print(f'  ❌ Student IDs must be between 1 and {N_S}')
        sys.exit(1)

    log = load_log()

    if str(day_number) not in log['predictions']:
        print(f'  ❌ No prediction found for Day {day_number}.')
        print(f'     Run: python prediction_api.py predict {day_number}  first.')
        sys.exit(1)

    if str(day_number) in log['corrections']:
        print(f'  ⚠  Day {day_number} already has a correction.')
        corr = log['corrections'][str(day_number)]
        print(f'     Actual: {corr["actual"]}  Best overlap: {corr["best_overlap"]}/6')
        return

    pred       = log['predictions'][str(day_number)]
    groups     = pred['groups']
    actual_set = set(actual_students)

    # Score every group
    overlaps = [len(actual_set & set(g)) for g in groups]
    best_ov  = max(overlaps)
    best_pct = best_ov / K_SEL * 100.0
    best_idx = overlaps.index(best_ov)

    print(f'\n  Actual students selected: {sorted(actual_students)}')
    print(f'\n  {"Grp":<5} {"Students":^40} {"Overlap":>8}  {"Acc":>6}')
    print(f'  {"─"*5} {"─"*40} {"─"*8}  {"─"*6}')
    for i, (g, ov) in enumerate(zip(groups, overlaps)):
        star = ' ⭐' if i == best_idx else ''
        print(f'  {i+1:<5} {str([s+1 for s in g]):<40} {ov}/6     {ov/K_SEL:>5.1%}{star}')

    print(f'\n  🎯 Best: Group {best_idx+1}  →  {best_ov}/6 correct  ({best_pct:.1f}%)')

    # Save correction record
    log['corrections'][str(day_number)] = {
        'day':          day_number,
        'timestamp':    datetime.now().isoformat(),
        'actual':       sorted(actual_students),
        'overlaps':     overlaps,
        'best_overlap': best_ov,
        'best_acc':     round(best_pct, 2),
    }
    save_log(log)

    # Online update: feed this day's result into the model
    model = load_state()
    if model is None:
        print('\n  ⚠  No model state — rebuilding from database first...')
        binary, avg_pos, _ = load_database()
        model = M2UltraWide()
        model.fit(binary, avg_pos)

    obs = np.zeros(N_S, dtype=np.float32)
    for sid in actual_students:
        obs[sid - 1] = 1.0
    model.update(obs)
    save_state(model)

    print(f'\n  ✓ Model updated online with Day {day_number} result.')
    print(f'  💾 State saved — ready for the next prediction.')


def cmd_status() -> None:
    log   = load_log()
    preds = log['predictions']
    corrs = log['corrections']

    print('=' * 70)
    print('  STATUS — M2+ Ultra-Wide Prediction Tool')
    print('=' * 70)

    model = load_state()
    if model:
        print(f'\n  Model: M2+ Ultra-Wide  λ=[{LAM_MIN}, {LAM_MAX}]  T={TEMPERATURE}  '
              f'{N_GROUPS} groups/day')
        print(f'  State: {int(model.n_obs)} total observations  |  '
              f'λ range=[{model.lam_s.min():.4f}, {model.lam_s.max():.4f}]')
    else:
        print('\n  Model state: not initialised yet (run predict first)')

    n_pred    = len(preds)
    n_corr    = len(corrs)
    n_pending = n_pred - n_corr

    print(f'\n  Predictions made     : {n_pred}')
    print(f'  Corrections received : {n_corr}')
    print(f'  Awaiting correction  : {n_pending}')

    if n_corr > 0:
        accs = [c['best_acc'] for c in corrs.values()]
        print(f'\n  {"─"*68}')
        print(f'  Accuracy over {n_corr} corrected day(s):')
        print(f'    Average : {np.mean(accs):.2f}%   '
              f'Best : {max(accs):.2f}%   Worst : {min(accs):.2f}%')
        print(f'\n  {"Day":<8} {"Actual students":<34} {"Best":>6}  {"Acc":>6}')
        print(f'  {"─"*8} {"─"*34} {"─"*6}  {"─"*6}')
        for day_str in sorted(corrs, key=int):
            c = corrs[day_str]
            print(f'  {c["day"]:<8} {str(c["actual"]):<34} '
                  f'{c["best_overlap"]}/6     {c["best_acc"]:>5.1f}%')

    if n_pending > 0:
        pending_days = [d for d in sorted(preds, key=int) if d not in corrs]
        print(f'\n  Pending:')
        for d in pending_days:
            print(f'    Day {d}  (predicted {preds[d]["timestamp"][:10]})')

    print('=' * 70)


def cmd_update() -> None:
    print('=' * 70)
    print('  UPDATE — Rebuilding M2+ Ultra-Wide from full database')
    print('=' * 70)

    print(f'\n  Loading {EXCEL_PATH.name}...')
    binary, avg_pos, day_nums = load_database()
    T = len(binary)
    print(f'  ✓ {T} days loaded  (day {day_nums[0]} → day {day_nums[-1]})')

    model = M2UltraWide()
    model.fit(binary, avg_pos)
    save_state(model)

    print(f'\n  ✓ Model rebuilt on all {T} days')
    print(f'  ✓ λ range : [{model.lam_s.min():.4f}, {model.lam_s.max():.4f}]'
          f'   mean={model.lam_s.mean():.4f}')
    print(f'  💾 State saved → {STATE_FILE.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────────────────────────────────────
def _print_groups(groups: list) -> None:
    print(f'\n  {"Grp":<5} {"Students (IDs 1–55)"}')
    print(f'  {"─"*5} {"─"*42}')
    for i, g in enumerate(groups, 1):
        print(f'  {i:<5} {[s+1 for s in g]}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) == 1:
        print('=' * 70)
        print('  Prediction API  —  M2+ Ultra-Wide  (10 predictions/day)')
        print('=' * 70)
        print('\n  Commands:')
        print('    python prediction_api.py predict 1309')
        print('    python prediction_api.py correct 1309 5,10,15,20,25,30')
        print('    python prediction_api.py status')
        print('    python prediction_api.py update       ← rebuild model from DB')
        print()
        return

    parser = argparse.ArgumentParser(description='M2+ Ultra-Wide Prediction Tool')
    parser.add_argument('command', choices=['predict', 'correct', 'status', 'update'])
    parser.add_argument('day',      nargs='?', type=int)
    parser.add_argument('students', nargs='?', type=str)
    args = parser.parse_args()

    if args.command == 'predict':
        if args.day is None:
            print('❌  Usage: python prediction_api.py predict <day>')
            sys.exit(1)
        cmd_predict(args.day)

    elif args.command == 'correct':
        if args.day is None or args.students is None:
            print('❌  Usage: python prediction_api.py correct <day> <id1,id2,id3,id4,id5,id6>')
            sys.exit(1)
        try:
            students = [int(s.strip()) for s in args.students.split(',')]
        except ValueError:
            print('❌  Student IDs must be integers separated by commas')
            sys.exit(1)
        cmd_correct(args.day, students)

    elif args.command == 'status':
        cmd_status()

    elif args.command == 'update':
        cmd_update()


if __name__ == '__main__':
    main()
