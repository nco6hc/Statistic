"""
M2+ Narrow Adaptive Predictor — Production API
================================================
Best confirmed model for a 5-predictions-per-day tool.

  Model  : M2+ Narrow Adaptive λ [0.950 – 0.995]
  Temp   : 1.0
  Groups : 5 per day
  Avg    : 26.33%  (measured on 100 test days)

Commands:
    python m2_predictor.py update                           # first-time fit / re-fit
    python m2_predictor.py predict 1310                     # generate today's 5 groups
    python m2_predictor.py correct 1310 6,11,22,33,44,55   # submit actual result
    python m2_predictor.py status                           # view accuracy history
"""
from __future__ import annotations
import sys, copy, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
EXCEL_PATH  = ROOT / 'Database.xlsx'
STATE_DIR   = ROOT / 'predictions'
STATE_FILE  = STATE_DIR / 'm2_narrow_state.pkl'
LOG_FILE    = STATE_DIR / 'm2_narrow_log.json'

# ── Model constants ───────────────────────────────────────────────────────────
N_S         = 55
K_SEL       = 6
N_GROUPS    = 5
TEMPERATURE = 1.0
LAM_MIN     = 0.950
LAM_MAX     = 0.995
N_TIER      = 3
REFRESH     = 10
SEED        = 42


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load Database.xlsx → binary matrix + avg call-order position."""
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    T       = len(df)
    binary  = np.zeros((T, N_S), dtype=np.float32)
    pos_sum = np.zeros(N_S, dtype=np.float64)
    pos_cnt = np.zeros(N_S, dtype=np.float64)
    days    = df['Days'].tolist()

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
    return binary, avg_pos, days


# ══════════════════════════════════════════════════════════════════════════════
# M2+ Narrow Adaptive model
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


class M2NarrowAdaptive:
    """
    Per-student adaptive λ in [0.950, 0.995].
    Bursty (high-variance) students → lower λ (shorter memory).
    Stable students → higher λ (longer memory).
    """

    def __init__(self):
        self.lam_s = np.full(N_S, (LAM_MIN + LAM_MAX) / 2.)
        self.S     = np.zeros(N_S, dtype=np.float64)
        self.N     = np.zeros(N_S, dtype=np.float64)
        self.grp   = np.zeros(N_S, dtype=np.int32)
        self.mu_g  = np.array([K_SEL / N_S])
        self.ka_g  = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary)
        stds    = binary.std(axis=0)
        smin, smax = stds.min(), stds.max()
        t = (stds - smin) / (smax - smin) if smax > smin else np.zeros(N_S)
        self.lam_s = LAM_MAX - t * (LAM_MAX - LAM_MIN)
        for s in range(N_S):
            w = self.lam_s[s] ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S[s] = float((binary[:, s] * w).sum())
            self.N[s] = float(w.sum())
        self.grp          = _qgroups(avg_pos, N_TIER)
        self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)
        self.n_obs        = float(n)

    def probs(self):
        a, b = _post_ab(self.S, self.N, self.grp, self.mu_g, self.ka_g)
        p    = np.clip((a / (a + b)).astype(np.float32), 1e-9, None)
        return p / p.sum()

    def update(self, obs: np.ndarray):
        """Update with one new binary observation vector (0-indexed, shape N_S)."""
        self.S     = self.lam_s * self.S + obs.astype(np.float64)
        self.N     = self.lam_s * self.N + 1.0
        self.n_obs += 1.0
        if REFRESH > 0 and int(self.n_obs) % REFRESH == 0:
            self.mu_g, self.ka_g = _hyperpriors(self.S, self.N, self.grp)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_or_empty(cls) -> 'M2NarrowAdaptive':
        m = cls()
        if STATE_FILE.exists():
            with open(STATE_FILE, 'rb') as f:
                state = pickle.load(f)
            for k, v in state.items():
                setattr(m, k, copy.deepcopy(v))
        return m

    @property
    def is_fitted(self): return self.n_obs > 0


# ══════════════════════════════════════════════════════════════════════════════
# Group sampling
# ══════════════════════════════════════════════════════════════════════════════
def sample_groups(probs, n_groups=N_GROUPS, temperature=TEMPERATURE):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
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


# ══════════════════════════════════════════════════════════════════════════════
# Log helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_log() -> dict:
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'predictions': {}, 'corrections': {}}

def save_log(log: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════════════
def cmd_update():
    """Re-fit model on all data in Database.xlsx."""
    print('=' * 70)
    print('  UPDATE — Re-fit M2+ Narrow Adaptive on all data')
    print('=' * 70)
    print(f'\n  Loading {EXCEL_PATH.name}...')
    binary, avg_pos, days = load_data()
    print(f'  {len(binary)} days  |  Day {days[0]} → {days[-1]}')
    print('\n  Fitting model (this takes ~1 second)...')
    model = M2NarrowAdaptive()
    model.fit(binary, avg_pos)
    model.save()
    print(f'\n  λ range assigned:  [{model.lam_s.min():.3f},  {model.lam_s.max():.3f}]')
    print(f'  State saved  →  {STATE_FILE}')
    print(f'\n  ✅ Model ready. You can now run:')
    print(f'     python m2_predictor.py predict DAY')
    print('=' * 70)


def cmd_predict(day: int):
    """Generate 5 candidate groups for the given day."""
    print('=' * 70)
    print(f'  PREDICT — Day {day}')
    print(f'  Model: M2+ Narrow Adaptive λ [{LAM_MIN}-{LAM_MAX}]  T={TEMPERATURE}')
    print('=' * 70)

    model = M2NarrowAdaptive.load_or_empty()
    if not model.is_fitted:
        print('\n  ❌ Model not fitted yet.')
        print('     Run first:  python m2_predictor.py update')
        sys.exit(1)

    probs  = model.probs()
    groups = sample_groups(probs)

    # ── Display groups ────────────────────────────────────────────────────
    print(f'\n  📋 Day {day} — {N_GROUPS} Candidate Groups:\n')
    print(f'  {"Group":<8}  {"Students (1-based)":<34}  Confidence')
    print(f'  {"─"*8}  {"─"*34}  {"─"*20}')
    for i, g in enumerate(groups, 1):
        students = [s + 1 for s in g]
        conf     = sum(probs[s] for s in g) * 100
        bar      = '█' * max(1, int(conf * 2.5))
        print(f'  Group {i:<2}   {str(students):<34}  {conf:5.1f}%  {bar}')

    # ── Top 10 individual probabilities ───────────────────────────────────
    top10 = np.argsort(probs)[::-1][:10]
    print(f'\n  📊 Top 10 students by individual probability:\n')
    print(f'  {"Rank":<5} {"Student":<10} {"Prob":>6}  Bar')
    print(f'  {"─"*5} {"─"*10} {"─"*6}  {"─"*20}')
    for rank, s in enumerate(top10, 1):
        bar = '█' * max(1, int(probs[s] * 250))
        print(f'  {rank:<5} Student {s+1:<4}  {probs[s]*100:5.2f}%  {bar}')

    # ── Save to log ───────────────────────────────────────────────────────
    log = load_log()
    log['predictions'][str(day)] = {
        'groups'   : [[s + 1 for s in g] for g in groups],
        'top_probs': {str(s + 1): float(probs[s]) for s in top10},
    }
    save_log(log)

    print(f'\n  ✅ Saved. After class, submit actual result:')
    print(f'     python m2_predictor.py correct {day} S1,S2,S3,S4,S5,S6')
    print('=' * 70)


def cmd_correct(day: int, actual_ids: list):
    """Submit actual result and update model."""
    print('=' * 70)
    print(f'  CORRECT — Day {day}')
    print('=' * 70)

    if len(actual_ids) != K_SEL:
        print(f'  ❌ Need exactly {K_SEL} student IDs, got {len(actual_ids)}.')
        sys.exit(1)

    log = load_log()

    # ── Accuracy vs saved prediction ──────────────────────────────────────
    best_overlap = 0
    accuracy     = 0.0
    if str(day) in log['predictions']:
        saved_groups = log['predictions'][str(day)]['groups']
        actual_set   = set(actual_ids)
        overlaps     = [len(set(g) & actual_set) for g in saved_groups]
        best_overlap = max(overlaps)
        best_group   = saved_groups[overlaps.index(best_overlap)]
        accuracy     = best_overlap / K_SEL * 100

        grades = {0: '❌ No match', 1: '🟡 Weak (1/6)',
                  2: '🟡 Moderate (2/6)', 3: '🟢 Good (3/6 = 50%)',
                  4: '🟢 Strong (4/6 = 67%)', 5: '🔵 Excellent (5/6)',
                  6: '🏆 Perfect! (6/6)'}

        print(f'\n  Actual result :  {sorted(actual_ids)}')
        print(f'  Best predicted:  {best_group}')
        print(f'  Overlap       :  {best_overlap}/{K_SEL}  ({accuracy:.1f}%)')
        print(f'  Grade         :  {grades.get(best_overlap, "?")}')

        # Show all group overlaps
        print(f'\n  All groups comparison:')
        for i, (g, ov) in enumerate(zip(saved_groups, overlaps), 1):
            marker = '  ← best' if ov == best_overlap and g == best_group else ''
            print(f'    Group {i}: {g}  →  {ov}/6{marker}')
    else:
        print(f'\n  ⚠️  No saved prediction for Day {day}.')
        print(f'       Model will still be updated with actual result.')
        print(f'  Actual: {sorted(actual_ids)}')

    # ── Update model state ────────────────────────────────────────────────
    model = M2NarrowAdaptive.load_or_empty()
    if not model.is_fitted:
        print('\n  ⚠️  Model not fitted. Run:  python m2_predictor.py update')
    else:
        obs = np.zeros(N_S, dtype=np.float32)
        for sid in actual_ids:
            if 1 <= sid <= N_S:
                obs[sid - 1] = 1.0
        model.update(obs)
        model.save()
        print(f'\n  ✅ Model updated  (total obs: {int(model.n_obs)})')

    # ── Save correction to log ────────────────────────────────────────────
    log['corrections'][str(day)] = {
        'actual'      : sorted(actual_ids),
        'best_overlap': best_overlap,
        'accuracy'    : accuracy,
    }
    save_log(log)
    print('=' * 70)


def cmd_status():
    """Show accuracy history and model info."""
    print('=' * 70)
    print('  STATUS — M2+ Narrow Adaptive Predictor')
    print('=' * 70)

    # Model info
    model = M2NarrowAdaptive.load_or_empty()
    if model.is_fitted:
        probs = model.probs()
        top5  = [int(s) + 1 for s in np.argsort(probs)[::-1][:5]]
        print(f'\n  Model state   : ✅ Fitted  (observations: {int(model.n_obs)})')
        print(f'  λ range       : [{model.lam_s.min():.3f},  {model.lam_s.max():.3f}]')
        print(f'  Top-5 students: {top5}')
    else:
        print('\n  Model state   : ❌ Not fitted')
        print('  Run: python m2_predictor.py update')

    log   = load_log()
    preds = log.get('predictions', {})
    corrs = log.get('corrections', {})

    print(f'\n  Predictions made  : {len(preds)}')
    print(f'  Corrections done  : {len(corrs)}')

    if corrs:
        accs = [v['accuracy'] for v in corrs.values()]
        print(f'\n  📊 Accuracy Summary (all corrected days):')
        print(f'     Average : {np.mean(accs):.2f}%')
        print(f'     Best    : {np.max(accs):.2f}%')
        print(f'     Worst   : {np.min(accs):.2f}%')
        print(f'     Std     : {np.std(accs):.4f}')

        dist = {k: 0 for k in range(K_SEL + 1)}
        for v in corrs.values():
            dist[v['best_overlap']] += 1
        print(f'\n  📈 Score Distribution:')
        for k in range(K_SEL + 1):
            if dist[k] > 0:
                pct = dist[k] / len(corrs) * 100
                bar = '█' * dist[k]
                print(f'     {k}/6 correct : {dist[k]:3d} days  ({pct:5.1f}%)  {bar}')

        recent = sorted(corrs.keys(), key=int)[-10:]
        print(f'\n  📅 Recent {len(recent)} corrected days:')
        for d in recent:
            c   = corrs[d]
            ovl = c['best_overlap']
            acc = c['accuracy']
            bar = '█' * ovl + '░' * (K_SEL - ovl)
            print(f'     Day {d:>5}: [{bar}]  {ovl}/6  ({acc:5.1f}%)')

    print('=' * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def print_usage():
    print('=' * 70)
    print('  M2+ Narrow Adaptive Predictor')
    print('=' * 70)
    print('\n  First-time setup (or after adding new data):')
    print('    python m2_predictor.py update')
    print('\n  Generate 5 predictions for a day:')
    print('    python m2_predictor.py predict 1310')
    print('\n  Submit actual result after class:')
    print('    python m2_predictor.py correct 1310 6,11,22,33,44,55')
    print('\n  View accuracy history:')
    print('    python m2_predictor.py status')
    print('=' * 70)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'predict':
        if len(sys.argv) < 3:
            print('Usage: python m2_predictor.py predict DAY')
            sys.exit(1)
        cmd_predict(int(sys.argv[2]))

    elif cmd == 'correct':
        if len(sys.argv) < 4:
            print('Usage: python m2_predictor.py correct DAY S1,S2,S3,S4,S5,S6')
            sys.exit(1)
        students = [int(s.strip()) for s in sys.argv[3].split(',')]
        cmd_correct(int(sys.argv[2]), students)

    elif cmd == 'status':
        cmd_status()

    elif cmd in ('update', 'refit', 'train'):
        cmd_update()

    else:
        print(f'  Unknown command: {cmd}')
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
