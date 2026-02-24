"""
Advanced Beta-Binomial: Time-Decay Bayesian  &  Thompson Sampling + Beam Search
================================================================================

Two independent techniques applied to the BB prediction problem:

A. TIME-DECAY BAYESIAN (TD-HBB)
   Exponential forgetting of past observations:
     S_eff(t) = lambda * S_eff(t-1) + x(t)
     N_eff(t) = lambda * N_eff(t-1) + 1
   Effective memory = 1/(1-lambda) days.
   lambda=0.999->~1000d   0.99->~100d   0.97->~33d   0.95->~20d

B. THOMPSON SAMPLING + BEAM SEARCH (TS-BS-BB)
   Draw theta_s ~ Beta(alpha_s, beta_s) [Thompson Sampling].
   Beam search adds a co-occurrence diversity penalty:
     score(G U {s}) = score(G) + theta_s - gamma * mean_{t in G}[cooc_rate(s,t)]
   FULLY VECTORIZED: batch draw (n_samples, N_S); beam state as numpy tensors.

Usage:
    cd dl_pipeline
    python hbb_advanced/run_td_thompson.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import time

# ── Global config ─────────────────────────────────────────────────────────────
SEED          = 42
EXCEL_PATH    = str(_DL_DIR.parent / 'Database.xlsx')
N_S           = 55
K_SEL         = 6
TRAIN_RATIO   = 0.9
SIM_DAYS      = 100
N_GROUPS      = 10
TEMPERATURE   = 1.0
REFRESH_EVERY = 10

TS_SAMPLES    = 300
BEAM_WIDTH    = 15
SYNERGY_GAMMA = 0.5
DECAY_VALUES  = [0.999, 0.99, 0.97, 0.95]

BENCHMARKS = {
    'Flat BB (original)    ': 26.33,
    'HBB-Call-Order        ': 30.33,
    'DPM optimised         ': 30.83,
    'Order-3 Markov        ': 31.67,
    'Hybrid optimised      ': 32.17,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    T       = len(df)
    binary    = np.zeros((T, N_S), dtype=np.float32)
    pos_sum   = np.zeros(N_S, dtype=np.float64)
    pos_count = np.zeros(N_S, dtype=np.float64)
    cooc      = np.zeros((N_S, N_S), dtype=np.float64)

    for row_idx, row in df.iterrows():
        sel = []
        for pos_idx, col in enumerate(id_cols):
            sid = int(row[col]) - 1
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
                pos_sum[sid]   += pos_idx + 1
                pos_count[sid] += 1.0
                sel.append(sid)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                cooc[sel[i], sel[j]] += 1
                cooc[sel[j], sel[i]] += 1

    avg_pos = np.where(pos_count > 0, pos_sum / pos_count, 3.5).astype(np.float64)
    return binary, avg_pos, cooc


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sample_groups_from_p(probs, n_groups, temperature=TEMPERATURE):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups = []
    for _ in range(n_groups * 6):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SEL:].tolist()))
    return groups[:n_groups]


def _quantile_groups(values, n):
    cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)


def _estimate_hyperpriors(S, N, grp):
    G = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G) * 10.0
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


def _posterior_ab(S, N, grp, mu_g, kappa_g):
    alpha = mu_g[grp] * kappa_g[grp] + S
    beta_ = (1.0 - mu_g[grp]) * kappa_g[grp] + np.maximum(N - S, 0.0)
    return alpha, beta_


# ══════════════════════════════════════════════════════════════════════════════
# A. TIME-DECAY BAYESIAN BB
# ══════════════════════════════════════════════════════════════════════════════

class TimeDecayBB:
    """Exponential forgetting + optional HBB call-order grouping."""

    def __init__(self, decay=0.99, grouping='call_order',
                 n_tier=3, refresh_every=REFRESH_EVERY):
        self.decay = decay; self.grouping = grouping
        self.n_tier = n_tier; self.refresh_every = refresh_every
        self.S_eff = np.zeros(N_S); self.N_eff = np.zeros(N_S)
        self.grp   = np.zeros(N_S, dtype=np.int32)
        self.mu_g  = np.array([K_SEL / N_S]); self.kappa_g = np.array([2.0])
        self.n_obs = 0.0

    def fit(self, binary, avg_pos):
        n = len(binary); lam = self.decay
        # Weight: oldest obs = lam^(n-1), newest = 1
        w          = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
        self.S_eff = (binary * w[:, None]).sum(axis=0).astype(np.float64)
        self.N_eff = np.full(N_S, w.sum(), dtype=np.float64)
        self.grp   = _quantile_groups(avg_pos, self.n_tier) \
                     if self.grouping == 'call_order' \
                     else np.zeros(N_S, dtype=np.int32)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S_eff, self.N_eff, self.grp)
        self.n_obs = float(n)
        eff_w = 1.0 / (1.0 - lam) if lam < 1.0 else float(n)
        print(f"    lambda={lam:.3f}  eff_window~{eff_w:.0f}d  "
              f"grouping={self.grouping}({self.n_tier})")

    def predict(self, n_groups=N_GROUPS, temperature=TEMPERATURE):
        a, b = _posterior_ab(self.S_eff, self.N_eff, self.grp,
                             self.mu_g, self.kappa_g)
        return _sample_groups_from_p((a / (a + b)).astype(np.float32),
                                     n_groups, temperature)

    def update(self, obs):
        self.S_eff  = self.decay * self.S_eff + obs.astype(np.float64)
        self.N_eff  = self.decay * self.N_eff + 1.0
        self.n_obs += 1.0
        if self.refresh_every > 0 and int(self.n_obs) % self.refresh_every == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S_eff, self.N_eff, self.grp)


# ══════════════════════════════════════════════════════════════════════════════
# B. THOMPSON SAMPLING + VECTORIZED BEAM SEARCH
# ══════════════════════════════════════════════════════════════════════════════

class ThompsonBeamBB:
    """
    Thompson Sampling + optional Diversity-Beam-Search (fully vectorized).

    gamma=0 : pure TS, batch draw (n_samples, N_S) -> argsort top-6 (no loop).
    gamma>0 : per-sample vectorized beam search with co-occurrence penalty.
              Beam state: scores(W,), sel_mask(W,N_S) bool, cooc_acc(W,N_S).
              Expansion: score_cand[a,s] = scores[a] + theta[s]
                                         - gamma * cooc_acc[a,s] / step
              -> flatten -> top-W -> decode (beam_idx, student_idx) -> update.
    """

    def __init__(self, n_samples=TS_SAMPLES, beam_width=BEAM_WIDTH,
                 gamma=SYNERGY_GAMMA, grouping='call_order', n_tier=3,
                 refresh_every=REFRESH_EVERY, use_td=False, td_decay=0.99):
        self.n_samples = n_samples; self.bw = beam_width; self.gamma = gamma
        self.grouping = grouping; self.n_tier = n_tier
        self.refresh_every = refresh_every
        self.use_td = use_td; self.td_decay = td_decay
        self.S = np.zeros(N_S); self.N = np.zeros(N_S)
        self.grp = np.zeros(N_S, dtype=np.int32)
        self.mu_g = np.array([K_SEL / N_S]); self.kappa_g = np.array([2.0])
        self.cooc_rate = np.zeros((N_S, N_S), dtype=np.float32)
        self.n_obs = 0.0
        self._rng  = np.random.default_rng(SEED)

    def fit(self, binary, avg_pos, cooc_full):
        n = len(binary)
        if self.use_td:
            lam = self.td_decay
            w   = lam ** np.arange(n - 1, -1, -1, dtype=np.float64)
            self.S = (binary * w[:, None]).sum(axis=0).astype(np.float64)
            self.N = np.full(N_S, w.sum(), dtype=np.float64)
        else:
            self.S = binary.sum(axis=0).astype(np.float64)
            self.N = np.full(N_S, float(n), dtype=np.float64)

        # Co-occurrence rate from training data
        c = np.zeros((N_S, N_S), dtype=np.float64)
        for t in range(n):
            sel = np.where(binary[t] == 1)[0]
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    c[sel[i], sel[j]] += 1; c[sel[j], sel[i]] += 1
        self.cooc_rate = (c / (n + 1e-9)).astype(np.float32)

        self.grp = _quantile_groups(avg_pos, self.n_tier) \
                   if self.grouping == 'call_order' \
                   else np.zeros(N_S, dtype=np.int32)
        self.mu_g, self.kappa_g = _estimate_hyperpriors(
            self.S, self.N, self.grp)
        self.n_obs = float(n)
        td_str = f", td_decay={self.td_decay}" if self.use_td else ""
        print(f"    n_samples={self.n_samples}  W={self.bw}  gamma={self.gamma}"
              f"  grouping={self.grouping}({self.n_tier}){td_str}")

    def _beam_search_vec(self, theta):
        """
        Numpy-only vectorized beam search for one Thompson draw.
        State: scores(W,), sel_mask(W,N_S) bool, cooc_acc(W,N_S) float32.
        """
        W = self.bw
        scores   = np.zeros(W, dtype=np.float64)
        sel_mask = np.zeros((W, N_S), dtype=bool)
        cooc_acc = np.zeros((W, N_S), dtype=np.float64)
        n_active = 1

        for step in range(K_SEL):
            sa = scores[:n_active]    # (a,)
            sm = sel_mask[:n_active]  # (a, N)
            ca = cooc_acc[:n_active]  # (a, N)

            # Score candidates: (a, N)
            cand = sa[:, None] + theta[None, :]
            if self.gamma > 0.0 and step > 0:
                cand -= self.gamma * (ca / step)
            cand[sm] = -1e18          # mask already-selected

            flat  = cand.ravel()
            top_k = min(W, int((~sm).sum()))
            if top_k < len(flat):
                top_idx = np.argpartition(flat, -top_k)[-top_k:]
            else:
                top_idx = np.arange(len(flat))
            top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

            bi = top_idx // N_S   # previous beam item
            si = top_idx  % N_S   # new student

            new_sel  = sm[bi].copy()
            new_sel[np.arange(len(top_idx)), si] = True
            new_cooc = ca[bi] + self.cooc_rate[si]  # (top_k, N)

            n_active = len(top_idx)
            scores[:n_active]   = flat[top_idx]
            sel_mask[:n_active] = new_sel
            cooc_acc[:n_active] = new_cooc

        return tuple(sorted(np.where(sel_mask[0])[0].tolist()))

    def predict(self, n_groups=N_GROUPS, temperature=TEMPERATURE):
        a, b = _posterior_ab(self.S, self.N, self.grp, self.mu_g, self.kappa_g)
        alpha = a.astype(np.float64); beta_ = b.astype(np.float64)
        group_cnt = {}

        if self.gamma == 0.0 or self.bw <= 1:
            # Batch TS: draw all samples at once — pure numpy
            theta_batch = self._rng.beta(alpha, beta_, size=(self.n_samples, N_S))
            top6 = np.argsort(theta_batch, axis=1)[:, -K_SEL:]
            for row in top6:
                g = tuple(sorted(row.tolist()))
                group_cnt[g] = group_cnt.get(g, 0) + 1
        else:
            # TS + Vectorized Beam Search (one draw at a time)
            for _ in range(self.n_samples):
                theta = self._rng.beta(alpha, beta_).astype(np.float32)
                g     = self._beam_search_vec(theta)
                group_cnt[g] = group_cnt.get(g, 0) + 1

        ranked = sorted(group_cnt, key=group_cnt.__getitem__, reverse=True)
        groups = [list(g) for g in ranked[:n_groups]]
        if len(groups) < n_groups:
            p_m = (a / (a + b)).astype(np.float32)
            for g in _sample_groups_from_p(p_m, n_groups, temperature):
                if g not in groups:
                    groups.append(g)
                if len(groups) >= n_groups:
                    break
        return groups[:n_groups]

    def update(self, obs):
        if self.use_td:
            self.S = self.td_decay * self.S + obs.astype(np.float64)
            self.N = self.td_decay * self.N + 1.0
        else:
            self.S += obs.astype(np.float64); self.N += 1.0
        self.n_obs += 1.0
        if self.refresh_every > 0 and int(self.n_obs) % self.refresh_every == 0:
            self.mu_g, self.kappa_g = _estimate_hyperpriors(
                self.S, self.N, self.grp)
        sel = np.where(obs == 1)[0]
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                n = self.n_obs
                v = float(self.cooc_rate[sel[i], sel[j]]) * (n - 1) / n + 1.0 / n
                self.cooc_rate[sel[i], sel[j]] = v
                self.cooc_rate[sel[j], sel[i]] = v


# ══════════════════════════════════════════════════════════════════════════════
# Simulation harness
# ══════════════════════════════════════════════════════════════════════════════

def _run_sim(binary, split, model, label, fit_extra=None):
    np.random.seed(SEED)
    if fit_extra is not None:
        model.fit(binary[:split], *fit_extra)
    else:
        model.fit(binary[:split])

    n_sim = min(SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()
    for i in range(n_sim):
        actual = set(np.where(binary[split + i] == 1)[0].tolist())
        groups = model.predict()
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
        if (i + 1) in {1, 10, 25, 50, 75, 100}:
            print(f"    Day {i+1:3d}: {best*100:.1f}%  actual={sorted(actual)}")
        model.update(binary[split + i])

    arr  = np.array(accs) * 100.0
    dist = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
    nz   = {k: v for k, v in dist.items() if v > 0}
    print(f"\n  -- {label} --")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 76)
    print("ADVANCED BB: TIME-DECAY  +  THOMPSON SAMPLING + BEAM SEARCH".center(76))
    print("=" * 76)

    binary, avg_pos, cooc = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  N_GROUPS={N_GROUPS}, T={TEMPERATURE}, SIM_DAYS={SIM_DAYS}\n")
    results = {}

    # ── A. TIME-DECAY SWEEP ──────────────────────────────────────────────────
    print("=" * 76)
    print("  PART A -- TIME-DECAY BAYESIAN BB  (lambda sweep + HBB call-order)")
    print("=" * 76)
    td_results = {}
    for lam in DECAY_VALUES:
        eff_w = 1.0 / (1.0 - lam) if lam < 1.0 else float(split)
        label = f"TD-HBB lam={lam} (~{eff_w:.0f}d)"
        print(f"\n{'─'*76}\n  {label}\n{'─'*76}")
        m   = TimeDecayBB(decay=lam, grouping='call_order', n_tier=3)
        arr = _run_sim(binary, split, m, label, fit_extra=(avg_pos,))
        results[label] = arr; td_results[lam] = arr

    best_lam = max(td_results, key=lambda l: td_results[l].mean())
    print(f"\n  Best decay: lam={best_lam}  ({td_results[best_lam].mean():.2f}%)")

    print(f"\n{'─'*76}\n  TD-BB-flat lam=0.99 (no grouping)\n{'─'*76}")
    m_flat = TimeDecayBB(decay=0.99, grouping='none')
    results['TD-BB-flat lam=0.99'] = _run_sim(binary, split, m_flat,
                                               'TD-BB-flat lam=0.99',
                                               fit_extra=(avg_pos,))

    # ── B. PURE THOMPSON SAMPLING ────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  PART B -- THOMPSON SAMPLING  (gamma=0, vectorized batch draw)")
    print("=" * 76)
    label_ts = f'TS-BB (gamma=0, n={TS_SAMPLES})'
    print(f"\n{'─'*76}\n  {label_ts}\n{'─'*76}")
    m_ts   = ThompsonBeamBB(n_samples=TS_SAMPLES, beam_width=1, gamma=0.0,
                            grouping='call_order', n_tier=3)
    arr_ts = _run_sim(binary, split, m_ts, label_ts, fit_extra=(avg_pos, cooc))
    results[label_ts] = arr_ts

    # ── C. THOMPSON + BEAM SEARCH  (gamma sweep) ─────────────────────────────
    print(f"\n{'='*76}")
    print(f"  PART C -- THOMPSON + BEAM SEARCH  (W={BEAM_WIDTH}, gamma sweep)")
    print("=" * 76)
    gamma_vals   = [0.2, 0.5, 1.0]
    tsbs_results = {}
    for gam in gamma_vals:
        label = f'TS-BS-BB gamma={gam}'
        print(f"\n{'─'*76}\n  {label}  W={BEAM_WIDTH}  n={TS_SAMPLES}\n{'─'*76}")
        m_tsbs = ThompsonBeamBB(n_samples=TS_SAMPLES, beam_width=BEAM_WIDTH,
                                gamma=gam, grouping='call_order', n_tier=3)
        arr = _run_sim(binary, split, m_tsbs, label, fit_extra=(avg_pos, cooc))
        results[label] = arr; tsbs_results[gam] = arr

    best_gam = max(tsbs_results, key=lambda g: tsbs_results[g].mean())
    print(f"\n  Best gamma: {best_gam}  ({tsbs_results[best_gam].mean():.2f}%)")

    # ── D. COMBINED: TD + TS + BEAM SEARCH ───────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  PART D -- COMBINED: TD-TS-BS  (lam={best_lam}, gamma={best_gam})")
    print("=" * 76)
    label_comb = f'TD-TS-BS lam={best_lam} gamma={best_gam}'
    print(f"\n{'─'*76}\n  {label_comb}\n{'─'*76}")
    m_comb = ThompsonBeamBB(n_samples=TS_SAMPLES, beam_width=BEAM_WIDTH,
                            gamma=best_gam, grouping='call_order', n_tier=3,
                            use_td=True, td_decay=best_lam)
    arr_comb = _run_sim(binary, split, m_comb, label_comb,
                        fit_extra=(avg_pos, cooc))
    results[label_comb] = arr_comb

    # ── LEADERBOARD ───────────────────────────────────────────────────────────
    prev_best = max(BENCHMARKS.values())
    print(f"\n{'='*76}")
    print("  FULL LEADERBOARD")
    print("=" * 76)
    print(f"\n  {'Model':<38}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*38}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<38}  {v:>8.2f}%  {'—':>8}  {'—':>8}  {'—':>7}  [prev]")
    print(f"  {'─'*38}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for n, a in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        flag = '  * NEW BEST' if a.mean() > prev_best else ''
        print(f"  {n:<38}  {a.mean():>8.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    # ── TD INSIGHT ────────────────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print("  TIME-DECAY INSIGHT  (effective memory window vs performance)")
    print("─" * 76)
    print(f"  {'lam':>6}  {'eff_window':>12}  {'Avg':>8}  {'Std':>8}")
    for lam in DECAY_VALUES:
        eff_w = 1.0 / (1.0 - lam) if lam < 1.0 else float(split)
        a = td_results[lam]
        print(f"  {lam:>6.3f}  {eff_w:>11.0f}d  {a.mean():>8.2f}%  {a.std():>8.4f}")

    # ── BEAM SEARCH INSIGHT ───────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print("  BEAM SEARCH INSIGHT  (diversity penalty gamma vs performance)")
    print("─" * 76)
    print(f"  {'gamma':>6}  {'Avg':>8}  {'Std':>8}  {'Best':>8}  Notes")
    print(f"  {0.0:>6.1f}  {arr_ts.mean():>8.2f}%  {arr_ts.std():>8.4f}  "
          f"{arr_ts.max():>7.2f}%  pure TS (greedy)")
    for gam in gamma_vals:
        a = tsbs_results[gam]
        print(f"  {gam:>6.1f}  {a.mean():>8.2f}%  {a.std():>8.4f}  "
              f"{a.max():>7.2f}%  diversity beam")

    # ── VARIANCE ANALYSIS ─────────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print("  VARIANCE ANALYSIS  (Std -- lower = more consistent)")
    print("─" * 76)
    hbb_co_std = 9.2436
    print(f"  {'Model':<38}  {'Avg':>8}  {'Std':>8}  {'dStd vs HBB-CO':>15}")
    for n, a in sorted(results.items(), key=lambda x: x[1].std()):
        ds   = a.std() - hbb_co_std
        flag = '  lower' if ds < -0.05 else ('  higher' if ds > 0.1 else '')
        print(f"  {n:<38}  {a.mean():>8.2f}%  {a.std():>8.4f}  {ds:>+14.4f}{flag}")

    overall_best = max(results, key=lambda x: results[x].mean())
    print(f"\n{'='*76}")
    print(f"  BEST MODEL: {overall_best}  ({results[overall_best].mean():.2f}%)")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
