"""
Hierarchical Beta-Binomial by Student Group  (HBB-Group)
==========================================================

Core idea
─────────
Students are not homogeneous — the teacher has structured calling patterns:
  • Some students are called very frequently (favourites / monitors)
  • Some are called early in the session (class leaders, position-1 = seat monitor)
  • Some are densely co-connected (social/group leaders)

Grouping students by these attributes and fitting a separate Beta hyperprior
per group creates a 3-level hierarchy:

  Level 3 — Global :  μ₀, κ₀           (grand mean & concentration)
  Level 2 — Group g:  μ_g, κ_g          (group mean & concentration)
  Level 1 — Student:  θ_s | group g     (individual rate)

Posterior per student (conjugate Beta-Binomial):
  α_s = μ_g · κ_g + S_s         (successes)
  β_s = (1-μ_g) · κ_g + F_s     (failures)
  θ̂_s = α_s / (α_s + β_s)       (posterior mean)

Variance reduction mechanism:
  • Flat BB uses prior Beta(1,1) → κ_prior = 2 → weak pull toward 0.5
  • Group BB uses prior Beta(μ_g·κ_g, (1-μ_g)·κ_g):
      κ_g >> 2 → strong pull toward μ_g (the group's empirical rate)
  • Students in a "high-frequency" group are pulled toward 0.15, not 0.5
  • Students in a "low-frequency" group are anchored near 0.08
  → Posterior variance is lower for outlier students → better predictions

Three grouping strategies (derived purely from call data):
  1. Frequency   — tertile split of calling rate
                   "Favourite" / "Normal" / "Rarely-called"
  2. Call-order  — avg position in Student ID 1–6 columns (call ORDER)
                   position ≈ 1 → called first → class monitor / leader (proxy for Score)
                   position ≈ 6 → called last  → less prominent
  3. Centrality  — co-occurrence network degree centrality (leader proxy)
                   Students co-called with many others → high centrality
  4. Combined    — k-means on (freq, call-order, centrality), 4 groups

Online Bayesian group update:
  • Student posteriors updated every day (conjugate update)
  • Group hyperpriors (μ_g, κ_g) refreshed every `refresh_every` days
    via Empirical Bayes re-estimation on all accumulated data
  → Groups adapt as teacher calling patterns evolve during test period

Benchmarks (100-day, best-of-10 oracle):
  Flat Bayesian BB original  : 26.33%  (κ_prior ≈ 2, uniform)
  DPM optimised              : 30.83%
  Order-3 Markov             : 31.67%
  Hybrid optimised           : 32.17%

Usage:
    cd dl_pipeline
    python hbb_group/run_hbb_group.py
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

# ── Config ──────────────────────────────────────────────────────────────────
SEED           = 42
EXCEL_PATH     = str(_DL_DIR.parent / 'Database.xlsx')
N_S            = 55
K_SEL          = 6
TRAIN_RATIO    = 0.9
SIM_DAYS       = 100
N_GROUPS       = 10         # prediction candidates
TEMPERATURE    = 1.0
REFRESH_EVERY  = 10         # days between group-hyperprior refreshes

BENCHMARKS = {
    'Flat BB (original)    ': 26.33,
    'DPM optimised         ': 30.83,
    'Order-3 Markov        ': 31.67,
    'Hybrid optimised      ': 32.17,
}


# ── Data + position features ──────────────────────────────────────────────

def load_data():
    """
    Returns binary matrix  (T, N_S)
            avg_position    (N_S,)  — mean call-order position (1=first, 6=last)
    """
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = sorted([c for c in df.columns if c.startswith('Student ID')])
    # id_cols = ['Student ID 1', ..., 'Student ID 6']  — position order

    binary       = np.zeros((len(df), N_S), dtype=np.float32)
    pos_sum      = np.zeros(N_S, dtype=np.float64)   # sum of call positions
    pos_count    = np.zeros(N_S, dtype=np.float64)   # times called

    for row_idx, row in df.iterrows():
        for pos_idx, col in enumerate(id_cols):      # pos_idx 0→5
            sid = int(row[col]) - 1                  # 0-indexed
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
                pos_sum[sid]   += pos_idx + 1        # 1-indexed position
                pos_count[sid] += 1.0

    # Students never called → neutral position 3.5
    avg_position = np.where(pos_count > 0,
                            pos_sum / pos_count,
                            3.5).astype(np.float64)
    return binary, avg_position


# ── K-means (no sklearn dependency) ────────────────────────────────────────

def _kmeans(X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Simple k-means++, returns labels (N,)."""
    rng = np.random.default_rng(seed)
    # k-means++ initialisation
    idx     = [int(rng.integers(len(X)))]
    for _ in range(k - 1):
        dists = np.min(np.stack([((X - X[i])**2).sum(axis=1) for i in idx]), axis=0)
        prob  = dists / (dists.sum() + 1e-10)
        idx.append(int(rng.choice(len(X), p=prob)))
    centers = X[idx].copy().astype(np.float64)

    labels = np.zeros(len(X), dtype=np.int32)
    for _ in range(200):
        dists      = np.array([((X - c)**2).sum(axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)
    return labels


def _norm(x: np.ndarray) -> np.ndarray:
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Hierarchical Beta-Binomial by Group
# ══════════════════════════════════════════════════════════════════════════════

class HBBGroup:
    """
    3-level Hierarchical Beta-Binomial with student grouping.

    Parameters
    ----------
    grouping      : 'frequency' | 'call_order' | 'centrality' | 'combined'
    n_tier        : number of student groups (ignored for 'combined' → uses 4)
    refresh_every : days between empirical-Bayes hyperprior refreshes (0=never)
    """

    def __init__(self, grouping: str = 'frequency', n_tier: int = 3,
                 refresh_every: int = REFRESH_EVERY):
        self.grouping      = grouping
        self.n_tier        = n_tier if grouping != 'combined' else 4
        self.refresh_every = refresh_every

        self.G             = self.n_tier          # actual number of groups
        self.student_group = np.zeros(N_S, dtype=np.int32)
        self.mu_g          = np.zeros(self.G)
        self.kappa_g       = np.ones(self.G) * 10.0
        self.S             = np.zeros(N_S, dtype=np.float64)
        self.F             = np.zeros(N_S, dtype=np.float64)
        self.n_obs         = 0.0

        # For centrality computation
        self._cooc         = np.zeros((N_S, N_S), dtype=np.float64)
        self._n_train      = 0.0

    # ── Group assignment ───────────────────────────────────────────────────

    def _quantile_assign(self, values: np.ndarray, n: int) -> np.ndarray:
        """Split `values` into `n` quantile tiers → labels 0..n-1."""
        cuts = np.percentile(values, np.linspace(0, 100, n + 1)[1:-1])
        return np.digitize(values, cuts).astype(np.int32)

    def _assign_groups(self, rates: np.ndarray,
                       avg_position: np.ndarray,
                       centrality: np.ndarray) -> np.ndarray:
        if self.grouping == 'frequency':
            return self._quantile_assign(rates, self.n_tier)
        elif self.grouping == 'call_order':
            # low position = called early = leader → group 0
            return self._quantile_assign(avg_position, self.n_tier)
        elif self.grouping == 'centrality':
            return self._quantile_assign(centrality, self.n_tier)
        elif self.grouping == 'combined':
            X = np.column_stack([_norm(rates),
                                 _norm(avg_position),
                                 _norm(centrality)])
            return _kmeans(X, k=self.G)
        else:
            return np.zeros(N_S, dtype=np.int32)

    # ── Hyperprior estimation (Empirical Bayes, Method of Moments) ──────────

    def _estimate_hyperpriors(self, rates: np.ndarray) -> None:
        """
        For each group g:
          μ_g = mean(rates[group g])
          κ_g = μ_g(1-μ_g) / Var(rates[group g])  − 1   (MoM)
          Clipped to [2, 1000].
        """
        for g in range(self.G):
            m = self.student_group == g
            if m.sum() < 2:
                self.mu_g[g]    = rates.mean()
                self.kappa_g[g] = 10.0
                continue
            r   = rates[m]
            mu  = float(r.mean())
            var = float(r.var())
            self.mu_g[g] = mu
            if var > 1e-10:
                kappa = mu * (1.0 - mu) / var - 1.0
                self.kappa_g[g] = float(np.clip(kappa, 2.0, 1000.0))
            else:
                self.kappa_g[g] = 1000.0   # perfectly homogeneous group

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, binary: np.ndarray, avg_position: np.ndarray) -> None:
        n            = len(binary)
        self._n_train = float(n)
        rates        = binary.mean(axis=0).astype(np.float64)

        # Co-occurrence → centrality
        self._cooc[:] = 0.0
        for t in range(n):
            sel = np.where(binary[t] == 1)[0]
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    self._cooc[sel[i], sel[j]] += 1
                    self._cooc[sel[j], sel[i]] += 1
        centrality = self._cooc.sum(axis=1) / (n + 1e-9)

        # Assign groups
        self.student_group = self._assign_groups(rates, avg_position, centrality)
        self.G             = int(self.student_group.max()) + 1
        self.mu_g          = np.zeros(self.G)
        self.kappa_g       = np.ones(self.G) * 10.0

        # Empirical Bayes hyperpriors
        self._estimate_hyperpriors(rates)

        # Student posteriors warm-started from training data
        self.S     = binary.sum(axis=0).astype(np.float64)
        self.F     = (n - self.S)
        self.n_obs = float(n)

        self._print_summary(rates, avg_position, centrality)

    def _print_summary(self, rates, avg_position, centrality) -> None:
        prior_var_flat  = self._prior_variance(2.0, rates.mean())
        prior_var_group = self._mean_prior_variance()
        reduction       = (1.0 - prior_var_group / prior_var_flat) * 100.0

        print(f"    Grouping      : {self.grouping}  ({self.G} groups)")
        print(f"    Prior var (flat):  {prior_var_flat:.6f}  "
              f"→  Group: {prior_var_group:.6f}  "
              f"(↓{reduction:.1f}% reduction)")
        print(f"\n    {'Grp':>4}  {'N':>4}  {'μ_g':>8}  {'κ_g':>8}  "
              f"{'Rate [min,max]':>16}  {'AvgPos':>7}  {'Central':>9}")
        print(f"    {'─'*4}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*16}  {'─'*7}  {'─'*9}")
        for g in range(self.G):
            m = self.student_group == g
            if not m.any():
                continue
            r_g = rates[m]; p_g = avg_position[m]; c_g = centrality[m]
            print(f"    {g:>4}  {m.sum():>4}  {self.mu_g[g]:>8.4f}  "
                  f"{self.kappa_g[g]:>8.1f}  "
                  f"[{r_g.min():.3f}, {r_g.max():.3f}]   "
                  f"{p_g.mean():>7.2f}  {c_g.mean():>9.2f}")

    @staticmethod
    def _prior_variance(kappa: float, mu: float) -> float:
        return mu * (1.0 - mu) / (kappa + 1.0)

    def _mean_prior_variance(self) -> float:
        vars = []
        for g in range(self.G):
            m = self.student_group == g
            if m.any():
                vars.append(self._prior_variance(self.kappa_g[g], self.mu_g[g]))
        return float(np.mean(vars)) if vars else 0.0

    # ── Posterior ───────────────────────────────────────────────────────────

    def _posterior_mean(self) -> np.ndarray:
        alpha = self.mu_g[self.student_group] * self.kappa_g[self.student_group] + self.S
        beta  = (1.0 - self.mu_g[self.student_group]) * self.kappa_g[self.student_group] + self.F
        return (alpha / (alpha + beta)).astype(np.float32)

    def _posterior_variance(self) -> np.ndarray:
        alpha = self.mu_g[self.student_group] * self.kappa_g[self.student_group] + self.S
        beta  = (1.0 - self.mu_g[self.student_group]) * self.kappa_g[self.student_group] + self.F
        ab    = alpha + beta
        return (alpha * beta / (ab ** 2 * (ab + 1))).astype(np.float32)

    def predict(self, n_groups: int = N_GROUPS,
                temperature: float = TEMPERATURE) -> list[list[int]]:
        return _sample_groups(self._posterior_mean(), n_groups, temperature)

    # ── Online update ────────────────────────────────────────────────────────

    def update(self, obs: np.ndarray) -> None:
        self.S     += obs.astype(np.float64)
        self.F     += (1.0 - obs).astype(np.float64)
        self.n_obs += 1.0
        if self.refresh_every > 0 and int(self.n_obs) % self.refresh_every == 0:
            rates = self.S / self.n_obs
            self._estimate_hyperpriors(rates)


# ── Flat BB baseline (κ_prior = 2, uniform) ─────────────────────────────────

class FlatBB:
    """Original Bayesian Beta-Binomial: Beta(1+S, 1+F) per student."""
    def __init__(self):
        self.S = np.zeros(N_S, dtype=np.float64)
        self.F = np.zeros(N_S, dtype=np.float64)

    def fit(self, binary: np.ndarray, avg_position=None) -> None:
        self.S = binary.sum(axis=0).astype(np.float64)
        self.F = (len(binary) - self.S)
        base = K_SEL / N_S
        prior_var = base * (1 - base) / 3.0
        print(f"    Flat BB prior: Beta(1,1) κ=2 → prior_var={prior_var:.6f}")

    def predict(self, n_groups=N_GROUPS, temperature=TEMPERATURE):
        p = (1.0 + self.S) / (2.0 + self.S + self.F)
        return _sample_groups(p.astype(np.float32), n_groups, temperature)

    def update(self, obs: np.ndarray) -> None:
        self.S += obs.astype(np.float64)
        self.F += (1.0 - obs).astype(np.float64)


# ── Sampling ─────────────────────────────────────────────────────────────────

def _sample_groups(probs: np.ndarray, n_groups: int,
                   temperature: float) -> list[list[int]]:
    p = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p /= p.sum()
    groups: list[list[int]] = []
    for _ in range(n_groups * 4):
        g = sorted(np.random.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SEL:].tolist()))
    return groups[:n_groups]


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(binary: np.ndarray, split: int,
             avg_position: np.ndarray,
             model, label: str) -> np.ndarray:
    np.random.seed(SEED)
    model.fit(binary[:split], avg_position)

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
    print(f"\n  ── {label} ──")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    return arr


# ── Variance reduction analysis ──────────────────────────────────────────────

def variance_analysis(binary: np.ndarray, split: int,
                      avg_position: np.ndarray,
                      models: dict) -> None:
    print(f"\n  {'Model':<30}  {'Prior Var':>12}  {'Post Var (train)':>18}  "
          f"{'Post Var (after 10d)':>20}  {'Reduction':>10}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*18}  {'─'*20}  {'─'*10}")

    rates = binary[:split].mean(axis=0)

    for label, model in models.items():
        # Train
        m_copy = type(model).__new__(type(model))
        m_copy.__dict__.update(model.__dict__)

        # Get posterior variance after training
        if hasattr(m_copy, '_posterior_variance'):
            pv_train = float(m_copy._posterior_variance().mean())
            # Simulate 10 days and recheck
            m2 = type(model)(
                **{k: v for k, v in vars(model).items()
                   if k in ('grouping', 'n_tier', 'refresh_every')}
            ) if hasattr(model, 'grouping') else FlatBB()
            m2.fit(binary[:split], avg_position)
            for i in range(10):
                m2.update(binary[split + i])
            pv_10d = float(m2._posterior_variance().mean()) if hasattr(m2, '_posterior_variance') else 0.0

            if isinstance(model, HBBGroup):
                pv_prior = model._prior_variance(float(model.kappa_g.mean()),
                                                  float(model.mu_g.mean()))
            else:
                pv_prior = rates.mean() * (1 - rates.mean()) / 3.0

            red = (1 - pv_train / (rates.mean() * (1 - rates.mean()) / 3.0)) * 100
            print(f"  {label:<30}  {pv_prior:>12.2e}  {pv_train:>18.2e}  "
                  f"{pv_10d:>20.2e}  {red:>9.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("HIERARCHICAL BETA-BINOMIAL BY STUDENT GROUP".center(72))
    print("Frequency | Call-Order (Score) | Centrality (Leader) | Combined".center(72))
    print("=" * 72)

    binary, avg_position = load_data()
    split = int(len(binary) * TRAIN_RATIO)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  Prediction: N_GROUPS={N_GROUPS}, T={TEMPERATURE}, refresh={REFRESH_EVERY}d")
    print(f"\n  Position feature  : [{avg_position.min():.2f}, {avg_position.max():.2f}]  "
          f"mean={avg_position.mean():.2f}  (1=first called, 6=last)")
    print(f"  Students always first (pos<2): "
          f"{(avg_position < 2.0).sum()}  "
          f"| always last (pos>5): {(avg_position > 5.0).sum()}")

    results: dict[str, np.ndarray] = {}

    # ── 0. Flat BB baseline ────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [0/5]  FLAT BB BASELINE  (uniform prior, no grouping)")
    print("─" * 72)
    flat = FlatBB()
    results['Flat BB (no grouping)'] = simulate(binary, split, avg_position,
                                                 flat, 'Flat BB')

    # ── 1. HBB-Frequency ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [1/5]  HBB-FREQUENCY  (3 tiers: rarely / normal / favourite)")
    print("─" * 72)
    hbb_freq = HBBGroup(grouping='frequency', n_tier=3)
    results['HBB-Frequency'] = simulate(binary, split, avg_position,
                                         hbb_freq, 'HBB-Frequency')

    # ── 2. HBB-Call-Order (Score proxy) ───────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [2/5]  HBB-CALL-ORDER  (3 tiers by avg call position = score proxy)")
    print("         Tier 0: pos 1–2 (called first = monitor/leader)")
    print("         Tier 1: pos 3–4 (mid-rank)")
    print("         Tier 2: pos 5–6 (called last)")
    print("─" * 72)
    hbb_pos = HBBGroup(grouping='call_order', n_tier=3)
    results['HBB-Call-Order'] = simulate(binary, split, avg_position,
                                          hbb_pos, 'HBB-Call-Order')

    # ── 3. HBB-Centrality (Leader proxy) ──────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [3/5]  HBB-CENTRALITY  (3 tiers by co-occurrence network degree)")
    print("         Tier 2: high centrality = bridge students = leader proxy")
    print("─" * 72)
    hbb_cent = HBBGroup(grouping='centrality', n_tier=3)
    results['HBB-Centrality'] = simulate(binary, split, avg_position,
                                          hbb_cent, 'HBB-Centrality')

    # ── 4. HBB-Combined (k-means) ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [4/5]  HBB-COMBINED  (k-means on freq + call-order + centrality, k=4)")
    print("─" * 72)
    hbb_comb = HBBGroup(grouping='combined', n_tier=4)
    results['HBB-Combined'] = simulate(binary, split, avg_position,
                                        hbb_comb, 'HBB-Combined')

    # ── 5. HBB-Combined fine-grained (k=6) ───────────────────────────────
    print(f"\n{'─'*72}")
    print("  [5/5]  HBB-COMBINED-6  (k-means k=6, finer grouping)")
    print("─" * 72)
    hbb_comb6 = HBBGroup(grouping='combined', n_tier=6)
    results['HBB-Combined-6'] = simulate(binary, split, avg_position,
                                          hbb_comb6, 'HBB-Combined-6')

    # ── Variance reduction table ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  POSTERIOR VARIANCE ANALYSIS  (prior variance reduction vs Flat BB)")
    print(f"{'─'*72}")
    print(f"\n  {'Model':<22}  {'κ_prior':>8}  {'Prior Var':>12}  "
          f"{'Post Var/student':>18}  {'Reduction':>10}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*12}  {'─'*18}  {'─'*10}")
    base_mu  = binary[:split].mean(axis=0).mean()
    flat_pvar = base_mu * (1 - base_mu) / 3.0   # κ=2 flat

    for label, model in [
        ('Flat BB',        flat),
        ('HBB-Frequency',  hbb_freq),
        ('HBB-Call-Order', hbb_pos),
        ('HBB-Centrality', hbb_cent),
        ('HBB-Combined',   hbb_comb),
        ('HBB-Combined-6', hbb_comb6),
    ]:
        if isinstance(model, FlatBB):
            kappa_mean = 2.0
            pvar       = flat_pvar
            reduction  = 0.0
        else:
            kappa_mean = float(model.kappa_g.mean())
            pvar       = model._mean_prior_variance()
            reduction  = (1.0 - pvar / flat_pvar) * 100.0
        print(f"  {label:<22}  {kappa_mean:>8.1f}  {pvar:>12.2e}  "
              f"{'—':>18}  {reduction:>9.1f}%")

    # ── Group membership quality ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  GROUP MEMBERSHIP — HBB-COMBINED  (which students are in each group)")
    print(f"{'─'*72}")
    rates_train = binary[:split].mean(axis=0)
    for g in range(hbb_comb.G):
        m   = hbb_comb.student_group == g
        sids = (np.where(m)[0] + 1).tolist()   # 1-indexed student IDs
        avg_r = float(rates_train[m].mean())
        avg_p = float(avg_position[m].mean())
        print(f"  Group {g}  [{m.sum():2d} students]  "
              f"μ_g={hbb_comb.mu_g[g]:.4f}  κ_g={hbb_comb.kappa_g[g]:.0f}  "
              f"AvgRate={avg_r:.3f}  AvgPos={avg_p:.2f}")
        print(f"    Students: {sids}")

    # ── Call-order group insight ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  CALL-ORDER INSIGHT  (Score / Leader proxy via call position)")
    print(f"{'─'*72}")
    for g in range(3):
        m    = hbb_pos.student_group == g
        sids = (np.where(m)[0] + 1).tolist()
        r    = rates_train[m]
        p    = avg_position[m]
        label = ['Position 1-2 (Leader/Monitor)', 'Position 3-4 (Mid-rank)', 'Position 5-6 (Called last)'][g]
        print(f"\n  Group {g} — {label}  [{m.sum()} students]")
        print(f"    μ_g={hbb_pos.mu_g[g]:.4f}  κ_g={hbb_pos.kappa_g[g]:.0f}  "
              f"Rate=[{r.min():.3f},{r.max():.3f}]  AvgPos=[{p.min():.2f},{p.max():.2f}]")
        print(f"    Students: {sids}")

    # ── Full leaderboard ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FULL LEADERBOARD")
    print(f"{'=' * 72}")
    print(f"\n  {'Model':<35}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    for name, val in BENCHMARKS.items():
        print(f"  {name:<35}  {val:>8.2f}%  {'—':>8}  {'—':>8}  {'—':>7}  [prev]")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    best_prev = max(BENCHMARKS.values())
    for name, arr in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        flag = '  ★ NEW BEST' if arr.mean() > best_prev else ''
        print(f"  {name:<35}  {arr.mean():>8.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>7.4f}{flag}")

    # ── Summary insight ───────────────────────────────────────────────────────
    best_name  = max(results, key=lambda x: results[x].mean())
    best_avg   = results[best_name].mean()
    flat_avg   = results['Flat BB (no grouping)'].mean()
    delta      = best_avg - flat_avg
    print(f"\n  Best HBB grouping: {best_name}  ({best_avg:.2f}%)  "
          f"vs Flat BB ({flat_avg:.2f}%)  Δ={delta:+.2f}%")

    print(f"\n{'=' * 72}")
    print("  HBB-GROUP COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
