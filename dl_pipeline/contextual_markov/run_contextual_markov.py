"""
Contextual Markov Chain
========================
Extends the FactoredMarkovChain with 4 richer contextual signals:

  1. Days-since-last-call  — exponential recency decay (student-level)
  2. Rolling frequency     — multi-scale momentum: 5 / 10 / 20-day windows
  3. Pairwise co-selection — contextual group-cohesion boost from recent day
  4. Entropy weighting     — per-student predictability; high-entropy students
                             shrink toward the base rate

Combination rule (multiplicative factors, then normalize):
    score_s = base_rate_s
            * recency_factor_s        # days-since component
            * momentum_factor_s       # rolling-freq component
            * cooc_factor_s           # pairwise co-selection
    final_s = (1 - conf_s) * base_rate_s + conf_s * score_s
            ← conf_s = 1 - H_s (entropy weight)

Ablation study (inside same run):
    Each feature is toggled off to measure its individual contribution.

Benchmarks (100-day, best-of-n_groups overlap):
    Markov only (α=0.0)   :  26.17%  (from optimize_hybrid Stage 1)
    DPM original          :  27.67%
    Hybrid original       :  26.83%
    DPM optimized (ng=10) :  30.83%
    Hybrid optimized      :  32.17%

Usage:
    cd dl_pipeline
    python contextual_markov/run_contextual_markov.py
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

# ── Config ─────────────────────────────────────────────────────────────────
SEED        = 42
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_STUDENTS  = 55
K           = 6
TRAIN_RATIO = 0.9
N_SIM_DAYS  = 100

# Prediction generation
N_GROUPS    = 10      # match optimized benchmark (best-of-10 oracle)
TEMPERATURE = 1.0

# Contextual Markov hyperparams (tunable)
RECENCY_LAMBDA   = 0.20   # λ for exp(-λ * days_since); larger = stronger recency penalty
ROLLING_WEIGHTS  = (0.50, 0.30, 0.20)   # w5, w10, w20
ROLLING_WINDOWS  = (5, 10, 20)
COOC_CLIP        = (0.3, 3.0)           # min / max co-occurrence ratio
ENTROPY_ALPHA    = 0.6                   # 0 = ignore entropy, 1 = full entropy modulation

BENCHMARKS = {
    'Markov only (α=0)': 26.17,
    'DPM original      ': 27.67,
    'Hybrid original   ': 26.83,
    'DPM optimized     ': 30.83,
    'Hybrid optimized  ': 32.17,
}


# ── Data ───────────────────────────────────────────────────────────────────

def load_binary() -> np.ndarray:
    df = pd.read_excel(EXCEL_PATH)
    df = df.sort_values('Days').reset_index(drop=True)
    binary = np.zeros((len(df), N_STUDENTS), dtype=np.float32)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]
    for row_idx, row in df.iterrows():
        for col in id_cols:
            sid = int(row[col]) - 1
            if 0 <= sid < N_STUDENTS:
                binary[row_idx, sid] = 1.0
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# Contextual Markov
# ══════════════════════════════════════════════════════════════════════════════

class ContextualMarkov:
    """
    Markov Chain with 4 contextual feature components.

    Feature 1 — Days-since-last-call (recency)
    ─────────────────────────────────────────────
    days_since_s = how many days ago student s was last called
                   (within the observation window; capped at window length)
    recency_s = exp(-λ * max(0, threshold - days_since_s))
              × exp(+λ * max(0, days_since_s - threshold))
    Simplified: recency_s = sigmoid((days_since_s - threshold) / scale)
    Interpretation: called very recently → lower score (teacher rotates);
                    absent for many days  → higher score (overdue).

    Feature 2 — Rolling frequency (momentum)
    ──────────────────────────────────────────
    freq_w_s = count_s_last_w / w  for w in {5, 10, 20}
    momentum_s = Σ_w weight_w × freq_w_s   (weighted blend)
    Captures whether the student is in a "hot" or "cold" streak recently.

    Feature 3 — Pairwise co-selection (group context)
    ───────────────────────────────────────────────────
    From training: cooc[i, j] = count of days both i and j were called.
    Given yesterday's group G_{t-1}:
        cooc_raw_s = Σ_{j ∈ G} cooc[s, j]
    Normalized: cooc_ratio_s = cooc_raw_s / (|G| × base_freq_s × n_train)
    Clipped to [cooc_min, cooc_max].
    Interpretation: students who historically co-occur with yesterday's group
                    get a multiplicative boost.

    Feature 4 — Entropy weighting (predictability)
    ────────────────────────────────────────────────
    Binary entropy per student based on overall calling rate:
        H_s = -p_s log2 p_s - (1-p_s) log2(1-p_s),  p_s = freq_s / n_days
    H_s ∈ [0, 1].  H_s near 0 → very predictable (rare or very common).
    Confidence: conf_s = 1 - H_s
    Entropy-modulated prediction:
        final_s = (1 - α × conf_s) × base_rate_s + α × conf_s × score_s
    High-entropy (unpredictable) students lean toward the base rate;
    low-entropy students have their combined score trusted more.

    Also tracks teacher-level daily entropy to report behavioral consistency.
    """

    def __init__(self,
                 n_students:     int   = N_STUDENTS,
                 recency_lambda: float = RECENCY_LAMBDA,
                 recency_thresh: float = 4.0,
                 rolling_weights: tuple = ROLLING_WEIGHTS,
                 rolling_windows: tuple = ROLLING_WINDOWS,
                 cooc_clip:       tuple = COOC_CLIP,
                 entropy_alpha:   float = ENTROPY_ALPHA,
                 # feature flags for ablation
                 use_recency:  bool = True,
                 use_rolling:  bool = True,
                 use_cooc:     bool = True,
                 use_entropy:  bool = True):

        self.N               = n_students
        self.lam             = recency_lambda
        self.thresh          = recency_thresh
        self.rw              = rolling_weights
        self.wins            = rolling_windows
        self.cooc_min        = cooc_clip[0]
        self.cooc_max        = cooc_clip[1]
        self.entropy_alpha   = entropy_alpha

        self.use_recency = use_recency
        self.use_rolling = use_rolling
        self.use_cooc    = use_cooc
        self.use_entropy = use_entropy

        # Learned statistics (filled in fit())
        self.base_rate   = np.full(n_students, K / n_students, dtype=np.float64)
        self.cooc        = np.zeros((n_students, n_students), dtype=np.float64)
        self.n_train     = 0.0
        self.student_H   = np.zeros(n_students, dtype=np.float64)  # per-student entropy
        self.daily_H_train = []   # teacher-level entropy per training day

        # Online cumulative counters (updated during simulation)
        self.cum_counts  = np.zeros(n_students, dtype=np.float64)
        self.cum_days    = 0.0

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, binary: np.ndarray) -> None:
        n_days  = len(binary)
        self.n_train = float(n_days)
        p       = binary.mean(axis=0).astype(np.float64)
        self.base_rate  = p
        self.cum_counts = binary.sum(axis=0).astype(np.float64)
        self.cum_days   = float(n_days)

        # Co-occurrence matrix (upper + lower filled)
        self.cooc[:] = 0.0
        for t in range(n_days):
            sel = np.where(binary[t] == 1)[0]
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    self.cooc[sel[i], sel[j]] += 1
                    self.cooc[sel[j], sel[i]] += 1

        # Per-student binary entropy  H = -p log2 p - (1-p) log2(1-p)
        eps = 1e-9
        pc  = np.clip(p, eps, 1 - eps)
        self.student_H = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))

        # Daily teacher entropy: treat daily selection as distribution
        for t in range(n_days):
            daily_p = binary[t].mean()           # fraction of students called
            if 0 < daily_p < 1:
                h = -(daily_p * np.log2(daily_p) + (1-daily_p) * np.log2(1-daily_p))
                self.daily_H_train.append(h)

        self._print_fit_summary()

    def _print_fit_summary(self) -> None:
        p  = self.base_rate
        H  = self.student_H
        dH = np.mean(self.daily_H_train) if self.daily_H_train else 0.0
        print(f"    Base rate   : [{p.min():.4f}, {p.max():.4f}]  mean={p.mean():.4f}")
        print(f"    Student H   : [{H.min():.4f}, {H.max():.4f}]  mean={H.mean():.4f}")
        print(f"    Teacher daily entropy (train): {dH:.4f}")
        conf = 1.0 - H
        print(f"    Confidence  : [{conf.min():.4f}, {conf.max():.4f}]  "
              f"(high = predictable student)")

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, window: np.ndarray,
                n_groups: int = N_GROUPS,
                temperature: float = TEMPERATURE) -> list[list[int]]:
        """
        window: recent days binary matrix (n_recent, N_STUDENTS)
                at least 20 days recommended for rolling features.
        """
        probs = self._compute_probs(window)
        return self._sample_groups(probs, n_groups, temperature)

    def _compute_probs(self, window: np.ndarray) -> np.ndarray:
        base = self.base_rate.copy()                    # (N,)
        score = base.copy()

        # ── Feature 1: Days-since recency ──────────────────────────────────
        if self.use_recency:
            days_since = self._days_since(window)
            # Sigmoid centred at thresh; >thresh = positive boost
            recency_f = 1.0 / (1.0 + np.exp(-self.lam * (days_since - self.thresh)))
            recency_f = recency_f / (recency_f.mean() + 1e-9)   # mean-normalise → ≈1
            score = score * recency_f

        # ── Feature 2: Rolling frequency momentum ──────────────────────────
        if self.use_rolling:
            momentum = np.zeros(self.N, dtype=np.float64)
            for w, wt in zip(self.wins, self.rw):
                seg = window[-w:] if len(window) >= w else window
                momentum += wt * seg.mean(axis=0).astype(np.float64)
            # Normalise as ratio to base rate (>1 = hot streak, <1 = cold)
            roll_ratio = momentum / (base + 1e-9)
            roll_ratio = np.clip(roll_ratio, 0.2, 5.0)
            roll_ratio = roll_ratio / (roll_ratio.mean() + 1e-9)
            score = score * roll_ratio

        # ── Feature 3: Pairwise co-selection ───────────────────────────────
        if self.use_cooc and len(window) > 0:
            yesterday = window[-1]
            yesterday_sel = np.where(yesterday == 1)[0]
            if len(yesterday_sel) > 0:
                cooc_raw = self.cooc[yesterday_sel, :].sum(axis=0)   # (N,)
                # Expected under independence: base_rate * n_train * |G|
                expected = base * self.n_train * len(yesterday_sel)
                cooc_ratio = cooc_raw / (expected + 1e-9)
                cooc_ratio = np.clip(cooc_ratio, self.cooc_min, self.cooc_max)
                cooc_ratio = cooc_ratio / (cooc_ratio.mean() + 1e-9)
                score = score * cooc_ratio

        # ── Feature 4: Entropy weighting ────────────────────────────────────
        if self.use_entropy:
            conf   = 1.0 - self.student_H                            # (N,)
            # α controls how much entropy modulation matters
            weight = self.entropy_alpha * conf
            weight = np.clip(weight, 0.0, 1.0)
            # Blend: low-confidence → pull toward base rate
            score  = (1.0 - weight) * base + weight * score

        # Normalise to probability distribution
        score = np.clip(score, 1e-9, None)
        return (score / score.sum()).astype(np.float32)

    def _days_since(self, window: np.ndarray) -> np.ndarray:
        """For each student, days since last call within window."""
        cap = float(len(window) + 1)
        days = np.full(self.N, cap, dtype=np.float64)
        for t in range(len(window) - 1, -1, -1):
            called = window[t] == 1
            days[called & (days == cap)] = float(len(window) - t)
        return days

    def _sample_groups(self, probs: np.ndarray,
                       n_groups: int, temperature: float) -> list[list[int]]:
        p = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
        p /= p.sum()
        groups: list[list[int]] = []
        for _ in range(n_groups * 4):
            g = sorted(np.random.choice(self.N, K, replace=False, p=p).tolist())
            if g not in groups:
                groups.append(g)
            if len(groups) >= n_groups:
                break
        if not groups:
            groups.append(sorted(np.argsort(probs)[-K:].tolist()))
        return groups[:n_groups]

    # ── Online update ───────────────────────────────────────────────────────

    def update(self, obs: np.ndarray) -> None:
        self.cum_counts += obs.astype(np.float64)
        self.cum_days   += 1.0
        # Refresh base rate (online Bayesian update)
        self.base_rate = self.cum_counts / self.cum_days
        # Refresh entropy
        eps = 1e-9
        pc  = np.clip(self.base_rate, eps, 1 - eps)
        self.student_H = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
        # Update co-occurrence
        sel = np.where(obs == 1)[0]
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                self.cooc[sel[i], sel[j]] += 1
                self.cooc[sel[j], sel[i]] += 1
        self.n_train += 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def simulate(binary: np.ndarray, split: int, label: str,
             n_groups: int = N_GROUPS, temperature: float = TEMPERATURE,
             **kwargs) -> np.ndarray:
    np.random.seed(SEED)
    model = ContextualMarkov(**kwargs)
    model.fit(binary[:split])

    n_sim = min(N_SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()

    for i in range(n_sim):
        # Use all available history up to current test day as window
        win_start = max(0, split + i - 30)
        window    = binary[win_start: split + i]

        actual = set(np.where(binary[split + i] == 1)[0].tolist())
        groups = model.predict(window, n_groups=n_groups, temperature=temperature)
        best   = max(len(set(g) & actual) for g in groups) / K
        accs.append(best)

        if (i + 1) in {1, 10, 25, 50, 75, 100}:
            print(f"    Day {i+1:3d}: {best*100:.1f}%  actual={sorted(actual)}")

        model.update(binary[split + i])

    arr = np.array(accs) * 100.0
    dist = {k: int((arr == k / K * 100).sum()) for k in range(K + 1)}
    non_zero = {k: v for k, v in dist.items() if v > 0}
    elapsed  = time.perf_counter() - t0
    print(f"\n  ── {label} ──")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in non_zero.items()))
    print(f"  Time: {elapsed:.1f}s")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("CONTEXTUAL MARKOV CHAIN".center(72))
    print("Days-since | Rolling Freq | Pairwise Co-selection | Entropy".center(72))
    print("=" * 72)

    binary = load_binary()
    split  = int(len(binary) * TRAIN_RATIO)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  Prediction: n_groups={N_GROUPS}, T={TEMPERATURE}\n")

    results: dict[str, np.ndarray] = {}

    # ── 1. Full Contextual Markov ───────────────────────────────────────────
    print("─" * 72)
    print("  [1/6]  FULL CONTEXTUAL MARKOV  (all 4 features)")
    print("─" * 72)
    results['Full Contextual'] = simulate(
        binary, split, 'Full Contextual Markov',
        use_recency=True, use_rolling=True, use_cooc=True, use_entropy=True
    )

    # ── 2–5. Ablation: turn off one feature at a time ──────────────────────
    ablation = [
        ('No Recency',  dict(use_recency=False, use_rolling=True,  use_cooc=True,  use_entropy=True)),
        ('No Rolling',  dict(use_recency=True,  use_rolling=False, use_cooc=True,  use_entropy=True)),
        ('No Co-occ',   dict(use_recency=True,  use_rolling=True,  use_cooc=False, use_entropy=True)),
        ('No Entropy',  dict(use_recency=True,  use_rolling=True,  use_cooc=True,  use_entropy=False)),
    ]

    for idx, (name, flags) in enumerate(ablation, start=2):
        print(f"\n{'─'*72}")
        print(f"  [{idx}/6]  ABLATION: {name}")
        print(f"{'─'*72}")
        results[name] = simulate(binary, split, name, **flags)

    # ── 6. Baseline: bare Markov (no contextual features) ──────────────────
    print(f"\n{'─'*72}")
    print(f"  [6/6]  BASELINE: No features (pure base-rate only)")
    print(f"{'─'*72}")
    results['Base-rate only'] = simulate(
        binary, split, 'Base-rate only',
        use_recency=False, use_rolling=False, use_cooc=False, use_entropy=False
    )

    # ── Feature contribution summary ────────────────────────────────────────
    full_avg = results['Full Contextual'].mean()
    print(f"\n{'─'*72}")
    print("  FEATURE CONTRIBUTION (Full avg minus ablated avg)")
    print(f"{'─'*72}")
    print(f"\n  {'Feature':<22}  {'Δ vs Full':>10}  {'Interpretation'}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*30}")
    for name, flags in ablation:
        delta = full_avg - results[name].mean()
        sign  = '+' if delta >= 0 else ''
        interp = 'helps' if delta > 0.5 else ('neutral' if abs(delta) <= 0.5 else 'hurts')
        print(f"  {name:<22}  {sign}{delta:>9.2f}%  {interp}")

    base_delta = full_avg - results['Base-rate only'].mean()
    print(f"  {'All features (total)':<22}  +{base_delta:>8.2f}%  vs pure base-rate")

    # ── Full leaderboard ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FULL LEADERBOARD")
    print(f"{'=' * 72}")
    print(f"\n  {'Model':<35}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    # Benchmarks (prev)
    for name, val in BENCHMARKS.items():
        print(f"  {name:<35}  {val:>8.2f}%  {'—':>8}  {'—':>8}  {'—':>7}  [prev]")

    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    # New results sorted by avg
    sorted_res = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    for name, arr in sorted_res:
        flag = '  ★ NEW BEST' if arr.mean() > max(BENCHMARKS.values()) else ''
        print(f"  {name:<35}  {arr.mean():>8.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>7.4f}{flag}")

    print(f"\n{'=' * 72}")
    print("  CONTEXTUAL MARKOV COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
