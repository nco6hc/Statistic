"""
Variable-Order Markov Chain  (Orders 1, 2, 3)  +  Bayesian Model Averaging
=============================================================================

Per-student state at order k:
    s = (binary_{t-1}, binary_{t-2}, ..., binary_{t-k})   ∈ {0,1}^k
    → 2^k possible contexts per student

Transition counts:
    counts[s, context]  = times student s was called given this k-day context
    totals[s, context]  = times this context was seen for student s

Laplace-smoothed probability:
    P(s_t=1 | context) = (counts[s, context] + α) / (totals[s, context] + 2α)

Bayesian Model Averaging  (BMA):
    Each order k is a separate model M_k.
    Weight estimation via leave-last-15% log-likelihood on training data:
        log w_k  ∝  Σ_{t ∈ val} Σ_s log P_k(s_t | context_{t,s})
    Posterior weights normalised with softmax (temperature τ=1).
    Final probability:
        P_BMA(s) = Σ_k w_k × P_k(s)

Benchmarks (100-day, best-of-10-groups oracle):
    Markov only (α=0)       :  26.17%
    Contextual Markov       :  31.17%
    DPM optimized           :  30.83%
    Hybrid optimized        :  32.17%

Usage:
    cd dl_pipeline
    python variable_markov/run_variable_markov.py
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
from itertools import product

# ── Config ─────────────────────────────────────────────────────────────────
SEED        = 42
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_STUDENTS  = 55
K_SELECT    = 6       # students selected per day
TRAIN_RATIO = 0.9
N_SIM_DAYS  = 100
N_GROUPS    = 10
TEMPERATURE = 1.0
SMOOTHING   = 1.0     # Laplace α
VAL_FRAC    = 0.15    # fraction of train used for BMA weight estimation
BMA_TEMP    = 1.0     # softmax temperature for weight normalisation

ORDERS = [1, 2, 3]

BENCHMARKS = {
    'Markov only (α=0)  ': 26.17,
    'Contextual Markov  ': 31.17,
    'DPM optimized      ': 30.83,
    'Hybrid optimized   ': 32.17,
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
# Variable-Order Markov model (single order)
# ══════════════════════════════════════════════════════════════════════════════

class VOMCModel:
    """
    Per-student Variable-order Markov Chain of a fixed order k.

    Internal storage:
        counts[s, ctx]   (N, 2^k)  — times student s called given context ctx
        totals[s, ctx]   (N, 2^k)  — times context ctx appeared for student s

    Context encoding:
        ctx = binary int from (b_{t-1}, b_{t-2}, ..., b_{t-k})
        e.g. order-2: (b_{t-1}=1, b_{t-2}=0) → ctx = 1*2^0 + 0*2^1 = 1
    """

    def __init__(self, order: int, n_students: int = N_STUDENTS,
                 smoothing: float = SMOOTHING):
        self.order    = order
        self.N        = n_students
        self.alpha    = smoothing
        n_ctx         = 2 ** order
        self.counts   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.totals   = np.zeros((n_students, n_ctx), dtype=np.float64)
        self.base_cnt = np.zeros(n_students, dtype=np.float64)   # for fallback
        self.n_days   = 0.0

    @staticmethod
    def _encode(history: np.ndarray, order: int) -> int:
        """
        Encode last `order` days into an integer context index.
        history shape: (order, N_STUDENTS) — history[-1] = yesterday
        Returns integer ctx per student as array of shape (N_STUDENTS,).
        """
        ctx = np.zeros(history.shape[1], dtype=np.int32)
        for lag in range(order):
            # lag=0 → yesterday (most recent), bit position 0
            ctx += history[-(lag + 1)].astype(np.int32) * (2 ** lag)
        return ctx

    def fit(self, binary: np.ndarray) -> None:
        n = len(binary)
        self.n_days = float(n)
        self.base_cnt = binary.sum(axis=0).astype(np.float64)

        self.counts[:] = 0.0
        self.totals[:] = 0.0

        for t in range(self.order, n):
            history = binary[t - self.order: t]    # (order, N)
            ctx     = self._encode(history, self.order)   # (N,)
            today   = binary[t]                          # (N,)

            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if today[s] == 1:
                    self.counts[s, c] += 1.0

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        """
        window: recent days matrix (n_recent, N_STUDENTS), n_recent >= order.
        Returns probability vector (N_STUDENTS,).
        """
        if len(window) < self.order:
            # fall back to base rate
            base = (self.base_cnt + self.alpha) / (self.n_days + 2 * self.alpha)
            return (base / base.sum()).astype(np.float32)

        history = window[-self.order:]              # (order, N)
        ctx     = self._encode(history, self.order) # (N,)
        probs   = np.zeros(self.N, dtype=np.float64)

        for s in range(self.N):
            c     = ctx[s]
            cnt   = self.counts[s, c]
            tot   = self.totals[s, c]
            if tot > 0:
                probs[s] = (cnt + self.alpha) / (tot + 2 * self.alpha)
            else:
                # Context never seen → fall back to global rate
                probs[s] = (self.base_cnt[s] + self.alpha) / (self.n_days + 2 * self.alpha)

        probs = np.clip(probs, 1e-9, None)
        return (probs / probs.sum()).astype(np.float32)

    def log_likelihood(self, binary: np.ndarray) -> float:
        """Compute log-likelihood on a held-out segment."""
        ll   = 0.0
        n    = len(binary)
        for t in range(self.order, n):
            history = binary[t - self.order: t]
            ctx     = self._encode(history, self.order)
            today   = binary[t]
            for s in range(self.N):
                c     = ctx[s]
                cnt   = self.counts[s, c]
                tot   = self.totals[s, c]
                if tot > 0:
                    p = (cnt + self.alpha) / (tot + 2 * self.alpha)
                else:
                    p = (self.base_cnt[s] + self.alpha) / (self.n_days + 2 * self.alpha)
                p   = float(np.clip(p, 1e-9, 1 - 1e-9))
                obs = int(today[s])
                ll += obs * np.log(p) + (1 - obs) * np.log(1 - p)
        return ll

    def update(self, obs: np.ndarray, window: np.ndarray) -> None:
        """Online update with one new observation."""
        if len(window) >= self.order:
            history = window[-self.order:]
            ctx     = self._encode(history, self.order)
            for s in range(self.N):
                c = ctx[s]
                self.totals[s, c] += 1.0
                if obs[s] == 1:
                    self.counts[s, c] += 1.0
        self.base_cnt += obs.astype(np.float64)
        self.n_days   += 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Bayesian Model Averaging
# ══════════════════════════════════════════════════════════════════════════════

class BMAMarkov:
    """
    Bayesian Model Averaging over orders {1, 2, 3}.

    Training:
        1. Fit each VOMCModel on the first (1-val_frac) of training data.
        2. Compute log-likelihood of each on the last val_frac of training data.
        3. Weights = softmax(log_likelihoods / τ).

    Prediction:
        P_BMA(s) = Σ_k w_k × P_k(s | context)

    Online update:
        All sub-models updated simultaneously with each new observation.
    """

    def __init__(self, orders: list[int] = None, n_students: int = N_STUDENTS,
                 smoothing: float = SMOOTHING, val_frac: float = VAL_FRAC,
                 bma_temp: float = BMA_TEMP):
        self.orders    = orders or ORDERS
        self.val_frac  = val_frac
        self.bma_temp  = bma_temp
        self.models    = {k: VOMCModel(k, n_students, smoothing) for k in self.orders}
        self.weights   = np.ones(len(self.orders)) / len(self.orders)   # uniform prior

    def fit(self, binary: np.ndarray) -> None:
        n       = len(binary)
        val_cut = int(n * (1 - self.val_frac))

        # Sub-models fit on fit split only
        for k, m in self.models.items():
            m.fit(binary[:val_cut])

        # Log-likelihoods on held-out val segment
        log_lls = []
        for k in self.orders:
            ll = self.models[k].log_likelihood(binary[val_cut:])
            log_lls.append(ll)

        log_lls = np.array(log_lls, dtype=np.float64)

        # Softmax with temperature
        log_w = log_lls / self.bma_temp
        log_w -= log_w.max()
        w     = np.exp(log_w)
        self.weights = w / w.sum()

        print(f"    BMA weights after val log-likelihood:")
        for k, wt, ll in zip(self.orders, self.weights, log_lls):
            print(f"      Order-{k}: weight={wt:.4f}  val_ll={ll:.1f}")

        # Re-fit sub-models on full training data for simulation
        for k, m in self.models.items():
            m.fit(binary)

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        combined = np.zeros(N_STUDENTS, dtype=np.float64)
        for wt, k in zip(self.weights, self.orders):
            combined += wt * self.models[k].predict_probs(window).astype(np.float64)
        combined = np.clip(combined, 1e-9, None)
        return (combined / combined.sum()).astype(np.float32)

    def update(self, obs: np.ndarray, window: np.ndarray) -> None:
        for m in self.models.values():
            m.update(obs, window)


# ══════════════════════════════════════════════════════════════════════════════
# Group sampling
# ══════════════════════════════════════════════════════════════════════════════

def _sample_groups(probs: np.ndarray, n_groups: int, temperature: float) -> list[list[int]]:
    np.random.seed(None)   # allow fresh samples each call
    p = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p /= p.sum()
    groups: list[list[int]] = []
    for _ in range(n_groups * 4):
        g = sorted(np.random.choice(N_STUDENTS, K_SELECT, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SELECT:].tolist()))
    return groups[:n_groups]


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def simulate_vomc(binary: np.ndarray, split: int,
                  model,          # VOMCModel | BMAMarkov
                  label: str) -> np.ndarray:
    np.random.seed(SEED)
    n_sim = min(N_SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()

    for i in range(n_sim):
        idx    = split + i
        # Window: all available history up to current day
        win    = binary[max(0, idx - 30): idx]

        actual = set(np.where(binary[idx] == 1)[0].tolist())
        probs  = model.predict_probs(win)
        groups = _sample_groups(probs, N_GROUPS, TEMPERATURE)
        best   = max(len(set(g) & actual) for g in groups) / K_SELECT
        accs.append(best)

        if (i + 1) in {1, 10, 25, 50, 75, 100}:
            print(f"    Day {i+1:3d}: {best*100:.1f}%  actual={sorted(actual)}")

        model.update(binary[idx], win)

    arr  = np.array(accs) * 100.0
    dist = {k: int((arr == k / K_SELECT * 100).sum()) for k in range(K_SELECT + 1)}
    nz   = {k: v for k, v in dist.items() if v > 0}
    print(f"\n  ── {label} ──")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("VARIABLE-ORDER MARKOV  +  BAYESIAN MODEL AVERAGING".center(72))
    print("Orders 1, 2, 3  →  BMA".center(72))
    print("=" * 72)

    binary = load_binary()
    split  = int(len(binary) * TRAIN_RATIO)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  Prediction: n_groups={N_GROUPS}, T={TEMPERATURE}, smoothing={SMOOTHING}")
    print(f"  BMA val fraction: {VAL_FRAC*100:.0f}% of train for weight estimation\n")

    results: dict[str, np.ndarray] = {}

    # ── Orders 1, 2, 3 individually ────────────────────────────────────────
    for order in ORDERS:
        n_ctx = 2 ** order
        print(f"{'─'*72}")
        print(f"  ORDER-{order} MARKOV   [{n_ctx} contexts per student]")
        print(f"{'─'*72}")

        model = VOMCModel(order)
        model.fit(binary[:split])
        base  = model.base_cnt / model.n_days
        print(f"    Base rate range : [{base.min():.4f}, {base.max():.4f}]")
        # Show context population for this order
        filled = (model.totals > 0).sum()
        total  = model.totals.size
        print(f"    Contexts seen   : {filled}/{total}  "
              f"({filled/total*100:.1f}% populated)")

        results[f'Order-{order}'] = simulate_vomc(binary, split, model, f'Order-{order} Markov')

    # ── Bayesian Model Averaging ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  BAYESIAN MODEL AVERAGING  (Orders 1+2+3)")
    print(f"{'─'*72}")

    bma = BMAMarkov(orders=ORDERS)
    bma.fit(binary[:split])
    results['BMA (1+2+3)'] = simulate_vomc(binary, split, bma, 'BMA (Orders 1+2+3)')

    # ── Order comparison table ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  ORDER COMPARISON")
    print(f"{'─'*72}")
    print(f"\n  {'Model':<20}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>8}  {'0-wrong':>8}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for name, arr in results.items():
        zero_wrong = int((arr >= 16.67).sum())   # days with ≥ 1 correct
        print(f"  {name:<20}  {arr.mean():>8.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>8.4f}  {zero_wrong:>6}/100")

    # ── BMA weight analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  BMA WEIGHT ANALYSIS  (posterior model probabilities)")
    print(f"{'─'*72}")
    bma2 = BMAMarkov(orders=ORDERS)     # fresh fit to re-print weights cleanly
    bma2.fit(binary[:split])
    print(f"\n  Interpretation:")
    best_order = ORDERS[int(np.argmax(bma2.weights))]
    for k, wt in zip(ORDERS, bma2.weights):
        bar = '█' * int(wt * 40)
        print(f"    Order-{k}: {wt:.4f}  {bar}")
    print(f"\n  Dominant order: {best_order}  "
          f"(highest log-likelihood on held-out training data)")

    # ── Full leaderboard ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  FULL LEADERBOARD")
    print(f"{'=' * 72}")
    print(f"\n  {'Model':<35}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    for name, val in BENCHMARKS.items():
        print(f"  {name:<35}  {val:>8.2f}%  {'—':>8}  {'—':>8}  {'—':>7}  [prev]")

    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    all_results = list(results.items())
    all_results.sort(key=lambda x: x[1].mean(), reverse=True)
    best_prev = max(BENCHMARKS.values())
    for name, arr in all_results:
        flag = '  ★ NEW BEST' if arr.mean() > best_prev else ''
        print(f"  {name:<35}  {arr.mean():>8.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>7.4f}{flag}")

    print(f"\n{'=' * 72}")
    print("  VARIABLE-ORDER MARKOV COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
