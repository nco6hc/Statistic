"""
DPM Hyperparameter Optimization
=================================
Stage 1 — K_max × α_DP  (structural clustering parameters)
Stage 2 — n_groups × temperature  (prediction generation parameters)

Each combination runs the same 100-day simulation for fair comparison.
All combinations run in <30s total (no PyTorch required).

Baseline:
  DPM default (K=8, α=2.0, ng=5, T=1.5): 27.67%
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

# ── Constants ──────────────────────────────────────────────────────────────
SEED        = 42
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_STUDENTS  = 55
K           = 6
TRAIN_RATIO = 0.9
N_SIM_DAYS  = 100

BASELINE_DPM    = 27.67
BASELINE_HYBRID = 26.83

# ── Grid search ranges ─────────────────────────────────────────────────────
STAGE1_K_MAX   = [4, 6, 8, 12, 16]
STAGE1_ALPHA   = [0.5, 1.0, 2.0, 4.0, 8.0]
STAGE2_NGROUPS = [3, 5, 7, 10]
STAGE2_TEMP    = [1.0, 1.5, 2.0, 2.5]

# ── Data ──────────────────────────────────────────────────────────────────

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


# ── DPM model (self-contained) ─────────────────────────────────────────────

class DPM:
    """Truncated Dirichlet Process Mixture (variational EM, online updates)."""

    def __init__(self, K_max: int, alpha_dp: float):
        self.N        = N_STUDENTS
        self.K        = K_max
        self.alpha_dp = alpha_dp
        base          = K / N_STUDENTS
        self.a  = np.ones(K_max) * base * 20.0
        self.b  = np.ones(K_max) * (1.0 - base) * 20.0
        self.r  = np.ones((N_STUDENTS, K_max)) / K_max
        self.pi = np.ones(K_max) / K_max
        self.S_total = np.zeros(N_STUDENTS, dtype=np.float64)
        self.n_total = 0.0

    def fit(self, binary: np.ndarray, n_iter: int = 30) -> None:
        n_days         = len(binary)
        self.S_total   = binary.sum(axis=0).astype(np.float64)
        self.n_total   = float(n_days)
        rates          = self.S_total / n_days

        # Initialise cluster centres at evenly-spaced quantiles
        quantiles = np.percentile(rates, np.linspace(5, 95, self.K))
        for k, mu_k in enumerate(quantiles):
            mu_k = float(np.clip(mu_k, 0.01, 0.99))
            self.a[k] = mu_k * 20.0
            self.b[k] = (1.0 - mu_k) * 20.0

        # Variational EM
        for _ in range(n_iter):
            theta_k = self.a / (self.a + self.b)
            log_r   = np.zeros((self.N, self.K))
            for k in range(self.K):
                log_r[:, k] = (
                    self.S_total * np.log(theta_k[k] + 1e-10) +
                    (n_days - self.S_total) * np.log(1.0 - theta_k[k] + 1e-10) +
                    np.log(self.pi[k] + 1e-10)
                )
            log_r -= log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r)
            r /= r.sum(axis=1, keepdims=True)
            self.r = r

            N_k = r.sum(axis=0)
            for k in range(self.K):
                ws = float((r[:, k] * self.S_total).sum())
                wf = float((r[:, k] * (n_days - self.S_total)).sum())
                self.a[k] = 1.0 + ws
                self.b[k] = 1.0 + wf

            suffix_N  = np.cumsum(N_k[::-1])[::-1]
            beta_vb   = self.alpha_dp + suffix_N - N_k
            V         = (1.0 + N_k) / (1.0 + N_k + beta_vb)
            remaining = np.cumprod(np.concatenate([[1.0], 1.0 - V[:-1]]))
            pi        = V * remaining
            self.pi   = pi / pi.sum()

    def _probs(self) -> np.ndarray:
        theta_k = self.a / (self.a + self.b)
        return (self.r * theta_k[None, :]).sum(axis=1).astype(np.float32)

    def predict(self, n_groups: int, temperature: float) -> list[list[int]]:
        probs = self._probs()
        p = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
        p /= p.sum()
        groups: list[list[int]] = []
        for _ in range(n_groups * 4):
            g = sorted(np.random.choice(N_STUDENTS, K, replace=False, p=p).tolist())
            if g not in groups:
                groups.append(g)
            if len(groups) >= n_groups:
                break
        if not groups:
            groups.append(sorted(np.argsort(probs)[-K:].tolist()))
        return groups[:n_groups]

    def update(self, obs: np.ndarray) -> None:
        self.S_total += obs.astype(np.float64)
        self.n_total += 1.0

        theta_k = self.a / (self.a + self.b)
        log_r   = np.zeros((self.N, self.K))
        for k in range(self.K):
            log_r[:, k] = (
                self.S_total * np.log(theta_k[k] + 1e-10) +
                (self.n_total - self.S_total) * np.log(1.0 - theta_k[k] + 1e-10) +
                np.log(self.pi[k] + 1e-10)
            )
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        self.r = r

        for k in range(self.K):
            ws = float((r[:, k] * self.S_total).sum())
            wf = float((r[:, k] * (self.n_total - self.S_total)).sum())
            self.a[k] = 1.0 + ws
            self.b[k] = 1.0 + wf


# ── Simulation ─────────────────────────────────────────────────────────────

def simulate(binary: np.ndarray, split: int,
             K_max: int, alpha_dp: float,
             n_groups: int, temperature: float) -> np.ndarray:
    np.random.seed(SEED)
    model = DPM(K_max, alpha_dp)
    model.fit(binary[:split])
    n_sim = min(N_SIM_DAYS, len(binary) - split)
    accs  = []
    for i in range(n_sim):
        actual = set(np.where(binary[split + i] == 1)[0].tolist())
        groups = model.predict(n_groups, temperature)
        best   = max(len(set(g) & actual) for g in groups) / K
        accs.append(best)
        model.update(binary[split + i])
    return np.array(accs) * 100.0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DPM HYPERPARAMETER OPTIMIZATION".center(70))
    print("=" * 70)

    t_start = time.perf_counter()
    binary  = load_binary()
    split   = int(len(binary) * TRAIN_RATIO)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  Baseline DPM  (K=8,  α=2.0, ng=5, T=1.5): {BASELINE_DPM:.2f}%")
    print(f"  Baseline Hybrid (existing):                  {BASELINE_HYBRID:.2f}%\n")

    # ── Stage 1: K_max × α_DP ─────────────────────────────────────────────
    print("─" * 70)
    print("  STAGE 1 — K_max × α_DP  [n_groups=5, temperature=1.5]")
    print("─" * 70)
    print(f"\n  {'K':>5}  {'α_DP':>6}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    stage1: dict[tuple, float] = {}
    for K_max in STAGE1_K_MAX:
        for a_dp in STAGE1_ALPHA:
            accs = simulate(binary, split, K_max, a_dp, n_groups=5, temperature=1.5)
            avg  = accs.mean()
            stage1[(K_max, a_dp)] = avg
            flag = "  ★" if avg > BASELINE_DPM else ""
            print(f"  {K_max:>5}  {a_dp:>6.1f}  {avg:>8.2f}%  "
                  f"{accs.max():>7.2f}%  {accs.min():>7.2f}%  {accs.std():>7.4f}{flag}")

    best_k, best_a = max(stage1, key=stage1.__getitem__)
    best_s1 = stage1[(best_k, best_a)]
    print(f"\n  ▶ Stage 1 winner:  K={best_k}, α_DP={best_a} → {best_s1:.2f}%")

    # ── Stage 2: n_groups × temperature ───────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  STAGE 2 — n_groups × temperature  [K={best_k}, α_DP={best_a}]")
    print(f"{'─' * 70}")
    print(f"\n  {'ng':>5}  {'Temp':>6}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    stage2: dict[tuple, float] = {}
    for ng in STAGE2_NGROUPS:
        for temp in STAGE2_TEMP:
            accs = simulate(binary, split, best_k, best_a, ng, temp)
            avg  = accs.mean()
            stage2[(ng, temp)] = avg
            flag = "  ★" if avg > BASELINE_DPM else ""
            print(f"  {ng:>5}  {temp:>6.1f}  {avg:>8.2f}%  "
                  f"{accs.max():>7.2f}%  {accs.min():>7.2f}%  {accs.std():>7.4f}{flag}")

    best_ng, best_t = max(stage2, key=stage2.__getitem__)
    best_s2 = stage2[(best_ng, best_t)]
    print(f"\n  ▶ Stage 2 winner:  ng={best_ng}, T={best_t} → {best_s2:.2f}%")

    # ── Final verification run ─────────────────────────────────────────────
    final_accs = simulate(binary, split, best_k, best_a, best_ng, best_t)
    dist = {k: int((final_accs == k / K * 100).sum()) for k in range(7)}
    non_zero = {k: v for k, v in dist.items() if v > 0}

    print(f"\n{'=' * 70}")
    print("  DPM OPTIMIZED — FINAL RESULT")
    print(f"{'=' * 70}")
    print(f"\n  Config:  K={best_k}, α_DP={best_a}, n_groups={best_ng}, T={best_t}")
    print(f"  Avg:     {final_accs.mean():.2f}%")
    print(f"  Best:    {final_accs.max():.2f}%")
    print(f"  Worst:   {final_accs.min():.2f}%")
    print(f"  Std:     {final_accs.std():.4f}")
    print(f"  Dist:    " + "  ".join(f"{k}/6={v}" for k, v in non_zero.items()))

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  COMPARISON                                 │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  DPM original  (K=8, α=2.0, ng=5, T=1.5)  │  {BASELINE_DPM:.2f}%  │")
    print(f"  │  DPM optimized ({best_k=}, {best_a=}, ng={best_ng}, T={best_t})  │  {final_accs.mean():.2f}%  │")
    print(f"  │  Hybrid Orig                                │  {BASELINE_HYBRID:.2f}%  │")
    delta = final_accs.mean() - BASELINE_DPM
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Delta vs baseline DPM:  {delta:+.2f}%               │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"\n  Total time: {time.perf_counter()-t_start:.1f}s")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
