"""
Dirichlet Process Mixture of Markov Chains  (DPMoMC)
=====================================================

Generative model:
    π ~ StickBreaking(α_DP)                         ← regime weights
    θ_{s,k,b} ~ Beta(a₀, b₀)  ∀ s, k, b ∈ {0,1}  ← per-regime, per-context transitions
    z_t  ~ Categorical(π)                            ← latent regime for day t
    binary[t,s] ~ Bernoulli(θ_{s, z_t, binary[t-1,s]})

  Context b = binary[t-1, s] ∈ {0,1}:
    b=0 → P(s called today | s NOT called yesterday, regime k)
    b=1 → P(s called today | s WAS called yesterday, regime k)

  Each regime k learns different "teacher calling style" — e.g.:
    k=0: strong rotation (high P(s|b=0), low P(s|b=1))
    k=1: repetition    (low P(s|b=0), high P(s|b=1))
    k=2: neutral       (uniform)

Optimisation:
  A) EM + Simulated Annealing
       E-step  : r[t,k] ∝ π_k × P(binary_t | ctx_t, θ_k)^{1/τ}
       M-step  : θ̂_{s,k,b}  = a_{s,k,b} / (a_{s,k,b} + b_{s,k,b})
                  π̂_k       = N_k / Σ N_k
       Anneal  : τ: τ_start → 1.0  (exponential, over n_iter steps)
       Effect  : high τ → diffuse (explore), τ→1 → sharp (converge)

  B) Variational Inference  (mean-field)
       q(z_t)        = Categorical(φ_t)
       q(θ_{s,k,b})  = Beta(â_{s,k,b}, b̂_{s,k,b})
       q(V_k)        = Beta(γ1_k, γ2_k)   ← stick-breaking
       E-step  : log φ_{t,k} ∝ E[log π_k] + Σ_s E[log P(binary_t | θ)]
                  expectations via digamma function
       M-step  : â, b̂ updated with φ-weighted counts
                  γ1_k = 1 + N_k,  γ2_k = α_DP + Σ_{j>k} N_j
       Tracks  : ELBO = E[log-lik] + H[q(Z)] per iteration

Prediction (both methods):
    P(s selected | ctx_s=b) = Σ_k π_k × â_{s,k,b}/(â_{s,k,b}+b̂_{s,k,b})

Online update (simulation):
    For each new observation day: one E-step → soft-update θ, π

Benchmarks (100-day, best-of-10 oracle):
    Markov only (α=0)  :  26.17%
    DPM opt (ng=10)    :  30.83%
    Order-3 Markov     :  31.67%
    Contextual Markov  :  31.17%
    Hybrid optimised   :  32.17%

Usage:
    cd dl_pipeline
    python dpmm_markov/run_dpmm_markov.py
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

# ── Digamma (try scipy; fall back to Stirling series) ─────────────────────
try:
    from scipy.special import digamma as _digamma
except ImportError:
    def _digamma(x: np.ndarray) -> np.ndarray:   # type: ignore[misc]
        x  = np.asarray(x, dtype=np.float64)
        r  = np.zeros_like(x)
        y  = x.copy()
        # Shift to y >= 6 using recurrence: ψ(x) = ψ(x+1) − 1/x
        while np.any(y < 6.0):
            m = y < 6.0
            r[m] -= 1.0 / y[m]
            y[m] += 1.0
        # Asymptotic expansion (accurate for y >= 6)
        r += np.log(y) - 0.5/y - 1.0/(12*y**2) + 1.0/(120*y**4)
        return r


# ── Config ─────────────────────────────────────────────────────────────────
SEED        = 42
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55          # students
K_SEL       = 6           # selected per day
TRAIN_R     = 0.9
SIM_DAYS    = 100
N_GROUPS    = 10
TEMPERATURE = 1.0

K_MAX       = 8           # truncated DP clusters
ALPHA_DP    = 2.0
N_ITER      = 40
TAU_START   = 5.0         # EM annealing: high τ = soft
TAU_END     = 1.0         # EM annealing: τ=1 = standard EM

BASE        = K_SEL / N_S           # ~0.1091
A0          = BASE * 20             # Beta prior α  ≈ 2.18
B0          = (1 - BASE) * 20       # Beta prior β  ≈ 17.82

BENCHMARKS = {
    'Markov only (α=0)    ': 26.17,
    'DPM opt (ng=10)      ': 30.83,
    'Contextual Markov    ': 31.17,
    'Order-3 Markov       ': 31.67,
    'Hybrid optimised     ': 32.17,
}


# ── Data ───────────────────────────────────────────────────────────────────

def load_binary() -> np.ndarray:
    df = pd.read_excel(EXCEL_PATH)
    df = df.sort_values('Days').reset_index(drop=True)
    binary = np.zeros((len(df), N_S), dtype=np.float32)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]
    for row_idx, row in df.iterrows():
        for col in id_cols:
            sid = int(row[col]) - 1
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# Shared base: prediction + online update
# ══════════════════════════════════════════════════════════════════════════════

class DPMoMCBase:
    """
    Shared state and inference for both optimisers.

    Parameters
    ----------
    a, b : ndarray (N_S, K, 2)  — variational Beta params; dim-2 indexes context b ∈ {0,1}
    pi   : ndarray (K,)         — cluster mixing weights (point estimate)
    N_k  : ndarray (K,)         — cumulative soft counts (for online π update)
    """

    def __init__(self, K_max: int = K_MAX, alpha_dp: float = ALPHA_DP):
        self.K        = K_max
        self.alpha_dp = alpha_dp
        # Beta variational / EM params: (N_S, K, 2)
        self.a  = np.full((N_S, K_max, 2), A0,  dtype=np.float64)
        self.b  = np.full((N_S, K_max, 2), B0,  dtype=np.float64)
        self.pi = np.ones(K_max, dtype=np.float64) / K_max
        self.N_k = np.ones(K_max, dtype=np.float64)   # soft cluster counts

    def _theta(self) -> np.ndarray:
        """Posterior mean of Beta: (N_S, K, 2)."""
        return self.a / (self.a + self.b)

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        """
        Marginalise over clusters: P(s) = Σ_k π_k × θ_{s,k,ctx_s}
        window: (n_recent, N_S) — window[-1] = yesterday
        """
        if len(window) < 1:
            return np.full(N_S, 1.0 / N_S, dtype=np.float32)
        ctx   = window[-1].astype(np.int32)                       # (N_S,)
        theta = self._theta()                                      # (N_S, K, 2)
        # ctx_theta[s, k] = θ_{s,k,ctx_s}
        ctx_theta = (theta[:, :, 0] * (1 - ctx[:, None]) +
                     theta[:, :, 1] *      ctx[:, None])          # (N_S, K)
        probs = (self.pi[None, :] * ctx_theta).sum(axis=1)        # (N_S,)
        probs = np.clip(probs, 1e-9, None)
        return (probs / probs.sum()).astype(np.float32)

    def _e_step_day(self, obs: np.ndarray, ctx: np.ndarray,
                    log_pi: np.ndarray) -> np.ndarray:
        """E-step for a single day → soft assignment r (K,)."""
        theta     = self._theta()                                  # (N_S, K, 2)
        ctx_theta = (theta[:, :, 0] * (1 - ctx[:, None]) +
                     theta[:, :, 1] *      ctx[:, None])          # (N_S, K)
        log_lik   = (obs[:, None] * np.log(ctx_theta + 1e-10) +
                     (1 - obs[:, None]) * np.log(1 - ctx_theta + 1e-10)
                     ).sum(axis=0)                                 # (K,)
        log_r = log_pi + log_lik
        log_r -= log_r.max()
        r = np.exp(log_r)
        return r / r.sum()

    def update(self, obs: np.ndarray, window: np.ndarray) -> None:
        """Online Bayesian update after observing one new day."""
        if len(window) < 1:
            return
        ctx   = window[-1].astype(np.float64)
        obs_f = obs.astype(np.float64)
        r     = self._e_step_day(obs_f, ctx, np.log(self.pi + 1e-10))  # (K,)

        for b in range(2):
            m = (window[-1].astype(np.int32) == b)                    # (N_S,)
            # self.a[m, :, b] += r * obs_f[m, None]
            self.a[m, :, b] += r[None, :] * obs_f[m, None]
            self.b[m, :, b] += r[None, :] * (1.0 - obs_f[m, None])

        self.N_k  += r
        self.pi    = self.N_k / self.N_k.sum()


# ══════════════════════════════════════════════════════════════════════════════
# A) EM + Simulated Annealing
# ══════════════════════════════════════════════════════════════════════════════

class DPMoMC_EM(DPMoMCBase):
    """
    EM with simulated annealing for DPMoMC.

    Annealing role
    ──────────────
    Standard EM maximises a lower bound; it gets trapped in poor local optima.
    We temper the E-step assignment:
        r[t,k] ∝ (π_k × P(data_t | cluster k))^{1/τ}
    High τ → r is nearly uniform (ignores likelihood → explores all clusters).
    As τ → 1 the algorithm recovers standard EM.
    Schedule: τ(i) = τ_start × (τ_end / τ_start)^{i / (n_iter−1)}
    """

    def __init__(self, K_max: int = K_MAX, alpha_dp: float = ALPHA_DP,
                 n_iter: int = N_ITER, tau_start: float = TAU_START,
                 tau_end: float = TAU_END):
        super().__init__(K_max, alpha_dp)
        self.n_iter    = n_iter
        self.tau_start = tau_start
        self.tau_end   = tau_end

    def fit(self, binary: np.ndarray) -> None:
        n       = len(binary)
        ctx_all = binary[:-1].astype(np.float64)   # (T, N_S)
        obs_all = binary[1:].astype(np.float64)    # (T, N_S)
        T       = len(obs_all)

        # Initialise cluster centres at quantile-spaced rates
        rates     = binary.mean(axis=0)
        quantiles = np.percentile(rates, np.linspace(5, 95, self.K))
        for k, q in enumerate(quantiles):
            q = float(np.clip(q, 0.01, 0.99))
            self.a[:, k, :] = q * 20.0
            self.b[:, k, :] = (1.0 - q) * 20.0

        self.pi  = np.ones(self.K, dtype=np.float64) / self.K
        self.N_k = np.ones(self.K, dtype=np.float64)

        log_liks: list[float] = []
        t0 = time.perf_counter()

        for it in range(self.n_iter):
            tau = (self.tau_start *
                   (self.tau_end / self.tau_start) ** (it / max(self.n_iter - 1, 1)))

            # ── E-step (vectorised) ───────────────────────────────────────
            theta  = self._theta()             # (N_S, K, 2)
            theta0 = theta[:, :, 0]            # (N_S, K)
            theta1 = theta[:, :, 1]

            # ctx_theta[t, s, k] = θ_{s,k,ctx[t,s]}
            ctx_theta = ((1 - ctx_all[:, :, None]) * theta0[None, :, :] +
                              ctx_all[:, :, None]  * theta1[None, :, :])  # (T, N_S, K)

            log_lik = (obs_all[:, :, None] * np.log(ctx_theta + 1e-10) +
                       (1 - obs_all[:, :, None]) * np.log(1 - ctx_theta + 1e-10)
                       ).sum(axis=1)                                       # (T, K)

            log_r = (np.log(self.pi + 1e-10)[None, :] + log_lik) / tau    # (T, K)
            log_r -= log_r.max(axis=1, keepdims=True)
            r      = np.exp(log_r)
            r     /= r.sum(axis=1, keepdims=True)                         # (T, K)

            log_liks.append(float((r * log_lik).sum()))

            # ── M-step ───────────────────────────────────────────────────
            N_k       = r.sum(axis=0) + 1e-8                               # (K,)
            self.pi   = N_k / N_k.sum()
            self.N_k  = N_k

            for b in range(2):
                mask     = (ctx_all == b)                                  # (T, N_S)
                w_r      = mask[:, :, None] * r[:, None, :]               # (T, N_S, K)
                self.a[:, :, b] = A0 + (w_r * obs_all[:, :, None]).sum(axis=0)
                self.b[:, :, b] = B0 + (w_r * (1 - obs_all[:, :, None])).sum(axis=0)

        eff_k  = int((self.pi > 1.0 / (self.K * 2)).sum())
        delta5 = (log_liks[-1] - log_liks[-6]) if len(log_liks) >= 6 else 0.0
        print(f"    τ schedule: {self.tau_start:.1f} → {self.tau_end:.1f}  "
              f"({self.n_iter} iters, {time.perf_counter()-t0:.1f}s)")
        print(f"    Final log-lik: {log_liks[-1]:.1f}   "
              f"Δ(last 5 iters): {delta5:+.1f}")
        print(f"    Effective clusters: {eff_k}/{self.K}  "
              f"π=[{self.pi.min():.3f}, {self.pi.max():.3f}]")
        print(f"    Cluster π: {self.pi.round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# B) Variational Inference  (mean-field)
# ══════════════════════════════════════════════════════════════════════════════

class DPMoMC_VI(DPMoMCBase):
    """
    Mean-field VI for DPMoMC.

    Variational family
    ──────────────────
    q(Z) = Π_t Categorical(φ_t)
    q(θ_{s,k,b}) = Beta(â_{s,k,b}, b̂_{s,k,b})
    q(V_k) = Beta(γ1_k, γ2_k)   (stick-breaking representation of π)

    CAVI updates
    ────────────
    φ_{t,k}  ∝ exp( E[log π_k] + Σ_s E[log P(y_{t,s} | θ_{s,k,b})] )

    E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1−V_j)]
        E[log V_k]    = ψ(γ1_k) − ψ(γ1_k + γ2_k)
        E[log(1−V_k)] = ψ(γ2_k) − ψ(γ1_k + γ2_k)

    E[log θ_{s,k,b}]     = ψ(â_{s,k,b}) − ψ(â_{s,k,b} + b̂_{s,k,b})
    E[log(1−θ_{s,k,b})]  = ψ(b̂_{s,k,b}) − ψ(â_{s,k,b} + b̂_{s,k,b})

    β param updates:
        â_{s,k,b}  = a₀ + Σ_{t: ctx=b} φ_{t,k} · y_{t,s}
        b̂_{s,k,b}  = b₀ + Σ_{t: ctx=b} φ_{t,k} · (1−y_{t,s})

    Stick-breaking updates (DP-VB, Blei & Jordan 2006):
        γ1_k = 1 + N_k
        γ2_k = α_DP + Σ_{j>k} N_j    where  N_k = Σ_t φ_{t,k}

    ELBO (tracked, approximate):
        L ≈ Σ_{t,k} φ_{t,k}(E[log π_k] + E[log-lik_{t,k}]) − Σ_t H[q(z_t)]
    """

    def __init__(self, K_max: int = K_MAX, alpha_dp: float = ALPHA_DP,
                 n_iter: int = N_ITER):
        super().__init__(K_max, alpha_dp)
        self.n_iter = n_iter
        # Stick-breaking variational params
        self.gamma1 = np.ones(K_max, dtype=np.float64)
        self.gamma2 = np.full(K_max, alpha_dp, dtype=np.float64)

    def _e_log_pi(self) -> np.ndarray:
        """E[log π_k] under stick-breaking q(V)."""
        dg1  = _digamma(self.gamma1)
        dg2  = _digamma(self.gamma2)
        dg12 = _digamma(self.gamma1 + self.gamma2)
        e_log_v   = dg1 - dg12                                    # (K,)
        e_log_1mv = dg2 - dg12                                    # (K,)
        prefix    = np.concatenate([[0.0], np.cumsum(e_log_1mv[:-1])])
        return e_log_v + prefix                                    # (K,)

    def _e_log_theta(self):
        """
        E[log θ] and E[log(1−θ)] under q(θ)=Beta(a,b).
        Returns (N_S, K, 2) each.
        """
        dg_a  = _digamma(self.a)
        dg_b  = _digamma(self.b)
        dg_ab = _digamma(self.a + self.b)
        return dg_a - dg_ab, dg_b - dg_ab

    def fit(self, binary: np.ndarray) -> None:
        n       = len(binary)
        ctx_all = binary[:-1].astype(np.float64)   # (T, N_S)
        obs_all = binary[1:].astype(np.float64)    # (T, N_S)
        T       = len(obs_all)

        # Initialise as EM does (quantile-spaced cluster centres)
        rates     = binary.mean(axis=0)
        quantiles = np.percentile(rates, np.linspace(5, 95, self.K))
        for k, q in enumerate(quantiles):
            q = float(np.clip(q, 0.01, 0.99))
            self.a[:, k, :] = q * 20.0
            self.b[:, k, :] = (1.0 - q) * 20.0

        self.gamma1 = np.ones(self.K, dtype=np.float64)
        self.gamma2 = np.full(self.K, self.alpha_dp, dtype=np.float64)

        elbos: list[float] = []
        t0 = time.perf_counter()

        for it in range(self.n_iter):
            e_log_pi       = self._e_log_pi()         # (K,)
            e_log_t, e_log_1t = self._e_log_theta()   # (N_S, K, 2) each

            # ── E-step (CAVI, digamma expectations) ───────────────────────
            elt0 = e_log_t[:, :, 0]    # (N_S, K)  context=0
            elt1 = e_log_t[:, :, 1]    # (N_S, K)  context=1
            el10 = e_log_1t[:, :, 0]
            el11 = e_log_1t[:, :, 1]

            # ctx_elt[t, s, k] = E[log θ_{s,k,ctx[t,s]}]
            ctx_elt  = ((1 - ctx_all[:, :, None]) * elt0[None, :, :] +
                             ctx_all[:, :, None]  * elt1[None, :, :])  # (T, N_S, K)
            ctx_el1t = ((1 - ctx_all[:, :, None]) * el10[None, :, :] +
                             ctx_all[:, :, None]  * el11[None, :, :])  # (T, N_S, K)

            # Expected log-likelihood per day per cluster
            e_ll = (obs_all[:, :, None] * ctx_elt +
                    (1 - obs_all[:, :, None]) * ctx_el1t
                    ).sum(axis=1)                                       # (T, K)

            log_phi = e_log_pi[None, :] + e_ll                         # (T, K)
            log_phi -= log_phi.max(axis=1, keepdims=True)
            phi      = np.exp(log_phi)
            phi     /= phi.sum(axis=1, keepdims=True)                  # (T, K) ← q(Z)

            # ELBO (approximate: reconstruction + H[q(Z)])
            recon   = float((phi * e_ll).sum())
            h_qz    = float(-(phi * np.log(phi + 1e-10)).sum())
            elbos.append(recon + h_qz)

            # ── M-step (coordinate ascent) ────────────────────────────────
            N_k = phi.sum(axis=0)                                      # (K,)

            # Stick-breaking variational params
            suffix_N      = np.cumsum(N_k[::-1])[::-1]                # Σ_{j≥k} N_j
            self.gamma1   = 1.0 + N_k
            self.gamma2   = self.alpha_dp + suffix_N - N_k

            # Point-estimate π for predict/online-update
            V_bar    = self.gamma1 / (self.gamma1 + self.gamma2)
            log_1mv  = np.log(np.clip(1.0 - V_bar, 1e-10, None))
            log_pi   = np.log(V_bar + 1e-10) + np.concatenate([[0.0], np.cumsum(log_1mv[:-1])])
            log_pi  -= log_pi.max()
            pi_pt    = np.exp(log_pi)
            self.pi  = pi_pt / pi_pt.sum()
            self.N_k = N_k

            # Beta variational params
            for b in range(2):
                mask  = (ctx_all == b)                                 # (T, N_S)
                w_phi = mask[:, :, None] * phi[:, None, :]            # (T, N_S, K)
                self.a[:, :, b] = A0 + (w_phi * obs_all[:, :, None]).sum(axis=0)
                self.b[:, :, b] = B0 + (w_phi * (1 - obs_all[:, :, None])).sum(axis=0)

        eff_k  = int((self.pi > 1.0 / (self.K * 2)).sum())
        delta5 = (elbos[-1] - elbos[-6]) if len(elbos) >= 6 else 0.0
        print(f"    VI CAVI  ({self.n_iter} iters, {time.perf_counter()-t0:.1f}s)")
        print(f"    Final ELBO: {elbos[-1]:.1f}   Δ(last 5 iters): {delta5:+.2f}")
        print(f"    Effective clusters: {eff_k}/{self.K}  "
              f"π=[{self.pi.min():.3f}, {self.pi.max():.3f}]")
        print(f"    Cluster π: {self.pi.round(3)}")
        # Stick-breaking posterior means
        V_bar = self.gamma1 / (self.gamma1 + self.gamma2)
        print(f"    V̄_k (stick prob): {V_bar.round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

def _sample_groups(probs: np.ndarray, n_groups: int, temperature: float) -> list[list[int]]:
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


def simulate(binary: np.ndarray, split: int,
             model: DPMoMCBase, label: str) -> np.ndarray:
    np.random.seed(SEED)
    n_sim = min(SIM_DAYS, len(binary) - split)
    accs  = []
    t0    = time.perf_counter()

    for i in range(n_sim):
        idx    = split + i
        window = binary[max(0, idx - 30): idx]

        actual = set(np.where(binary[idx] == 1)[0].tolist())
        probs  = model.predict_probs(window)
        groups = _sample_groups(probs, N_GROUPS, TEMPERATURE)
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)

        if (i + 1) in {1, 10, 25, 50, 75, 100}:
            print(f"    Day {i+1:3d}: {best*100:.1f}%  actual={sorted(actual)}")

        model.update(binary[idx], window)

    arr  = np.array(accs) * 100.0
    dist = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
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
    print("DIRICHLET PROCESS MIXTURE OF MARKOV CHAINS".center(72))
    print("EM + Simulated Annealing  |  Variational Inference".center(72))
    print("=" * 72)

    binary = load_binary()
    split  = int(len(binary) * TRAIN_R)
    print(f"\n  Data: {len(binary)} days | Train: {split} | Test: {len(binary)-split}")
    print(f"  K_max={K_MAX}, α_DP={ALPHA_DP}, order=1, n_iter={N_ITER}")
    print(f"  Prior: Beta(a₀={A0:.2f}, b₀={B0:.2f})")
    print(f"  Prediction: n_groups={N_GROUPS}, T={TEMPERATURE}\n")

    results: dict[str, np.ndarray] = {}

    # ── A) EM + Annealing ──────────────────────────────────────────────────
    print("─" * 72)
    print("  [1/2]  EM + SIMULATED ANNEALING")
    print(f"  Temperature: τ = {TAU_START:.1f} → {TAU_END:.1f}  "
          f"(exponential, {N_ITER} iters)")
    print("─" * 72)
    model_em = DPMoMC_EM()
    model_em.fit(binary[:split])
    results['DPMoMC EM+Anneal'] = simulate(binary, split, model_em, 'DPMoMC EM+Anneal')

    # ── B) Variational Inference ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  [2/2]  VARIATIONAL INFERENCE  (mean-field CAVI)")
    print(f"  ELBO optimisation, {N_ITER} coordinate ascent iterations")
    print(f"  ψ-based expectations  |  stick-breaking q(π)  |  Beta q(θ)")
    print("─" * 72)
    model_vi = DPMoMC_VI()
    model_vi.fit(binary[:split])
    results['DPMoMC VI'] = simulate(binary, split, model_vi, 'DPMoMC VI')

    # ── Detailed comparison ─────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  ALGORITHM COMPARISON")
    print(f"{'─'*72}")
    print(f"\n  {'Method':<25}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>8}  "
          f"{'≥1 correct':>12}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")
    for name, arr in results.items():
        n_ok = int((arr >= 16.67).sum())
        print(f"  {name:<25}  {arr.mean():>8.2f}%  {arr.max():>7.2f}%  "
              f"{arr.min():>7.2f}%  {arr.std():>8.4f}  {n_ok:>9}/100")

    # ── Cluster profile (EM) ────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  CLUSTER PROFILES  (EM — transition θ̄ means by context)")
    print(f"{'─'*72}")
    theta_em = model_em._theta()   # (N_S, K, 2)
    print(f"\n  {'Cluster':>8}  {'π_k':>7}  "
          f"{'θ̄_k(b=0)':>12}  {'θ̄_k(b=1)':>12}  {'Δ(b=1−b=0)':>12}  {'Style':>12}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for k in range(K_MAX):
        t0m = float(theta_em[:, k, 0].mean())
        t1m = float(theta_em[:, k, 1].mean())
        delta = t1m - t0m
        if delta > 0.02:
            style = "repetitive"
        elif delta < -0.02:
            style = "rotation"
        else:
            style = "neutral"
        print(f"  {k:>8}  {model_em.pi[k]:>7.4f}  {t0m:>12.4f}  "
              f"{t1m:>12.4f}  {delta:>+12.4f}  {style:>12}")

    # ── VI stick-breaking analysis ──────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  VI STICK-BREAKING  POSTERIOR  q(V_k) = Beta(γ1_k, γ2_k)")
    print(f"{'─'*72}")
    V_bar    = model_vi.gamma1 / (model_vi.gamma1 + model_vi.gamma2)
    V_var    = (model_vi.gamma1 * model_vi.gamma2 /
                ((model_vi.gamma1 + model_vi.gamma2)**2 *
                 (model_vi.gamma1 + model_vi.gamma2 + 1)))
    print(f"\n  {'k':>4}  {'γ1':>8}  {'γ2':>8}  {'V̄_k':>8}  {'Var':>10}  "
          f"{'π_k':>8}  bar")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*24}")
    for k in range(K_MAX):
        bar = '█' * max(1, int(V_bar[k] * 20))
        print(f"  {k:>4}  {model_vi.gamma1[k]:>8.1f}  {model_vi.gamma2[k]:>8.1f}  "
              f"{V_bar[k]:>8.4f}  {V_var[k]:>10.6f}  "
              f"{model_vi.pi[k]:>8.4f}  {bar}")

    # ── Full leaderboard ─────────────────────────────────────────────────────
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

    print(f"\n{'=' * 72}")
    print("  DPMoMC COMPLETE")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
