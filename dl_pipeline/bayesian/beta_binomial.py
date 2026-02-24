"""
Bayesian Beta-Binomial Model for Student Selection Prediction
=============================================================

Mathematical Foundation
-----------------------

PRIOR — Per-student belief before seeing data:

    θ_s ~ Beta(α₀, β₀)

    where α₀ = κ · (k/n),   β₀ = κ · (1 - k/n)
          κ  = prior_concentration  (strength; κ=2 means "2 pseudo-obs")
          k/n = 6/55 ≈ 0.1091       (expected selection rate)

LIKELIHOOD — Each day is independent per student:

    x_s(t) | θ_s ~ Bernoulli(θ_s)

CONJUGATE POSTERIOR UPDATE — Closed-form, O(1) per student:

    θ_s | x ~ Beta(α₀ + Σ_selected, β₀ + Σ_not_selected)

TEMPORAL DISCOUNTING (decay γ ∈ (0,1)):

    count_s(t+1) = γ · count_s(t) + x_s(t)
    total(t+1)   = γ · total(t)   + 1

    α_s = α₀ + count_s(t)
    β_s = β₀ + total(t) − count_s(t)

    Effective window size ≈ 1 / (1 − γ) days
      γ=0.97 → ~33 days   γ=0.95 → ~20 days   γ=0.99 → ~100 days

COOLDOWN PENALTY:

    After student s is selected, add δ to their β_s:
        β_s += cooling_penalty
    This temporarily reduces re-selection probability.
    It decays multiplicatively each day: β_cooling(t+1) = γ · β_cooling(t)

PREDICTION — Thompson Sampling:

    For each candidate group:
        Sample θ_s ~ Beta(α_s, β_s) independently for all s
        Select the k students with highest sampled θ

    This provides natural exploration-exploitation balance:
      - High-posterior students get picked often (exploitation)
      - Uncertain students occasionally get picked (exploration)

EMPIRICAL BAYES (hyperprior update):

    Periodically re-estimate (α₀, β₀) from current posteriors via
    Method of Moments on the collection of posterior means:

        κ̂  = Ê[θ](1 − Ê[θ]) / V̂ar[θ] − 1
        α̂₀ = κ̂ · Ê[θ]
        β̂₀ = κ̂ · (1 − Ê[θ])

    This lets the hyperprior adapt to the observed data distribution.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from scipy.special import betaln, digamma
    from scipy.stats import beta as _scipy_beta
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# Core Beta-Binomial Model
# ============================================================
class BetaBinomialModel:
    """
    Per-student Beta-Binomial model with temporal discounting and cooldown.

    Each student s independently maintains a Beta posterior over θ_s,
    their probability of being selected on any given day.

    Quick API:
        model.fit(binary_history)     → initialize from historical data
        model.update(binary_day)      → O(n) Bayesian update
        model.posterior_mean          → E[θ_s | data]  (n_students,)
        model.thompson_sample()       → sample θ_s ~ Beta(α_s, β_s)
        model.predict_groups(k=6)     → k-student groups via Thompson Sampling
        model.credible_interval()     → 95% CI per student
        model.empirical_bayes_update()→ re-estimate prior from posteriors
    """

    def __init__(self,
                 n_students: int = 55,
                 k_per_day: int = 6,
                 prior_concentration: float = 2.0,
                 decay: float = 0.97,
                 cooling_penalty: float = 3.0):
        """
        Args:
            n_students:           Number of students in the class.
            k_per_day:            Students selected per day (6).
            prior_concentration:  κ — prior strength in pseudo-observations.
                                  κ=2  → weak prior (~2 observations worth)
                                  κ=10 → stronger prior (~10 observations)
            decay:                γ — temporal discount factor (0 < γ < 1).
                                  Recent days receive more weight than old days.
            cooling_penalty:      δ — extra β added after selection.
                                  Suppresses immediate re-selection; fades with γ.
        """
        self.n_students = n_students
        self.k_per_day = k_per_day
        self.decay = decay
        self.cooling_penalty = cooling_penalty

        # Informative prior: encode the expected selection rate k/n
        base_rate    = k_per_day / n_students
        self.alpha_0 = prior_concentration * base_rate
        self.beta_0  = prior_concentration * (1.0 - base_rate)

        # Running soft counts (populated via fit() or update())
        self._count   = np.zeros(n_students, dtype=np.float64)  # discounted successes
        self._total   = 0.0                                       # discounted days
        self._cooling = np.zeros(n_students, dtype=np.float64)   # cooldown buffer

    # ----------------------------------------------------------
    # Posterior parameter properties
    # ----------------------------------------------------------

    @property
    def alpha(self) -> np.ndarray:
        """α_s = α₀ + discounted_successes   shape: (n_students,)"""
        return self.alpha_0 + self._count

    @property
    def beta(self) -> np.ndarray:
        """β_s = β₀ + discounted_failures + cooling   shape: (n_students,)"""
        failures = np.maximum(self._total - self._count, 0.0)
        return self.beta_0 + failures + self._cooling

    @property
    def posterior_mean(self) -> np.ndarray:
        """E[θ_s | data] = α_s / (α_s + β_s)   shape: (n_students,)"""
        a, b = self.alpha, self.beta
        return a / (a + b)

    @property
    def posterior_std(self) -> np.ndarray:
        """Std[θ_s | data]   shape: (n_students,)"""
        a, b = self.alpha, self.beta
        n = a + b
        var = (a * b) / (n * n * (n + 1.0))
        return np.sqrt(np.maximum(var, 0.0))

    @property
    def effective_n(self) -> float:
        """Effective number of observations in the discounted window."""
        return float(self._total)

    # ----------------------------------------------------------
    # Fit & Update
    # ----------------------------------------------------------

    def fit(self, binary_history: np.ndarray) -> 'BetaBinomialModel':
        """
        Initialize posteriors from historical data using temporal discounting.

        Processes history chronologically so that earlier days have less
        influence than recent ones (controlled by decay).

        Args:
            binary_history: (n_days, n_students) binary selection matrix.
                            binary_history[t, s] = 1 if student s was
                            selected on day t, else 0.
        Returns:
            self  (for method chaining)
        """
        self._count   = np.zeros(self.n_students, dtype=np.float64)
        self._total   = 0.0
        self._cooling = np.zeros(self.n_students, dtype=np.float64)

        for t in range(len(binary_history)):
            self._apply_update(binary_history[t])

        return self

    def update(self, binary_day: np.ndarray) -> None:
        """
        Single-day Bayesian update — O(n_students).

        Args:
            binary_day: (n_students,) binary vector for today's selections.
        """
        self._apply_update(binary_day)

    def _apply_update(self, binary_day: np.ndarray) -> None:
        """Core temporal-discounted conjugate update."""
        # 1. Discount all previous observations
        self._count   *= self.decay
        self._total   *= self.decay
        self._cooling *= self.decay      # cooldown naturally fades

        # 2. Add today's observation
        self._count += binary_day.astype(np.float64)
        self._total += 1.0

        # 3. Apply cooldown penalty to selected students
        selected = np.where(binary_day == 1)[0]
        self._cooling[selected] += self.cooling_penalty

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def thompson_sample(self) -> np.ndarray:
        """
        Draw θ_s ~ Beta(α_s, β_s) independently for each student.

        Thompson Sampling: each call produces a different random ranking,
        creating natural exploration across candidate groups.

        Returns:
            (n_students,) sampled probability vector.
        """
        a = np.maximum(self.alpha, 1e-4)
        b = np.maximum(self.beta,  1e-4)
        return np.random.beta(a, b)

    def predict_groups(self,
                       n_groups: int = 5,
                       k: int = 6,
                       use_thompson: bool = True,
                       temperature: float = 1.2) -> Tuple[list, np.ndarray]:
        """
        Generate n_groups diverse candidate groups.

        For Thompson Sampling groups: each group independently samples
        θ ~ Beta(α, β), selecting top-k by sampled values.
        Later groups get slightly higher temperature → more diversity.

        Args:
            n_groups:      Number of candidate groups to generate.
            k:             Students per group.
            use_thompson:  True = Thompson Sampling; False = posterior mean
                           with temperature softmax.
            temperature:   Base temperature for posterior mean mode.

        Returns:
            groups: list of n_groups lists, each with k student IDs (1-indexed).
            probs:  (n_students,) posterior mean probabilities.
        """
        base_probs = self.posterior_mean
        groups = []

        for g in range(n_groups):
            if use_thompson:
                sampled = self.thompson_sample()
                if g > 0:
                    # Increase temperature for later groups → more diversity
                    temp = 1.0 + 0.3 * g
                    sampled = np.power(np.maximum(sampled, 1e-9), 1.0 / temp)
                sampled = sampled / sampled.sum()
            else:
                temp = temperature * (0.8 + 0.15 * g)
                sampled = np.power(np.maximum(base_probs, 1e-9), 1.0 / temp)
                sampled = sampled / sampled.sum()

            selected_idx = np.random.choice(
                self.n_students, size=k, replace=False, p=sampled
            )
            # Sort within group by posterior mean (most likely first)
            selected_idx = selected_idx[np.argsort(-base_probs[selected_idx])]
            groups.append((selected_idx + 1).tolist())

        return groups, base_probs

    # ----------------------------------------------------------
    # Uncertainty Quantification
    # ----------------------------------------------------------

    def credible_interval(self, ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Equal-tailed posterior credible interval per student.

        Args:
            ci: Width of credible interval (default 0.95 = 95% CI).

        Returns:
            lower: (n_students,) lower credible bounds.
            upper: (n_students,) upper credible bounds.
        """
        a, b = self.alpha, self.beta

        if SCIPY_AVAILABLE:
            lo = (1.0 - ci) / 2.0
            hi = 1.0 - lo
            lower = _scipy_beta.ppf(lo, a, b)
            upper = _scipy_beta.ppf(hi, a, b)
        else:
            # Normal approximation  (accurate when α,β are not tiny)
            mean = a / (a + b)
            std  = self.posterior_std
            z    = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(round(ci, 2), 1.960)
            lower = np.clip(mean - z * std, 0.0, 1.0)
            upper = np.clip(mean + z * std, 0.0, 1.0)

        return lower.astype(np.float32), upper.astype(np.float32)

    def posterior_entropy(self) -> np.ndarray:
        """
        Differential entropy of Beta(α_s, β_s) per student.

        H[Beta(α,β)] = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)

        Higher entropy → more uncertain about that student's probability.

        Returns:
            (n_students,) entropy values.
        """
        if SCIPY_AVAILABLE:
            a, b = self.alpha, self.beta
            ent = (betaln(a, b)
                   - (a - 1.0) * digamma(a)
                   - (b - 1.0) * digamma(b)
                   + (a + b - 2.0) * digamma(a + b))
            return ent.astype(np.float32)
        return self.posterior_std   # proxy when scipy unavailable

    # ----------------------------------------------------------
    # Empirical Bayes
    # ----------------------------------------------------------

    def empirical_bayes_update(self) -> Tuple[float, float]:
        """
        Re-estimate prior (α₀, β₀) from current posteriors via
        Method of Moments (Empirical Bayes).

        Treats the posterior means {E[θ_s]} as draws from Beta(α₀, β₀)
        and fits the hyperparameters accordingly:

            κ̂  = Ê[θ](1 − Ê[θ]) / V̂ar[θ] − 1
            α̂₀ = κ̂ · Ê[θ],   β̂₀ = κ̂ · (1 − Ê[θ])

        Returns:
            (new_alpha_0, new_beta_0)
        """
        means = self.posterior_mean
        m = float(means.mean())
        v = float(means.var())

        if v > 1e-10 and 1e-6 < m < 1.0 - 1e-6:
            kappa = m * (1.0 - m) / v - 1.0
            kappa = float(np.clip(kappa, 0.5, 200.0))
            new_alpha_0 = kappa * m
            new_beta_0  = kappa * (1.0 - m)
        else:
            # Fallback: informative prior from selection rate
            base_rate   = self.k_per_day / self.n_students
            new_alpha_0 = 2.0 * base_rate
            new_beta_0  = 2.0 * (1.0 - base_rate)

        self.alpha_0 = new_alpha_0
        self.beta_0  = new_beta_0
        return new_alpha_0, new_beta_0

    # ----------------------------------------------------------
    # Diagnostics & Serialisation
    # ----------------------------------------------------------

    def top_k_students(self, k: int = 10) -> List[dict]:
        """Top-k students by posterior mean with uncertainty bounds."""
        means = self.posterior_mean
        stds  = self.posterior_std
        lower, upper = self.credible_interval()
        top_idx = np.argsort(-means)[:k]
        return [
            {
                'student_id':    int(i + 1),
                'post_mean':     float(means[i]),
                'post_std':      float(stds[i]),
                'ci_lower_95':   float(lower[i]),
                'ci_upper_95':   float(upper[i]),
            }
            for i in top_idx
        ]

    def most_uncertain_students(self, k: int = 10) -> List[dict]:
        """Top-k most uncertain students by posterior std."""
        means = self.posterior_mean
        stds  = self.posterior_std
        top_idx = np.argsort(-stds)[:k]
        return [
            {
                'student_id': int(i + 1),
                'post_mean':  float(means[i]),
                'post_std':   float(stds[i]),
            }
            for i in top_idx
        ]

    def summary_stats(self) -> dict:
        """Summary of current posterior state."""
        means = self.posterior_mean
        stds  = self.posterior_std
        return {
            'effective_n':   float(self._total),
            'alpha_0':       float(self.alpha_0),
            'beta_0':        float(self.beta_0),
            'mean_of_means': float(means.mean()),
            'std_of_means':  float(means.std()),
            'min_mean':      float(means.min()),
            'max_mean':      float(means.max()),
            'mean_std':      float(stds.mean()),
            'cooling_sum':   float(self._cooling.sum()),
        }

    def get_state(self) -> dict:
        """Serialize model state for checkpointing."""
        return {
            'n_students':      self.n_students,
            'k_per_day':       self.k_per_day,
            'decay':           self.decay,
            'cooling_penalty': self.cooling_penalty,
            'alpha_0':         float(self.alpha_0),
            'beta_0':          float(self.beta_0),
            '_count':          self._count.tolist(),
            '_total':          float(self._total),
            '_cooling':        self._cooling.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> 'BetaBinomialModel':
        """Restore model from serialized state."""
        model = cls(
            n_students=state['n_students'],
            k_per_day=state['k_per_day'],
            decay=state['decay'],
            cooling_penalty=state['cooling_penalty'],
        )
        model.alpha_0   = state['alpha_0']
        model.beta_0    = state['beta_0']
        model._count    = np.array(state['_count'])
        model._total    = state['_total']
        model._cooling  = np.array(state['_cooling'])
        return model
