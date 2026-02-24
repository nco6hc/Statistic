"""
Group Scorer — Joint Evaluation of a 6-Student Group
=====================================================

Why a group score instead of just individual probabilities?
-----------------------------------------------------------
Greedy selection (top-6 by individual p_s) ignores INTERACTIONS between
students. Two students with moderate individual probability but high
co-occurrence rate may form a better pair than two high-probability
students who rarely appear together.

The joint group score combines:

  score(G) = Σ_{s∈G} log(p_s + ε)              [individual log-probs]
           + α · (1/C(k,2)) Σ_{i<j} R[si,sj]   [co-occurrence bonus]

Where:
  R[i,j] = cooc[i,j] / max(count[i], count[j])  [normalized ∈ [0,1]]
  C(k,2) = k(k-1)/2 pairs normalizer

Co-occurrence is tracked with exponential decay so recent patterns
weigh more than old ones (same decay as Bayesian BB model).

This score is used by BOTH the Beam Search and Genetic Algorithm
as their FITNESS / OBJECTIVE FUNCTION.
"""

import numpy as np


class GroupScorer:
    """
    Scores a candidate group of k students using:
      - Sum of log-probabilities (individual signal)
      - Normalized pairwise co-occurrence bonus (group signal)

    Co-occurrence is tracked with temporal exponential decay so that
    recent group patterns weigh more heavily than historical ones.

    Usage:
        scorer = GroupScorer().fit(binary_history)
        score  = scorer.score_group([0, 4, 12, 23, 37, 51], probs)
        scorer.update(today_binary)
    """

    def __init__(self, n_students: int = 55, k: int = 6,
                 cooc_weight: float = 2.0, decay: float = 0.97):
        """
        Args:
            n_students:   Number of students.
            k:            Students per group (6).
            cooc_weight:  α — weight of co-occurrence bonus relative to
                          log-probability term.
                          cooc_weight=0  → pure log-prob (greedy equivalent)
                          cooc_weight=2  → co-occurrence matters ~10% of log-prob
            decay:        γ — exponential decay for co-occurrence statistics.
                          Same as Bayesian BB: recent days weigh more.
        """
        self.n_students  = n_students
        self.k           = k
        self.cooc_weight = cooc_weight
        self.decay       = decay

        # Running discounted co-occurrence counts
        self._cooc       = np.zeros((n_students, n_students), dtype=np.float64)
        self._ind_counts = np.zeros(n_students,               dtype=np.float64)

        # Cache for normalized co-occurrence matrix (recomputed when dirty)
        self._cooc_norm  = None
        self._dirty      = True

        # Number of pairs = C(k, 2)
        self._n_pairs    = k * (k - 1) / 2.0

    # ----------------------------------------------------------
    # Fit & Update
    # ----------------------------------------------------------

    def fit(self, binary_history: np.ndarray) -> 'GroupScorer':
        """
        Populate co-occurrence statistics from historical binary data.

        Args:
            binary_history: (n_days, n_students) selection matrix.

        Returns:
            self
        """
        for row in binary_history:
            self._apply_update(row)
        return self

    def update(self, binary_day: np.ndarray) -> None:
        """One-day online update. Call after observing the actual selection."""
        self._apply_update(binary_day)

    def _apply_update(self, binary_day: np.ndarray) -> None:
        # Temporal decay
        self._cooc       *= self.decay
        self._ind_counts *= self.decay

        # Add today
        selected = np.where(binary_day == 1)[0]
        self._ind_counts[selected] += 1.0
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                si, sj = selected[i], selected[j]
                self._cooc[si, sj] += 1.0
                self._cooc[sj, si] += 1.0

        self._dirty = True

    # ----------------------------------------------------------
    # Normalized Co-Occurrence
    # ----------------------------------------------------------

    def _get_cooc_norm(self) -> np.ndarray:
        """
        Lazily compute normalized co-occurrence matrix.

        R[i,j] = cooc[i,j] / max(count[i], count[j])

        Values in [0, 1]:  1 = always together,  0 = never together.
        """
        if not self._dirty and self._cooc_norm is not None:
            return self._cooc_norm

        max_cnt = np.maximum(self._ind_counts[:, None],
                             self._ind_counts[None, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            self._cooc_norm = np.where(max_cnt > 0,
                                       self._cooc / max_cnt,
                                       0.0)
        self._dirty = False
        return self._cooc_norm

    # ----------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------

    def score_group(self, group_indices: list, probs: np.ndarray) -> float:
        """
        Compute the joint score for a complete group.

        score(G) = Σ log(p_s) + α/C(k,2) * Σ_{i<j} R[si,sj]

        Args:
            group_indices: list of student indices (0-based), length k.
            probs:         (n_students,) individual probability vector.

        Returns:
            Scalar score (higher = better group).
        """
        # --- Term 1: sum of log-probabilities ---
        log_p = float(np.sum(np.log(probs[group_indices] + 1e-9)))

        # --- Term 2: co-occurrence bonus ---
        cooc_norm   = self._get_cooc_norm()
        cooc_bonus  = 0.0
        for i in range(len(group_indices)):
            for j in range(i + 1, len(group_indices)):
                cooc_bonus += cooc_norm[group_indices[i], group_indices[j]]

        if self._n_pairs > 0:
            cooc_bonus /= self._n_pairs

        return log_p + self.cooc_weight * cooc_bonus

    def score_partial(self, partial: list, new_s: int,
                      probs: np.ndarray) -> float:
        """
        Score after adding student new_s to a partial group (beam expansion).
        Equivalent to score_group(partial + [new_s]) but avoids rebuilding.
        """
        return self.score_group(partial + [new_s], probs)

    def cooc_bonus_for_student(self, student: int, partial: list) -> float:
        """
        Marginal co-occurrence contribution of student `student`
        given the already-selected `partial` group.
        Used by Beam Search for incremental scoring display.
        """
        cooc_norm = self._get_cooc_norm()
        return float(sum(cooc_norm[student, s] for s in partial))

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    def top_cooc_pairs(self, k: int = 10) -> list:
        """Return top-k most co-selected student pairs."""
        cooc_norm = self._get_cooc_norm()
        pairs = []
        for i in range(self.n_students):
            for j in range(i + 1, self.n_students):
                pairs.append((i + 1, j + 1, float(cooc_norm[i, j])))
        return sorted(pairs, key=lambda x: -x[2])[:k]

    def student_cooc_summary(self, student_idx: int, top_k: int = 5) -> list:
        """Top-k co-selection partners for a given student."""
        cooc_norm = self._get_cooc_norm()
        row = cooc_norm[student_idx].copy()
        row[student_idx] = 0.0
        top_idx = np.argsort(-row)[:top_k]
        return [(int(i + 1), float(row[i])) for i in top_idx]
