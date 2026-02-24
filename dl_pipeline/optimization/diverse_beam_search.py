"""
Diverse Beam Search for Maximum Coverage
=========================================

INSIGHT FROM PREVIOUS EXPERIMENTS
-----------------------------------
Standard Beam Search + GA failed to beat simple Thompson Sampling because
they all converge to SIMILAR groups (the same high-scoring students appear
in all 5 candidates). This gives poor coverage of the actual selection space.

Thompson Sampling works well (27.67%) because it generates DIVERSE groups —
each draw from Beta(α, β) produces a completely different ranking, so the
5 candidates cover different regions of the student space.

THE FIX: Diverse Beam Search
-----------------------------
Generate N groups sequentially, where each new group is PENALIZED for
overlapping with previously chosen groups:

  score_with_diversity(G, prev_groups) =
      joint_score(G) - λ * max_overlap(G, prev_groups)

This forces each candidate to be different from previous ones while still
scoring high individually. The λ parameter controls the exploration-vs-exploitation
tradeoff:
  λ=0   → standard beam search (all groups similar)
  λ=∞   → pure diversity (groups share no students)
  λ=5   → balance: good individual score + forced diversity

This is the n-best diverse decoding algorithm used in NLP/MT beam search.

COVERAGE METRIC
---------------
To verify diversity is working, we track:
  - Unique students covered across all N groups
  - Average pairwise overlap between groups
  - Best-of-N overlap with actual selection (accuracy)
"""

from __future__ import annotations
import heapq
import numpy as np
from typing import List, Tuple

from group_scorer import GroupScorer


class DiverseBeamSearchOptimizer:
    """
    Diverse Beam Search: generates N groups that are BOTH high-scoring
    and maximally diverse from each other.

    Unlike standard Beam Search (which returns the top-N beams, often
    very similar), this explicitly penalizes overlap with already-chosen
    groups.

    Algorithm for generating N groups:
      groups = []
      for i in range(N):
          G_i* = argmax_{G} [ joint_score(G) - lambda * max_overlap(G, groups) ]
          groups.append(G_i*)
      return groups

    Each G_i* is found via a standard Beam Search with the diversity penalty
    added to the scoring function.

    Args:
        beam_width:        B — beams to maintain during group construction.
        k:                 Group size (6).
        n_groups:          Number of diverse groups to generate.
        diversity_lambda:  λ — penalty per overlapping student with previous groups.
                           Tuning: 5.0 balances score vs diversity for log-prob scoring.
        temperature:       Temperature for sharpening/flattening input probs.
    """

    def __init__(self, beam_width: int = 50, k: int = 6,
                 n_groups: int = 5, diversity_lambda: float = 5.0,
                 temperature: float = 1.0):
        self.beam_width       = beam_width
        self.k                = k
        self.n_groups         = n_groups
        self.diversity_lambda = diversity_lambda
        self.temperature      = temperature

    # ----------------------------------------------------------
    # Main API
    # ----------------------------------------------------------

    def search(self, probs: np.ndarray,
               scorer: GroupScorer) -> Tuple[List[List[int]], List[float], dict]:
        """
        Generate n_groups diverse high-scoring groups.

        Args:
            probs:   (n_students,) probability vector.
            scorer:  GroupScorer for fitness evaluation.

        Returns:
            groups:   List of n_groups groups (0-indexed), best-first.
            scores:   Joint scores for each group (pre-diversity penalty).
            stats:    Diversity statistics dict.
        """
        if self.temperature != 1.0:
            log_p  = np.log(probs + 1e-9) / self.temperature
            log_p -= log_p.max()
            probs  = np.exp(log_p)
            probs  = probs / probs.sum()

        n_students = len(probs)
        chosen_groups : List[List[int]]  = []
        chosen_scores : List[float]      = []

        for group_idx in range(self.n_groups):
            # Build frozen sets of already-chosen students for fast overlap check
            prev_sets = [frozenset(g) for g in chosen_groups]

            # Beam Search to find best group given diversity constraint
            group, score = self._beam_search_one(
                probs, scorer, prev_sets, n_students)

            chosen_groups.append(group)
            chosen_scores.append(score)

        stats = self._diversity_stats(chosen_groups)
        return chosen_groups, chosen_scores, stats

    # ----------------------------------------------------------
    # Internal: single beam search with diversity penalty
    # ----------------------------------------------------------

    def _beam_search_one(self, probs: np.ndarray,
                          scorer: GroupScorer,
                          prev_sets: list,
                          n_students: int) -> Tuple[List[int], float]:
        """
        Find the best single group via Beam Search with diversity penalty.

        For a candidate group G:
          penalized_score(G) = joint_score(G) - λ * max_student_overlap(G, prev_groups)

        where max_student_overlap = maximum #overlapping students with any single
        previous group (so we especially penalize groups that DUPLICATE an existing one).
        """
        beam: List[Tuple[tuple, float]] = [((), 0.0)]

        for _step in range(self.k):
            candidates: List[Tuple[float, tuple]] = []

            for partial_tuple, _prev in beam:
                partial     = list(partial_tuple)
                partial_set = set(partial)

                for s in range(n_students):
                    if s in partial_set:
                        continue
                    new_partial = partial + [s]

                    # Joint score (GroupScorer)
                    base_score = scorer.score_group(new_partial, probs)

                    # Diversity penalty: overlap with EACH previous chosen group
                    if prev_sets:
                        new_set  = set(new_partial)
                        max_overlap = max(len(new_set & ps) for ps in prev_sets)
                        penalty  = self.diversity_lambda * max_overlap
                    else:
                        penalty = 0.0

                    penalized = base_score - penalty
                    candidates.append((penalized, tuple(new_partial)))

            # Keep top beam_width by penalized score
            top_candidates = heapq.nlargest(self.beam_width, candidates,
                                             key=lambda x: x[0])
            beam = [(grp, scr) for scr, grp in top_candidates]

        # Best complete group: use joint_score (not penalized) for reporting
        best_tuple = beam[0][0]
        best_group = sorted(best_tuple)
        best_score = scorer.score_group(best_group, probs)
        return best_group, best_score

    # ----------------------------------------------------------
    # Diversity Statistics
    # ----------------------------------------------------------

    def _diversity_stats(self, groups: List[List[int]]) -> dict:
        """Compute diversity statistics for the generated groups."""
        n_groups = len(groups)
        if n_groups == 0:
            return {}

        # Unique students covered
        all_students = set()
        for g in groups:
            all_students.update(g)

        # Average pairwise overlap
        total_overlap = 0.0
        n_pairs = 0
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                total_overlap += len(set(groups[i]) & set(groups[j]))
                n_pairs += 1

        avg_overlap = total_overlap / max(1, n_pairs)

        return {
            'n_unique_students': len(all_students),
            'max_possible'     : self.k * n_groups,
            'coverage_ratio'   : len(all_students) / max(1, self.k * n_groups),
            'avg_pairwise_overlap': avg_overlap,
        }

    # ----------------------------------------------------------
    # Temperature schedule for diverse groups
    # ----------------------------------------------------------

    @staticmethod
    def thompson_diverse_search(bb_model, markov_probs: np.ndarray,
                                 searcher: 'DiverseBeamSearchOptimizer',
                                 group_scorer: GroupScorer,
                                 n_groups: int = 5) -> Tuple[list, dict]:
        """
        Run DiverseBeamSearch on a SINGLE hybrid Thompson sample.
        The diversity mechanism (lambda penalty) ensures N diverse groups
        WITHOUT needing multiple Thompson draws.

        Returns (groups, stats)
        """
        # One Thompson-Markov blended sample
        from run_optimization import hybrid_thompson_sample, MARKOV_ALPHA  # noqa
        theta  = hybrid_thompson_sample(bb_model, markov_probs)
        groups, scores, stats = searcher.search(theta, group_scorer)
        return groups, stats
