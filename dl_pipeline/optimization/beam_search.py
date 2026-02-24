"""
Beam Search Optimizer for Student Group Selection
==================================================

Standard machine-learning approach: pick the student with the highest
individual probability. This is GREEDY and ignores interactions.

Beam Search approaches the problem differently:
  - Build the group ONE student at a time (6 sequential decisions)
  - Maintain B candidate partial-groups ("beams") at each step
  - Expand each beam by adding every remaining student
  - Keep only the TOP-B complete evaluations after scoring
  - Return the best complete group (and top-n_groups for diversity)

Because the GroupScorer includes a pairwise co-occurrence bonus, the
beam that looks suboptimal after step 1 (greedy) can become optimal by
step 6 if it discovers a highly co-selected cluster of students.

Algorithm:
  beam = [ ([], 0.0) ]  # list of (partial_group, score)
  for step in range(k):
      candidates = []
      for partial, score in beam:
          remaining = all students NOT in partial
          for s in remaining:
              new_score = scorer.score_group(partial + [s], probs)
              candidates.append((partial + [s], new_score))
      beam = top-B candidates by score     ← PRUNE step
  return beam[0][0]  # best complete group

Time complexity per prediction:  O(B × n × k)
  B=50, n=55, k=6 → ~16,500 evaluations, each O(k²) → very fast (<2 ms)
"""

from __future__ import annotations
import heapq
import numpy as np
from typing import List, Tuple

from group_scorer import GroupScorer


# ──────────────────────────────────────────────────────────────
#  Beam Search
# ──────────────────────────────────────────────────────────────

class BeamSearchOptimizer:
    """
    Beam Search over the combinatorial space of k-student groups.

    Each step selects ONE more student and keeps the top-B partial groups
    (beams) alive.  After k steps every beam is a complete group and the
    top-n_groups beams are returned as diverse candidate groups.

    Args:
        beam_width:  Number of beams to maintain. Larger → more thorough
                     search but slower. B=50 is a good default for n=55, k=6.
        k:           Group size (6 students).
        n_groups:    How many top beams to return as candidates.
        temperature: Temperature applied to probabilities before scoring.
                     temperature=1.0  → use probs as-is.
                     temperature<1.0  → sharpen distribution (more confident).
                     temperature>1.0  → flatten distribution (more exploration).
    """

    def __init__(self, beam_width: int = 50, k: int = 6,
                 n_groups: int = 5, temperature: float = 1.0):
        self.beam_width  = beam_width
        self.k           = k
        self.n_groups    = n_groups
        self.temperature = temperature

    # ----------------------------------------------------------
    # Main API
    # ----------------------------------------------------------

    def search(self, probs: np.ndarray,
               scorer: GroupScorer) -> Tuple[List[List[int]], List[float]]:
        """
        Run beam search to find the top-n_groups highest-scoring groups.

        Args:
            probs:  (n_students,) probability vector from base model.
            scorer: GroupScorer used to evaluate candidate groups.

        Returns:
            groups: List of groups (each = list of n_students indices, 0-based).
                    Sorted best-first.  Length = min(n_groups, beam_width).
            scores: Corresponding joint scores.
        """
        # Apply temperature scaling (optional exploration control)
        if self.temperature != 1.0:
            log_p  = np.log(probs + 1e-9) / self.temperature
            log_p -= log_p.max()
            probs  = np.exp(log_p)
            probs  = probs / probs.sum()

        n_students = len(probs)
        k          = self.k

        # Beam: list of (partial_group_as_tuple, score)
        # We use a list; after each step we prune to top beam_width.
        beam: List[Tuple[tuple, float]] = [((), 0.0)]

        for step in range(k):
            candidates: List[Tuple[float, tuple]] = []   # (score, group_tuple)

            for partial_tuple, _prev_score in beam:
                partial = list(partial_tuple)
                # Avoid duplicates: iterate only over students NOT in partial
                partial_set = set(partial)
                for s in range(n_students):
                    if s in partial_set:
                        continue
                    new_partial = partial + [s]
                    score       = scorer.score_group(new_partial, probs)
                    # heapq in Python is min-heap; push (score, group)
                    candidates.append((score, tuple(new_partial)))

            # Keep top beam_width by score (descending)
            # Use nlargest for clarity — efficient for small beam_width
            top_candidates = heapq.nlargest(self.beam_width, candidates,
                                            key=lambda x: x[0])
            beam = [(grp, scr) for scr, grp in top_candidates]

        # Collect top-n_groups unique complete groups
        seen   : set              = set()
        groups : List[List[int]]  = []
        scores : List[float]      = []

        for grp_tuple, scr in beam:
            key = frozenset(grp_tuple)
            if key not in seen:
                seen.add(key)
                groups.append(sorted(grp_tuple))
                scores.append(scr)
            if len(groups) >= self.n_groups:
                break

        return groups, scores

    # ----------------------------------------------------------
    # Convenience: top-1 group
    # ----------------------------------------------------------

    def best_group(self, probs: np.ndarray,
                   scorer: GroupScorer) -> List[int]:
        """Return the single best group."""
        groups, _ = self.search(probs, scorer)
        return groups[0] if groups else []

    # ----------------------------------------------------------
    # Greedy Baseline (beam_width=1)
    # ----------------------------------------------------------

    @staticmethod
    def greedy_top_k(probs: np.ndarray, k: int = 6) -> List[int]:
        """
        Pure greedy selection: top-k students by individual probability.
        This is what beam search degenerates to when beam_width=1 AND
        the scorer has cooc_weight=0.

        Returns list of k student indices sorted ascending.
        """
        return sorted(np.argsort(-probs)[:k].tolist())

    # ----------------------------------------------------------
    # Beam Trace (diagnostics)
    # ----------------------------------------------------------

    def trace_search(self, probs: np.ndarray,
                     scorer: GroupScorer) -> dict:
        """
        Run beam search and return a trace for visualization/debugging.

        Returns:
            dict with keys:
              'steps'       : list of (step, top-5 beams) per step
              'best_group'  : best final group
              'best_score'  : its score
              'greedy_group': greedy top-6 for comparison
              'greedy_score': greedy group's score
        """
        n_students = len(probs)
        k          = self.k
        beam       = [((), 0.0)]
        steps      = []

        for step in range(k):
            candidates = []
            for partial_tuple, _ in beam:
                partial     = list(partial_tuple)
                partial_set = set(partial)
                for s in range(n_students):
                    if s in partial_set:
                        continue
                    new_partial = partial + [s]
                    score       = scorer.score_group(new_partial, probs)
                    candidates.append((score, tuple(new_partial)))

            top_candidates = heapq.nlargest(self.beam_width, candidates,
                                            key=lambda x: x[0])
            beam = [(grp, scr) for scr, grp in top_candidates]
            steps.append([(list(grp), scr) for grp, scr in beam[:5]])

        best_group  = sorted(beam[0][0])
        best_score  = beam[0][1]
        greedy      = self.greedy_top_k(probs, k)
        greedy_score = scorer.score_group(greedy, probs)

        return {
            'steps'        : steps,
            'best_group'   : best_group,
            'best_score'   : best_score,
            'greedy_group' : greedy,
            'greedy_score' : greedy_score,
        }
