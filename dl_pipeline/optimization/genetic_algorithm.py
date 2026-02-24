"""
Genetic Algorithm Optimizer for Student Group Selection
=======================================================

While Beam Search explores the group space systematically (left-to-right),
the Genetic Algorithm evolves a POPULATION of complete groups simultaneously.

WHY THIS WORKS DIFFERENTLY:
  - Beam Search is deterministic and optimal within its beam width
  - GA is stochastic — it can escape local optima via mutation
  - GA is better at finding globally diverse solutions
  - GA can parallelize across all 6 positions simultaneously

CHROMOSOME REPRESENTATION:
  Each individual = a sorted tuple of k unique student indices
  e.g. (2, 14, 27, 33, 44, 51)  → "pick students 3, 15, 28, 34, 45, 52"

GENETIC OPERATORS:
  Initialization:
      Bias toward high-probability students (prob-weighted sampling without
      replacement). A fraction of the population is randomly initialized to
      maintain diversity.

  Selection (Tournament):
      Randomly pick tournament_k individuals; winner = highest fitness.
      Stochastic but favors better individuals.

  Crossover (Union Sampling):
      parent1 = {A, B, C, D, E, F}
      parent2 = {B, C, D, G, H, I}
      common  = {B, C, D}          → keep in child
      pool    = {A,E,F,G,H,I}      → sample k-len(common) from pool
      child   = common ∪ sample    → still exactly k students, all unique

  Mutation (Swap):
      With probability mutation_rate, replace ONE random student with
      a random student not currently in the group.
      The new student is chosen with probability proportional to probs
      (so high-prob students get selected more often during mutation).

  Elitism:
      Top elite_n individuals from previous generation survive unchanged.
      Prevents regression from generation to generation.

FITNESS FUNCTION:
  Same GroupScorer.score_group(group, probs) used by Beam Search.
  This ensures fair comparison between the two optimizers.

TIME COMPLEXITY per prediction:
  pop_size × n_generations evaluations = 80 × 40 = 3,200 × O(k²) = fast
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from group_scorer import GroupScorer


# ──────────────────────────────────────────────────────────────
#  Genetic Algorithm
# ──────────────────────────────────────────────────────────────

class GeneticAlgorithmOptimizer:
    """
    Evolves a population of k-student groups to maximize GroupScorer fitness.

    Args:
        pop_size:       Number of individuals in the population.
        n_generations:  Number of generations to evolve.
        k:              Group size (6).
        n_students:     Total student pool (55).
        tournament_k:   Tournament size for selection (3).
        mutation_rate:  Probability that any individual undergoes mutation.
        elite_fraction: Fraction of top individuals copied unchanged to next gen.
        n_groups:       How many top individuals to return as candidates.
        biased_init_frac: Fraction of population initialized with prob-biased
                          sampling (vs purely random). Higher → faster convergence
                          but less diversity.
    """

    def __init__(self, pop_size: int = 80, n_generations: int = 40,
                 k: int = 6, n_students: int = 55,
                 tournament_k: int = 3, mutation_rate: float = 0.25,
                 elite_fraction: float = 0.1, n_groups: int = 5,
                 biased_init_frac: float = 0.6):
        self.pop_size          = pop_size
        self.n_generations     = n_generations
        self.k                 = k
        self.n_students        = n_students
        self.tournament_k      = tournament_k
        self.mutation_rate     = mutation_rate
        self.n_groups          = n_groups
        self.biased_init_frac  = biased_init_frac
        self.elite_n           = max(1, int(pop_size * elite_fraction))

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def optimize(self, probs: np.ndarray,
                 scorer: GroupScorer,
                 rng: np.random.Generator | None = None
                 ) -> Tuple[List[List[int]], List[float], List[float]]:
        """
        Run the genetic algorithm and return best groups.

        Args:
            probs:   (n_students,) probability vector from base model.
            scorer:  GroupScorer for fitness evaluation.
            rng:     Optional random generator (for reproducibility).

        Returns:
            groups:          Top-n_groups unique individuals, best-first.
            scores:          Corresponding fitness scores.
            fitness_history: Best fitness score per generation (for plotting).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Normalize probs for sampling
        p = np.array(probs, dtype=np.float64)
        p = np.clip(p, 1e-9, None)
        p_norm = p / p.sum()

        # ── 1. Initialise population ──────────────────────────
        population = self._init_population(p_norm, rng)

        fitness_history: List[float] = []

        for _gen in range(self.n_generations):
            # ── 2. Evaluate fitness ───────────────────────────
            fitnesses = np.array([scorer.score_group(ind, probs)
                                   for ind in population], dtype=np.float64)

            best_fit = float(fitnesses.max())
            fitness_history.append(best_fit)

            # ── 3. Elitism ────────────────────────────────────
            elite_idx  = np.argsort(-fitnesses)[:self.elite_n]
            new_pop    = [population[i] for i in elite_idx]

            # ── 4. Breed remainder ────────────────────────────
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)
                child = self._crossover(p1, p2, rng)
                child = self._mutate(child, p_norm, rng)
                new_pop.append(child)

            population = new_pop

        # ── 5. Final evaluation & return top-n_groups ─────────
        fitnesses = np.array([scorer.score_group(ind, probs)
                               for ind in population], dtype=np.float64)
        best_fit  = float(fitnesses.max())
        fitness_history.append(best_fit)

        # Collect unique top individuals
        order = np.argsort(-fitnesses)
        seen   : set              = set()
        groups : List[List[int]]  = []
        scores : List[float]      = []
        for idx in order:
            key = frozenset(population[idx])
            if key not in seen:
                seen.add(key)
                groups.append(sorted(population[idx]))
                scores.append(float(fitnesses[idx]))
            if len(groups) >= self.n_groups:
                break

        return groups, scores, fitness_history

    # ----------------------------------------------------------
    # Genetic Operators
    # ----------------------------------------------------------

    def _init_population(self, p_norm: np.ndarray,
                         rng: np.random.Generator) -> List[list]:
        """
        Mixed initialisation:
          - biased_init_frac of pop: sample WITHOUT replacement weighted by probs
          - remainder: uniform random sample without replacement
        """
        pop   : List[list] = []
        n_biased = int(self.pop_size * self.biased_init_frac)
        all_idx  = np.arange(self.n_students)

        for i in range(self.pop_size):
            if i < n_biased:
                # Biased: higher-probability students more likely to appear
                chosen = rng.choice(all_idx, size=self.k,
                                    replace=False, p=p_norm)
            else:
                # Random (diversity)
                chosen = rng.choice(all_idx, size=self.k, replace=False)
            pop.append(sorted(chosen.tolist()))

        return pop

    def _tournament_select(self, population: List[list],
                            fitnesses: np.ndarray,
                            rng: np.random.Generator) -> list:
        """
        Tournament selection: randomly pick tournament_k individuals,
        return the one with the highest fitness.
        """
        idx      = rng.choice(len(population), size=self.tournament_k,
                               replace=False)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx]

    def _crossover(self, parent1: list, parent2: list,
                   rng: np.random.Generator) -> list:
        """
        Union-sampling crossover for set-valued chromosomes.

        1. Identify students common to both parents (always kept).
        2. Pool the students unique to each parent.
        3. Sample from the pool to fill remaining k - len(common) slots.

        Result: a valid group of exactly k unique students.
        """
        s1     = set(parent1)
        s2     = set(parent2)
        common = s1 & s2
        pool   = list((s1 | s2) - common)

        n_needed = self.k - len(common)
        if n_needed <= 0:
            # Both parents are identical: return one of them unchanged
            return sorted(list(common)[:self.k])

        if n_needed > len(pool):
            # Edge case: not enough unique students in pool — top up from ALL
            all_remaining = list(set(range(self.n_students)) - common)
            rng.shuffle(all_remaining)
            pool = pool + all_remaining
            pool = list(dict.fromkeys(pool))  # deduplicate preserving order

        chosen_from_pool = rng.choice(pool, size=n_needed,
                                       replace=False).tolist()
        child = sorted(list(common) + chosen_from_pool)
        return child

    def _mutate(self, individual: list, p_norm: np.ndarray,
                rng: np.random.Generator) -> list:
        """
        Swap mutation: with probability mutation_rate, replace ONE randomly
        chosen student with a new student sampled from p_norm.

        The new student is chosen probability-proportional so that high-prob
        students enter the group more often during mutation.
        """
        if rng.random() >= self.mutation_rate:
            return individual

        ind_set = set(individual)
        # Remove one random student
        out_idx = rng.integers(0, self.k)
        out_s   = individual[out_idx]
        ind_set.discard(out_s)

        # Eligible new students: not already in group
        eligible = np.array([s for s in range(self.n_students)
                              if s not in ind_set])
        if len(eligible) == 0:
            return individual

        p_eligible = p_norm[eligible]
        p_eligible = p_eligible / p_eligible.sum()
        new_s      = int(rng.choice(eligible, p=p_eligible))

        ind_set.add(new_s)
        return sorted(list(ind_set))

    # ----------------------------------------------------------
    # Convergence Statistics
    # ----------------------------------------------------------

    def diversity_score(self, population: List[list]) -> float:
        """
        Fraction of unique individuals in the population.
        1.0 = fully diverse,  0.0 = all identical (converged).
        """
        unique = len({frozenset(ind) for ind in population})
        return unique / max(1, len(population))
