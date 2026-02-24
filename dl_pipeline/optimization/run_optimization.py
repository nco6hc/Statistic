"""
run_optimization.py — Combinatorial Optimization Pipeline
==========================================================

APPROACH
--------
Rather than independently predicting each student's probability and
taking the top-6, this pipeline treats group selection as a combinatorial
OPTIMIZATION problem:

  Find G* = argmax_{G ⊆ S, |G|=6}  score(G, probs)

where:
  score(G) = Σ_{s∈G} log(p_s)   +   α * co_occurrence_bonus(G)

Three strategies are compared head-to-head:

  1. Greedy    — top-6 by individual probability (ignores interactions)
  2. BeamSearch— systematic width-B search building the group slot-by-slot
  3. GeneticAlg— evolutionary search over population of complete groups

Base probability model: Markov Chain + Bayesian Beta-Binomial HYBRID
  (same signal as the best prior model — LSTM+Markov Hybrid at 29%)
  Markov captures transitions/cooldown/co-occurrence (sharp peaks)
  BB adds smoothed frequency prior (handles cold-start)
Group scorer: GroupScorer with cooc_weight=0 (Markov already handles co-occurrence)

SIMULATION
----------
  - 100-day rolling window, predict every day
  - n_groups=5 diverse candidate groups per strategy per prediction
  - Accuracy = best overlap of any candidate with actual selection / 6 * 100%
  - Both Bayesian BB and GroupScorer updated with observed data each day

USAGE
-----
  cd dl_pipeline/optimization
  python run_optimization.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ──────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # optimization/
_DL_DIR   = _THIS_DIR.parent                         # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                          # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'optimization'
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_DL_DIR))
sys.path.insert(0, str(_THIS_DIR))

from data_loader import DataManager
from bayesian.beta_binomial import BetaBinomialModel
from markov_chain import FactoredMarkovChain
from group_scorer import GroupScorer
from beam_search import BeamSearchOptimizer
from genetic_algorithm import GeneticAlgorithmOptimizer
from diverse_beam_search import DiverseBeamSearchOptimizer

# ── Configuration ────────────────────────────────────────────────────────────
EXCEL_PATH      = str(_ROOT_DIR / 'Database.xlsx')
N_STUDENTS      = 55
K               = 6          # students per group
TRAIN_RATIO     = 0.9
SIM_DAYS        = 100        # online simulation window
PRED_INTERVAL   = 1          # predict every N days (1 = every day)
N_GROUPS        = 5          # candidate groups per prediction

# Bayesian BB hyperparams
BB_ALPHA0       = 1.0
BB_BETA0        = 5.0
BB_DECAY        = 0.97
BB_COOLING      = 3.0

# Markov Chain hyperparams
MARKOV_SMOOTHING = 1.0
MARKOV_WINDOW    = 14   # days of history for predict_probabilities
MARKOV_ALPHA     = 0.5  # blend weight: hybrid = MARKOV_ALPHA*markov + (1-MARKOV_ALPHA)*bb
                        # 0.5 matches the best-performing LSTM+Markov hybrid weights

# GroupScorer hyperparams
# cooc_weight=1.5: Markov captures transition + conditional co-occurrence (given yesterday).
# GroupScorer adds ABSOLUTE pairwise co-occurrence (how often any two students appear
# together regardless of yesterday). These are complementary signals.
# With Markov's sharper probs (range 0.003-0.03), cooc_weight=1.5 is well-balanced.
COOC_WEIGHT     = 1.5
SCORER_DECAY    = 0.97

# Beam Search hyperparams
BEAM_WIDTH      = 50
DIVERSITY_LAMBDA = 5.0  # penalty per overlapping student in Diverse Beam Search

# Genetic Algorithm hyperparams
POP_SIZE        = 80
N_GENERATIONS   = 40
MUTATION_RATE   = 0.25

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG_SEED        = 42


# ═══════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def overlap_accuracy(groups: list, actual_binary: np.ndarray) -> float:
    """
    Best-of-N-groups accuracy.
    accuracy = max overlap across groups / k * 100
    Groups are 0-indexed student indices.
    """
    actual_set = set(np.where(actual_binary == 1)[0].tolist())
    best = 0
    for g in groups:
        overlap = len(set(g) & actual_set)
        if overlap > best:
            best = overlap
    return best / K * 100.0


def hybrid_thompson_sample(bb_model, markov_probs: np.ndarray) -> np.ndarray:
    """
    Generate one Thompson sample blended with Markov probabilities.

    Strategy: sample theta ~ Beta(alpha, beta) from BB, then re-weight by
    Markov signal and renormalize. This preserves Thompson Sampling's
    diversity while injecting the sharper Markov signal.

    Returns: (n_students,) probability vector.
    """
    theta = bb_model.thompson_sample()          # draw from Beta posterior
    # Blend: Thompson gives diversity, Markov gives sharp individual signal
    blended = MARKOV_ALPHA * markov_probs + (1.0 - MARKOV_ALPHA) * theta
    blended = np.clip(blended, 1e-9, None)
    return blended / blended.sum()


def deterministic_groups(hybrid_probs: np.ndarray,
                          n_groups: int = N_GROUPS) -> list:
    """
    Generate n_groups groups deterministically from hybrid_probs via
    temperature variation (no stochastic sampling).  Group 1 = pure top-6.
    Groups 2-N sharpen/soften the distribution for diversity.
    Returns list of 0-indexed groups.
    """
    groups: list = []
    seen:   set  = set()
    temperatures = [1.0, 0.6, 0.4, 1.8, 3.0]   # diverse coverage
    for temp in temperatures[:n_groups]:
        log_p = np.log(hybrid_probs + 1e-9) / temp
        log_p -= log_p.max()
        p = np.exp(log_p)
        p /= p.sum()
        g   = sorted(np.argsort(-p)[:K].tolist())
        key = frozenset(g)
        if key not in seen:
            seen.add(key)
            groups.append(g)
    while len(groups) < n_groups:
        g = sorted(np.argsort(-hybrid_probs)[:K].tolist())
        groups.append(g)
    return groups[:n_groups]


def greedy_groups_thompson(bb_model, markov_probs: np.ndarray,
                            n_groups: int = N_GROUPS) -> list:
    """
    Generate n_groups diverse groups via hybrid Thompson Sampling.
    Returns list of 0-indexed groups.
    """
    groups: list = []
    seen:   set  = set()
    max_tries = n_groups * 4
    for _ in range(max_tries):
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        g = sorted(np.argsort(-theta)[:K].tolist())
        key = frozenset(g)
        if key not in seen:
            seen.add(key)
            groups.append(g)
        if len(groups) >= n_groups:
            break
    while len(groups) < n_groups:
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        groups.append(sorted(np.argsort(-theta)[:K].tolist()))
    return groups[:n_groups]


def ga_groups_thompson(bb_model, markov_probs: np.ndarray,
                        ga_opt: 'GeneticAlgorithmOptimizer',
                        group_scorer: 'GroupScorer',
                        n_groups: int = N_GROUPS,
                        rng: np.random.Generator | None = None) -> list:
    """
    Run N independent mini-GA searches, each on a fresh hybrid Thompson sample.
    Markov-blended signal gives the GA sharper peaks to evolve toward.
    Each mini-GA uses fewer generations (15) for speed.
    Returns list of n_groups unique 0-indexed groups.
    """
    mini_opt = GeneticAlgorithmOptimizer(
        pop_size=40, n_generations=15, k=K, n_students=N_STUDENTS,
        mutation_rate=MUTATION_RATE, n_groups=1)

    groups: list = []
    seen:   set  = set()
    max_tries = n_groups * 3

    for _ in range(max_tries):
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        grps, _, _ = mini_opt.optimize(theta, group_scorer, rng=rng)
        for g in grps:
            key = frozenset(g)
            if key not in seen:
                seen.add(key)
                groups.append(g)
        if len(groups) >= n_groups:
            break

    while len(groups) < n_groups:
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        groups.append(sorted(np.argsort(-theta)[:K].tolist()))
    return groups[:n_groups]


def beam_groups_thompson(bb_model, markov_probs: np.ndarray,
                          beam_opt: 'BeamSearchOptimizer',
                          group_scorer: 'GroupScorer',
                          n_groups: int = N_GROUPS) -> list:
    """
    Run Beam Search N times, each on a fresh hybrid Thompson sample.
    Each run uses Markov-blended Thompson probabilities for a strong signal
    with stochastic diversity across the N groups.
    Returns list of 0-indexed unique groups.
    """
    groups : list  = []
    seen   : set   = set()
    max_tries = n_groups * 3
    for _ in range(max_tries):
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        best  = beam_opt.best_group(theta, group_scorer)
        key   = frozenset(best)
        if key not in seen:
            seen.add(key)
            groups.append(best)
        if len(groups) >= n_groups:
            break
    while len(groups) < n_groups:
        theta = hybrid_thompson_sample(bb_model, markov_probs)
        groups.append(sorted(np.argsort(-theta)[:K].tolist()))
    return groups[:n_groups]


# ═══════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print(" COMBINATORIAL OPTIMIZATION PIPELINE")
    print(" Beam Search  +  Genetic Algorithm  +  Greedy Baseline")
    print("=" * 64)

    t0 = time.perf_counter()

    # ── 1. Load Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data ...")
    dm     = DataManager(EXCEL_PATH)
    dm.load_data()
    binary = dm.convert_to_binary()           # (n_days, 55) numpy array
    n_days = binary.shape[0]

    split_idx  = int(n_days * TRAIN_RATIO)
    train_data = binary[:split_idx]
    test_data  = binary[split_idx:]

    print(f"      Total days   : {n_days}")
    print(f"      Train days   : {split_idx}")
    print(f"      Test days    : {len(test_data)}")

    # ── 2. Fit Probability Model ─────────────────────────────────────────
    print("\n[2/5] Fitting Bayesian Beta-Binomial model ...")
    bb = BetaBinomialModel(n_students=N_STUDENTS,
                           prior_concentration=BB_ALPHA0,
                           decay=BB_DECAY, cooling_penalty=BB_COOLING)
    t_bb = time.perf_counter()
    bb.fit(train_data)
    print(f"      Fit time : {time.perf_counter() - t_bb:.3f}s")

    # ── 2b. Fit Markov Chain ─────────────────────────────────────────────
    print("\n[2b] Fitting Markov Chain (transition + cooldown + co-occurrence) ...")
    markov = FactoredMarkovChain(n_students=N_STUDENTS,
                                  smoothing=MARKOV_SMOOTHING)
    t_mc = time.perf_counter()
    markov.fit(train_data)
    print(f"      Fit time : {time.perf_counter() - t_mc:.3f}s")
    print(f"      Signal range (sample): [{markov.predict_probabilities(train_data[-MARKOV_WINDOW:]).min():.4f}, "
          f"{markov.predict_probabilities(train_data[-MARKOV_WINDOW:]).max():.4f}]")

    # ── 3. Fit GroupScorer ───────────────────────────────────────────────
    print("\n[3/5] Fitting GroupScorer (co-occurrence) ...")
    scorer = GroupScorer(n_students=N_STUDENTS, k=K,
                         cooc_weight=COOC_WEIGHT, decay=SCORER_DECAY)
    t_sc = time.perf_counter()
    scorer.fit(train_data)
    print(f"      Fit time : {time.perf_counter() - t_sc:.3f}s")

    # Top co-selected pairs diagnostic
    top_pairs = scorer.top_cooc_pairs(k=5)
    print("      Top co-selected pairs (students 1-based):")
    for i, j, r in top_pairs:
        print(f"        S{int(i):02d} <-> S{int(j):02d}  rate={r:.3f}")

    # ── 4. Initialize Optimizers ─────────────────────────────────────────
    print("\n[4/5] Initializing optimizers ...")
    beam_optimizer  = BeamSearchOptimizer(beam_width=BEAM_WIDTH, k=K,
                                           n_groups=N_GROUPS)
    diverse_optimizer = DiverseBeamSearchOptimizer(beam_width=BEAM_WIDTH, k=K,
                                                    n_groups=N_GROUPS,
                                                    diversity_lambda=DIVERSITY_LAMBDA)
    ga_optimizer    = GeneticAlgorithmOptimizer(pop_size=POP_SIZE,
                                                n_generations=N_GENERATIONS,
                                                k=K,
                                                n_students=N_STUDENTS,
                                                mutation_rate=MUTATION_RATE,
                                                n_groups=N_GROUPS)
    rng_ga = np.random.default_rng(RNG_SEED)

    print(f"      BeamSearch       : beam_width={BEAM_WIDTH}")
    print(f"      DiverseBeamSearch: beam_width={BEAM_WIDTH}, lambda={DIVERSITY_LAMBDA}")
    print(f"      GeneticAlg       : pop={POP_SIZE}, gen={N_GENERATIONS}, "
          f"mut={MUTATION_RATE}")

    # ── 5. Online Simulation ─────────────────────────────────────────────
    print("\n[5/5] Running online simulation ...")

    # Use last SIM_DAYS of test data (or all if shorter)
    sim_data = test_data[-SIM_DAYS:]
    n_sim    = len(sim_data)

    acc_greedy = []
    acc_beam   = []
    acc_ga     = []
    acc_determ = []   # deterministic optimizer: pure Markov+BB, no sampling
    acc_diverse = []  # Diverse Beam Search: one Thompson sample, N forced-diverse groups

    # Timing accumulators
    t_greedy_total = 0.0
    t_beam_total   = 0.0
    t_ga_total     = 0.0
    t_determ_total = 0.0
    t_diverse_total = 0.0

    # GA convergence trace (for one sample day)
    ga_sample_day    = max(0, n_sim // 2)
    ga_fitness_trace = []

    print(f"      Simulating {n_sim} days ...")
    print(f"      {'Day':>4}  {'Greedy':>8}  {'Beam':>8}  {'GA':>8}  {'Determ':>8}  {'Diverse':>8}")
    print("      " + "-" * 56)

    # Rolling window of binary history for Markov prediction
    # Start with the last MARKOV_WINDOW days of training data
    window_start = split_idx - len(test_data)  # index into full binary array
    rolling_hist = list(binary[max(0, split_idx - MARKOV_WINDOW): split_idx])

    for d in range(n_sim):
        # ---- Hybrid probability signal ----------------------------------------
        # Markov: sharp transition/cooldown/co-occurrence signal
        recent = np.array(rolling_hist[-MARKOV_WINDOW:], dtype=np.float32)
        markov_probs = markov.predict_probabilities(recent)   # (55,)

        # BB: smoothed frequency prior
        bb_probs = bb.posterior_mean.copy()                   # (55,)

        # Hybrid blend (same as the 29% LSTM+Markov model, minus LSTM)
        hybrid_probs = (MARKOV_ALPHA * markov_probs
                        + (1.0 - MARKOV_ALPHA) * bb_probs)
        hybrid_probs = np.clip(hybrid_probs, 1e-9, None)
        hybrid_probs /= hybrid_probs.sum()

        # ---- Strategy A: Greedy (hybrid Thompson) ----------------------------
        t_s = time.perf_counter()
        g_groups = greedy_groups_thompson(bb, markov_probs, n_groups=N_GROUPS)
        t_greedy_total += time.perf_counter() - t_s

        # ---- Strategy B: Beam Search (hybrid Thompson) -----------------------
        t_s = time.perf_counter()
        b_groups = beam_groups_thompson(bb, markov_probs, beam_optimizer, scorer,
                                         n_groups=N_GROUPS)
        t_beam_total += time.perf_counter() - t_s

        # ---- Strategy C: Genetic Algorithm (hybrid Thompson mini-runs) -------
        t_s = time.perf_counter()
        ga_groups = ga_groups_thompson(bb, markov_probs, ga_optimizer, scorer,
                                        n_groups=N_GROUPS, rng=rng_ga)
        # GA convergence trace on sample day (use deterministic hybrid_probs)
        if d == ga_sample_day:
            _, _, ga_fitness_trace = ga_optimizer.optimize(
                hybrid_probs, scorer, rng=rng_ga)
        t_ga_total += time.perf_counter() - t_s

        # ---- Strategy D: Deterministic Hybrid (pure Markov+BB, no sampling) --
        t_s = time.perf_counter()
        d_groups = deterministic_groups(hybrid_probs, n_groups=N_GROUPS)
        t_determ_total += time.perf_counter() - t_s

        # ---- Strategy E: Diverse Beam Search (one Thompson, λ-diverse) ------
        t_s = time.perf_counter()
        theta_div   = hybrid_thompson_sample(bb, markov_probs)
        div_groups, div_scores, div_stats = diverse_optimizer.search(
            theta_div, scorer)
        t_diverse_total += time.perf_counter() - t_s

        # ---- Evaluate --------------------------------------------------------
        actual = sim_data[d]
        a_g = overlap_accuracy(g_groups,   actual)
        a_b = overlap_accuracy(b_groups,   actual)
        a_a = overlap_accuracy(ga_groups,  actual)
        a_d = overlap_accuracy(d_groups,   actual)
        a_e = overlap_accuracy(div_groups, actual)

        acc_greedy.append(a_g)
        acc_beam.append(a_b)
        acc_ga.append(a_a)
        acc_determ.append(a_d)
        acc_diverse.append(a_e)

        # ---- Update models ---------------------------------------------------
        bb.update(actual)
        scorer.update(actual)
        # Markov online update needs today, yesterday, day-before
        hist = rolling_hist
        yesterday  = np.array(hist[-1],   dtype=np.float32) if len(hist) >= 1 else np.zeros(N_STUDENTS)
        day_before = np.array(hist[-2],   dtype=np.float32) if len(hist) >= 2 else np.zeros(N_STUDENTS)
        markov.update(actual, yesterday, day_before)
        rolling_hist.append(actual)

        if (d + 1) % 10 == 0 or d < 3:
            print(f"      {d+1:>4}  {a_g:>7.1f}%  {a_b:>7.1f}%  {a_a:>7.1f}%  {a_d:>7.1f}%  {a_e:>7.1f}%")

    # ── 6. Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)

    stats_g = _stats(acc_greedy)
    stats_b = _stats(acc_beam)
    stats_a = _stats(acc_ga)
    stats_d = _stats(acc_determ)
    stats_e = _stats(acc_diverse)

    t_per_g = t_greedy_total  / n_sim * 1000
    t_per_b = t_beam_total    / n_sim * 1000
    t_per_a = t_ga_total      / n_sim * 1000
    t_per_d = t_determ_total  / n_sim * 1000
    t_per_e = t_diverse_total / n_sim * 1000

    print(f"\n  {'Strategy':<28} {'Avg':>7} {'Best':>7} {'Worst':>7} "
          f"{'Std':>7} {'ms/pred':>9}")
    print("  " + "-" * 72)
    for name, s, t_ms in [("Greedy+Thompson (hybrid)",  stats_g, t_per_g),
                           ("Beam Search+Thompson",      stats_b, t_per_b),
                           ("GA+Thompson",               stats_a, t_per_a),
                           ("Deterministic Hybrid",      stats_d, t_per_d),
                           ("Diverse Beam Search",       stats_e, t_per_e)]:
        print(f"  {name:<28} {s['avg']:>6.2f}%  {s['best']:>6.2f}%  "
              f"{s['worst']:>6.2f}%  {s['std']:>6.2f}  {t_ms:>8.1f}ms")

    best_opt = max(stats_g['avg'], stats_b['avg'], stats_a['avg'],
                   stats_d['avg'], stats_e['avg'])
    best_name = max([("Greedy", stats_g['avg']), ("Beam", stats_b['avg']),
                      ("GA", stats_a['avg']), ("Determ", stats_d['avg']),
                      ("DiverseBeam", stats_e['avg'])],
                     key=lambda x: x[1])[0]
    print(f"\n  Best optimizer       : {best_name}  ({best_opt:.2f}%)")
    print(f"  vs LSTM+Markov (29%) : {best_opt - 29.0:+.2f}%")

    # ── 7. Comparison with Previous Models ───────────────────────────────
    PRIOR_MODELS = [
        ("LSTM+Markov Hybrid",  29.00),
        ("Enhanced LSTM",       28.33),
        ("Bayesian BB",         27.67),
        ("Gradient Boosting",   26.67),
        ("Baseline LSTM",       26.00),
        ("Meta-Ensemble",       24.33),
    ]

    print("\n" + "=" * 64)
    print("  LEADERBOARD — ALL MODELS")
    print("=" * 64)
    all_models = PRIOR_MODELS + [
        ("Greedy+Thompson",    stats_g['avg']),
        ("Beam Search+Th.",    stats_b['avg']),
        ("GA+Thompson",        stats_a['avg']),
        ("Deterministic Hybr.",stats_d['avg']),
        ("Diverse Beam Search",stats_e['avg']),
    ]
    all_models.sort(key=lambda x: -x[1])
    print(f"  {'Rank':<5} {'Model':<24} {'Avg Acc':>8}")
    print("  " + "-" * 40)
    for rank, (name, avg) in enumerate(all_models, 1):
        marker = " <-- NEW" if name in {"Greedy+Thompson", "Beam Search+Th.",
                                          "GA+Thompson", "Deterministic Hybr.",
                                          "Diverse Beam Search"} else ""
        print(f"  {rank:<5} {name:<24} {avg:>7.2f}%{marker}")

    total_time = time.perf_counter() - t0
    print(f"\n  Total runtime : {total_time:.1f}s")

    # ── 8. Visualization ─────────────────────────────────────────────────
    _save_charts(acc_greedy, acc_beam, acc_ga, acc_determ, acc_diverse,
                 all_models, ga_fitness_trace, ga_sample_day)

    print(f"\n  Charts saved to: {_CKPT_DIR}")
    print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _stats(acc_list: list) -> dict:
    a = np.array(acc_list)
    return {
        'avg'  : float(np.mean(a)),
        'best' : float(np.max(a)),
        'worst': float(np.min(a)),
        'std'  : float(np.std(a)),
    }


def _save_charts(acc_greedy, acc_beam, acc_ga, acc_determ, acc_diverse,
                  all_models, ga_fitness_trace, ga_sample_day):
    """4-panel comparison dashboard."""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Combinatorial Optimization — Beam Search & Genetic Algorithm',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    days  = np.arange(1, len(acc_greedy) + 1)
    ALPHA = 0.65
    LW    = 1.4

    # ── Panel A: Accuracy Over Time ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(days, acc_greedy, color='steelblue',  lw=LW, alpha=ALPHA,
             label='Greedy')
    ax0.plot(days, acc_beam,   color='darkorange', lw=LW, alpha=ALPHA,
             label='Beam Search')
    ax0.plot(days, acc_ga,     color='mediumseagreen', lw=LW, alpha=ALPHA,
             label='Genetic Algorithm')

    # Rolling averages (window=10)
    w = 10
    for arr, c, ls in [(acc_greedy, 'steelblue', '--'),
                        (acc_beam,   'darkorange', '--'),
                        (acc_ga,     'mediumseagreen', '--')]:
        roll = np.convolve(arr, np.ones(w) / w, mode='valid')
        ax0.plot(np.arange(w, len(arr) + 1), roll, color=c, lw=2.0,
                 ls=ls, alpha=0.9)

    ax0.set_xlabel('Simulation Day')
    ax0.set_ylabel('Accuracy (%)')
    ax0.set_title('Accuracy Over Time  (dashed = 10-day rolling avg)')
    ax0.legend(fontsize=8)
    ax0.set_ylim(-5, 110)
    ax0.grid(axis='y', alpha=0.3)

    # ── Panel B: Leaderboard Bar Chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    new_set = {"Greedy (Optim.)", "Beam Search", "Genetic Alg."}
    colors_bar = ['tomato' if m in new_set else 'steelblue'
                  for m, _ in all_models]
    names  = [m for m, _ in all_models]
    avgs   = [a for _, a in all_models]
    y_pos  = np.arange(len(names))
    bars   = ax1.barh(y_pos, avgs, color=colors_bar, alpha=0.8, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Accuracy (%)')
    ax1.set_title('All Models — Leaderboard')
    for bar, val in zip(bars, avgs):
        ax1.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=7)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color='tomato', alpha=0.8, label='New (Optimization)'),
                         Patch(color='steelblue', alpha=0.8, label='Prior models')],
               fontsize=7, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # ── Panel C: Score Distribution (violin) ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    data_dict  = {'Greedy\n+Th.': acc_greedy, 'Beam\n+Th.': acc_beam,
                  'GA\n+Th.': acc_ga, 'Determ.\nHybrid': acc_determ,
                  'Diverse\nBeam': acc_diverse}
    labels_v   = list(data_dict.keys())
    values_v   = [data_dict[k] for k in labels_v]
    parts      = ax2.violinplot(values_v, showmedians=True, showextrema=True)
    colors_v   = ['steelblue', 'darkorange', 'mediumseagreen', 'crimson', 'purple']
    for pc, c in zip(parts['bodies'], colors_v):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    ax2.set_xticks(np.arange(1, len(labels_v) + 1))
    ax2.set_xticklabels(labels_v)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Score Distribution (Violin Plot)')
    ax2.set_ylim(-5, 110)
    ax2.grid(axis='y', alpha=0.3)

    # Add mean markers
    for i, arr in enumerate(values_v, 1):
        ax2.scatter([i], [np.mean(arr)], color='black', zorder=5, s=30,
                    marker='D')

    # ── Panel D: GA Fitness Convergence ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if ga_fitness_trace:
        gens = np.arange(len(ga_fitness_trace))
        ax3.plot(gens, ga_fitness_trace, color='mediumseagreen', lw=2.0,
                 marker='o', markersize=3, alpha=0.8)
        ax3.fill_between(gens, ga_fitness_trace, alpha=0.15,
                          color='mediumseagreen')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Best Fitness (log-prob + cooc)')
        ax3.set_title(f'GA Convergence  (Day {ga_sample_day + 1} of simulation)')
        ax3.grid(alpha=0.3)

        # Annotate improvement
        if len(ga_fitness_trace) > 1:
            init_f = ga_fitness_trace[0]
            best_f = max(ga_fitness_trace)
            ax3.annotate(f'Init: {init_f:.2f}',
                          xy=(0, init_f), xytext=(5, init_f + 0.5),
                          fontsize=8, color='gray')
            ax3.annotate(f'Best: {best_f:.2f}',
                          xy=(np.argmax(ga_fitness_trace), best_f),
                          xytext=(len(ga_fitness_trace) * 0.5, best_f - 0.5),
                          fontsize=8, color='darkgreen',
                          arrowprops=dict(arrowstyle='->', color='darkgreen',
                                          lw=1.2))
    else:
        ax3.text(0.5, 0.5, 'No GA trace available',
                  ha='center', va='center', transform=ax3.transAxes)

    out_path = _CKPT_DIR / 'optimization_dashboard.png'
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

    # ── Second figure: cumulative accuracy comparison ─────────────────────
    fig2, ax = plt.subplots(figsize=(11, 5))
    cum_g = np.cumsum(acc_greedy) / (np.arange(len(acc_greedy)) + 1)
    cum_b = np.cumsum(acc_beam)   / (np.arange(len(acc_beam))   + 1)
    cum_a = np.cumsum(acc_ga)     / (np.arange(len(acc_ga))     + 1)
    cum_d = np.cumsum(acc_determ) / (np.arange(len(acc_determ)) + 1)
    cum_e = np.cumsum(acc_diverse)/ (np.arange(len(acc_diverse))+ 1)
    ax.plot(days, cum_g, color='steelblue',      lw=2.0, label='Greedy+Thompson')
    ax.plot(days, cum_b, color='darkorange',     lw=2.0, label='Beam Search+Th.')
    ax.plot(days, cum_a, color='mediumseagreen', lw=2.0, label='GA+Thompson')
    ax.plot(days, cum_d, color='crimson',        lw=2.0, label='Deterministic Hybrid')
    ax.plot(days, cum_e, color='purple',         lw=2.5, label='Diverse Beam Search', ls='--')
    ax.set_xlabel('Simulation Day')
    ax.set_ylabel('Cumulative Average Accuracy (%)')
    ax.set_title('Cumulative Accuracy — Optimization Strategies')
    ax.legend()
    ax.grid(alpha=0.3)
    out2 = _CKPT_DIR / 'optimization_cumulative.png'
    fig2.tight_layout()
    fig2.savefig(str(out2), dpi=130)
    plt.close(fig2)
    print(f"  Saved: {out2.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
