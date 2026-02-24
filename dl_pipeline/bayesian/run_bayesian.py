"""
Bayesian Beta-Binomial Pipeline for Student Selection Prediction
================================================================

Model:
  - Per-student Beta-Binomial posteriors with temporal discounting
  - Thompson Sampling for diverse group generation
  - Cooldown penalty after selection
  - Empirical Bayes hyperprior update every N simulation days

Mathematical guarantees:
  - Fully probabilistic: maintains uncertainty over every student
  - Conjugate updates: O(n_students) per day, no retraining needed
  - No data leakage: only past observations inform each prediction

Usage:
    cd dl_pipeline/bayesian
    python run_bayesian.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent         # bayesian/
_DL_DIR   = _THIS_DIR.parent                        # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                         # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'bayesian'

sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

from bayesian.beta_binomial import BetaBinomialModel


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'excel_path':            str(_ROOT_DIR / 'Database.xlsx'),
    'train_ratio':           0.9,
    'n_students':            55,
    'k_per_day':             6,
    'n_simulation_days':     100,
    'prediction_interval':   1,
    'n_groups':              5,
    'random_seed':           42,
    # --- Beta-Binomial hyperparameters ---
    'prior_concentration':   2.0,   # κ: prior strength (pseudo-observations)
    'decay':                 0.97,  # γ: temporal discount; eff. window ~33 days
    'cooling_penalty':       3.0,   # δ: cooldown β-bonus after selection
    'use_thompson':          True,  # Thompson Sampling vs posterior mean
    'temperature':           1.2,   # Temperature for posterior-mean mode
    # --- Empirical Bayes ---
    'empirical_bayes':       True,
    'eb_interval':           10,    # Re-estimate hyperprior every N sim days
}

PREVIOUS_RESULTS = {
    'Baseline LSTM': {
        'average_accuracy': 26.00, 'best_accuracy': 66.67,
        'worst_accuracy': 0.00, 'std_deviation': 12.54,
        'total_predictions': 50,
        'distribution': {0: 4.0, 1: 46.0, 2: 42.0, 3: 6.0, 4: 2.0, 5: 0.0, 6: 0.0},
    },
    'Enhanced LSTM': {
        'average_accuracy': 28.33, 'best_accuracy': 50.00,
        'worst_accuracy': 0.00, 'std_deviation': 12.13,
        'total_predictions': 50,
        'distribution': {0: 2.0, 1: 40.0, 2: 44.0, 3: 14.0, 4: 0.0, 5: 0.0, 6: 0.0},
    },
    'LSTM+Markov Hybrid': {
        'average_accuracy': 29.00, 'best_accuracy': 50.00,
        'worst_accuracy': 0.00, 'std_deviation': 10.96,
        'total_predictions': 50,
        'distribution': {0: 2.0, 1: 32.0, 2: 56.0, 3: 10.0, 4: 0.0, 5: 0.0, 6: 0.0},
    },
    'Gradient Boosting': {
        'average_accuracy': 26.67, 'best_accuracy': 50.00,
        'worst_accuracy': 16.67, 'std_deviation': 11.55,
        'total_predictions': 50,
        'distribution': {0: 0.0, 1: 52.0, 2: 36.0, 3: 12.0, 4: 0.0, 5: 0.0, 6: 0.0},
    },
}


# ============================================================
# Statistics helpers
# ============================================================
def compute_statistics(accuracies: list) -> dict:
    distribution = {}
    for correct in range(7):
        acc_val = (correct / 6) * 100
        count = sum(1 for a in accuracies if abs(a - acc_val) < 1.0)
        distribution[correct] = (count / len(accuracies)) * 100

    first_10 = float(np.mean(accuracies[:10]))  if len(accuracies) >= 10 else 0.0
    last_10  = float(np.mean(accuracies[-10:])) if len(accuracies) >= 10 else 0.0

    return {
        'average_accuracy':  float(np.mean(accuracies)),
        'best_accuracy':     float(np.max(accuracies)),
        'worst_accuracy':    float(np.min(accuracies)),
        'std_deviation':     float(np.std(accuracies)),
        'total_predictions': len(accuracies),
        'distribution':      distribution,
        'first_10_avg':      first_10,
        'last_10_avg':       last_10,
    }


def print_statistics(stats: dict, title: str):
    print(f"\n{'='*70}")
    print(title.center(70))
    print(f"{'='*70}")
    print(f"\n  Total predictions : {stats['total_predictions']}")
    print(f"  Average accuracy  : {stats['average_accuracy']:.2f}%")
    print(f"  Best accuracy     : {stats['best_accuracy']:.2f}%")
    print(f"  Worst accuracy    : {stats['worst_accuracy']:.2f}%")
    print(f"  Std deviation     : {stats['std_deviation']:.4f}")
    print(f"\n  Accuracy distribution:")
    for c in range(7):
        pct = stats['distribution'].get(c, 0)
        n   = round(pct * stats['total_predictions'] / 100)
        bar = '█' * n
        print(f"    {c}/6 correct: {n:3d} ({pct:5.1f}%)  {bar}")
    imp = stats['last_10_avg'] - stats['first_10_avg']
    print(f"\n  Trend analysis:")
    print(f"    First 10 avg : {stats['first_10_avg']:.2f}%")
    print(f"    Last 10 avg  : {stats['last_10_avg']:.2f}%")
    print(f"    Improvement  : {imp:+.2f}%")


def print_comparison(bayes_stats: dict):
    all_models = {**PREVIOUS_RESULTS, 'Bayesian BB': bayes_stats}
    col_w = 16

    print(f"\n{'='*85}")
    print("COMPLETE MODEL COMPARISON".center(85))
    print(f"{'='*85}")

    header = f"  {'Metric':<22}" + ''.join(f" {n:<{col_w}}" for n in all_models)
    print(f"\n{header}")
    print("  " + "-" * (22 + (col_w + 1) * len(all_models)))

    metrics = [
        ('Avg Accuracy (%)',   'average_accuracy'),
        ('Best Accuracy (%)',  'best_accuracy'),
        ('Worst Accuracy (%)', 'worst_accuracy'),
        ('Std Deviation',      'std_deviation'),
    ]
    for label, key in metrics:
        vals = [m[key] for m in all_models.values()]
        is_lower_better = 'worst' in key or 'std' in key
        best_val = min(vals) if is_lower_better else max(vals)
        row = f"  {label:<22}"
        for m in all_models.values():
            v = m[key]
            mark = "*" if abs(v - best_val) < 0.01 else " "
            row += f" {v:>8.2f}{mark:<{col_w-9}}"
        print(row)

    print(f"\n  Accuracy Distribution (% of predictions):")
    print(f"  {'Correct':<10}" + ''.join(f" {n:<{col_w}}" for n in all_models))
    print("  " + "-" * (10 + (col_w + 1) * len(all_models)))
    for c in range(7):
        row = f"  {c}/6{'':<6}"
        for m in all_models.values():
            pct = m['distribution'].get(c, 0)
            row += f" {pct:>6.1f}%{'':<{col_w - 8}}"
        print(row)

    print(f"\n{'='*85}")
    best_avg  = max(all_models, key=lambda n: all_models[n]['average_accuracy'])
    best_peak = max(all_models, key=lambda n: all_models[n]['best_accuracy'])
    best_std  = min(all_models, key=lambda n: all_models[n]['std_deviation'])
    print(f"  Best avg accuracy  : {best_avg}  ({all_models[best_avg]['average_accuracy']:.2f}%)")
    print(f"  Best peak accuracy : {best_peak} ({all_models[best_peak]['best_accuracy']:.2f}%)")
    print(f"  Most consistent    : {best_std}  (std {all_models[best_std]['std_deviation']:.4f})")
    print(f"{'='*85}")


# ============================================================
# Online Simulation
# ============================================================
def run_bayesian_simulation(model: BetaBinomialModel,
                             all_binary: np.ndarray,
                             split_idx: int,
                             config: dict) -> tuple:
    """
    Run online Beta-Binomial simulation.

    Every `prediction_interval` days: predict → observe → update.
    Every `eb_interval` days:         re-estimate hyperprior (Empirical Bayes).
    """
    n_sim = min(config['n_simulation_days'], len(all_binary) - split_idx)
    print(f"\n  Simulation: {n_sim} days | "
          f"predict every {config['prediction_interval']} days | "
          f"EB update every {config['eb_interval']} days")

    accuracies   = []
    predictions_log = []
    days_since_eb = 0

    for test_idx in range(0, n_sim, config['prediction_interval']):
        actual_idx = split_idx + test_idx

        # ---- Posterior state printout every 20 predictions ----
        if test_idx > 0 and (test_idx // config['prediction_interval']) % 20 == 0:
            ss = model.summary_stats()
            print(f"  [Posterior state @ day {test_idx+1}]: "
                  f"eff_n={ss['effective_n']:.1f} | "
                  f"α₀={ss['alpha_0']:.3f} β₀={ss['beta_0']:.3f} | "
                  f"mean_of_means={ss['mean_of_means']:.4f}")

        # ---- Predict ----
        groups, probs = model.predict_groups(
            n_groups=config['n_groups'],
            k=config['k_per_day'],
            use_thompson=config['use_thompson'],
            temperature=config['temperature'],
        )

        # ---- Evaluate ----
        actual_binary = all_binary[actual_idx]
        actual_ids    = (np.where(actual_binary == 1)[0] + 1).tolist()

        overlap_counts = [len(set(g) & set(actual_ids)) for g in groups]
        best_overlap   = max(overlap_counts)
        best_accuracy  = (best_overlap / 6) * 100
        accuracies.append(best_accuracy)

        best_idx = overlap_counts.index(best_overlap)
        day_label = test_idx + 1
        print(f"  Day {day_label:3d}: {best_accuracy:5.1f}% ({best_overlap}/6) | "
              f"Group {best_idx+1}: {groups[best_idx]} | Actual: {actual_ids}")

        predictions_log.append({
            'day':          day_label,
            'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
        })

        # ---- Bayesian update (observe actual day) ----
        model.update(actual_binary)
        days_since_eb += config['prediction_interval']

        # ---- Empirical Bayes: re-estimate hyperprior ----
        if config['empirical_bayes'] and days_since_eb >= config['eb_interval']:
            days_since_eb = 0
            new_a0, new_b0 = model.empirical_bayes_update()
            print(f"    [EB update] α₀={new_a0:.4f}  β₀={new_b0:.4f}")

    return accuracies, predictions_log


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(CONFIG['random_seed'])

    print("=" * 70)
    print("BAYESIAN BETA-BINOMIAL MODEL PIPELINE".center(70))
    print("=" * 70)
    print("""
  Model:  Beta-Binomial with conjugate updates
  Prior:  θ_s ~ Beta(α₀, β₀)  [informative: encodes 6/55 base rate]
  Update: α_s += successes,  β_s += failures  (temporal-discounted)
  Pred:   Thompson Sampling from Beta(α_s, β_s)
  EB:     Re-estimate (α₀,β₀) from posteriors every 10 sim days
""")

    # ========================================================
    # PHASE 1: Load Data
    # ========================================================
    print(f"{'='*70}")
    print("PHASE 1: DATA PREPARATION".center(70))
    print(f"{'='*70}")

    print(f"\n  Loading: {CONFIG['excel_path']}")
    df = pd.read_excel(CONFIG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)

    n_students  = CONFIG['n_students']
    student_cols = [c for c in df.columns if c != 'Days']
    binary = np.zeros((len(df), n_students), dtype=np.float32)

    for idx, row in df.iterrows():
        for col in student_cols:
            sid = row[col]
            if pd.notna(sid):
                sid = int(sid)
                if 1 <= sid <= n_students:
                    binary[idx, sid - 1] = 1

    split_idx = int(len(binary) * CONFIG['train_ratio'])
    print(f"  Total days: {len(binary)} | Train: {split_idx} | Test: {len(binary)-split_idx}")
    print(f"  Selections per day: {binary.sum(axis=1).mean():.2f} (expected {CONFIG['k_per_day']})")

    # ========================================================
    # PHASE 2: Fit Bayesian Model on Training Data
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 2: FITTING BETA-BINOMIAL POSTERIORS".center(70))
    print(f"{'='*70}")

    model = BetaBinomialModel(
        n_students=n_students,
        k_per_day=CONFIG['k_per_day'],
        prior_concentration=CONFIG['prior_concentration'],
        decay=CONFIG['decay'],
        cooling_penalty=CONFIG['cooling_penalty'],
    )

    print(f"\n  Prior:  α₀={model.alpha_0:.4f}  β₀={model.beta_0:.4f}")
    print(f"  Decay:  γ={CONFIG['decay']}  (eff. window ≈ {1/(1-CONFIG['decay']):.0f} days)")
    print(f"  Cooldown penalty: δ={CONFIG['cooling_penalty']}")
    print(f"\n  Fitting on {split_idx} training days...")

    t0 = time.time()
    model.fit(binary[:split_idx])
    fit_time = time.time() - t0

    ss = model.summary_stats()
    print(f"  Fitting time: {fit_time:.3f}s")
    print(f"\n  Posterior summary (after training):")
    print(f"    Effective observations : {ss['effective_n']:.1f}")
    print(f"    Mean of posterior means: {ss['mean_of_means']:.4f}  "
          f"(expected {CONFIG['k_per_day']/n_students:.4f})")
    print(f"    Std of posterior means : {ss['std_of_means']:.4f}")
    print(f"    Range [min, max]       : [{ss['min_mean']:.4f}, {ss['max_mean']:.4f}]")
    print(f"    Mean posterior std     : {ss['mean_std']:.4f}  (uncertainty per student)")

    # Top students by posterior mean
    print(f"\n  Top-10 students by posterior mean (P(selected)):")
    print(f"  {'ID':>4}  {'Mean':>7}  {'Std':>7}  {'95% CI':>17}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*17}")
    for s in model.top_k_students(k=10):
        ci_str = f"[{s['ci_lower_95']:.4f}, {s['ci_upper_95']:.4f}]"
        print(f"  {s['student_id']:>4}  {s['post_mean']:>7.4f}  "
              f"{s['post_std']:>7.4f}  {ci_str:>17}")

    # Run Empirical Bayes update after training
    if CONFIG['empirical_bayes']:
        new_a0, new_b0 = model.empirical_bayes_update()
        print(f"\n  Post-training Empirical Bayes:")
        print(f"    Updated prior: α₀={new_a0:.4f}  β₀={new_b0:.4f}")

    # ========================================================
    # PHASE 3: Training Set Accuracy (Top-6 Overlap)
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 3: TRAINING SET ACCURACY CHECK".center(70))
    print(f"{'='*70}")

    # Refit a fresh model to get per-day posteriors for training evaluation
    eval_model = BetaBinomialModel(
        n_students=n_students,
        k_per_day=CONFIG['k_per_day'],
        prior_concentration=CONFIG['prior_concentration'],
        decay=CONFIG['decay'],
        cooling_penalty=CONFIG['cooling_penalty'],
    )
    train_overlaps = []
    for t in range(split_idx):
        probs = eval_model.posterior_mean
        top6_pred   = set(np.argsort(-probs)[:6])
        top6_actual = set(np.where(binary[t] == 1)[0])
        train_overlaps.append(len(top6_pred & top6_actual))
        eval_model.update(binary[t])

    train_acc = np.mean(train_overlaps) / 6
    print(f"\n  Train top-6 overlap accuracy : {train_acc:.4f}")
    print(f"  (Baseline random guess       : {6/n_students:.4f})")

    # ========================================================
    # PHASE 4: Online Simulation
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 4: ONLINE LEARNING SIMULATION".center(70))
    print(f"{'='*70}")

    t0 = time.time()
    accuracies, pred_log = run_bayesian_simulation(
        model, binary, split_idx, CONFIG
    )
    sim_time = time.time() - t0

    # ========================================================
    # PHASE 5: Results & Diagnostics
    # ========================================================
    bayes_stats = compute_statistics(accuracies)
    print_statistics(bayes_stats, "BAYESIAN MODEL - ONLINE SIMULATION RESULTS")

    # Final posterior diagnostics
    print(f"\n  Final posterior state:")
    ss = model.summary_stats()
    print(f"    Effective obs : {ss['effective_n']:.1f}")
    print(f"    α₀={ss['alpha_0']:.4f}  β₀={ss['beta_0']:.4f}  "
          f"(mean_of_means={ss['mean_of_means']:.4f})")

    print(f"\n  Top-10 students after simulation (updated posteriors):")
    print(f"  {'ID':>4}  {'Mean':>7}  {'Std':>7}  {'95% CI':>17}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*17}")
    for s in model.top_k_students(k=10):
        ci_str = f"[{s['ci_lower_95']:.4f}, {s['ci_upper_95']:.4f}]"
        print(f"  {s['student_id']:>4}  {s['post_mean']:>7.4f}  "
              f"{s['post_std']:>7.4f}  {ci_str:>17}")

    print(f"\n  Most uncertain students (highest posterior std):")
    for s in model.most_uncertain_students(k=5):
        print(f"    Student {s['student_id']:>3}: "
              f"mean={s['post_mean']:.4f}  std={s['post_std']:.4f}")

    # Save
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    with open(_CKPT_DIR / 'model_state.json', 'w') as f:
        json.dump(model.get_state(), f, indent=2)

    results_data = {
        'model_type': 'Bayesian Beta-Binomial',
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        'fit_time_s':  float(fit_time),
        'sim_time_s':  float(sim_time),
        'train_accuracy': float(train_acc),
        'results': bayes_stats,
        'final_posterior': ss,
    }
    with open(_CKPT_DIR / 'bayes_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    pd.DataFrame(pred_log).to_csv(
        _CKPT_DIR / 'bayes_predictions_log.csv', index=False
    )

    print(f"\n  Model state saved : {_CKPT_DIR / 'model_state.json'}")
    print(f"  Results saved     : {_CKPT_DIR / 'bayes_results.json'}")
    print(f"  Log saved         : {_CKPT_DIR / 'bayes_predictions_log.csv'}")

    print_comparison(bayes_stats)

    print(f"\n{'='*70}")
    print("BAYESIAN PIPELINE COMPLETE".center(70))
    print(f"{'='*70}")

    return bayes_stats


if __name__ == '__main__':
    results = main()
