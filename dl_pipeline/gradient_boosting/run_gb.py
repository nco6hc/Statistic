"""
Gradient Boosting + Feature Engineering Pipeline

Trains a HistGradientBoostingClassifier (one shared model for all students)
with 6 rich feature groups per (day, student) pair, then runs an online
simulation with periodic retraining on a sliding window.

Model approach:
  - One GBM model trained on ALL (day × student) pairs
  - Input:  9 features describing one (day, student) context
  - Output: P(student selected on that day)
  - Prediction: build 55 rows for next day, get 55 probs, sample top-6

Online updates:
  - Every `update_interval` days, retrain on the last `train_window` days
  - Retraining is fast (seconds) since GBM trains on tabular data

Usage:
    cd dl_pipeline/gradient_boosting
    python run_gb.py
"""

import sys
from pathlib import Path

# Resolve paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent         # gradient_boosting/
_DL_DIR   = _THIS_DIR.parent                        # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                         # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'gradient_boosting'

sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

from gradient_boosting.gb_features import (
    build_gb_features, IncrementalGBState
)


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'excel_path':          str(_ROOT_DIR / 'Database.xlsx'),
    'train_ratio':         0.9,
    'n_students':          55,
    'n_simulation_days':   100,
    'prediction_interval': 1,
    'n_groups':            5,
    'k_students':          6,
    'temperature':         1.5,
    'update_interval':     10,      # Retrain GBM every N simulation days
    'train_window':        300,     # Use last N days for retraining
    'random_seed':         42,
    # GBM hyperparameters
    'max_iter':            300,
    'learning_rate':       0.05,
    'max_depth':           5,
    'min_samples_leaf':    20,
    'l2_regularization':   1.0,
    'pos_weight':          9.0,     # Weight for positive class (selected)
}

# Previous results for comparison
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
}


# ============================================================
# GBM Model
# ============================================================
def build_sample_weights(y: np.ndarray, pos_weight: float = 9.0) -> np.ndarray:
    """Upweight positive samples to handle class imbalance."""
    return np.where(y == 1, pos_weight, 1.0)


def train_gbm(X: np.ndarray, y: np.ndarray, config: dict) -> HistGradientBoostingClassifier:
    """
    Train a HistGradientBoostingClassifier.

    Args:
        X: (n_samples, 9) feature matrix
        y: (n_samples,) binary target
        config: CONFIG dict

    Returns:
        Fitted GBM model
    """
    model = HistGradientBoostingClassifier(
        max_iter=config['max_iter'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        min_samples_leaf=config['min_samples_leaf'],
        l2_regularization=config['l2_regularization'],
        random_state=config['random_seed'],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=0,
    )
    sample_weight = build_sample_weights(y, config['pos_weight'])
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ============================================================
# Prediction with temperature sampling
# ============================================================
def gb_predict_groups(model: HistGradientBoostingClassifier,
                      X_day: np.ndarray,
                      n_groups: int = 5, k: int = 6,
                      temperature: float = 1.5) -> tuple:
    """
    Use GBM probabilities to generate diverse student groups.

    Args:
        model: Fitted GBM
        X_day: (n_students, 9) features for the target day
        n_groups: Number of groups to generate
        k: Students per group
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        groups: list of n_groups × k student ID lists
        probs: (n_students,) raw probability vector
    """
    probs = model.predict_proba(X_day)[:, 1]  # P(selected) for each student

    groups = []
    for g in range(n_groups):
        temp = temperature * (0.7 + 0.15 * g)
        adj = np.power(np.clip(probs, 1e-9, 1.0), 1.0 / temp)
        adj = adj / adj.sum()

        selected = np.random.choice(len(adj), size=k, replace=False, p=adj)
        selected = selected[np.argsort(-probs[selected])]
        groups.append((selected + 1).tolist())

    return groups, probs


# ============================================================
# Online Simulation
# ============================================================
def run_gb_simulation(model: HistGradientBoostingClassifier,
                      state: IncrementalGBState,
                      all_binary: np.ndarray,
                      split_idx: int,
                      config: dict) -> tuple:
    """
    Run online simulation: predict every `prediction_interval` days,
    retrain GBM every `update_interval` days.
    """
    n_sim = min(config['n_simulation_days'], len(all_binary) - split_idx)
    print(f"\n  Simulation: {n_sim} days, "
          f"predict every {config['prediction_interval']} days, "
          f"retrain GBM every {config['update_interval']} days")

    accuracies = []
    predictions_log = []
    days_since_retrain = 0

    for test_idx in range(0, n_sim, config['prediction_interval']):
        actual_idx = split_idx + test_idx

        # ---- Build features for prediction day ----
        X_day = state.get_features_for_next_day(actual_idx)

        # ---- Predict ----
        groups, probs = gb_predict_groups(
            model, X_day,
            n_groups=config['n_groups'],
            k=config['k_students'],
            temperature=config['temperature']
        )

        # ---- Accuracy ----
        actual_binary = all_binary[actual_idx]
        actual_ids = (np.where(actual_binary == 1)[0] + 1).tolist()

        overlap_counts = [len(set(g) & set(actual_ids)) for g in groups]
        best_overlap = max(overlap_counts)
        best_accuracy = (best_overlap / 6) * 100
        accuracies.append(best_accuracy)

        best_idx = overlap_counts.index(best_overlap)
        day_label = test_idx + 1
        print(f"  Day {day_label:3d}: {best_accuracy:5.1f}% ({best_overlap}/6) | "
              f"Best group {best_idx+1}: {groups[best_idx]} | Actual: {actual_ids}")

        predictions_log.append({
            'day': day_label, 'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
        })

        # ---- Update state with actual outcome ----
        state.update(actual_binary, actual_idx)
        days_since_retrain += config['prediction_interval']

        # ---- Periodic GBM retrain ----
        if days_since_retrain >= config['update_interval']:
            days_since_retrain = 0
            hist = state.binary_history
            window_start = max(0, len(hist) - config['train_window'])
            window_binary = hist[window_start:]
            abs_start = window_start  # binary_history[i] is absolute day i

            print(f"    [Retrain GBM on {len(window_binary)} days "
                  f"({len(window_binary) * config['n_students']:,} samples)]", end='')
            t0 = time.time()

            X_w, y_w = build_gb_features(
                window_binary, n_students=config['n_students'],
                start_day=abs_start
            )
            model = train_gbm(X_w, y_w, config)
            print(f" — {time.time()-t0:.1f}s")

    return model, accuracies, predictions_log


# ============================================================
# Statistics
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
        'average_accuracy': float(np.mean(accuracies)),
        'best_accuracy':    float(np.max(accuracies)),
        'worst_accuracy':   float(np.min(accuracies)),
        'std_deviation':    float(np.std(accuracies)),
        'total_predictions': len(accuracies),
        'distribution':      distribution,
        'first_10_avg':      first_10,
        'last_10_avg':       last_10,
    }


def print_statistics(stats: dict, title: str):
    print(f"\n{'='*70}")
    print(title.center(70))
    print(f"{'='*70}")
    print(f"\nTotal predictions: {stats['total_predictions']}")
    print(f"Average accuracy:  {stats['average_accuracy']:.2f}%")
    print(f"Best accuracy:     {stats['best_accuracy']:.2f}%")
    print(f"Worst accuracy:    {stats['worst_accuracy']:.2f}%")
    print(f"Std deviation:     {stats['std_deviation']:.4f}")
    print(f"\nAccuracy distribution:")
    for c in range(7):
        pct = stats['distribution'].get(c, 0)
        n   = round(pct * stats['total_predictions'] / 100)
        print(f"  {c}/6 correct: {n:3d} ({pct:5.1f}%)")
    imp = stats['last_10_avg'] - stats['first_10_avg']
    print(f"\nTrend analysis:")
    print(f"  First 10 predictions: {stats['first_10_avg']:.2f}%")
    print(f"  Last 10 predictions:  {stats['last_10_avg']:.2f}%")
    print(f"  Trend: {imp:+.2f}%")


def print_comparison(gb_stats: dict):
    all_models = {**PREVIOUS_RESULTS, 'Gradient Boosting': gb_stats}

    print(f"\n{'='*80}")
    print("COMPLETE MODEL COMPARISON".center(80))
    print(f"{'='*80}")

    col_w = 18
    header = f"  {'Metric':<22}" + ''.join(f" {n:<{col_w}}" for n in all_models)
    print(f"\n{header}")
    print("  " + "-" * (22 + (col_w + 1) * len(all_models)))

    metrics = [
        ('Avg Accuracy (%)',  'average_accuracy'),
        ('Best Accuracy (%)', 'best_accuracy'),
        ('Worst Accuracy (%)', 'worst_accuracy'),
        ('Std Deviation',     'std_deviation'),
        ('Predictions',       'total_predictions'),
    ]
    for label, key in metrics:
        row = f"  {label:<22}"
        vals = [m[key] for m in all_models.values()]
        best_val = max(vals) if 'worst' not in key.lower() and 'std' not in key.lower() else None
        for name, m in all_models.items():
            v = m[key]
            mark = " *" if best_val is not None and abs(v - best_val) < 0.01 else ""
            row += f" {v:<{col_w-2}.2f}{mark:<2}"
        print(row)

    print(f"\n  Accuracy Distribution (% of predictions):")
    print(f"  {'Correct':<10}" + ''.join(f" {n:<{col_w}}" for n in all_models))
    print("  " + "-" * (10 + (col_w + 1) * len(all_models)))
    for c in range(7):
        row = f"  {c}/6{'':<6}"
        for m in all_models.values():
            pct = m['distribution'].get(c, 0)
            row += f" {pct:>6.1f}%{'':<{col_w-8}}"
        print(row)

    # Winner
    print(f"\n{'='*80}")
    best_avg_name = max(all_models, key=lambda n: all_models[n]['average_accuracy'])
    best_avg = all_models[best_avg_name]['average_accuracy']
    best_peak_name = max(all_models, key=lambda n: all_models[n]['best_accuracy'])
    best_peak = all_models[best_peak_name]['best_accuracy']
    print(f"  Best average accuracy : {best_avg_name} ({best_avg:.2f}%)")
    print(f"  Best peak accuracy    : {best_peak_name} ({best_peak:.2f}%)")
    
    gb_dist = gb_stats['distribution']
    high_preds = sum(gb_dist.get(c, 0) for c in [4, 5, 6])
    if high_preds > 0:
        print(f"  Gradient Boosting 4+/6 predictions: {high_preds:.1f}% of the time!")
    print(f"{'='*80}")


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(CONFIG['random_seed'])

    print("=" * 70)
    print("GRADIENT BOOSTING + FEATURE ENGINEERING PIPELINE".center(70))
    print("=" * 70)
    print("\nFeature Groups:")
    print("  1. Days since last call")
    print("  2. Frequency last 5/10/20 days")
    print("  3. Pairwise co-selection score")
    print("  4. Position in class (1-55)")
    print("  5. Day-of-week pattern")
    print("  6. Entropy of teacher behavior (7d + 20d)")

    # ========================================================
    # PHASE 1: Load Data
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 1: DATA PREPARATION".center(70))
    print(f"{'='*70}")

    print(f"\nLoading data from: {CONFIG['excel_path']}")
    df = pd.read_excel(CONFIG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)

    student_cols = [c for c in df.columns if c != 'Days']
    n_students = CONFIG['n_students']
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

    # ========================================================
    # PHASE 2: Build Training Features
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 2: BUILDING TRAINING FEATURES".center(70))
    print(f"{'='*70}")

    t0 = time.time()
    X_train, y_train = build_gb_features(
        binary[:split_idx], n_students=n_students, start_day=0
    )
    print(f"  Feature computation: {time.time()-t0:.1f}s")

    # ========================================================
    # PHASE 3: Train GBM
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 3: TRAINING GRADIENT BOOSTING MODEL".center(70))
    print(f"{'='*70}")

    print(f"\n  Model: HistGradientBoostingClassifier")
    print(f"  Samples: {len(X_train):,}  (= {split_idx} days × {n_students} students)")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive class weight: {CONFIG['pos_weight']}x")

    t0 = time.time()
    gbm = train_gbm(X_train, y_train, CONFIG)
    train_time = time.time() - t0

    # Validation accuracy on train set
    y_pred_proba = gbm.predict_proba(X_train)[:, 1]
    y_pred_proba_mat = y_pred_proba.reshape(split_idx, n_students)

    correct_top6 = 0
    total_top6 = 0
    for t in range(split_idx):
        top6_pred  = set(np.argsort(-y_pred_proba_mat[t])[:6])
        top6_actual = set(np.where(binary[t] == 1)[0])
        correct_top6 += len(top6_pred & top6_actual)
        total_top6 += 6

    train_acc = correct_top6 / total_top6
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Train top-6 overlap accuracy: {train_acc:.4f}")

    # ========================================================
    # PHASE 4: Online Simulation
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 4: ONLINE LEARNING SIMULATION".center(70))
    print(f"{'='*70}")

    # Initialize incremental state from training data
    state = IncrementalGBState(binary[:split_idx], n_students=n_students, start_day=0)

    t0 = time.time()
    gbm, accuracies, pred_log = run_gb_simulation(
        gbm, state, binary, split_idx, CONFIG
    )
    sim_time = time.time() - t0

    # ========================================================
    # PHASE 5: Results
    # ========================================================
    gb_stats = compute_statistics(accuracies)
    print_statistics(gb_stats, "GRADIENT BOOSTING - ONLINE LEARNING STATISTICS")

    # Feature importances
    try:
        feature_names = [
            'days_since_last', 'freq_5', 'freq_10', 'freq_20',
            'pairwise_score', 'student_position', 'dow_freq',
            'entropy_7d', 'entropy_20d'
        ]
        importances = gbm.feature_importances_
        print(f"\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances),
                                 key=lambda x: -x[1]):
            bar = '█' * int(imp * 50)
            print(f"  {name:<22} {imp:.4f}  {bar}")
    except Exception:
        pass

    # Save results
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(gbm, _CKPT_DIR / 'gb_model.pkl')

    results_data = {
        'model_type': 'Gradient Boosting + Feature Engineering',
        'n_features': int(X_train.shape[1]),
        'n_training_samples': int(len(X_train)),
        'training_time_s': float(train_time),
        'simulation_time_s': float(sim_time),
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        'results': gb_stats,
    }
    with open(_CKPT_DIR / 'gb_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    pd.DataFrame(pred_log).to_csv(
        _CKPT_DIR / 'gb_predictions_log.csv', index=False
    )

    print(f"\n  Model saved:   {_CKPT_DIR / 'gb_model.pkl'}")
    print(f"  Results saved: {_CKPT_DIR / 'gb_results.json'}")
    print(f"  Log saved:     {_CKPT_DIR / 'gb_predictions_log.csv'}")

    # Full comparison
    print_comparison(gb_stats)

    print(f"\n{'='*70}")
    print("GRADIENT BOOSTING PIPELINE COMPLETE".center(70))
    print(f"{'='*70}")

    return gb_stats


if __name__ == "__main__":
    results = main()
