"""
Meta-Ensemble Pipeline
======================

Combines Markov Chain, XGBoost/HistGB, Logistic Regression, and LSTM
into a single blended prediction with adaptive online weight learning.

Simulation tracks EVERY model's standalone accuracy at each step,
enabling a fair per-model comparison and rich visualisation.

Usage:
    cd dl_pipeline/meta_ensemble
    python run_meta_ensemble.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
_ROOT_DIR  = _DL_DIR.parent
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'meta_ensemble'

sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import json
import time
import warnings
warnings.filterwarnings('ignore')

from meta_ensemble.base_models import (
    MarkovWrapper, XGBWrapper, LogisticWrapper, LSTMWrapper,
    MetaEnsemble, probs_to_groups,
)


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'excel_path':          str(_ROOT_DIR / 'Database.xlsx'),
    'train_ratio':         0.9,
    'n_students':          55,
    'k_per_day':           6,
    'n_simulation_days':   100,
    'prediction_interval': 1,
    'n_groups':            5,
    'random_seed':         42,
    # LSTM settings
    'lstm_hidden':         64,
    'lstm_epochs':         20,
    'lstm_seq_len':        14,
    # Ensemble weight learning
    'ema_alpha':           0.15,
    'ensemble_temperature': 0.5,
}

# Previous model benchmarks (for comparison table)
PREVIOUS_RESULTS = {
    'Baseline LSTM':       {'average_accuracy': 26.00, 'best_accuracy': 66.67, 'worst_accuracy': 0.00, 'std_deviation': 12.54, 'total_predictions': 50, 'distribution': {0: 4.0, 1: 46.0, 2: 42.0, 3: 6.0, 4: 2.0, 5: 0.0, 6: 0.0}},
    'Enhanced LSTM':       {'average_accuracy': 28.33, 'best_accuracy': 50.00, 'worst_accuracy': 0.00, 'std_deviation': 12.13, 'total_predictions': 50, 'distribution': {0: 2.0, 1: 40.0, 2: 44.0, 3: 14.0, 4: 0.0, 5: 0.0, 6: 0.0}},
    'LSTM+Markov Hybrid':  {'average_accuracy': 29.00, 'best_accuracy': 50.00, 'worst_accuracy': 0.00, 'std_deviation': 10.96, 'total_predictions': 50, 'distribution': {0: 2.0, 1: 32.0, 2: 56.0, 3: 10.0, 4: 0.0, 5: 0.0, 6: 0.0}},
    'Gradient Boosting':   {'average_accuracy': 26.67, 'best_accuracy': 50.00, 'worst_accuracy': 16.67, 'std_deviation': 11.55, 'total_predictions': 50, 'distribution': {0: 0.0, 1: 52.0, 2: 36.0, 3: 12.0, 4: 0.0, 5: 0.0, 6: 0.0}},
    'Bayesian BB':         {'average_accuracy': 27.67, 'best_accuracy': 50.00, 'worst_accuracy': 0.00, 'std_deviation': 11.36, 'total_predictions': 50, 'distribution': {0: 2.0, 1: 40.0, 2: 48.0, 3: 10.0, 4: 0.0, 5: 0.0, 6: 0.0}},
}


# ============================================================
# Statistics
# ============================================================
def compute_statistics(accuracies: list) -> dict:
    distribution = {}
    for c in range(7):
        threshold = (c / 6) * 100
        count = sum(1 for a in accuracies if abs(a - threshold) < 1.0)
        distribution[c] = (count / len(accuracies)) * 100

    return {
        'average_accuracy':  float(np.mean(accuracies)),
        'best_accuracy':     float(np.max(accuracies)),
        'worst_accuracy':    float(np.min(accuracies)),
        'std_deviation':     float(np.std(accuracies)),
        'total_predictions': len(accuracies),
        'distribution':      distribution,
        'first_10_avg':      float(np.mean(accuracies[:10])) if len(accuracies) >= 10 else 0.0,
        'last_10_avg':       float(np.mean(accuracies[-10:])) if len(accuracies) >= 10 else 0.0,
    }


def print_model_stats(stats: dict, name: str):
    print(f"\n  {name}:")
    print(f"    Avg: {stats['average_accuracy']:5.2f}%  "
          f"Best: {stats['best_accuracy']:5.2f}%  "
          f"Std: {stats['std_deviation']:.4f}  "
          f"Trend: {stats['last_10_avg']-stats['first_10_avg']:+.2f}%")
    dist_str = "  ".join(f"{c}/6={round(stats['distribution'].get(c,0)*stats['total_predictions']/100)}({stats['distribution'].get(c,0):.0f}%)"
                          for c in range(5))
    print(f"    Dist: {dist_str}")


def print_comparison(new_results: dict):
    """Print full comparison table including previous models."""
    all_models = {**PREVIOUS_RESULTS, **new_results}
    col_w = 14

    print(f"\n{'='*90}")
    print("COMPLETE MODEL COMPARISON".center(90))
    print(f"{'='*90}")
    header = f"  {'Metric':<22}" + ''.join(f" {n[:col_w-1]:<{col_w}}" for n in all_models)
    print(f"\n{header}")
    print("  " + "-" * (22 + (col_w + 1) * len(all_models)))

    for label, key, lower_better in [
        ('Avg Accuracy (%)',   'average_accuracy',  False),
        ('Best Accuracy (%)',  'best_accuracy',     False),
        ('Worst Accuracy (%)', 'worst_accuracy',    True),
        ('Std Deviation',      'std_deviation',     True),
    ]:
        vals = [m[key] for m in all_models.values()]
        best = min(vals) if lower_better else max(vals)
        row  = f"  {label:<22}"
        for m in all_models.values():
            v    = m[key]
            mark = "*" if abs(v - best) < 0.01 else " "
            row += f" {v:>7.2f}{mark:<{col_w-8}}"
        print(row)

    print(f"\n  Accuracy Distribution (% of predictions with k/6 correct):")
    print(f"  {'k':<6}" + ''.join(f" {n[:col_w-1]:<{col_w}}" for n in all_models))
    print("  " + "-" * (6 + (col_w + 1) * len(all_models)))
    for c in range(5):
        row = f"  {c}/6   "
        for m in all_models.values():
            pct = m['distribution'].get(c, 0)
            row += f" {pct:>6.1f}%{'':<{col_w-8}}"
        print(row)

    print(f"\n{'='*90}")
    best_avg  = max(all_models, key=lambda n: all_models[n]['average_accuracy'])
    best_std  = min(all_models, key=lambda n: all_models[n]['std_deviation'])
    best_peak = max(all_models, key=lambda n: all_models[n]['best_accuracy'])
    print(f"  Best avg accuracy  : {best_avg}  ({all_models[best_avg]['average_accuracy']:.2f}%)")
    print(f"  Best consistency   : {best_std}  (std {all_models[best_std]['std_deviation']:.4f})")
    print(f"  Best single peak   : {best_peak}  ({all_models[best_peak]['best_accuracy']:.2f}%)")
    print(f"{'='*90}")


# ============================================================
# Online Simulation
# ============================================================
def run_simulation(wrappers: dict, ensemble: MetaEnsemble,
                   all_binary: np.ndarray, split_idx: int,
                   config: dict) -> dict:
    """
    Run the online simulation.

    For each prediction step:
      1. Each base model predicts independently (5 groups → best overlap)
      2. Ensemble blends the 4 probability vectors (5 groups → best overlap)
      3. All models are updated with the actual observed day
      4. Ensemble weights are adapted via EMA of top-6 overlap

    Returns:
        dict mapping model_name → list of accuracy values (50 floats)
    """
    n_sim   = min(config['n_simulation_days'], len(all_binary) - split_idx)
    n_grp   = config['n_groups']
    k       = config['k_per_day']
    p_int   = config['prediction_interval']
    names   = list(wrappers.keys())

    model_accs   = {n: [] for n in names}
    model_accs['Meta-Ensemble'] = []
    pred_log     = []

    print(f"\n  Simulation: {n_sim} days | "
          f"predict every {p_int} days | {len(names)+1} models tracked\n")

    for test_idx in range(0, n_sim, p_int):
        abs_day      = split_idx + test_idx
        actual_bin   = all_binary[abs_day]
        actual_ids   = set((np.where(actual_bin == 1)[0] + 1).tolist())

        # ---- Per-model predictions ----
        probs_list = []
        for name, wrapper in wrappers.items():
            if name in ('XGBoost', 'HistGB', 'Logistic'):
                p = wrapper.predict_proba(abs_day)
            else:
                p = wrapper.predict_proba()
            probs_list.append(p)

            groups      = probs_to_groups(p, n_groups=n_grp, k=k)
            best_overlap = max(len(set(g) & actual_ids) for g in groups)
            model_accs[name].append(best_overlap / k * 100)

        # ---- Ensemble prediction ----
        ens_probs    = ensemble.blend(probs_list)
        ens_groups   = probs_to_groups(ens_probs, n_groups=n_grp, k=k)
        ens_overlaps = [len(set(g) & actual_ids) for g in ens_groups]
        ens_best     = max(ens_overlaps)
        ens_best_idx = ens_overlaps.index(ens_best)
        model_accs['Meta-Ensemble'].append(ens_best / k * 100)

        # ---- Console log (ensemble result only) ----
        day_label = test_idx + 1
        print(f"  Day {day_label:3d}: Ensemble={ens_best/k*100:5.1f}% ({ens_best}/6) "
              f"| {ens_groups[ens_best_idx]} | Actual: {sorted(actual_ids)}")

        # Weight display every 20 steps
        if day_label % 20 == 0:
            w_dict = {n: f"{w:.3f}" for n, w in zip(names, ensemble.weights)}
            print(f"         Weights → {w_dict}")

        pred_log.append({
            'day':          day_label,
            **{f"acc_{n}": model_accs[n][-1] for n in names},
            'acc_Ensemble': ens_best / k * 100,
            'ens_group':    str(ens_groups[ens_best_idx]),
            'actual':       str(sorted(actual_ids)),
        })

        # ---- Updates ----
        for name, wrapper in wrappers.items():
            if name in ('XGBoost', 'HistGB', 'Logistic'):
                wrapper.update(actual_bin, abs_day)
            else:
                wrapper.update(actual_bin)

        ensemble.update_weights(probs_list, actual_bin, k=k)

    return model_accs, pred_log


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(CONFIG['random_seed'])
    torch_seed = CONFIG['random_seed']
    import torch; torch.manual_seed(torch_seed)

    print("=" * 75)
    print("META-ENSEMBLE PIPELINE  (Markov + XGBoost + Logistic + LSTM)".center(75))
    print("=" * 75)

    # ========================================================
    # PHASE 1: Load Data
    # ========================================================
    print(f"\n{'='*75}")
    print("PHASE 1: DATA PREPARATION".center(75))
    print(f"{'='*75}")

    df = pd.read_excel(CONFIG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)

    n_students   = CONFIG['n_students']
    student_cols = [c for c in df.columns if c != 'Days']
    binary       = np.zeros((len(df), n_students), dtype=np.float32)
    for idx, row in df.iterrows():
        for col in student_cols:
            sid = row[col]
            if pd.notna(sid):
                sid = int(sid)
                if 1 <= sid <= n_students:
                    binary[idx, sid - 1] = 1

    split_idx = int(len(binary) * CONFIG['train_ratio'])
    print(f"\n  Total days: {len(binary)} | Train: {split_idx} | Test: {len(binary)-split_idx}")

    # ========================================================
    # PHASE 2: Fit Base Models
    # ========================================================
    print(f"\n{'='*75}")
    print("PHASE 2: FITTING BASE MODELS".center(75))
    print(f"{'='*75}\n")

    train_bin = binary[:split_idx]
    t_total   = time.time()

    markov_w  = MarkovWrapper(n_students=n_students)
    xgb_w     = XGBWrapper(n_students=n_students)
    lr_w      = LogisticWrapper(n_students=n_students)
    lstm_w    = LSTMWrapper(
        n_students=n_students,
        hidden_size=CONFIG['lstm_hidden'],
        n_epochs=CONFIG['lstm_epochs'],
        seq_len=CONFIG['lstm_seq_len'],
    )

    t0 = time.time()
    print(f"  [1/4] Fitting Markov Chain...")
    markov_w.fit(train_bin)
    print(f"        Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"\n  [2/4] Fitting {xgb_w.name}...")
    xgb_w.fit(train_bin, split_idx)
    print(f"        Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"\n  [3/4] Fitting Logistic Regression...")
    lr_w.fit(train_bin, split_idx)
    print(f"        Done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"\n  [4/4] Fitting LSTM...")
    lstm_w.fit(train_bin)
    print(f"        Done in {time.time()-t0:.1f}s")

    print(f"\n  Total fitting time: {time.time()-t_total:.1f}s")

    # Ordered dict to control the blend order
    wrappers = {
        'Markov':           markov_w,
        xgb_w.name:        xgb_w,
        'Logistic':         lr_w,
        'LSTM':             lstm_w,
    }
    model_names = list(wrappers.keys())

    ensemble = MetaEnsemble(
        n_models=len(wrappers),
        temperature=CONFIG['ensemble_temperature'],
        ema_alpha=CONFIG['ema_alpha'],
    )
    print(f"\n  Ensemble initial weights: {dict(zip(model_names, ensemble.weights.round(3)))}")

    # ========================================================
    # PHASE 3: Online Simulation
    # ========================================================
    print(f"\n{'='*75}")
    print("PHASE 3: ONLINE LEARNING SIMULATION".center(75))
    print(f"{'='*75}")

    t0 = time.time()
    model_accs, pred_log = run_simulation(
        wrappers, ensemble, binary, split_idx, CONFIG
    )
    sim_time = time.time() - t0

    # ========================================================
    # PHASE 4: Statistics
    # ========================================================
    print(f"\n{'='*75}")
    print("PHASE 4: PER-MODEL RESULTS".center(75))
    print(f"{'='*75}")

    all_stats = {}
    for name, accs in model_accs.items():
        all_stats[name] = compute_statistics(accs)
        print_model_stats(all_stats[name], name)

    # Final ensemble weights
    print(f"\n  Final ensemble weights:")
    for n, w in zip(model_names, ensemble.weights):
        bar = '█' * int(w * 40)
        print(f"    {n:<12} {w:.4f}  {bar}")

    # ========================================================
    # PHASE 5: Save Results
    # ========================================================
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results_data = {
        'model_type': 'Meta-Ensemble (Markov + XGB + Logistic + LSTM)',
        'config':     {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        'sim_time_s': float(sim_time),
        'final_weights': dict(zip(model_names, ensemble.weights.tolist())),
        'per_model_results': {n: all_stats[n] for n in all_stats},
    }
    with open(_CKPT_DIR / 'meta_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    pd.DataFrame(pred_log).to_csv(_CKPT_DIR / 'predictions_log.csv', index=False)
    print(f"\n  Results saved: {_CKPT_DIR / 'meta_results.json'}")

    # ========================================================
    # PHASE 6: Comparison Table
    # ========================================================
    print_comparison(all_stats)

    # ========================================================
    # PHASE 7: Visualisation
    # ========================================================
    print(f"\n{'='*75}")
    print("PHASE 5: GENERATING VISUALISATIONS".center(75))
    print(f"{'='*75}\n")

    try:
        from meta_ensemble.visualize import create_full_dashboard

        weight_history = np.array(ensemble.weight_history)   # (steps, 4)

        create_full_dashboard(
            previous_results=PREVIOUS_RESULTS,
            new_results=all_stats,
            accuracy_traces=model_accs,
            weight_history=weight_history,
            model_names=model_names,
            output_dir=_CKPT_DIR,
        )
        print(f"  Charts saved to: {_CKPT_DIR}")
    except Exception as e:
        print(f"  Visualisation skipped: {e}")

    print(f"\n{'='*75}")
    print("META-ENSEMBLE PIPELINE COMPLETE".center(75))
    print(f"{'='*75}")

    return all_stats


if __name__ == '__main__':
    results = main()
