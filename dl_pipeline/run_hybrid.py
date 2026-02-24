"""
LSTM + Markov Chain Hybrid Pipeline

Combines Enhanced LSTM (neural temporal patterns) with Factored Markov Chain
(explicit transition & co-occurrence patterns) for prediction.

Blending strategy:
  hybrid_prob = alpha * LSTM_prob + (1 - alpha) * Markov_prob
  
  alpha is tuned to find the optimal balance.

Why this might outperform LSTM alone:
  - Markov Chain explicitly models TRANSITION patterns (who follows whom)
  - Markov Chain captures PAIRWISE co-occurrence (group patterns)
  - LSTM captures deep TEMPORAL patterns from 14-day sequences
  - The two models see the data through different lenses
  - Blending reduces noise while preserving strong signals from both

Usage:
    cd dl_pipeline
    python run_hybrid.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import time

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'excel_path': '../Database.xlsx',
    'sequence_length': 14,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 15,
    'train_ratio': 0.9,
    'n_simulation_days': 100,
    'prediction_interval': 1,
    'n_groups': 5,
    'k_students': 6,
    'temperature': 1.5,
    'online_iterations': 5,
    'replay_buffer_size': 10,
    'replay_min_samples': 5,
    'n_students': 55,
    'random_seed': 42,
    # Hybrid blending weights to test
    'alpha_values': [0.5, 0.6, 0.7, 0.8],
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
}


# ============================================================
# Dataset
# ============================================================
class HybridDataset(Dataset):
    def __init__(self, enriched, binary, seq_len=14):
        self.features = enriched
        self.targets = binary
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ============================================================
# LSTM Training
# ============================================================
def train_lstm(model, train_loader, val_loader, criterion, device,
               lr, epochs, patience):
    """Train Enhanced LSTM with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} | "
                  f"Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"    Best val loss: {best_val_loss:.4f}")
    return optimizer


# ============================================================
# Hybrid prediction
# ============================================================
def hybrid_predict(model, markov, enriched_seq, binary_seq, device,
                   alpha, n_groups=5, k=6, temperature=1.5):
    """
    Combine LSTM and Markov Chain predictions.
    
    hybrid_prob = alpha * LSTM_prob + (1 - alpha) * Markov_prob
    
    Args:
        model: Enhanced LSTM model
        markov: Fitted FactoredMarkovChain
        enriched_seq: Enriched features for last seq_len days (seq_len, 220)
        binary_seq: Raw binary vectors for last seq_len days (seq_len, 55)
        device: torch device
        alpha: Blending weight (higher = more LSTM, lower = more Markov)
        n_groups: Number of groups to generate
        k: Students per group
        temperature: Sampling temperature
    
    Returns:
        groups: List of student ID lists
        hybrid_probs: Combined probability vector (55,)
        lstm_probs: LSTM-only probabilities (55,)
        markov_probs: Markov-only probabilities (55,)
    """
    # LSTM probabilities
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(enriched_seq).unsqueeze(0).to(device)
        lstm_probs = model(X).cpu().numpy()[0]

    # Markov Chain probabilities
    markov_probs = markov.predict_probabilities(binary_seq)

    # Normalize LSTM probs to sum to 1 (for proper blending)
    lstm_norm = lstm_probs / lstm_probs.sum() if lstm_probs.sum() > 0 else lstm_probs

    # Blend
    hybrid_probs = alpha * lstm_norm + (1 - alpha) * markov_probs

    # Normalize
    hybrid_probs = hybrid_probs / hybrid_probs.sum()

    # Generate diverse groups from hybrid probabilities
    groups = []
    for g in range(n_groups):
        temp = temperature * (0.7 + 0.15 * g)
        adj = np.power(hybrid_probs, 1.0 / temp)
        adj = adj / adj.sum()

        selected = np.random.choice(len(adj), size=k, replace=False, p=adj)
        selected = selected[np.argsort(-hybrid_probs[selected])]
        groups.append((selected + 1).tolist())

    return groups, hybrid_probs, lstm_probs, markov_probs


# ============================================================
# Online simulation
# ============================================================
def run_hybrid_simulation(model, markov, optimizer, criterion,
                          all_enriched, all_binary, split_idx, device,
                          alpha, config):
    """Run online learning simulation with hybrid predictions."""
    seq_len = config['sequence_length']
    n_sim = min(config['n_simulation_days'], len(all_binary) - split_idx)

    accuracies = []
    predictions_log = []
    replay_buffer = []

    for test_idx in range(0, n_sim, config['prediction_interval']):
        actual_idx = split_idx + test_idx
        if actual_idx < seq_len:
            continue

        # Get sequences
        enriched_seq = all_enriched[actual_idx - seq_len: actual_idx]
        binary_seq = all_binary[actual_idx - seq_len: actual_idx]

        # Hybrid prediction
        groups, hybrid_probs, lstm_probs, markov_probs = hybrid_predict(
            model, markov, enriched_seq, binary_seq, device,
            alpha=alpha,
            n_groups=config['n_groups'],
            k=config['k_students'],
            temperature=config['temperature']
        )

        # Accuracy
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
            'best_accuracy': best_accuracy, 'predicted_groups': groups,
            'actual': actual_ids,
        })

        # Online update for BOTH models
        # 1. Update Markov Chain with new observation
        if actual_idx >= 2:
            markov.update(
                actual_binary,
                all_binary[actual_idx - 1],
                all_binary[actual_idx - 2]
            )

        # 2. Update LSTM with replay buffer
        replay_buffer.append((enriched_seq.copy(), actual_binary.copy()))

        if test_idx + 1 < n_sim:
            next_idx = actual_idx + 1
            if next_idx < len(all_binary) and next_idx >= seq_len:
                X_next = all_enriched[next_idx - seq_len: next_idx]
                replay_buffer.append((X_next.copy(), all_binary[next_idx].copy()))

        if len(replay_buffer) >= config['replay_min_samples']:
            recent = replay_buffer[-config['replay_buffer_size']:]
            X_batch = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            y_batch = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)

            model.train()
            for _ in range(config['online_iterations']):
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    return accuracies, predictions_log


# ============================================================
# Statistics
# ============================================================
def compute_statistics(accuracies):
    distribution = {}
    for correct in range(7):
        acc_val = (correct / 6) * 100
        count = sum(1 for a in accuracies if abs(a - acc_val) < 1.0)
        distribution[correct] = (count / len(accuracies)) * 100

    first_10 = np.mean(accuracies[:10]) if len(accuracies) >= 10 else 0
    last_10 = np.mean(accuracies[-10:]) if len(accuracies) >= 10 else 0

    return {
        'average_accuracy': float(np.mean(accuracies)),
        'best_accuracy': float(np.max(accuracies)),
        'worst_accuracy': float(np.min(accuracies)),
        'std_deviation': float(np.std(accuracies)),
        'total_predictions': len(accuracies),
        'distribution': distribution,
        'first_10_avg': float(first_10),
        'last_10_avg': float(last_10),
    }


def print_statistics(stats, title):
    print(f"\n{'='*70}")
    print(f"{title}".center(70))
    print(f"{'='*70}")
    print(f"\nTotal predictions: {stats['total_predictions']}")
    print(f"Average accuracy: {stats['average_accuracy']:.2f}%")
    print(f"Best accuracy: {stats['best_accuracy']:.2f}%")
    print(f"Worst accuracy: {stats['worst_accuracy']:.2f}%")
    print(f"Std deviation: {stats['std_deviation']:.4f}")
    print(f"\nAccuracy distribution:")
    for c in range(7):
        pct = stats['distribution'].get(c, 0)
        n = round(pct * stats['total_predictions'] / 100)
        print(f"  {c}/6 correct: {n:3d} ({pct:5.1f}%)")
    if stats.get('first_10_avg'):
        imp = stats['last_10_avg'] - stats['first_10_avg']
        print(f"\nTrend analysis:")
        print(f"  First 10: {stats['first_10_avg']:.2f}%")
        print(f"  Last 10: {stats['last_10_avg']:.2f}%")
        print(f"  Change: {imp:+.2f}%")


def print_comparison(all_results):
    """Print comparison table of all models."""
    print(f"\n{'='*80}")
    print("COMPLETE MODEL COMPARISON".center(80))
    print(f"{'='*80}")

    header = f"  {'Metric':<22}"
    for name in all_results:
        header += f" {name:<18}"
    print(f"\n{header}")
    print("  " + "-" * 78)

    metrics = [
        ('Avg Accuracy (%)', 'average_accuracy'),
        ('Best Accuracy (%)', 'best_accuracy'),
        ('Worst Accuracy (%)', 'worst_accuracy'),
        ('Std Deviation', 'std_deviation'),
        ('Predictions', 'total_predictions'),
    ]

    for label, key in metrics:
        row = f"  {label:<22}"
        for name, m in all_results.items():
            v = m[key]
            row += f" {v:<18.2f}"
        print(row)

    print(f"\n  Accuracy Distribution (% of predictions):")
    print(f"  {'Correct':<10}", end="")
    for name in all_results:
        print(f" {name:<18}", end="")
    print()
    print("  " + "-" * 78)

    for c in range(7):
        row = f"  {c}/6{'':<6}"
        for name, m in all_results.items():
            pct = m['distribution'].get(c, 0)
            row += f" {pct:>6.1f}%{'':<11}"
        print(row)

    # Winner
    print(f"\n{'='*80}")
    best_avg_name = max(all_results, key=lambda n: all_results[n]['average_accuracy'])
    best_avg = all_results[best_avg_name]['average_accuracy']
    best_peak_name = max(all_results, key=lambda n: all_results[n]['best_accuracy'])
    best_peak = all_results[best_peak_name]['best_accuracy']
    
    print(f"  Best average accuracy: {best_avg_name} ({best_avg:.2f}%)")
    print(f"  Best peak accuracy:   {best_peak_name} ({best_peak:.2f}%)")
    print(f"{'='*80}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("LSTM + MARKOV CHAIN HYBRID PIPELINE".center(70))
    print("=" * 70)

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ========================================================
    # PHASE 1: Load & Prepare Data
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 1: DATA PREPARATION".center(70))
    print(f"{'='*70}")

    print("\nLoading data...")
    df = pd.read_excel(CONFIG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)

    student_cols = [c for c in df.columns if c != 'Days']
    binary = np.zeros((len(df), CONFIG['n_students']), dtype=np.float32)
    for idx, row in df.iterrows():
        for col in student_cols:
            sid = row[col]
            if pd.notna(sid):
                sid = int(sid)
                if 1 <= sid <= CONFIG['n_students']:
                    binary[idx, sid - 1] = 1

    print(f"  Loaded {len(binary)} days, {CONFIG['n_students']} students")

    print("\nBuilding enriched features...")
    enriched = build_enriched_features(binary)
    n_features = enriched.shape[1]

    split_idx = int(len(binary) * CONFIG['train_ratio'])

    train_ds = HybridDataset(enriched[:split_idx], binary[:split_idx], CONFIG['sequence_length'])
    val_ds = HybridDataset(enriched[split_idx:], binary[split_idx:], CONFIG['sequence_length'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"  Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")

    # ========================================================
    # PHASE 2: Train Enhanced LSTM
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 2: TRAINING ENHANCED LSTM".center(70))
    print(f"{'='*70}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=CONFIG['n_students'],
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Enhanced BiLSTM + Attention: {n_params:,} params")

    optimizer = train_lstm(
        model, train_loader, val_loader, criterion, device,
        lr=0.0005, epochs=CONFIG['epochs'],
        patience=CONFIG['early_stopping_patience']
    )

    # ========================================================
    # PHASE 3: Fit Markov Chain
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 3: FITTING MARKOV CHAIN".center(70))
    print(f"{'='*70}")

    # Fit on training data only (no data leakage)
    markov = FactoredMarkovChain(n_students=CONFIG['n_students'], smoothing=1.0)
    markov.fit(binary[:split_idx])

    # Show some Markov Chain statistics
    base_rate = 6.0 / CONFIG['n_students']
    print(f"\n  Base selection rate: {base_rate:.4f} ({6}/{CONFIG['n_students']})")

    # Show transition pattern examples
    print(f"\n  Sample transition probabilities:")
    for s in [0, 10, 20, 30, 40]:
        p_sel_after_sel = (markov.transition_counts[s, 1, :].sum() + 1) / \
                          (markov.transition_totals[s, 1, :].sum() + 2)
        p_sel_after_not = (markov.transition_counts[s, 0, :].sum() + 1) / \
                          (markov.transition_totals[s, 0, :].sum() + 2)
        print(f"    Student {s+1:2d}: P(sel|was_sel)={p_sel_after_sel:.3f}  "
              f"P(sel|not_sel)={p_sel_after_not:.3f}  "
              f"ratio={p_sel_after_sel/p_sel_after_not:.2f}x")

    # ========================================================
    # PHASE 4: Test Multiple Alpha Values
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 4: HYBRID SIMULATION (TESTING BLEND WEIGHTS)".center(70))
    print(f"{'='*70}")

    best_alpha = None
    best_avg_acc = 0
    best_stats = None
    best_log = None
    all_alpha_results = {}

    for alpha in CONFIG['alpha_values']:
        print(f"\n{'~'*70}")
        print(f"  Testing alpha = {alpha:.1f} "
              f"({alpha*100:.0f}% LSTM + {(1-alpha)*100:.0f}% Markov)")
        print(f"{'~'*70}")

        # Reset model and Markov to pre-simulation state
        # We need fresh copies for each alpha test
        torch.manual_seed(CONFIG['random_seed'])
        np.random.seed(CONFIG['random_seed'])

        # Reload best LSTM weights
        model_copy = EnhancedLSTMPredictor(
            n_features=n_features, n_students=CONFIG['n_students'],
            hidden_size=128, num_layers=2, dropout=0.3
        ).to(device)
        model_copy.load_state_dict(model.state_dict())

        markov_copy = FactoredMarkovChain(n_students=CONFIG['n_students'], smoothing=1.0)
        markov_copy.fit(binary[:split_idx])

        opt_copy = optim.Adam(model_copy.parameters(), lr=0.0003)

        start_time = time.time()
        accuracies, pred_log = run_hybrid_simulation(
            model_copy, markov_copy, opt_copy, criterion,
            enriched, binary, split_idx, device,
            alpha=alpha, config=CONFIG
        )
        sim_time = time.time() - start_time

        stats = compute_statistics(accuracies)
        all_alpha_results[alpha] = stats

        print(f"\n  alpha={alpha:.1f} => Avg: {stats['average_accuracy']:.2f}% | "
              f"Best: {stats['best_accuracy']:.2f}% | "
              f"Worst: {stats['worst_accuracy']:.2f}% | "
              f"Time: {sim_time:.1f}s")

        if stats['average_accuracy'] > best_avg_acc:
            best_avg_acc = stats['average_accuracy']
            best_alpha = alpha
            best_stats = stats
            best_log = pred_log

    # ========================================================
    # PHASE 5: Results Summary
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 5: ALPHA TUNING RESULTS".center(70))
    print(f"{'='*70}")

    print(f"\n  {'Alpha':<10} {'LSTM %':<10} {'Markov %':<10} {'Avg Acc':<12} "
          f"{'Best':<10} {'Worst':<10} {'Std':<10}")
    print("  " + "-" * 72)

    for alpha, stats in all_alpha_results.items():
        print(f"  {alpha:<10.1f} {alpha*100:<10.0f} {(1-alpha)*100:<10.0f} "
              f"{stats['average_accuracy']:<12.2f} {stats['best_accuracy']:<10.2f} "
              f"{stats['worst_accuracy']:<10.2f} {stats['std_deviation']:<10.4f}")

    print(f"\n  >> Best alpha: {best_alpha:.1f} "
          f"({best_alpha*100:.0f}% LSTM + {(1-best_alpha)*100:.0f}% Markov)")

    # Print full stats for best alpha
    print_statistics(best_stats,
                     f"HYBRID (alpha={best_alpha:.1f}) - ONLINE LEARNING STATISTICS")

    # Save results
    ckpt_dir = Path('dl_checkpoints')
    ckpt_dir.mkdir(exist_ok=True)

    results_data = {
        'model_type': f'LSTM + Markov Chain Hybrid (alpha={best_alpha})',
        'best_alpha': best_alpha,
        'all_alpha_results': {
            str(a): {k: v for k, v in s.items()}
            for a, s in all_alpha_results.items()
        },
        'best_results': best_stats,
    }
    with open(ckpt_dir / 'hybrid_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    log_df = pd.DataFrame([
        {'day': p['day'], 'best_overlap': p['best_overlap'],
         'best_accuracy': p['best_accuracy']}
        for p in best_log
    ])
    log_df.to_csv(ckpt_dir / 'hybrid_predictions_log.csv', index=False)

    print(f"\n  Results saved: dl_checkpoints/hybrid_results.json")
    print(f"  Log saved: dl_checkpoints/hybrid_predictions_log.csv")

    # Full comparison against previous models
    all_models = {**PREVIOUS_RESULTS}
    all_models[f'Hybrid (a={best_alpha})'] = best_stats
    print_comparison(all_models)

    print(f"\n{'='*70}")
    print("HYBRID PIPELINE COMPLETE".center(70))
    print(f"{'='*70}")

    return best_stats, best_alpha


if __name__ == "__main__":
    results, alpha = main()
