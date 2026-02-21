"""
Enhanced LSTM Pipeline - Training, Evaluation & Comparison

This script runs the complete enhanced LSTM pipeline and compares
results against the baseline LSTM model.

Improvements implemented:
1. Feature Engineering: 220 features (vs 55 raw binary)
   - 7-day selection frequency
   - 14-day selection frequency
   - Recency signal
2. Bidirectional LSTM: context from both directions
3. Attention Mechanism: learns which past days matter
4. Focal Loss: focuses on harder predictions
5. Label Smoothing: prevents overconfidence

Usage:
    cd dl_pipeline
    python run_enhanced.py
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


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'excel_path': '../Database.xlsx',
    'sequence_length': 14,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0005,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'train_ratio': 0.9,
    'n_simulation_days': 100,
    'prediction_interval': 2,
    'n_groups': 5,
    'k_students': 6,
    'temperature': 1.5,
    'online_lr': 0.0003,
    'online_iterations': 5,
    'replay_buffer_size': 10,
    'replay_min_samples': 5,
    'n_students': 55,
    'random_seed': 42,
}

# Baseline results for comparison (from previous 90/10 LSTM run)
BASELINE_RESULTS = {
    'average_accuracy': 26.00,
    'best_accuracy': 66.67,
    'worst_accuracy': 0.00,
    'std_deviation': 12.54,
    'total_predictions': 50,
    'distribution': {0: 4.0, 1: 46.0, 2: 42.0, 3: 6.0, 4: 2.0, 5: 0.0, 6: 0.0},
    'first_10_avg': 30.00,
    'last_10_avg': 30.00,
}


# ============================================================
# Dataset
# ============================================================
class EnhancedDataset(Dataset):
    """PyTorch Dataset with enriched features."""
    
    def __init__(self, enriched_features: np.ndarray, binary_targets: np.ndarray,
                 sequence_length: int = 14):
        self.features = enriched_features
        self.targets = binary_targets
        self.seq_len = sequence_length
    
    def __len__(self) -> int:
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx: int):
        X = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ============================================================
# Data Loading
# ============================================================
def load_data(excel_path: str, n_students: int = 55):
    """Load Excel data and convert to binary vectors."""
    df = pd.read_excel(excel_path)
    df = df.sort_values('Days').reset_index(drop=True)
    
    student_cols = [col for col in df.columns if col != 'Days']
    n_rows = len(df)
    binary = np.zeros((n_rows, n_students), dtype=np.float32)
    
    for idx, row in df.iterrows():
        for col in student_cols:
            sid = row[col]
            if pd.notna(sid):
                sid = int(sid)
                if 1 <= sid <= n_students:
                    binary[idx, sid - 1] = 1
    
    return binary, df['Days'].values


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, val_loader, criterion, device, config):
    """Train the enhanced model with early stopping."""
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config['epochs']):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Top-6 overlap accuracy
            _, pred_idx = torch.topk(out, 6, dim=1)
            _, actual_idx = torch.topk(y, 6, dim=1)
            for i in range(len(X)):
                pred_set = set(pred_idx[i].cpu().numpy())
                actual_set = set(actual_idx[i].cpu().numpy())
                train_correct += len(pred_set & actual_set)
            train_total += len(X) * 6
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                
                _, pred_idx = torch.topk(out, 6, dim=1)
                _, actual_idx = torch.topk(y, 6, dim=1)
                for i in range(len(X)):
                    pred_set = set(pred_idx[i].cpu().numpy())
                    actual_set = set(actual_idx[i].cpu().numpy())
                    val_correct += len(pred_set & actual_set)
                val_total += len(X) * 6
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | LR={lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final train accuracy: {history['train_acc'][-1]*100:.2f}%")
    print(f"  Final val accuracy: {history['val_acc'][-1]*100:.2f}%")
    
    return history, optimizer


# ============================================================
# Prediction
# ============================================================
def predict_groups(model, X_tensor, device, n_groups=5, k=6, temperature=1.5):
    """
    Generate multiple diverse prediction groups using temperature sampling.
    Same approach as baseline for fair comparison.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        probs = model(X_tensor).cpu().numpy()[0]
    
    groups = []
    for g in range(n_groups):
        # Different temperature per group for diversity
        temp = temperature * (0.7 + 0.15 * g)
        adj = np.power(probs, 1.0 / temp)
        adj = adj / adj.sum()
        
        # Sample k students without replacement
        selected = np.random.choice(len(adj), size=k, replace=False, p=adj)
        # Sort by original probability (highest first)
        selected = selected[np.argsort(-probs[selected])]
        # Convert to 1-indexed student IDs
        groups.append((selected + 1).tolist())
    
    return groups, probs


# ============================================================
# Online Learning Simulation
# ============================================================
def run_online_simulation(model, criterion, all_enriched, all_binary,
                          split_idx, device, config):
    """
    Run online learning simulation matching baseline evaluation format.
    """
    seq_len = config['sequence_length']
    n_sim_days = min(config['n_simulation_days'], len(all_binary) - split_idx)
    
    test_days_available = len(all_binary) - split_idx
    expected_predictions = n_sim_days // config['prediction_interval']
    
    print(f"\n  Simulation setup:")
    print(f"    Test data available: {test_days_available} days")
    print(f"    Simulation days: {n_sim_days}")
    print(f"    Prediction interval: Every {config['prediction_interval']} days")
    print(f"    Expected predictions: {expected_predictions}")
    
    # Online optimizer (lower LR than training)
    online_optimizer = optim.Adam(model.parameters(), lr=config['online_lr'])
    
    # Tracking
    accuracies = []
    predictions_log = []
    replay_buffer = []  # stores (X, y) pairs
    
    for test_idx in range(0, n_sim_days, config['prediction_interval']):
        actual_idx = split_idx + test_idx
        
        if actual_idx < seq_len:
            continue
        
        # ======== MAKE PREDICTION ========
        X_seq = all_enriched[actual_idx - seq_len : actual_idx]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0)
        
        groups, probs = predict_groups(
            model, X_tensor, device,
            n_groups=config['n_groups'],
            k=config['k_students'],
            temperature=config['temperature']
        )
        
        # ======== COMPUTE ACCURACY ========
        actual_binary = all_binary[actual_idx]
        actual_ids = (np.where(actual_binary == 1)[0] + 1).tolist()
        
        overlap_counts = []
        for group in groups:
            overlap = len(set(group) & set(actual_ids))
            overlap_counts.append(overlap)
        
        best_overlap = max(overlap_counts)
        best_accuracy = (best_overlap / 6) * 100
        accuracies.append(best_accuracy)
        
        # Print day result
        best_group_idx = overlap_counts.index(best_overlap)
        day_label = test_idx + 1
        print(f"  Day {day_label:3d}: {best_accuracy:5.1f}% ({best_overlap}/6) | "
              f"Best group {best_group_idx+1}: {groups[best_group_idx]} | "
              f"Actual: {actual_ids}")
        
        # Log prediction
        predictions_log.append({
            'day': day_label,
            'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
            'predicted_groups': groups,
            'actual': actual_ids,
            'overlap_counts': overlap_counts
        })
        
        # ======== ONLINE UPDATE ========
        # Add current day to replay buffer
        replay_buffer.append((X_seq.copy(), actual_binary.copy()))
        
        # Also add intermediate day if available
        if test_idx + 1 < n_sim_days:
            next_idx = actual_idx + 1
            if next_idx < len(all_binary) and next_idx >= seq_len:
                X_next = all_enriched[next_idx - seq_len : next_idx]
                replay_buffer.append((X_next.copy(), all_binary[next_idx].copy()))
        
        # Train on replay buffer
        if len(replay_buffer) >= config['replay_min_samples']:
            recent = replay_buffer[-config['replay_buffer_size']:]
            X_batch = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            y_batch = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            
            model.train()
            total_loss = 0.0
            for _ in range(config['online_iterations']):
                online_optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                online_optimizer.step()
                total_loss += loss.item()
    
    return accuracies, predictions_log


# ============================================================
# Statistics & Comparison
# ============================================================
def print_statistics(accuracies, title="ONLINE LEARNING STATISTICS"):
    """Print detailed statistics matching baseline format."""
    print(f"\n{'='*70}")
    print(f"{title}".center(70))
    print(f"{'='*70}")
    
    avg_acc = np.mean(accuracies)
    best_acc = np.max(accuracies)
    worst_acc = np.min(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\nTotal predictions: {len(accuracies)}")
    print(f"Average accuracy: {avg_acc:.2f}%")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Worst accuracy: {worst_acc:.2f}%")
    print(f"Std deviation: {std_acc:.4f}")
    
    # Accuracy distribution
    print(f"\nAccuracy distribution:")
    distribution = {}
    for correct in range(7):
        accuracy_val = (correct / 6) * 100
        count = sum(1 for acc in accuracies if abs(acc - accuracy_val) < 1.0)
        pct = (count / len(accuracies)) * 100
        distribution[correct] = pct
        print(f"  {correct}/6 correct: {count:3d} ({pct:5.1f}%)")
    
    # Trend analysis
    first_10 = last_10 = improvement = 0
    if len(accuracies) >= 20:
        first_10 = np.mean(accuracies[:10])
        last_10 = np.mean(accuracies[-10:])
        improvement = last_10 - first_10
        print(f"\nTrend analysis:")
        print(f"  First 10 predictions: {first_10:.2f}%")
        print(f"  Last 10 predictions: {last_10:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
    
    return {
        'average_accuracy': float(avg_acc),
        'best_accuracy': float(best_acc),
        'worst_accuracy': float(worst_acc),
        'std_deviation': float(std_acc),
        'total_predictions': len(accuracies),
        'distribution': distribution,
        'first_10_avg': float(first_10),
        'last_10_avg': float(last_10),
    }


def print_comparison(enhanced_results):
    """Print side-by-side comparison with baseline."""
    baseline = BASELINE_RESULTS
    enhanced = enhanced_results
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON: BASELINE vs ENHANCED LSTM".center(70))
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<28} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
    print("-" * 70)
    
    metrics = [
        ('Average Accuracy (%)', 'average_accuracy', True),
        ('Best Accuracy (%)', 'best_accuracy', True),
        ('Worst Accuracy (%)', 'worst_accuracy', True),
        ('Std Deviation', 'std_deviation', False),
        ('Total Predictions', 'total_predictions', None),
    ]
    
    for name, key, higher_better in metrics:
        b_val = baseline[key]
        e_val = enhanced[key]
        diff = e_val - b_val
        
        if higher_better is None:
            change_str = f"{diff:+.0f}" if diff != 0 else "same"
        elif higher_better:
            marker = " (+)" if diff > 0 else (" (-)" if diff < 0 else "")
            change_str = f"{diff:+.2f}{marker}"
        else:
            marker = " (+)" if diff < 0 else (" (-)" if diff > 0 else "")
            change_str = f"{diff:+.2f}{marker}"
        
        print(f"  {name:<26} {b_val:<15.2f} {e_val:<15.2f} {change_str}")
    
    # Trend comparison
    print(f"\n  {'First 10 avg (%)':<26} {baseline['first_10_avg']:<15.2f} "
          f"{enhanced['first_10_avg']:<15.2f} "
          f"{enhanced['first_10_avg'] - baseline['first_10_avg']:+.2f}")
    print(f"  {'Last 10 avg (%)':<26} {baseline['last_10_avg']:<15.2f} "
          f"{enhanced['last_10_avg']:<15.2f} "
          f"{enhanced['last_10_avg'] - baseline['last_10_avg']:+.2f}")
    
    # Distribution comparison
    print(f"\n  Accuracy Distribution:")
    print(f"  {'Correct':<12} {'Baseline %':<15} {'Enhanced %':<15} {'Change':<15}")
    print("  " + "-" * 55)
    
    for correct in range(7):
        b_pct = baseline['distribution'].get(correct, 0)
        e_pct = enhanced['distribution'].get(correct, 0)
        diff = e_pct - b_pct
        direction = "(+)" if (correct >= 2 and diff > 0) or (correct <= 1 and diff < 0) else ""
        if (correct >= 2 and diff < 0) or (correct <= 1 and diff > 0):
            direction = "(-)"
        print(f"  {correct}/6{'':<8} {b_pct:>8.1f}%{'':<6} {e_pct:>8.1f}%{'':<6} {diff:>+6.1f}% {direction}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY".center(70))
    print(f"{'='*70}")
    
    acc_diff = enhanced['average_accuracy'] - baseline['average_accuracy']
    
    print(f"\n  Improvements applied:")
    print(f"    1. Feature Engineering: 220 features (vs 55 raw binary)")
    print(f"    2. Bidirectional LSTM: context from both directions")
    print(f"    3. Attention Mechanism: learns which past days matter")
    print(f"    4. Focal Loss: focuses on harder predictions")
    print(f"    5. Label Smoothing: prevents overconfidence")
    
    if acc_diff > 2:
        print(f"\n  Result: Enhanced model OUTPERFORMS baseline by {acc_diff:+.2f}%")
        print(f"          ({enhanced['average_accuracy']:.2f}% vs {baseline['average_accuracy']:.2f}%)")
    elif acc_diff < -2:
        print(f"\n  Result: Baseline model performs better by {abs(acc_diff):.2f}%")
        print(f"          ({baseline['average_accuracy']:.2f}% vs {enhanced['average_accuracy']:.2f}%)")
        print(f"  Note: The enhanced model may need more training data or tuning.")
    else:
        print(f"\n  Result: Both models perform similarly (diff: {acc_diff:+.2f}%)")
        print(f"          Baseline: {baseline['average_accuracy']:.2f}%")
        print(f"          Enhanced: {enhanced['average_accuracy']:.2f}%")
    
    print(f"\n{'='*70}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 70)
    print("ENHANCED LSTM PIPELINE".center(70))
    print("BiLSTM + Attention + Feature Engineering + Focal Loss".center(70))
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        if key not in ('random_seed',):
            print(f"  {key}: {value}")
    
    # ========================================================
    # PHASE 1: Load and Prepare Data
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 1: DATA PREPARATION".center(70))
    print(f"{'='*70}")
    
    print("\nLoading data...")
    binary, days = load_data(CONFIG['excel_path'], CONFIG['n_students'])
    print(f"  Loaded {len(binary)} days, {CONFIG['n_students']} students")
    
    print("\nComputing enriched features...")
    start_time = time.time()
    enriched = build_enriched_features(binary)
    n_features = enriched.shape[1]
    print(f"  Feature computation time: {time.time() - start_time:.2f}s")
    
    # Split data (90/10 - same as baseline)
    split_idx = int(len(binary) * CONFIG['train_ratio'])
    
    train_enriched = enriched[:split_idx]
    train_binary = binary[:split_idx]
    test_enriched = enriched[split_idx:]
    test_binary = binary[split_idx:]
    
    # Create datasets
    train_dataset = EnhancedDataset(train_enriched, train_binary, CONFIG['sequence_length'])
    val_dataset = EnhancedDataset(test_enriched, test_binary, CONFIG['sequence_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"\n  Data split (same as baseline):")
    print(f"    Training days: {len(train_binary)} ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"    Test days: {len(test_binary)} ({(1-CONFIG['train_ratio'])*100:.0f}%)")
    print(f"    Train sequences: {len(train_dataset)}")
    print(f"    Val sequences: {len(val_dataset)}")
    
    # ========================================================
    # PHASE 2: Build and Train Model
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 2: MODEL TRAINING".center(70))
    print(f"{'='*70}")
    
    print(f"\nCreating Enhanced LSTM model...")
    model = EnhancedLSTMPredictor(
        n_features=n_features,
        n_students=CONFIG['n_students'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: BiLSTM({CONFIG['hidden_size']}x2) + Attention + FC")
    print(f"  Input features: {n_features} (vs 55 baseline)")
    print(f"  Parameters: {n_params:,} (vs 300,599 baseline)")
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    print(f"  Loss: Focal Loss (alpha=0.25, gamma=2.0, smoothing=0.05)")
    print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']}, wd={CONFIG['weight_decay']})")
    
    print(f"\nTraining for up to {CONFIG['epochs']} epochs...")
    start_time = time.time()
    history, optimizer = train_model(
        model, train_loader, val_loader, criterion, device, CONFIG
    )
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s")
    
    # Save model
    checkpoint_dir = Path('dl_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': CONFIG,
        'n_features': n_features,
    }, checkpoint_dir / 'enhanced_best_model.pth')
    print(f"  Model saved: dl_checkpoints/enhanced_best_model.pth")
    
    # ========================================================
    # PHASE 3: Online Learning Simulation
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 3: ONLINE LEARNING SIMULATION".center(70))
    print(f"{'='*70}")
    
    accuracies, predictions_log = run_online_simulation(
        model, criterion, enriched, binary, split_idx, device, CONFIG
    )
    
    # ========================================================
    # PHASE 4: Results & Comparison
    # ========================================================
    enhanced_results = print_statistics(accuracies, "ENHANCED LSTM - ONLINE LEARNING STATISTICS")
    
    # Save results
    results_data = {
        'model_type': 'Enhanced LSTM (BiLSTM + Attention + Focal Loss)',
        'improvements': [
            'Feature Engineering (220 features)',
            'Bidirectional LSTM',
            'Attention Mechanism',
            'Focal Loss (alpha=0.25, gamma=2.0)',
            'Label Smoothing (0.05)',
        ],
        'config': {k: v for k, v in CONFIG.items() if k != 'random_seed'},
        'parameters': n_params,
        'training_time_seconds': train_time,
        'results': enhanced_results,
        'predictions': [
            {
                'day': p['day'],
                'best_overlap': p['best_overlap'],
                'best_accuracy': p['best_accuracy'],
                'actual': p['actual'],
            }
            for p in predictions_log
        ]
    }
    
    with open(checkpoint_dir / 'enhanced_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: dl_checkpoints/enhanced_results.json")
    
    # Save predictions log
    log_df = pd.DataFrame([
        {'day': p['day'], 'best_overlap': p['best_overlap'], 
         'best_accuracy': p['best_accuracy']}
        for p in predictions_log
    ])
    log_df.to_csv(checkpoint_dir / 'enhanced_predictions_log.csv', index=False)
    print(f"Predictions log saved: dl_checkpoints/enhanced_predictions_log.csv")
    
    # Print comparison
    print_comparison(enhanced_results)
    
    print(f"\n{'='*70}")
    print("ENHANCED PIPELINE COMPLETE".center(70))
    print(f"{'='*70}")
    
    return enhanced_results


if __name__ == "__main__":
    results = main()
