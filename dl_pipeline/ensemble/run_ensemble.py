"""
Ensemble Model Pipeline

Trains 3 diverse model architectures on the same data,
then combines their predictions via weighted probability averaging.

Models in the ensemble:
  1. Enhanced BiLSTM + Attention (captures temporal patterns + feature richness)
  2. Transformer (captures global dependencies via self-attention)
  3. CNN-LSTM (captures local patterns via convolution + temporal via LSTM)

Why ensembles work:
  - Each model captures DIFFERENT patterns in the data
  - Combining reduces noise from any single model
  - High-confidence predictions from individual models still shine through
  - Errors are averaged out, reducing 0/6 misses
  - Peak predictions are preserved because when one model is confident, it dominates

Usage:
    cd dl_pipeline
    python run_ensemble.py
"""

import sys
from pathlib import Path

# Resolve paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent         # ensemble/
_DL_DIR   = _THIS_DIR.parent                        # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                         # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'ensemble'

sys.path.insert(0, str(_DL_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    'excel_path': str(_ROOT_DIR / 'Database.xlsx'),
    'sequence_length': 14,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 15,
    'train_ratio': 0.9,
    'n_simulation_days': 100,
    'prediction_interval': 2,
    'n_groups': 5,
    'k_students': 6,
    'temperature': 1.5,
    'online_iterations': 5,
    'replay_buffer_size': 10,
    'replay_min_samples': 5,
    'n_students': 55,
    'random_seed': 42,
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
    'Monte Carlo': {
        'average_accuracy': 23.48, 'best_accuracy': 50.00,
        'worst_accuracy': 0.00, 'std_deviation': 11.60,
        'total_predictions': 66,
        'distribution': {0: 6.1, 1: 53.0, 2: 34.8, 3: 6.1, 4: 0.0, 5: 0.0, 6: 0.0},
    },
}


# ============================================================
# Ensemble-aware Transformer (with enriched features)
# ============================================================
class EnhancedTransformer(nn.Module):
    """Transformer model adapted for enriched 220-dim features."""
    
    def __init__(self, n_features: int = 220, n_students: int = 55,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.n_students = n_students
        
        # Project enriched features to model dimension
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-position aggregation: use mean of all positions, not just last
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_students),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        x = self.encoder(x)
        # Use mean of all positions (more stable than just last)
        x = x.mean(dim=1)
        return self.fc(x)


# ============================================================
# Ensemble-aware CNN-LSTM (with enriched features)
# ============================================================
class EnhancedCNNLSTM(nn.Module):
    """CNN-LSTM hybrid adapted for enriched 220-dim features."""
    
    def __init__(self, n_features: int = 220, n_students: int = 55,
                 cnn_channels: int = 128, lstm_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_students = n_students
        
        # CNN for local pattern extraction
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.conv3 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(cnn_channels)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_students),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN: (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_cnn(x)
        
        # LSTM: (batch, n_features, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        
        return self.fc(x)


# ============================================================
# Dataset
# ============================================================
class EnsembleDataset(Dataset):
    """Dataset with enriched features for ensemble training."""
    
    def __init__(self, enriched: np.ndarray, binary: np.ndarray, seq_len: int = 14):
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
# Training helper
# ============================================================
def train_single_model(model, train_loader, val_loader, criterion, device,
                       lr, epochs, patience, model_name):
    """Train one model with early stopping."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"\n  Training {model_name}...")
    
    for epoch in range(epochs):
        # Train
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
        
        # Validate
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
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
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
    
    print(f"    Best val loss: {best_val_loss:.4f}, Final val acc: {val_acc:.4f}")
    return optimizer


# ============================================================
# Ensemble prediction
# ============================================================
def ensemble_predict(models, weights, X_tensor, device,
                     n_groups=5, k=6, temperature=1.5):
    """
    Combine predictions from multiple models using weighted averaging.
    
    Each model produces probability distributions over 55 students.
    The ensemble averages these (weighted) to get a combined probability,
    then samples groups from the combined distribution.
    
    This means:
    - If all models agree a student is likely -> very high combined probability
    - If only one model is confident -> still gets a moderate boost
    - Disagreements are smoothed out -> fewer 0/6 misses
    """
    combined_probs = np.zeros(models[0].n_students, dtype=np.float64)
    
    for model, weight in zip(models, weights):
        model.eval()
        with torch.no_grad():
            X = X_tensor.to(device)
            probs = model(X).cpu().numpy()[0]
        combined_probs += weight * probs
    
    combined_probs /= sum(weights)
    
    # Generate diverse groups from combined probabilities
    groups = []
    for g in range(n_groups):
        temp = temperature * (0.7 + 0.15 * g)
        adj = np.power(combined_probs, 1.0 / temp)
        adj = adj / adj.sum()
        
        selected = np.random.choice(len(adj), size=k, replace=False, p=adj)
        selected = selected[np.argsort(-combined_probs[selected])]
        groups.append((selected + 1).tolist())
    
    return groups, combined_probs


# ============================================================
# Online simulation
# ============================================================
def run_ensemble_simulation(models, model_weights, optimizers, criterion,
                            all_enriched, all_binary, split_idx, device, config):
    """Run online learning simulation with ensemble predictions."""
    
    seq_len = config['sequence_length']
    n_sim = min(config['n_simulation_days'], len(all_binary) - split_idx)
    
    print(f"\n  Simulation: {n_sim} days, predict every {config['prediction_interval']} days")
    
    accuracies = []
    predictions_log = []
    replay_buffer = []
    
    for test_idx in range(0, n_sim, config['prediction_interval']):
        actual_idx = split_idx + test_idx
        if actual_idx < seq_len:
            continue
        
        # ---- Ensemble prediction ----
        X_seq = all_enriched[actual_idx - seq_len : actual_idx]
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(0)
        
        groups, combined_probs = ensemble_predict(
            models, model_weights, X_tensor, device,
            n_groups=config['n_groups'], k=config['k_students'],
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
            'best_accuracy': best_accuracy, 'predicted_groups': groups,
            'actual': actual_ids, 'overlap_counts': overlap_counts
        })
        
        # ---- Online update for ALL models ----
        replay_buffer.append((X_seq.copy(), actual_binary.copy()))
        
        if test_idx + 1 < n_sim:
            next_idx = actual_idx + 1
            if next_idx < len(all_binary) and next_idx >= seq_len:
                X_next = all_enriched[next_idx - seq_len : next_idx]
                replay_buffer.append((X_next.copy(), all_binary[next_idx].copy()))
        
        if len(replay_buffer) >= config['replay_min_samples']:
            recent = replay_buffer[-config['replay_buffer_size']:]
            X_batch = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            y_batch = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            
            for model, opt in zip(models, optimizers):
                model.train()
                for _ in range(config['online_iterations']):
                    opt.zero_grad()
                    out = model(X_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
    
    return accuracies, predictions_log


# ============================================================
# Statistics
# ============================================================
def compute_statistics(accuracies):
    """Compute statistics dict from accuracy list."""
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
    """Print statistics block."""
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


def print_all_comparison(ensemble_stats):
    """Print comparison of ALL models tested so far."""
    all_models = {**PREVIOUS_RESULTS, 'Ensemble (3 Models)': ensemble_stats}
    
    print(f"\n{'='*80}")
    print("COMPLETE MODEL COMPARISON".center(80))
    print(f"{'='*80}")
    
    header = f"{'Metric':<24}"
    for name in all_models:
        header += f" {name:<16}"
    print(f"\n{header}")
    print("-" * 80)
    
    metrics = [
        ('Avg Accuracy (%)', 'average_accuracy'),
        ('Best Accuracy (%)', 'best_accuracy'),
        ('Worst Accuracy (%)', 'worst_accuracy'),
        ('Std Deviation', 'std_deviation'),
        ('Predictions', 'total_predictions'),
    ]
    
    for label, key in metrics:
        row = f"  {label:<22}"
        vals = [m[key] for m in all_models.values()]
        best_val = max(vals) if 'accuracy' in key.lower() and 'worst' not in key.lower() else None
        
        for name, m in all_models.items():
            v = m[key]
            marker = " *" if best_val is not None and v == best_val else ""
            row += f" {v:<14.2f}{marker}"
        print(row)
    
    print(f"\n  Accuracy Distribution (% of predictions):")
    print(f"  {'Correct':<10}", end="")
    for name in all_models:
        print(f" {name:<16}", end="")
    print()
    print("  " + "-" * 78)
    
    for c in range(7):
        row = f"  {c}/6{'':<6}"
        for name, m in all_models.items():
            pct = m['distribution'].get(c, 0)
            row += f" {pct:>6.1f}%{'':<9}"
        print(row)
    
    # Determine winner
    print(f"\n{'='*80}")
    print("WINNER".center(80))
    print(f"{'='*80}")
    
    best_name = max(all_models, key=lambda n: all_models[n]['average_accuracy'])
    best_avg = all_models[best_name]['average_accuracy']
    
    print(f"\n  Best average accuracy: {best_name} ({best_avg:.2f}%)")
    
    best_peak_name = max(all_models, key=lambda n: all_models[n]['best_accuracy'])
    best_peak = all_models[best_peak_name]['best_accuracy']
    print(f"  Best peak accuracy:   {best_peak_name} ({best_peak:.2f}%)")
    
    # Check if ensemble gets 4/6+ predictions
    ens_dist = ensemble_stats['distribution']
    higher_preds = sum(ens_dist.get(c, 0) for c in [4, 5, 6])
    if higher_preds > 0:
        print(f"\n  Ensemble 4+/6 predictions: {higher_preds:.1f}% of the time!")
    
    print(f"\n{'='*80}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ENSEMBLE MODEL PIPELINE".center(70))
    print("Enhanced BiLSTM + Transformer + CNN-LSTM".center(70))
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
    
    train_ds = EnsembleDataset(enriched[:split_idx], binary[:split_idx], CONFIG['sequence_length'])
    val_ds = EnsembleDataset(enriched[split_idx:], binary[split_idx:], CONFIG['sequence_length'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"  Train: {len(train_ds)} sequences | Val: {len(val_ds)} sequences")
    
    # ========================================================
    # PHASE 2: Train 3 Models
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 2: TRAINING 3 DIVERSE MODELS".center(70))
    print(f"{'='*70}")
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    
    # Model 1: Enhanced BiLSTM + Attention
    model1 = EnhancedLSTMPredictor(
        n_features=n_features, n_students=CONFIG['n_students'],
        hidden_size=CONFIG.get('hidden_size', 128),
        num_layers=2, dropout=0.3
    ).to(device)
    n1 = sum(p.numel() for p in model1.parameters())
    print(f"\n  Model 1: Enhanced BiLSTM + Attention ({n1:,} params)")
    opt1 = train_single_model(
        model1, train_loader, val_loader, criterion, device,
        lr=0.0005, epochs=CONFIG['epochs'], patience=CONFIG['early_stopping_patience'],
        model_name="Enhanced BiLSTM"
    )
    
    # Model 2: Transformer
    model2 = EnhancedTransformer(
        n_features=n_features, n_students=CONFIG['n_students'],
        d_model=128, nhead=4, num_layers=3,
        dim_feedforward=256, dropout=0.3
    ).to(device)
    n2 = sum(p.numel() for p in model2.parameters())
    print(f"\n  Model 2: Transformer ({n2:,} params)")
    opt2 = train_single_model(
        model2, train_loader, val_loader, criterion, device,
        lr=0.0005, epochs=CONFIG['epochs'], patience=CONFIG['early_stopping_patience'],
        model_name="Transformer"
    )
    
    # Model 3: CNN-LSTM
    model3 = EnhancedCNNLSTM(
        n_features=n_features, n_students=CONFIG['n_students'],
        cnn_channels=128, lstm_hidden=128, dropout=0.3
    ).to(device)
    n3 = sum(p.numel() for p in model3.parameters())
    print(f"\n  Model 3: CNN-LSTM ({n3:,} params)")
    opt3 = train_single_model(
        model3, train_loader, val_loader, criterion, device,
        lr=0.0005, epochs=CONFIG['epochs'], patience=CONFIG['early_stopping_patience'],
        model_name="CNN-LSTM"
    )
    
    total_params = n1 + n2 + n3
    print(f"\n  Total ensemble parameters: {total_params:,}")
    
    # ========================================================
    # PHASE 3: Online Simulation (Ensemble)
    # ========================================================
    print(f"\n{'='*70}")
    print("PHASE 3: ENSEMBLE ONLINE LEARNING SIMULATION".center(70))
    print(f"{'='*70}")
    
    models = [model1, model2, model3]
    # Equal weights - each model contributes equally
    model_weights = [1.0, 1.0, 1.0]
    optimizers = [
        optim.Adam(model1.parameters(), lr=0.0003),
        optim.Adam(model2.parameters(), lr=0.0003),
        optim.Adam(model3.parameters(), lr=0.0003),
    ]
    
    start_time = time.time()
    accuracies, predictions_log = run_ensemble_simulation(
        models, model_weights, optimizers, criterion,
        enriched, binary, split_idx, device, CONFIG
    )
    sim_time = time.time() - start_time
    
    # ========================================================
    # PHASE 4: Results
    # ========================================================
    ensemble_stats = compute_statistics(accuracies)
    print_statistics(ensemble_stats, "ENSEMBLE - ONLINE LEARNING STATISTICS")
    
    # Save results
    ckpt_dir = _CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model1_state': model1.state_dict(),
        'model2_state': model2.state_dict(),
        'model3_state': model3.state_dict(),
        'model_weights': model_weights,
        'config': CONFIG,
        'n_features': n_features,
    }, ckpt_dir / 'ensemble_models.pth')
    
    results_data = {
        'model_type': 'Ensemble (BiLSTM + Transformer + CNN-LSTM)',
        'components': ['Enhanced BiLSTM + Attention', 'Transformer (3-layer)', 'CNN-LSTM'],
        'total_parameters': total_params,
        'simulation_time_seconds': sim_time,
        'results': ensemble_stats,
    }
    with open(ckpt_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    log_df = pd.DataFrame([
        {'day': p['day'], 'best_overlap': p['best_overlap'],
         'best_accuracy': p['best_accuracy']}
        for p in predictions_log
    ])
    log_df.to_csv(ckpt_dir / 'ensemble_predictions_log.csv', index=False)
    
    print(f"\n  Model saved: dl_checkpoints/ensemble_models.pth")
    print(f"  Results saved: dl_checkpoints/ensemble_results.json")
    print(f"  Log saved: dl_checkpoints/ensemble_predictions_log.csv")
    
    # Full comparison
    print_all_comparison(ensemble_stats)
    
    print(f"\n{'='*70}")
    print("ENSEMBLE PIPELINE COMPLETE".center(70))
    print(f"{'='*70}")
    
    return ensemble_stats


if __name__ == "__main__":
    results = main()
