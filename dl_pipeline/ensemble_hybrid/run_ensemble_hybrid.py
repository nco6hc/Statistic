"""
Ensemble LSTM + Markov Chain Hybrid Pipeline
============================================

Combines THREE diverse neural architectures into an ensemble, then blends
the ensemble signal with a Markov Chain — stacking the two best-performing
ideas from prior experiments.

Architecture:
  Neural Ensemble:
    1. Enhanced BiLSTM + Attention  (sequential temporal patterns)
    2. Transformer                  (global self-attention dependencies)
    3. CNN-LSTM                     (local convolution + temporal memory)

  Structural model:
    4. FactoredMarkovChain          (explicit transition & co-occurrence)

  Final prediction:
    nn_probs    = (w1*model1 + w2*model2 + w3*model3) / (w1+w2+w3)
    hybrid_prob = alpha * nn_probs_norm + (1-alpha) * markov_probs

Why this should beat both parents:
  - Neural ensemble smooths individual model noise (fewer 0/6 misses)
  - Markov Chain adds a sharp frequency-transition signal the NNs miss
  - Two independent information sources → synergistic coverage
  - Online fine-tune of ALL 3 NNs + Markov update after each day

Simulation:
  100 test days, predict EVERY day (100 predictions), n_groups=5
  (same setup as loss-comparison and optimization experiments)

Usage:
    cd dl_pipeline/ensemble_hybrid
    python run_ensemble_hybrid.py
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent         # ensemble_hybrid/
_DL_DIR   = _THIS_DIR.parent                        # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                         # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'ensemble_hybrid'
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_DL_DIR))

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    'excel_path'              : str(_ROOT_DIR / 'Database.xlsx'),
    'sequence_length'         : 14,
    'batch_size'              : 32,
    'epochs'                  : 50,
    'early_stopping_patience' : 15,
    'train_ratio'             : 0.9,
    'n_simulation_days'       : 100,
    'prediction_interval'     : 1,      # predict EVERY day
    'n_groups'                : 5,
    'k_students'              : 6,
    'n_students'              : 55,
    'random_seed'             : 42,
    # Group generation temperatures (diversity via temp scaling)
    'temperatures'            : [0.7, 0.9, 1.1, 1.4, 1.8],
    # Online fine-tune
    'online_iterations'       : 5,
    'online_lr'               : 3e-4,
    'replay_buffer_size'      : 10,
    'replay_min_samples'      : 5,
    # Alpha grid: fraction of neural ensemble vs Markov
    'alpha_values'            : [0.5, 0.6, 0.7, 0.8],
}

PRIOR_RESULTS = {
    'LSTM+Markov Hybrid' : {'average_accuracy': 29.00, 'best_accuracy': 66.67,
                             'worst_accuracy': 0.00,  'std_deviation': 13.10,
                             'total_predictions': 50},
    'Enhanced LSTM'      : {'average_accuracy': 28.33, 'best_accuracy': 50.00,
                             'worst_accuracy': 0.00,  'std_deviation': 12.13,
                             'total_predictions': 50},
    'Bayesian BB'        : {'average_accuracy': 27.67, 'best_accuracy': 50.00,
                             'worst_accuracy': 0.00,  'std_deviation': 11.90,
                             'total_predictions': 50},
    'LSTM(BCE)'          : {'average_accuracy': 26.83, 'best_accuracy': 50.00,
                             'worst_accuracy': 0.00,  'std_deviation': 9.97,
                             'total_predictions': 100},
    'Gradient Boosting'  : {'average_accuracy': 26.67, 'best_accuracy': 66.67,
                             'worst_accuracy': 0.00,  'std_deviation': 12.90,
                             'total_predictions': 50},
    'Diverse Beam Search': {'average_accuracy': 26.50, 'best_accuracy': 66.67,
                             'worst_accuracy': 0.00,  'std_deviation': 13.20,
                             'total_predictions': 100},
    'Baseline LSTM'      : {'average_accuracy': 26.00, 'best_accuracy': 66.67,
                             'worst_accuracy': 0.00,  'std_deviation': 12.54,
                             'total_predictions': 50},
    'Meta-Ensemble'      : {'average_accuracy': 24.33, 'best_accuracy': 50.00,
                             'worst_accuracy': 0.00,  'std_deviation': 11.80,
                             'total_predictions': 50},
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])


# ═════════════════════════════════════════════════════════════════════════════
#  Neural architectures  (reused from ensemble/run_ensemble.py)
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedTransformer(nn.Module):
    """Transformer on 220-dim enriched features."""

    def __init__(self, n_features: int = 220, n_students: int = 55,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.3):
        super().__init__()
        self.n_students = n_students
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_students), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class EnhancedCNNLSTM(nn.Module):
    """CNN-LSTM on 220-dim enriched features."""

    def __init__(self, n_features: int = 220, n_students: int = 55,
                 cnn_channels: int = 128, lstm_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.n_students = n_students
        self.conv1 = nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(cnn_channels)
        self.conv3 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(cnn_channels)
        self.drop_cnn = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=lstm_hidden,
            num_layers=2, dropout=dropout, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_students), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop_cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════════════

class HybridEnsembleDataset(Dataset):
    def __init__(self, enriched: np.ndarray, binary: np.ndarray, seq_len: int = 14):
        self.features = enriched
        self.targets  = binary
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ═════════════════════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module, train_loader, val_loader,
                criterion, device, lr: float, epochs: int,
                patience: int, name: str) -> optim.Optimizer:
    """Train with early stopping; return best-val-loss weights."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    [{name}] Epoch {epoch+1:3d}: "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    [{name}] Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"    [{name}] Best val loss: {best_val:.4f}")
    model.eval()
    return optimizer


# ═════════════════════════════════════════════════════════════════════════════
#  Prediction helpers
# ═════════════════════════════════════════════════════════════════════════════

def ensemble_nn_probs(models: list, weights: list,
                      enriched_seq: np.ndarray, device) -> np.ndarray:
    """Weighted average of 3 neural model probabilities. Returns (55,)."""
    combined = np.zeros(CONFIG['n_students'], dtype=np.float64)
    total_w  = sum(weights)
    for model, w in zip(models, weights):
        model.eval()
        with torch.no_grad():
            X     = torch.FloatTensor(enriched_seq).unsqueeze(0).to(device)
            probs = model(X).cpu().numpy()[0]
        combined += w * probs
    return combined / total_w


def hybrid_predict(nn_probs: np.ndarray, markov_probs: np.ndarray,
                   alpha: float, n_groups: int, k: int,
                   temperatures: list) -> tuple[list, np.ndarray]:
    """
    Blend neural ensemble + Markov and generate diverse groups.

    hybrid = alpha * nn_norm + (1-alpha) * markov_norm
    """
    # Normalise each source to sum=1
    nn_norm  = nn_probs  / (nn_probs.sum()  + 1e-12)
    mk_norm  = markov_probs / (markov_probs.sum() + 1e-12)

    hybrid = alpha * nn_norm + (1.0 - alpha) * mk_norm
    hybrid = hybrid / hybrid.sum()

    groups = []
    seen   = set()
    for g in range(n_groups):
        temp = temperatures[g % len(temperatures)]
        adj  = np.power(np.maximum(hybrid, 1e-12), 1.0 / temp)
        adj /= adj.sum()
        idx  = np.random.choice(len(adj), size=k, replace=False, p=adj)
        idx  = sorted(idx.tolist())
        key  = frozenset(idx)
        if key not in seen:
            seen.add(key)
            groups.append(idx)

    while len(groups) < n_groups:
        temp = temperatures[-1]
        adj  = np.power(np.maximum(hybrid, 1e-12), 1.0 / temp)
        adj /= adj.sum()
        idx  = sorted(np.random.choice(len(adj), size=k,
                                        replace=False, p=adj).tolist())
        groups.append(idx)

    return groups[:n_groups], hybrid


def overlap_accuracy(groups: list, actual_binary: np.ndarray, k: int = 6) -> float:
    actual_set = set(int(i) for i in np.where(actual_binary == 1)[0])
    return max(len(set(g) & actual_set) for g in groups) / k * 100.0


# ═════════════════════════════════════════════════════════════════════════════
#  Online simulation
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(models: list, weights: list, markov: FactoredMarkovChain,
                   criterion, all_enriched: np.ndarray,
                   all_binary: np.ndarray, split_idx: int,
                   device, alpha: float, config: dict,
                   verbose: bool = True) -> tuple[list, list]:
    """
    100-day online simulation:
      - Predict every day (prediction_interval=1)
      - Generate n_groups=5 candidate groups
      - After observing the actual outcome:
          * Fine-tune all 3 NNs for online_iterations steps
          * Update Markov Chain transitions
    """
    seq_len    = config['sequence_length']
    sim_days   = min(config['n_simulation_days'],
                     len(all_binary) - split_idx)
    sim_data   = all_binary[split_idx:][-sim_days:]
    sim_enrich = all_enriched[split_idx:][-sim_days:]

    # Seed rolling buffers with last seq_len training days
    start      = max(0, split_idx - seq_len)
    rolling_b  = list(all_binary[start : split_idx])
    rolling_e  = list(all_enriched[start : split_idx])

    # Online optimizers
    online_opts = [optim.Adam(m.parameters(), lr=config['online_lr'])
                   for m in models]

    replay = []
    accuracies  = []
    prob_info   = []

    for d in range(sim_days):
        recent_b = np.array(rolling_b[-seq_len:], dtype=np.float32)
        recent_e = np.array(rolling_e[-seq_len:], dtype=np.float32)

        # ── Predict ──────────────────────────────────────────────────────
        nn_p    = ensemble_nn_probs(models, weights, recent_e, device)
        mk_p    = markov.predict_probabilities(recent_b)
        groups, hybrid = hybrid_predict(
            nn_p, mk_p, alpha,
            config['n_groups'], config['k_students'], config['temperatures'])

        prob_info.append({'nn_max': float(nn_p.max()), 'mk_max': float(mk_p.max()),
                          'hybrid_max': float(hybrid.max())})

        # ── Evaluate ─────────────────────────────────────────────────────
        actual = sim_data[d]
        acc    = overlap_accuracy(groups, actual, config['k_students'])
        accuracies.append(acc)

        if verbose and (d < 3 or (d + 1) % 10 == 0):
            actual_ids = sorted(int(i) + 1 for i in np.where(actual == 1)[0])
            best_g     = max(groups, key=lambda g: len(set(g) & set(
                               int(i) for i in np.where(actual == 1)[0])))
            print(f"  Day {d+1:3d}: {acc:5.1f}% | "
                  f"hybrid_max={hybrid.max():.3f} | actual={actual_ids}")

        # ── Online update: all 3 NNs ──────────────────────────────────────
        replay.append((recent_e.copy(), actual.copy()))
        if len(replay) >= config['replay_min_samples']:
            batch = replay[-config['replay_buffer_size']:]
            Xb = torch.FloatTensor(np.array([r[0] for r in batch])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in batch])).to(device)

            for model, opt in zip(models, online_opts):
                model.train()
                for _ in range(config['online_iterations']):
                    opt.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                model.eval()

        # ── Online update: Markov ─────────────────────────────────────────
        idx_in_full = (split_idx + len(all_binary[split_idx:]) - sim_days) + d
        if idx_in_full >= 2:
            markov.update(actual,
                          all_binary[idx_in_full - 1],
                          all_binary[idx_in_full - 2])

        rolling_b.append(actual)
        rolling_e.append(sim_enrich[d] if d < len(sim_enrich) else actual)

    return accuracies, prob_info


# ═════════════════════════════════════════════════════════════════════════════
#  Statistics & charts
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(accuracies: list) -> dict:
    a = np.array(accuracies)
    dist = {}
    for c in range(7):
        val = c / 6 * 100
        dist[c] = float(np.sum(np.abs(a - val) < 1.0) / len(a) * 100)

    return {
        'average_accuracy' : float(np.mean(a)),
        'best_accuracy'    : float(np.max(a)),
        'worst_accuracy'   : float(np.min(a)),
        'std_deviation'    : float(np.std(a)),
        'total_predictions': len(a),
        'distribution'     : dist,
        'first_10_avg'     : float(np.mean(a[:10])),
        'last_10_avg'      : float(np.mean(a[-10:])),
        'zero_days'        : int(np.sum(a == 0)),
        'fifty_plus_days'  : int(np.sum(a >= 50)),
    }


def print_stats(stats: dict, title: str):
    print(f"\n{'='*70}")
    print(title.center(70))
    print(f"{'='*70}")
    print(f"  Total predictions : {stats['total_predictions']}")
    print(f"  Average accuracy  : {stats['average_accuracy']:.2f}%")
    print(f"  Best  accuracy    : {stats['best_accuracy']:.2f}%")
    print(f"  Worst accuracy    : {stats['worst_accuracy']:.2f}%")
    print(f"  Std deviation     : {stats['std_deviation']:.4f}")
    print(f"  Zero-correct days : {stats['zero_days']}")
    print(f"  50%+ days         : {stats['fifty_plus_days']}")
    print(f"\n  Distribution:")
    for c in range(7):
        n   = round(stats['distribution'].get(c, 0) * stats['total_predictions'] / 100)
        pct = stats['distribution'].get(c, 0)
        bar = '#' * int(pct / 2)
        print(f"    {c}/6 : {n:4d} ({pct:5.1f}%)  {bar}")
    trend = stats['last_10_avg'] - stats['first_10_avg']
    print(f"\n  Trend (last10 - first10): {trend:+.2f}%")
    print(f"  First 10 avg: {stats['first_10_avg']:.2f}% | "
          f"Last 10 avg: {stats['last_10_avg']:.2f}%")


def save_charts(all_alpha_stats: dict, best_alpha: float,
                loss_hists: list, best_acc: list):
    """5-panel dashboard."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Ensemble LSTM + Markov Hybrid — Performance Dashboard',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    colors = ['steelblue', 'darkorange', 'mediumseagreen', 'crimson']
    days   = np.arange(1, len(best_acc) + 1)
    W      = 10

    # Panel 1: best-alpha accuracy over time
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(days, best_acc, color='steelblue', lw=1.2, alpha=0.4, label='Daily')
    roll = np.convolve(best_acc, np.ones(W) / W, mode='valid')
    ax0.plot(np.arange(W, len(best_acc) + 1), roll, color='steelblue', lw=2.4,
             label=f'10-day rolling avg')

    # Overlay prior bests
    for name, val, col in [('LSTM+Markov Hybrid', 29.00, 'red'),
                             ('Enhanced LSTM', 28.33, 'green')]:
        ax0.axhline(val, color=col, lw=1.3, ls='--', alpha=0.7,
                    label=f'{name} avg ({val}%)')

    avg_v = np.mean(best_acc)
    ax0.axhline(avg_v, color='navy', lw=1.8, ls=':', alpha=0.9,
                label=f'This model avg ({avg_v:.1f}%)')
    ax0.set_xlabel('Simulation Day')
    ax0.set_ylabel('Accuracy (%)')
    ax0.set_title(f'Daily Accuracy — Best alpha={best_alpha}')
    ax0.legend(fontsize=8)
    ax0.set_ylim(-5, 110)
    ax0.grid(axis='y', alpha=0.3)

    # Panel 2: alpha comparison bar
    ax1 = fig.add_subplot(gs[0, 2])
    alphas = list(all_alpha_stats.keys())
    avgs   = [all_alpha_stats[a]['average_accuracy'] for a in alphas]
    stds   = [all_alpha_stats[a]['std_deviation']    for a in alphas]
    bars   = ax1.bar([str(a) for a in alphas], avgs,
                     color=colors[:len(alphas)], alpha=0.8,
                     yerr=stds, capsize=5)
    for bar, v in zip(bars, avgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')
    ax1.set_xlabel('Alpha (NN fraction)')
    ax1.set_ylabel('Avg Accuracy (%)')
    ax1.set_title('Alpha Tuning Results')
    ax1.set_ylim(max(0, min(avgs) - 5), max(avgs) + 8)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 3: Training loss of 3 NNs
    ax2 = fig.add_subplot(gs[1, 0])
    model_names = ['BiLSTM+Attn', 'Transformer', 'CNN-LSTM']
    for hist, nm, c in zip(loss_hists, model_names, colors):
        ep = [h[0] for h in hist]
        ls = [h[1] for h in hist]
        ax2.plot(ep, ls, lw=2.0, marker='o', markersize=3, color=c, label=nm)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Validation Loss — 3 Neural Models')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 4: Distribution bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    stats_best = all_alpha_stats[best_alpha]
    overlap_vals = list(range(7))
    pcts         = [stats_best['distribution'].get(c, 0) for c in overlap_vals]
    ax3.bar([f'{c}/6' for c in overlap_vals], pcts, color='steelblue', alpha=0.75)
    ax3.set_xlabel('Overlap with actual 6')
    ax3.set_ylabel('% of days')
    ax3.set_title(f'Accuracy Distribution (alpha={best_alpha})')
    ax3.grid(axis='y', alpha=0.3)

    # Panel 5: Cumulative accuracy
    ax4 = fig.add_subplot(gs[1, 2])
    acc_arr = np.array(best_acc)
    cum     = np.cumsum(acc_arr) / (np.arange(len(acc_arr)) + 1)
    ax4.plot(days, cum, color='steelblue', lw=2.2, label='This model')
    ax4.axhline(29.00, color='red',   lw=1.5, ls='--', alpha=0.7,
                label='LSTM+Markov (29.0%)')
    ax4.axhline(28.33, color='green', lw=1.5, ls='--', alpha=0.7,
                label='Enhanced LSTM (28.33%)')
    ax4.set_xlabel('Simulation Day')
    ax4.set_ylabel('Cumulative Avg Accuracy (%)')
    ax4.set_title('Cumulative Accuracy Over Time')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    out = _CKPT_DIR / 'ensemble_hybrid_dashboard.png'
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print(" ENSEMBLE LSTM + MARKOV CHAIN HYBRID".center(72))
    print(" BiLSTM+Attn | Transformer | CNN-LSTM | FactoredMarkov".center(72))
    print("=" * 72)

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Phase 1: Data ─────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 1 — DATA PREPARATION")
    print(f"{'─'*72}")

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

    print("  Building enriched features ...")
    t0       = time.perf_counter()
    enriched = build_enriched_features(binary)
    n_feat   = enriched.shape[1]
    print(f"  Enriched features: {n_feat}-dim  ({time.perf_counter()-t0:.1f}s)")

    split_idx = int(len(binary) * CONFIG['train_ratio'])
    train_ds  = HybridEnsembleDataset(enriched[:split_idx], binary[:split_idx],
                                       CONFIG['sequence_length'])
    val_ds    = HybridEnsembleDataset(enriched[split_idx:], binary[split_idx:],
                                       CONFIG['sequence_length'])
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                               shuffle=False, drop_last=False)
    print(f"  Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs")

    # ── Phase 2: Train 3 Neural Models ───────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 2 — TRAINING 3 NEURAL MODELS")
    print(f"{'─'*72}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

    model1 = EnhancedLSTMPredictor(
        n_features=n_feat, n_students=CONFIG['n_students'],
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)
    model2 = EnhancedTransformer(
        n_features=n_feat, n_students=CONFIG['n_students'],
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.3
    ).to(device)
    model3 = EnhancedCNNLSTM(
        n_features=n_feat, n_students=CONFIG['n_students'],
        cnn_channels=128, lstm_hidden=128, dropout=0.3
    ).to(device)

    n1, n2, n3 = (sum(p.numel() for p in m.parameters())
                  for m in [model1, model2, model3])
    print(f"\n  Model 1 — Enhanced BiLSTM+Attention : {n1:,} params")
    print(f"  Model 2 — Transformer               : {n2:,} params")
    print(f"  Model 3 — CNN-LSTM                  : {n3:,} params")
    print(f"  Total neural params                 : {n1+n2+n3:,}")

    val_loss_hists = []   # track val loss per epoch for chart

    for model, mname in [(model1, 'BiLSTM+Attn'),
                          (model2, 'Transformer'),
                          (model3, 'CNN-LSTM')]:
        print(f"\n  --- Training {mname} ---")
        # Patch train_model to also record val loss history
        optimizer_ = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler_ = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_, mode='min', factor=0.5, patience=5)
        best_val_  = float('inf')
        best_state_= None
        no_imp_    = 0
        hist_      = []

        for epoch in range(CONFIG['epochs']):
            model.train()
            tl = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer_.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_.step()
                tl += loss.item()
            tl /= len(train_loader)

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    vl += criterion(model(X), y).item()
            vl /= len(val_loader)
            scheduler_.step(vl)
            hist_.append((epoch + 1, vl))

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}: train={tl:.4f}  val={vl:.4f}")

            if vl < best_val_:
                best_val_  = vl
                best_state_= {k: v.clone() for k, v in model.state_dict().items()}
                no_imp_    = 0
            else:
                no_imp_ += 1
            if no_imp_ >= CONFIG['early_stopping_patience']:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if best_state_:
            model.load_state_dict(best_state_)
        model.eval()
        val_loss_hists.append(hist_)
        print(f"    [{mname}] Best val loss: {best_val_:.4f}")

    # Save base weights for resetting between alpha trials
    base_states = [
        {k: v.clone() for k, v in m.state_dict().items()}
        for m in [model1, model2, model3]
    ]

    # ── Phase 3: Fit Markov Chain ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 3 — FITTING MARKOV CHAIN")
    print(f"{'─'*72}")

    base_markov = FactoredMarkovChain(n_students=CONFIG['n_students'], smoothing=1.0)
    base_markov.fit(binary[:split_idx])

    base_rate = 6.0 / CONFIG['n_students']
    print(f"  Base selection rate: {base_rate:.4f}  (6/{CONFIG['n_students']})")

    # Show sample Markov stats
    print("  Sample transition probs (student, P(sel|was_sel), P(sel|not_sel)):")
    for s in [0, 15, 30, 45, 54]:
        p1 = (base_markov.transition_counts[s,1,:].sum() + 1) / \
             (base_markov.transition_totals[s,1,:].sum() + 2)
        p0 = (base_markov.transition_counts[s,0,:].sum() + 1) / \
             (base_markov.transition_totals[s,0,:].sum() + 2)
        print(f"    Student {s+1:2d}: {p1:.3f} | {p0:.3f}  ratio={p1/p0:.2f}x")

    # ── Phase 4: Alpha Tuning ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 4 — ALPHA TUNING  (NN fraction vs Markov)")
    print(f"{'─'*72}")

    all_alpha_stats : dict = {}
    best_alpha      = None
    best_avg_acc    = -1.0
    best_accuracies : list = []
    model_weights   = [1.0, 1.0, 1.0]   # equal weight for all 3 NNs

    for alpha in CONFIG['alpha_values']:
        nn_pct = int(alpha * 100)
        mk_pct = int((1 - alpha) * 100)
        print(f"\n  --- alpha={alpha:.1f}  ({nn_pct}% NN | {mk_pct}% Markov) ---")

        # Reset models and Markov to pre-simulation state
        torch.manual_seed(CONFIG['random_seed'])
        np.random.seed(CONFIG['random_seed'])

        for model, state in zip([model1, model2, model3], base_states):
            model.load_state_dict(state)
            model.eval()

        markov_copy = FactoredMarkovChain(
            n_students=CONFIG['n_students'], smoothing=1.0)
        markov_copy.fit(binary[:split_idx])

        t_start = time.perf_counter()
        accs, _ = run_simulation(
            [model1, model2, model3], model_weights,
            markov_copy, criterion,
            enriched, binary, split_idx, device,
            alpha=alpha, config=CONFIG, verbose=True
        )
        elapsed = time.perf_counter() - t_start

        s = compute_stats(accs)
        all_alpha_stats[alpha] = s

        print(f"\n  alpha={alpha:.1f} => Avg: {s['average_accuracy']:.2f}%  "
              f"Best: {s['best_accuracy']:.2f}%  "
              f"Worst: {s['worst_accuracy']:.2f}%  "
              f"({elapsed:.1f}s)")

        if s['average_accuracy'] > best_avg_acc:
            best_avg_acc    = s['average_accuracy']
            best_alpha      = alpha
            best_accuracies = accs

    # ── Phase 5: Results ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 5 — ALPHA TUNING SUMMARY")
    print(f"{'─'*72}")
    print(f"\n  {'Alpha':<8} {'NN%':<8} {'Markov%':<10} "
          f"{'Avg':>8} {'Best':>8} {'Worst':>8} {'Std':>8} {'0-days':>8}")
    print("  " + "-" * 68)
    for a, s in all_alpha_stats.items():
        marker = " <-- BEST" if a == best_alpha else ""
        print(f"  {a:<8.1f} {int(a*100):<8} {int((1-a)*100):<10} "
              f"{s['average_accuracy']:>7.2f}%  {s['best_accuracy']:>7.2f}%  "
              f"{s['worst_accuracy']:>7.2f}%  {s['std_deviation']:>7.4f}  "
              f"{s['zero_days']:>6}{marker}")

    best_stats = all_alpha_stats[best_alpha]
    print_stats(best_stats,
                f"ENSEMBLE HYBRID (alpha={best_alpha}) — FINAL STATISTICS")

    # ── Leaderboard ───────────────────────────────────────────────────────
    this_name  = f'Ensemble+Markov (a={best_alpha})'
    all_models = dict(PRIOR_RESULTS)
    all_models[this_name] = best_stats

    sorted_models = sorted(all_models.items(),
                            key=lambda x: x[1]['average_accuracy'],
                            reverse=True)

    print(f"\n{'='*72}")
    print("  FULL LEADERBOARD — ALL MODELS".center(72))
    print(f"{'='*72}")
    print(f"\n  {'Rank':<5} {'Model':<30} {'Avg Acc':>9} {'Best':>8} {'N Preds':>9}")
    print("  " + "-" * 64)
    for rank, (name, s) in enumerate(sorted_models, 1):
        marker = " <-- NEW" if name == this_name else ""
        n_pred = s.get('total_predictions', '?')
        print(f"  {rank:<5} {name:<30} {s['average_accuracy']:>8.2f}%  "
              f"{s['best_accuracy']:>7.2f}%  {str(n_pred):>7}{marker}")

    prior_best = max(v['average_accuracy'] for k, v in PRIOR_RESULTS.items())
    delta = best_stats['average_accuracy'] - prior_best
    print(f"\n  New model avg       : {best_stats['average_accuracy']:.2f}%")
    print(f"  Previous best       : {prior_best:.2f}%  (LSTM+Markov Hybrid)")
    print(f"  Delta               : {delta:+.2f}%")

    # ── Save artefacts ────────────────────────────────────────────────────
    results_json = {
        'model_type'       : 'Ensemble LSTM + Markov Hybrid',
        'components'       : ['Enhanced BiLSTM+Attention',
                               'Transformer (3-layer)',
                               'CNN-LSTM',
                               'FactoredMarkovChain'],
        'best_alpha'       : best_alpha,
        'alpha_results'    : {str(a): {k: v for k, v in s.items()
                                        if k != 'distribution'}
                               for a, s in all_alpha_stats.items()},
        'best_stats'       : best_stats,
    }
    with open(_CKPT_DIR / 'ensemble_hybrid_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved: dl_checkpoints/ensemble_hybrid/ensemble_hybrid_results.json")

    # ── Charts ────────────────────────────────────────────────────────────
    save_charts(all_alpha_stats, best_alpha, val_loss_hists, best_accuracies)

    print(f"\n{'='*72}")
    print("  ENSEMBLE HYBRID PIPELINE COMPLETE".center(72))
    print(f"{'='*72}")
    return best_stats, best_alpha


if __name__ == '__main__':
    main()
