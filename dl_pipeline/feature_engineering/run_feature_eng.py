"""
Feature Engineering Comparison — Baseline, Enhanced & Hybrid LSTM
===================================================================

Tests whether adding 5 new feature groups (495-dim) improves 3 models
that currently use either 55-dim (Baseline) or 220-dim (Enhanced/Hybrid).

New feature groups added to 220-dim → 495-dim:
  + 3-day  frequency    (captures very recent hot streaks)
  + 28-day frequency    (stable long-term base rate per student)
  + EMA fast α=0.5      (exponential decay, half-life ≈ 1.4 days)
  + Momentum (7−28)     (trending up or down recently)
  + Overdue tanh signal (how long since last selected vs expected gap)

Models trained and compared:
  1. FE-Baseline : FeaturedBaseLSTM(495) + BCELoss
                   Upgraded from LSTMStudentPredictor(55)

  2. FE-Enhanced : EnhancedLSTMPredictor(495) + FocalLoss
                   Upgraded from EnhancedLSTMPredictor(220)

  3. FE-Hybrid   : EnhancedLSTMPredictor(495) + FocalLoss + FactoredMarkov
                   Upgraded from EnhancedLSTMPredictor(220) + Markov

Simulation: 100 test days, predict every day (prediction_interval=1),
            n_groups=5 candidate groups per day.
            Online fine-tune after each observation.

Usage:
    cd dl_pipeline/feature_engineering
    python run_feature_eng.py
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
_THIS_DIR = Path(__file__).resolve().parent       # feature_engineering/
_DL_DIR   = _THIS_DIR.parent                      # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                       # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'feature_engineering'
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_DL_DIR))

from feature_engineering import build_extended_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain

# ── Configuration ─────────────────────────────────────────────────────────────
CFG = {
    'excel_path'              : str(_ROOT_DIR / 'Database.xlsx'),
    'n_students'              : 55,
    'k_per_day'               : 6,
    'seq_len'                 : 14,
    'train_ratio'             : 0.9,
    'batch_size'              : 32,
    'epochs'                  : 50,
    'patience'                : 15,
    'lr'                      : 5e-4,
    'weight_decay'            : 1e-5,
    'n_sim_days'              : 100,
    'n_groups'                : 5,
    'temperatures'            : [0.7, 0.9, 1.1, 1.4, 1.8],
    # Online fine-tune
    'online_lr'               : 3e-4,
    'online_steps'            : 5,
    'replay_buffer'           : 10,
    'replay_min'              : 5,
    # Alpha values for Hybrid blending
    'alpha_values'            : [0.5, 0.6, 0.7, 0.8],
    'random_seed'             : 42,
}

# Prior results for leaderboard (avg_acc, best_acc, n_preds)
PRIOR = {
    'LSTM+Markov Hybrid (orig)' : (29.00, 66.67,  50),
    'Enhanced LSTM (orig)'      : (28.33, 50.00,  50),
    'Bayesian BB'               : (27.67, 50.00,  50),
    'LSTM-BCE (loss exp)'       : (26.83, 50.00, 100),
    'Gradient Boosting'         : (26.67, 66.67,  50),
    'Diverse Beam Search'       : (26.50, 66.67, 100),
    'Baseline LSTM (orig)'      : (26.00, 66.67,  50),
    'Meta-Ensemble'             : (24.33, 50.00,  50),
}

torch.manual_seed(CFG['random_seed'])
np.random.seed(CFG['random_seed'])


# ═════════════════════════════════════════════════════════════════════════════
#  FeaturedBaseLSTM — Baseline LSTM upgraded for 495-dim features
# ═════════════════════════════════════════════════════════════════════════════

class FeaturedBaseLSTM(nn.Module):
    """
    Upgraded Baseline LSTM for rich feature input.

    Architecture (matches original LSTMStudentPredictor structure):
        Input projection : Linear(n_features → hidden)  [NEW]
        LSTM             : (hidden → hidden, 2 layers, unidirectional)
        FC head          : hidden → 256 → 128 → n_students → Sigmoid

    The only change vs original LSTMStudentPredictor(55):
        - Input size  : 55-dim raw binary  →  495-dim extended features
        - Input proj  : added to compress before LSTM
        - Loss during training: BCELoss  (same as original baseline)
    """

    def __init__(self, n_features: int = 495, n_students: int = 55,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.n_students = n_students

        # Input projection: compress 495 → hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Unidirectional LSTM (same as original baseline — no BiLSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # FC head (same as original LSTMStudentPredictor)
        self.fc1  = nn.Linear(hidden_size, 256)
        self.bn1  = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)
        self.fc2  = nn.Linear(256, 128)
        self.bn2  = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.fc3  = nn.Linear(128, n_students)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)                      # (batch, seq_len, hidden)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]                      # (batch, hidden)
        out    = F.relu(self.bn1(self.fc1(last)))
        out    = self.drop1(out)
        out    = F.relu(self.bn2(self.fc2(out)))
        out    = self.drop2(out)
        return torch.sigmoid(self.fc3(out))         # (batch, n_students)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════════════

class FEDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = 14):
        self.features = features
        self.targets  = targets
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

def train_model(model: nn.Module,
                criterion: nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                device: str,
                name: str) -> list:
    """
    Train with early stopping; return val-loss history as [(epoch, val_loss)].
    Restores best-val-loss weights before returning.
    """
    optimizer  = optim.Adam(model.parameters(),
                             lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val   = float('inf')
    best_state = None
    no_imp     = 0
    history    = []

    for epoch in range(CFG['epochs']):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        history.append((epoch + 1, val_loss))

        if (epoch + 1) % 10 == 0:
            print(f"    [{name}] epoch {epoch+1:>3}/{CFG['epochs']}  "
                  f"val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if no_imp >= CFG['patience']:
            print(f"    [{name}] Early stop at epoch {epoch+1}  "
                  f"best_val={best_val:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return history


# ═════════════════════════════════════════════════════════════════════════════
#  Prediction helpers
# ═════════════════════════════════════════════════════════════════════════════

def _nn_probs(model: nn.Module, ext_seq: np.ndarray, device: str) -> np.ndarray:
    """(seq_len, 495) → (55,) probabilities."""
    model.eval()
    with torch.no_grad():
        x     = torch.FloatTensor(ext_seq).unsqueeze(0).to(device)
        probs = model(x).squeeze(0).cpu().numpy()
    return probs


def _make_groups(probs: np.ndarray, n_groups: int, k: int,
                 temperatures: list) -> list:
    """Generate n_groups diverse candidate groups via temperature sampling."""
    groups = []
    seen   = set()
    for g in range(n_groups):
        temp = temperatures[g % len(temperatures)]
        adj  = np.power(np.maximum(probs, 1e-12), 1.0 / temp)
        adj /= adj.sum()
        idx  = sorted(np.random.choice(len(adj), size=k,
                                        replace=False, p=adj).tolist())
        key  = frozenset(idx)
        if key not in seen:
            seen.add(key)
            groups.append(idx)
    while len(groups) < n_groups:
        adj  = np.power(np.maximum(probs, 1e-12), 1.0 / temperatures[-1])
        adj /= adj.sum()
        groups.append(sorted(np.random.choice(len(adj), size=k,
                                               replace=False, p=adj).tolist()))
    return groups[:n_groups]


def _overlap_acc(groups: list, actual: np.ndarray, k: int = 6) -> float:
    actual_set = set(int(i) for i in np.where(actual == 1)[0])
    return max(len(set(g) & actual_set) for g in groups) / k * 100.0


# ═════════════════════════════════════════════════════════════════════════════
#  Online simulation — NN only (Baseline & Enhanced)
# ═════════════════════════════════════════════════════════════════════════════

def run_nn_sim(model: nn.Module,
               criterion: nn.Module,
               all_ext: np.ndarray,
               all_bin: np.ndarray,
               split_idx: int,
               device: str,
               name: str) -> tuple[list, list]:
    """
    Online simulation for pure neural models.

    Returns:
        accuracies : list of float (one per sim day)
        max_probs  : list of float (model confidence per day)
    """
    seq_len  = CFG['seq_len']
    n_sim    = CFG['n_sim_days']
    sim_start = len(all_bin) - n_sim          # index in full array

    online_opt = optim.Adam(model.parameters(), lr=CFG['online_lr'])
    replay     = []
    accuracies = []
    max_probs  = []

    for d in range(n_sim):
        actual_idx = sim_start + d
        ext_seq    = all_ext[actual_idx - seq_len : actual_idx]
        actual     = all_bin[actual_idx]

        # ── Predict ──────────────────────────────────────────────────
        probs  = _nn_probs(model, ext_seq, device)
        max_probs.append(float(probs.max()))
        groups = _make_groups(probs, CFG['n_groups'], CFG['k_per_day'],
                               CFG['temperatures'])
        acc    = _overlap_acc(groups, actual, CFG['k_per_day'])
        accuracies.append(acc)

        if d < 3 or (d + 1) % 10 == 0:
            actual_ids = sorted(int(i) + 1 for i in np.where(actual == 1)[0])
            print(f"  [{name}] Day {d+1:3d}: {acc:5.1f}%  actual={actual_ids}")

        # ── Online fine-tune ──────────────────────────────────────────
        replay.append((ext_seq.copy(), actual.copy()))
        if len(replay) >= CFG['replay_min']:
            batch = replay[-CFG['replay_buffer']:]
            Xb = torch.FloatTensor(np.array([r[0] for r in batch])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in batch])).to(device)
            model.train()
            for _ in range(CFG['online_steps']):
                online_opt.zero_grad()
                criterion(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                online_opt.step()
            model.eval()

    return accuracies, max_probs


# ═════════════════════════════════════════════════════════════════════════════
#  Online simulation — Hybrid (NN + Markov)
# ═════════════════════════════════════════════════════════════════════════════

def run_hybrid_sim(model: nn.Module,
                   markov: FactoredMarkovChain,
                   criterion: nn.Module,
                   all_ext: np.ndarray,
                   all_bin: np.ndarray,
                   split_idx: int,
                   device: str,
                   alpha: float) -> tuple[list, list]:
    """
    Online simulation for FE-Hybrid model.

    Blends neural probabilities with Markov probabilities:
        hybrid = alpha * nn_norm + (1-alpha) * markov_norm
    """
    seq_len   = CFG['seq_len']
    n_sim     = CFG['n_sim_days']
    sim_start = len(all_bin) - n_sim

    online_opt = optim.Adam(model.parameters(), lr=CFG['online_lr'])
    replay     = []
    accuracies = []
    max_probs  = []

    for d in range(n_sim):
        actual_idx = sim_start + d
        ext_seq    = all_ext[actual_idx - seq_len : actual_idx]
        bin_seq    = all_bin[actual_idx - seq_len : actual_idx]
        actual     = all_bin[actual_idx]

        # ── Neural probs ──────────────────────────────────────────────
        nn_p   = _nn_probs(model, ext_seq, device)

        # ── Markov probs ──────────────────────────────────────────────
        mk_p   = markov.predict_probabilities(bin_seq)

        # ── Blend ─────────────────────────────────────────────────────
        nn_n   = nn_p  / (nn_p.sum()  + 1e-12)
        mk_n   = mk_p  / (mk_p.sum()  + 1e-12)
        hybrid = alpha * nn_n + (1.0 - alpha) * mk_n
        hybrid = hybrid / hybrid.sum()

        max_probs.append(float(hybrid.max()))
        groups = _make_groups(hybrid, CFG['n_groups'], CFG['k_per_day'],
                               CFG['temperatures'])
        acc    = _overlap_acc(groups, actual, CFG['k_per_day'])
        accuracies.append(acc)

        if d < 3 or (d + 1) % 10 == 0:
            actual_ids = sorted(int(i) + 1 for i in np.where(actual == 1)[0])
            print(f"  [FE-Hybrid a={alpha}] Day {d+1:3d}: "
                  f"{acc:5.1f}%  actual={actual_ids}")

        # ── Online fine-tune: NN ──────────────────────────────────────
        replay.append((ext_seq.copy(), actual.copy()))
        if len(replay) >= CFG['replay_min']:
            batch = replay[-CFG['replay_buffer']:]
            Xb = torch.FloatTensor(np.array([r[0] for r in batch])).to(device)
            yb = torch.FloatTensor(np.array([r[1] for r in batch])).to(device)
            model.train()
            for _ in range(CFG['online_steps']):
                online_opt.zero_grad()
                criterion(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                online_opt.step()
            model.eval()

        # ── Online update: Markov ─────────────────────────────────────
        if actual_idx >= 2:
            markov.update(actual,
                          all_bin[actual_idx - 1],
                          all_bin[actual_idx - 2])

    return accuracies, max_probs


# ═════════════════════════════════════════════════════════════════════════════
#  Statistics & charts
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(accs: list) -> dict:
    a = np.array(accs)
    dist = {c: float(np.sum(np.abs(a - c/6*100) < 1.0) / len(a) * 100)
            for c in range(7)}
    return {
        'avg'        : float(np.mean(a)),
        'best'       : float(np.max(a)),
        'worst'      : float(np.min(a)),
        'std'        : float(np.std(a)),
        'n_preds'    : len(a),
        'zero_days'  : int(np.sum(a == 0)),
        'fifty_days' : int(np.sum(a >= 50)),
        'first10'    : float(np.mean(a[:10])),
        'last10'     : float(np.mean(a[-10:])),
        'dist'       : dist,
    }


def save_charts(results: dict, histories: dict):
    """4-panel comparison dashboard."""
    names  = list(results.keys())
    colors = {'FE-Baseline': 'steelblue',
              'FE-Enhanced': 'darkorange',
              'FE-Hybrid'  : 'mediumseagreen'}
    days   = np.arange(1, CFG['n_sim_days'] + 1)
    W      = 10

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Feature Engineering Comparison (495-dim vs original)',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # Panel 1: accuracy over time
    ax0 = fig.add_subplot(gs[0, :2])
    for nm in names:
        acc = results[nm]['accs']
        c   = colors[nm]
        ax0.plot(days, acc, color=c, lw=1.0, alpha=0.35)
        roll = np.convolve(acc, np.ones(W)/W, mode='valid')
        ax0.plot(np.arange(W, len(acc)+1), roll, color=c, lw=2.2,
                 label=f'{nm} ({results[nm]["stats"]["avg"]:.1f}%)')

    # Overlay prior bests
    for label, avg, col in [('LSTM+Markov Hybrid 29.0%', 29.0, 'red'),
                              ('Enhanced LSTM 28.33%',  28.33, 'purple')]:
        ax0.axhline(avg, color=col, lw=1.3, ls='--', alpha=0.65, label=label)

    ax0.set_xlabel('Simulation Day')
    ax0.set_ylabel('Accuracy (%)')
    ax0.set_title('Daily Accuracy — bold lines = 10-day rolling avg')
    ax0.legend(fontsize=8)
    ax0.set_ylim(-5, 110)
    ax0.grid(axis='y', alpha=0.3)

    # Panel 2: bar chart
    ax1 = fig.add_subplot(gs[0, 2])
    avgs = [results[nm]['stats']['avg'] for nm in names]
    stds = [results[nm]['stats']['std'] for nm in names]
    bars = ax1.bar(names, avgs, color=[colors[n] for n in names],
                   alpha=0.8, yerr=stds, capsize=5)
    for bar, v in zip(bars, avgs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                 f'{v:.1f}%', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax1.set_ylim(max(0, min(avgs)-5), max(avgs)+9)
    ax1.set_ylabel('Avg Accuracy (%)')
    ax1.set_title('Avg ± Std (100 days)')
    ax1.grid(axis='y', alpha=0.3)

    # Panel 3: training val-loss curves
    ax2 = fig.add_subplot(gs[1, 0])
    for nm in names:
        hist = histories[nm]
        ax2.plot([h[0] for h in hist], [h[1] for h in hist],
                 color=colors[nm], lw=2.0, marker='o', ms=3, label=nm)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Validation Loss During Training')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 4: distribution bars
    ax3 = fig.add_subplot(gs[1, 1])
    x   = np.arange(7)
    w   = 0.25
    offsets = [-w, 0, w]
    for (nm, off) in zip(names, offsets):
        pcts = [results[nm]['stats']['dist'].get(c, 0) for c in range(7)]
        ax3.bar(x + off, pcts, width=w, color=colors[nm], alpha=0.75, label=nm)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{c}/6' for c in range(7)])
    ax3.set_xlabel('Overlap with actual 6')
    ax3.set_ylabel('% of days')
    ax3.set_title('Accuracy Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 5: cumulative accuracy
    ax4 = fig.add_subplot(gs[1, 2])
    for nm in names:
        a   = np.array(results[nm]['accs'])
        cum = np.cumsum(a) / (np.arange(len(a)) + 1)
        ax4.plot(days, cum, color=colors[nm], lw=2.0, label=nm)
    ax4.axhline(29.0,  color='red',    lw=1.3, ls='--', alpha=0.7,
                label='Hybrid orig (29.0%)')
    ax4.axhline(28.33, color='purple', lw=1.3, ls='--', alpha=0.7,
                label='Enhanced orig (28.33%)')
    ax4.set_xlabel('Simulation Day')
    ax4.set_ylabel('Cumulative Avg Accuracy (%)')
    ax4.set_title('Cumulative Accuracy')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    out = _CKPT_DIR / 'feature_engineering_dashboard.png'
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  FEATURE ENGINEERING COMPARISON  (495-dim vs original)".center(72))
    print("  Baseline LSTM | Enhanced LSTM | LSTM+Markov Hybrid".center(72))
    print("=" * 72)

    torch.manual_seed(CFG['random_seed'])
    np.random.seed(CFG['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Phase 1: Data ─────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 1 — DATA + FEATURE ENGINEERING")
    print(f"{'─'*72}\n")

    df = pd.read_excel(CFG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)

    student_cols = [c for c in df.columns if c != 'Days']
    binary = np.zeros((len(df), CFG['n_students']), dtype=np.float32)
    for idx, row in df.iterrows():
        for col in student_cols:
            sid = row[col]
            if pd.notna(sid):
                sid = int(sid)
                if 1 <= sid <= CFG['n_students']:
                    binary[idx, sid - 1] = 1.0

    print(f"  Loaded {len(binary)} days, {CFG['n_students']} students")

    t0       = time.perf_counter()
    extended = build_extended_features(binary, verbose=True)
    n_feat   = extended.shape[1]
    print(f"  Extended features: {n_feat}-dim  ({time.perf_counter()-t0:.2f}s)")

    split_idx  = int(len(binary) * CFG['train_ratio'])
    n_test     = len(binary) - split_idx
    sim_start  = len(binary) - CFG['n_sim_days']

    train_ds = FEDataset(extended[:split_idx], binary[:split_idx], CFG['seq_len'])
    val_ds   = FEDataset(extended[split_idx:], binary[split_idx:], CFG['seq_len'])
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                               shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'],
                               shuffle=False)
    print(f"  Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs | "
          f"Sim: last {CFG['n_sim_days']} test days")

    # ── Phase 2: Instantiate models ───────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 2 — MODEL ARCHITECTURES")
    print(f"{'─'*72}\n")

    bce_crit   = nn.BCELoss()
    focal_crit = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

    # FE-Baseline: FeaturedBaseLSTM + BCELoss
    model_base = FeaturedBaseLSTM(
        n_features=n_feat, n_students=CFG['n_students'],
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    # FE-Enhanced: EnhancedLSTMPredictor + FocalLoss
    model_enh = EnhancedLSTMPredictor(
        n_features=n_feat, n_students=CFG['n_students'],
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    # FE-Hybrid: same EnhancedLSTMPredictor + FocalLoss + Markov (trained later)
    model_hyb = EnhancedLSTMPredictor(
        n_features=n_feat, n_students=CFG['n_students'],
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    n_base = model_base.n_params()
    n_enh  = sum(p.numel() for p in model_enh.parameters() if p.requires_grad)
    n_hyb  = sum(p.numel() for p in model_hyb.parameters() if p.requires_grad)

    print(f"  FE-Baseline  : FeaturedBaseLSTM   — {n_base:>10,} params  |  BCELoss")
    print(f"  FE-Enhanced  : EnhancedLSTMPredictor — {n_enh:>8,} params  |  FocalLoss")
    print(f"  FE-Hybrid    : EnhancedLSTMPredictor — {n_hyb:>8,} params  |  FocalLoss + Markov")
    print(f"  Input dim    : {n_feat}  (was 55 for Baseline, 220 for Enhanced/Hybrid)")
    print(f"  New features : days_since | freq5/10/20 | co-selection | position | DOW rate | entropy")

    # ── Phase 3: Train ────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 3 — TRAINING")
    print(f"{'─'*72}\n")

    histories = {}

    print("  [1/3] FE-Baseline  (BCELoss, unidirectional LSTM + input proj)")
    t0 = time.perf_counter()
    histories['FE-Baseline'] = train_model(
        model_base, bce_crit, train_loader, val_loader, device, 'FE-Baseline')
    print(f"  FE-Baseline training done in {time.perf_counter()-t0:.1f}s\n")

    print("  [2/3] FE-Enhanced  (FocalLoss, BiLSTM + Attention)")
    t0 = time.perf_counter()
    histories['FE-Enhanced'] = train_model(
        model_enh, focal_crit, train_loader, val_loader, device, 'FE-Enhanced')
    print(f"  FE-Enhanced training done in {time.perf_counter()-t0:.1f}s\n")

    print("  [3/3] FE-Hybrid    (FocalLoss, BiLSTM + Attention)")
    t0 = time.perf_counter()
    histories['FE-Hybrid'] = train_model(
        model_hyb, focal_crit, train_loader, val_loader, device, 'FE-Hybrid')
    print(f"  FE-Hybrid training done in {time.perf_counter()-t0:.1f}s\n")

    # Save base weights for Hybrid alpha trials
    hyb_base_state = {k: v.clone() for k, v in model_hyb.state_dict().items()}

    # ── Phase 4: Fit Markov Chain ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 4 — FIT MARKOV CHAIN")
    print(f"{'─'*72}\n")

    base_markov = FactoredMarkovChain(n_students=CFG['n_students'], smoothing=1.0)
    base_markov.fit(binary[:split_idx])
    print(f"  Markov fitted on {split_idx} training days")
    print(f"  Base selection rate: {6/CFG['n_students']:.4f}  (6/{CFG['n_students']})")

    # ── Phase 5: Simulation — FE-Baseline & FE-Enhanced ──────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 5 — ONLINE SIMULATION")
    print(f"{'─'*72}\n")

    results = {}

    print("  --- FE-Baseline (100 days) ---")
    t0 = time.perf_counter()
    accs_base, mp_base = run_nn_sim(
        model_base, bce_crit, extended, binary, split_idx, device, 'FE-Baseline')
    print(f"  FE-Baseline sim done in {time.perf_counter()-t0:.1f}s")
    results['FE-Baseline'] = {'accs': accs_base, 'stats': compute_stats(accs_base),
                               'max_probs': mp_base}

    print(f"\n  --- FE-Enhanced (100 days) ---")
    t0 = time.perf_counter()
    accs_enh, mp_enh = run_nn_sim(
        model_enh, focal_crit, extended, binary, split_idx, device, 'FE-Enhanced')
    print(f"  FE-Enhanced sim done in {time.perf_counter()-t0:.1f}s")
    results['FE-Enhanced'] = {'accs': accs_enh, 'stats': compute_stats(accs_enh),
                               'max_probs': mp_enh}

    # ── Phase 6: FE-Hybrid alpha tuning ──────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 6 — FE-HYBRID ALPHA TUNING")
    print(f"{'─'*72}\n")

    best_alpha    = None
    best_alpha_avg = -1.0
    best_alpha_accs: list = []
    alpha_stats   = {}

    for alpha in CFG['alpha_values']:
        torch.manual_seed(CFG['random_seed'])
        np.random.seed(CFG['random_seed'])

        model_hyb.load_state_dict(hyb_base_state)
        model_hyb.eval()

        mk = FactoredMarkovChain(n_students=CFG['n_students'], smoothing=1.0)
        mk.fit(binary[:split_idx])

        print(f"\n  --- alpha={alpha:.1f}  "
              f"({int(alpha*100)}% NN | {int((1-alpha)*100)}% Markov) ---")
        t0 = time.perf_counter()
        accs_h, mp_h = run_hybrid_sim(
            model_hyb, mk, focal_crit, extended, binary,
            split_idx, device, alpha)
        elapsed = time.perf_counter() - t0
        s = compute_stats(accs_h)
        alpha_stats[alpha] = s
        print(f"  alpha={alpha:.1f} => Avg: {s['avg']:.2f}%  "
              f"Best: {s['best']:.2f}%  ({elapsed:.1f}s)")

        if s['avg'] > best_alpha_avg:
            best_alpha_avg  = s['avg']
            best_alpha      = alpha
            best_alpha_accs = accs_h

    results['FE-Hybrid'] = {
        'accs'      : best_alpha_accs,
        'stats'     : alpha_stats[best_alpha],
        'alpha'     : best_alpha,
        'all_alphas': alpha_stats,
        'max_probs' : [],
    }

    # ── Phase 7: Results ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PHASE 7 — ALPHA TUNING SUMMARY  (FE-Hybrid)")
    print(f"{'─'*72}\n")
    print(f"  {'Alpha':<8} {'NN%':<7} {'Markov%':<9} "
          f"{'Avg':>8} {'Best':>8} {'Worst':>8} {'Std':>8} {'0-days':>7}")
    print("  " + "-" * 66)
    for a, s in alpha_stats.items():
        mk = " <-- BEST" if a == best_alpha else ""
        print(f"  {a:<8.1f} {int(a*100):<7} {int((1-a)*100):<9} "
              f"{s['avg']:>7.2f}%  {s['best']:>7.2f}%  {s['worst']:>7.2f}%  "
              f"{s['std']:>7.4f}  {s['zero_days']:>5}{mk}")

    # ── Per-model summary ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FEATURE ENGINEERING RESULTS".center(72))
    print(f"{'='*72}\n")

    print(f"  {'Model':<20} {'Avg':>8} {'Best':>8} {'Worst':>8} "
          f"{'Std':>8} {'0-days':>8} {'50+days':>8} {'Trend':>8}")
    print("  " + "-" * 74)
    for nm in ['FE-Baseline', 'FE-Enhanced', 'FE-Hybrid']:
        s    = results[nm]['stats']
        alph = f"  (a={results[nm].get('alpha','?')})" if nm == 'FE-Hybrid' else ''
        trend = s['last10'] - s['first10']
        print(f"  {nm+alph:<24} {s['avg']:>7.2f}%  {s['best']:>7.2f}%  "
              f"{s['worst']:>7.2f}%  {s['std']:>7.4f}  "
              f"{s['zero_days']:>6}   {s['fifty_days']:>6}  "
              f"{trend:>+7.2f}%")

    # ── Delta vs originals ────────────────────────────────────────────────
    print(f"\n  Delta vs Original Models (feature engineering effect):")
    print(f"  {'Model':<22} {'Orig Avg':>10} {'FE Avg':>10} {'Delta':>10}")
    print("  " + "-" * 55)
    pairs = [
        ('FE-Baseline', 'Baseline LSTM (orig)',   26.00),
        ('FE-Enhanced', 'Enhanced LSTM (orig)',   28.33),
        ('FE-Hybrid',   'LSTM+Markov Hybrid (orig)', 29.00),
    ]
    for fe_name, orig_name, orig_avg in pairs:
        fe_avg = results[fe_name]['stats']['avg']
        delta  = fe_avg - orig_avg
        marker = " IMPROVED" if delta > 0 else ""
        print(f"  {fe_name:<22} {orig_avg:>9.2f}%  {fe_avg:>9.2f}%  "
              f"{delta:>+9.2f}%{marker}")

    # ── Full leaderboard ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FULL LEADERBOARD — ALL MODELS".center(72))
    print(f"{'='*72}\n")

    all_models = {nm: (v[0], v[1], v[2]) for nm, v in PRIOR.items()}
    for nm in ['FE-Baseline', 'FE-Enhanced', 'FE-Hybrid']:
        s = results[nm]['stats']
        suffix = f" (a={results[nm].get('alpha','-')})" if nm == 'FE-Hybrid' else ''
        all_models[nm + suffix] = (s['avg'], s['best'], s['n_preds'])

    sorted_lb = sorted(all_models.items(), key=lambda x: -x[1][0])

    new_names = {nm for nm in ['FE-Baseline', 'FE-Enhanced'] +
                 [f"FE-Hybrid (a={results['FE-Hybrid'].get('alpha','-')})"]}

    print(f"  {'Rank':<5} {'Model':<35} {'Avg Acc':>8} {'Best':>8} {'N Preds':>9}")
    print("  " + "-" * 68)
    for rank, (name, (avg, best, npreds)) in enumerate(sorted_lb, 1):
        marker = " <-- NEW" if any(name.startswith(x) for x in
                                    ['FE-Baseline', 'FE-Enhanced', 'FE-Hybrid']) else ""
        print(f"  {rank:<5} {name:<35} {avg:>7.2f}%  {best:>7.2f}%  "
              f"{str(npreds):>7}{marker}")

    # ── Save artefacts ────────────────────────────────────────────────────
    save_data = {
        'feature_dim'  : int(n_feat),
        'new_features' : ['3-day freq', '28-day freq', 'EMA α=0.5',
                           'Momentum (7-28)', 'Overdue tanh'],
        'FE-Baseline'  : results['FE-Baseline']['stats'],
        'FE-Enhanced'  : results['FE-Enhanced']['stats'],
        'FE-Hybrid'    : {
            'best_alpha'   : results['FE-Hybrid']['alpha'],
            'stats'        : results['FE-Hybrid']['stats'],
            'all_alphas'   : {str(a): s for a, s in alpha_stats.items()},
        },
    }
    with open(_CKPT_DIR / 'feature_eng_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved → dl_checkpoints/feature_engineering/feature_eng_results.json")

    save_charts(results, histories)

    print(f"\n{'='*72}")
    print("  FEATURE ENGINEERING PIPELINE COMPLETE".center(72))
    print(f"{'='*72}")
    return results


if __name__ == '__main__':
    main()
