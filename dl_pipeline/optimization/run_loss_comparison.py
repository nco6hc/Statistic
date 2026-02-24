"""
run_loss_comparison.py — Loss Function Comparison on LSTM
===========================================================

Trains the SAME LSTM architecture with 4 different loss functions and
compares group-selection accuracy head-to-head.

Loss functions tested:
  1. BCE       — standard binary cross-entropy (baseline)
  2. Ranking   — ListNet listwise loss (treats selection as ranked list)
  3. TopK      — hard-negative mining BCE (focus on rank 5–8 boundary)
  4. Focal     — down-weight easy negatives via (1-p_t)^2 factor

Architecture (identical for all 4):
  LSTM(input=55, hidden=64, layers=1) → FC(64→55) → [logits]
  sigmoid applied only during inference (not in forward pass)
  seq_len=14, no GPU required

Training:
  25 epochs on 90% training split, batch_size=32, Adam lr=1e-3
  LR halved when plateau detected

Online Simulation (same as other models):
  100 test days, 5 candidate groups per day, temperature-scaled sampling
  Model fine-tuned 3 steps after each observed day

Usage:
  cd dl_pipeline/optimization
  python run_loss_comparison.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # optimization/
_DL_DIR   = _THIS_DIR.parent                         # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                          # Statistic/
_CKPT_DIR  = _DL_DIR / 'dl_checkpoints' / 'loss_comparison'
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_DL_DIR))
sys.path.insert(0, str(_THIS_DIR))

from data_loader import DataManager
from loss_functions import BCELoss, RankingLoss, TopKLoss, FocalLoss

# ── Configuration ─────────────────────────────────────────────────────────────
EXCEL_PATH    = str(_ROOT_DIR / 'Database.xlsx')
N_STUDENTS    = 55
K             = 6
TRAIN_RATIO   = 0.9
SEQ_LEN       = 14
HIDDEN        = 64
N_LAYERS      = 1
DROPOUT       = 0.2
BATCH_SIZE    = 32
N_EPOCHS      = 25
LR            = 1e-3
N_GROUPS      = 5
SIM_DAYS      = 100
ONLINE_STEPS  = 3       # fine-tune steps per observed day
ONLINE_LR     = 3e-4    # lower LR for online updates

# Temperature schedule: groups 1-5 get increasing temperature for diversity
TEMPERATURES  = [0.7, 0.9, 1.1, 1.4, 1.8]

torch.manual_seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
#  LSTM Model — outputs LOGITS (sigmoid only at inference)
# ═══════════════════════════════════════════════════════════════════════════

class LSTMLogitPredictor(nn.Module):
    """
    Lightweight LSTM for student selection prediction.

    Returns RAW LOGITS (no sigmoid) so all loss functions can apply
    numerically stable operations on the pre-sigmoid values.
    """

    def __init__(self, n_students: int = N_STUDENTS,
                 hidden: int = HIDDEN, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_students, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_students)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_students)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]          # (batch, hidden)
        last   = self.drop(last)
        return self.fc(last)            # (batch, n_students)  ← LOGITS

    def predict_proba(self, seq: np.ndarray) -> np.ndarray:
        """
        Convenience: (seq_len, n_students) → (n_students,) probabilities.
        Applies sigmoid to convert logits.
        """
        self.eval()
        with torch.no_grad():
            x    = torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, n_students)
            logits = self(x)
            probs  = torch.sigmoid(logits).squeeze(0).numpy()
        return probs

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def build_sequences(binary: np.ndarray, seq_len: int = SEQ_LEN
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequence builder: X shape (N, seq_len, 55), y (N, 55)."""
    X, y = [], []
    for t in range(seq_len, len(binary)):
        X.append(binary[t - seq_len : t])
        y.append(binary[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(binary_train: np.ndarray,
                criterion: nn.Module,
                name: str,
                verbose: bool = True) -> tuple[LSTMLogitPredictor, list]:
    """
    Train one LSTM variant with the given loss function.

    Returns:
        model:         Trained LSTMLogitPredictor.
        loss_history:  List of (epoch, avg_train_loss) tuples.
    """
    X, y = build_sequences(binary_train)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model     = LSTMLogitPredictor()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

    loss_history = []
    t0 = time.perf_counter()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss   = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        loss_history.append((epoch, avg_loss))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    [{name}] epoch {epoch:>2}/{N_EPOCHS}  "
                  f"loss={avg_loss:.4f}  lr={lr_now:.2e}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    [{name}] Training done in {elapsed:.1f}s")

    model.eval()
    return model, loss_history


# ═══════════════════════════════════════════════════════════════════════════
#  Group Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_groups(probs: np.ndarray,
                    n_groups: int = N_GROUPS,
                    k: int = K) -> list:
    """
    Generate n_groups diverse candidate groups via temperature scaling.

    Each group uses a different temperature so later groups are more
    exploratory. Same strategy as the LSTM+Markov Hybrid (29% model).

    Returns list of 0-indexed group lists.
    """
    groups = []
    seen   = set()

    for g in range(n_groups):
        temp = TEMPERATURES[g % len(TEMPERATURES)]
        adj  = np.power(np.maximum(probs, 1e-9), 1.0 / temp)
        adj /= adj.sum()
        idx  = np.random.choice(len(adj), size=k, replace=False, p=adj)
        idx  = sorted(idx.tolist())
        key  = frozenset(idx)
        if key not in seen:
            seen.add(key)
            groups.append(idx)

    # Fill if not enough unique groups
    while len(groups) < n_groups:
        temp = TEMPERATURES[-1]
        adj  = np.power(np.maximum(probs, 1e-9), 1.0 / temp)
        adj /= adj.sum()
        idx  = sorted(np.random.choice(len(adj), size=k, replace=False,
                                        p=adj).tolist())
        groups.append(idx)

    return groups[:n_groups]


def overlap_accuracy(groups: list, actual_binary: np.ndarray) -> float:
    """Best-of-N-groups accuracy: max overlap / k * 100."""
    actual_set = set(np.where(actual_binary == 1)[0].tolist())
    return max(len(set(g) & actual_set) for g in groups) / K * 100.0


# ═══════════════════════════════════════════════════════════════════════════
#  Online Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(model: LSTMLogitPredictor,
                   criterion: nn.Module,
                   binary: np.ndarray,
                   split_idx: int,
                   name: str) -> tuple[list, list]:
    """
    Online simulation over the last SIM_DAYS test days.

    Each day:
      1. Predict from the last SEQ_LEN observed days.
      2. Generate N_GROUPS candidate groups.
      3. Evaluate against actual selection.
      4. Fine-tune model for ONLINE_STEPS steps on the new observation.

    Returns:
        accuracies:    List of float per day.
        prob_ranges:   List of (min, max) of predicted probs per day.
    """
    test_data = binary[split_idx:]
    sim_data  = test_data[-SIM_DAYS:]
    n_sim     = len(sim_data)

    # Seed rolling window with last SEQ_LEN training days
    rolling = list(binary[max(0, split_idx - SEQ_LEN) : split_idx])

    online_optimizer = optim.Adam(model.parameters(), lr=ONLINE_LR)
    accuracies  = []
    prob_ranges = []

    for d in range(n_sim):
        # ---- Predict --------------------------------------------------------
        recent = np.array(rolling[-SEQ_LEN:], dtype=np.float32)
        probs  = model.predict_proba(recent)        # (55,)

        prob_ranges.append((float(probs.min()), float(probs.max())))
        groups = generate_groups(probs, N_GROUPS)

        # ---- Evaluate -------------------------------------------------------
        actual = sim_data[d]
        acc    = overlap_accuracy(groups, actual)
        accuracies.append(acc)

        # ---- Online update --------------------------------------------------
        model.train()
        X_ft = torch.FloatTensor(recent).unsqueeze(0)       # (1, 14, 55)
        y_ft = torch.FloatTensor(actual).unsqueeze(0)       # (1, 55)

        for _ in range(ONLINE_STEPS):
            online_optimizer.zero_grad()
            logits = model(X_ft)
            loss   = criterion(logits, y_ft)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            online_optimizer.step()

        model.eval()
        rolling.append(actual)

    return accuracies, prob_ranges


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics & Charts
# ═══════════════════════════════════════════════════════════════════════════

def stats(arr: list) -> dict:
    a = np.array(arr)
    return {'avg': float(np.mean(a)), 'best': float(np.max(a)),
            'worst': float(np.min(a)), 'std': float(np.std(a))}


def save_charts(results: dict, loss_histories: dict):
    """5-panel comparison dashboard."""
    names  = list(results.keys())
    colors = {'BCE': 'steelblue', 'Ranking': 'darkorange',
              'TopK': 'mediumseagreen', 'Focal': 'crimson'}

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Loss Function Comparison — LSTM Student Selection',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    days = np.arange(1, SIM_DAYS + 1)
    W    = 10  # rolling window

    # ── Panel 1: Accuracy over time ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    for nm in names:
        acc = results[nm]['accuracies']
        c   = colors[nm]
        ax0.plot(days, acc, color=c, lw=1.2, alpha=0.45)
        roll = np.convolve(acc, np.ones(W) / W, mode='valid')
        ax0.plot(np.arange(W, len(acc) + 1), roll, color=c, lw=2.2,
                 label=f'{nm} ({results[nm]["stats"]["avg"]:.1f}%)')
    ax0.set_xlabel('Simulation Day')
    ax0.set_ylabel('Accuracy (%)')
    ax0.set_title('Daily Accuracy (bold = 10-day rolling avg)')
    ax0.legend(fontsize=9)
    ax0.set_ylim(-5, 110)
    ax0.grid(axis='y', alpha=0.3)

    # ── Panel 2: Bar chart average accuracy ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    avgs  = [results[nm]['stats']['avg'] for nm in names]
    stds  = [results[nm]['stats']['std'] for nm in names]
    bars  = ax1.bar(names, avgs, color=[colors[n] for n in names],
                    alpha=0.8, yerr=stds, capsize=5)
    for bar, val in zip(bars, avgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('Avg Accuracy ± Std (100 days)')
    ax1.grid(axis='y', alpha=0.3)
    ymin = max(0, min(avgs) - 5)
    ax1.set_ylim(ymin, max(avgs) + 7)

    # ── Panel 3: Training loss curves ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for nm in names:
        hist = loss_histories[nm]
        ep   = [h[0] for h in hist]
        ls   = [h[1] for h in hist]
        ax2.plot(ep, ls, color=colors[nm], lw=2.0, marker='o',
                 markersize=3, label=nm)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Curves')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 4: Probability distribution (violin) ────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    prob_data = [results[nm]['prob_ranges'] for nm in names]
    # Show the distribution of the max-prob each day (how sharp predictions are)
    max_probs = [[pr[1] for pr in pd] for pd in prob_data]
    parts = ax3.violinplot(max_probs, showmedians=True, showextrema=True)
    for pc, nm in zip(parts['bodies'], names):
        pc.set_facecolor(colors[nm])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    ax3.set_xticks(range(1, len(names) + 1))
    ax3.set_xticklabels(names, fontsize=8)
    ax3.set_ylabel('Max Predicted Probability per Day')
    ax3.set_title('Output Sharpness\n(higher = more confident predictions)')
    for i, (nm, mp) in enumerate(zip(names, max_probs), 1):
        ax3.scatter([i], [np.mean(mp)], color='black', s=35, zorder=5,
                    marker='D')
    ax3.grid(axis='y', alpha=0.3)

    # ── Panel 5: Cumulative accuracy ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    for nm in names:
        acc = np.array(results[nm]['accuracies'])
        cum = np.cumsum(acc) / (np.arange(len(acc)) + 1)
        ax4.plot(days, cum, color=colors[nm], lw=2.0, label=nm)
    ax4.set_xlabel('Simulation Day')
    ax4.set_ylabel('Cumulative Avg Accuracy (%)')
    ax4.set_title('Cumulative Accuracy')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    out = _CKPT_DIR / 'loss_comparison_dashboard.png'
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

LOSS_CONFIGS = [
    ('BCE',     BCELoss()),
    ('Ranking', RankingLoss()),
    ('TopK',    TopKLoss(k=K)),
    ('Focal',   FocalLoss(alpha=0.25, gamma=2.0)),
]

PRIOR_MODELS = [
    ("LSTM+Markov Hybrid",  29.00),
    ("Enhanced LSTM",       28.33),
    ("Bayesian BB",         27.67),
    ("Diverse Beam Search", 26.50),
    ("Gradient Boosting",   26.67),
    ("Baseline LSTM",       26.00),
    ("Meta-Ensemble",       24.33),
]


def main():
    print("=" * 66)
    print(" LOSS FUNCTION COMPARISON")
    print(" BCE  vs  Ranking Loss  vs  Top-K Loss  vs  Focal Loss")
    print("=" * 66)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    dm     = DataManager(EXCEL_PATH)
    dm.load_data()
    binary = dm.convert_to_binary()
    n_days = binary.shape[0]

    split_idx  = int(n_days * TRAIN_RATIO)
    train_data = binary[:split_idx]
    test_data  = binary[split_idx:]

    print(f"      Total: {n_days}d | Train: {split_idx}d | Test: {len(test_data)}d")
    print(f"      Simulation: last {SIM_DAYS} test days")
    print(f"      Model: LSTM(hidden={HIDDEN}, layers={N_LAYERS}) | "
          f"seq_len={SEQ_LEN}")

    dummy = LSTMLogitPredictor()
    print(f"      Params: {dummy.n_params():,}")

    # ── 2. Train each variant ─────────────────────────────────────────────
    print(f"\n[2/3] Training {len(LOSS_CONFIGS)} models x {N_EPOCHS} epochs ...")
    models        : dict = {}
    loss_histories: dict = {}

    for name, criterion in LOSS_CONFIGS:
        print(f"\n  --- {name} ---")
        model, hist = train_model(train_data, criterion, name)
        models[name]         = (model, criterion)
        loss_histories[name] = hist

    # ── 3. Online simulation ──────────────────────────────────────────────
    print(f"\n[3/3] Online simulation ({SIM_DAYS} days, {N_GROUPS} groups/pred) ...")
    results: dict = {}

    print(f"\n  {'Day':>4}", end="")
    for nm, _ in LOSS_CONFIGS:
        print(f"  {nm:>9}", end="")
    print()
    print("  " + "-" * (6 + 11 * len(LOSS_CONFIGS)))

    # Collect day-by-day for live progress
    running: dict = {nm: [] for nm, _ in LOSS_CONFIGS}
    sim_data = binary[split_idx:][-SIM_DAYS:]

    # Run all models in parallel day-by-day
    rollings = {}
    for nm, _ in LOSS_CONFIGS:
        rollings[nm] = list(binary[max(0, split_idx - SEQ_LEN): split_idx])

    online_opts: dict = {}
    for nm, (model, _) in models.items():
        online_opts[nm] = optim.Adam(model.parameters(), lr=ONLINE_LR)

    prob_ranges_all: dict = {nm: [] for nm, _ in LOSS_CONFIGS}

    for d in range(len(sim_data)):
        row = f"  {d+1:>4}"
        actual = sim_data[d]

        for nm, criterion in LOSS_CONFIGS:
            model, crit = models[nm]

            # Predict
            recent = np.array(rollings[nm][-SEQ_LEN:], dtype=np.float32)
            probs  = model.predict_proba(recent)
            prob_ranges_all[nm].append((float(probs.min()), float(probs.max())))

            groups = generate_groups(probs)
            acc    = overlap_accuracy(groups, actual)
            running[nm].append(acc)

            # Online fine-tune
            model.train()
            X_ft = torch.FloatTensor(recent).unsqueeze(0)
            y_ft = torch.FloatTensor(actual).unsqueeze(0)
            for _ in range(ONLINE_STEPS):
                online_opts[nm].zero_grad()
                loss = crit(model(X_ft), y_ft)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                online_opts[nm].step()
            model.eval()

            rollings[nm].append(actual)
            row += f"  {acc:>8.1f}%"

        if (d + 1) % 10 == 0 or d < 3:
            print(row)

    for nm, _ in LOSS_CONFIGS:
        results[nm] = {
            'accuracies' : running[nm],
            'prob_ranges': prob_ranges_all[nm],
            'stats'      : stats(running[nm]),
        }

    # ── Results table ────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  RESULTS SUMMARY")
    print("=" * 66)

    print(f"\n  {'Loss':<12} {'Avg':>7} {'Best':>7} {'Worst':>7} "
          f"{'Std':>7} {'0%-days':>9} {'50%-days':>10}")
    print("  " + "-" * 65)

    for nm, _ in LOSS_CONFIGS:
        s   = results[nm]['stats']
        arr = np.array(results[nm]['accuracies'])
        zero_days  = int(np.sum(arr == 0))
        fifty_days = int(np.sum(arr >= 50))
        print(f"  {nm:<12} {s['avg']:>6.2f}%  {s['best']:>6.2f}%  "
              f"{s['worst']:>6.2f}%  {s['std']:>6.2f}  "
              f"{zero_days:>6}d     {fifty_days:>6}d")

    # Print probability sharpness
    print(f"\n  {'Loss':<12} {'Avg max-prob':>14} {'Avg min-prob':>14}")
    print("  " + "-" * 42)
    for nm, _ in LOSS_CONFIGS:
        pr  = results[nm]['prob_ranges']
        avg_max = np.mean([x[1] for x in pr])
        avg_min = np.mean([x[0] for x in pr])
        print(f"  {nm:<12} {avg_max:>13.4f}  {avg_min:>13.4f}")

    print("\n  (Higher avg max-prob = model is more confident about top students)")

    # ── Leaderboard ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  FULL LEADERBOARD — ALL MODELS")
    print("=" * 66)

    new_models = [(f"LSTM({nm})", results[nm]['stats']['avg'])
                  for nm, _ in LOSS_CONFIGS]
    all_models = sorted(PRIOR_MODELS + new_models, key=lambda x: -x[1])

    new_names = {nm for nm, _ in new_models}
    print(f"\n  {'Rank':<5} {'Model':<28} {'Avg Acc':>8}")
    print("  " + "-" * 44)
    for rank, (name, avg) in enumerate(all_models, 1):
        marker = " <-- NEW" if name in new_names else ""
        print(f"  {rank:<5} {name:<28} {avg:>7.2f}%{marker}")

    # Best new model vs BCE baseline
    best_new = max(new_models, key=lambda x: x[1])
    bce_avg  = results['BCE']['stats']['avg']
    print(f"\n  Best new model : {best_new[0]} ({best_new[1]:.2f}%)")
    print(f"  vs BCE baseline: {best_new[1] - bce_avg:+.2f}%")

    # ── Charts ───────────────────────────────────────────────────────────
    save_charts(results, loss_histories)
    print(f"\n  Charts saved to: {_CKPT_DIR}")
    print("=" * 66)


if __name__ == '__main__':
    main()
