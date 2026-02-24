"""
Randomization / Permutation Test
=================================

Purpose:
    Verify that the ~26% benchmark accuracy reflects REAL learned patterns,
    not just exploitation of base-rate frequencies.

Method:
    Three conditions evaluated on the SAME 100 test days:

    1. ORIGINAL   — normal temporal order (what we've been measuring)
    2. SHUFFLED   — all 1308 rows randomly permuted before train/test split
                    → destroys every temporal dependency
                    → LSTM / Markov see random sequences
    3. RANDOM     — pure random predictor: pick 6 students uniformly at random
                    → theoretical baseline ≈ 6/55 × 6 / 6 ≈ 16.7% per day

    If SHUFFLED ≈ RANDOM (~16–17%) and ORIGINAL >> SHUFFLED:
        → The models learned REAL temporal patterns. ✓

    If SHUFFLED ≈ ORIGINAL (~26%):
        → Models are only learning static base rates (who is called most often),
          not temporal transitions.  Results are inflated by frequency bias.

Models tested:
    • Hybrid (LSTM + Markov, alpha=0.7)  — top model at 100-day benchmark
    • Bayesian BB                         — second-best at 100 days
    • Random baseline                     — control

Usage:
    cd dl_pipeline
    python randomization_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
CFG = {
    'excel_path':              '../Database.xlsx',
    'sequence_length':         14,
    'batch_size':              32,
    'epochs':                  50,
    'early_stopping_patience': 15,
    'lr':                      5e-4,
    'train_ratio':             0.9,
    'n_sim_days':              100,
    'n_students':              55,
    'k':                       6,
    'n_groups':                5,
    'temperature':             1.5,
    'alpha':                   0.7,      # best alpha from full benchmark
    'online_iterations':       5,
    'replay_buffer':           10,
    'replay_min':              5,
}
DEVICE = torch.device('cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_binary():
    """Return (n_days, 55) binary matrix in ORIGINAL temporal order."""
    df = pd.read_excel(CFG['excel_path'])
    df = df.sort_values('Days').reset_index(drop=True)
    n_days = len(df)
    binary = np.zeros((n_days, CFG['n_students']), dtype=np.float32)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]
    for _, row in df.iterrows():
        day_idx = int(row['Days']) - 1
        for col in id_cols:
            sid = int(row[col]) - 1
            if 0 <= sid < CFG['n_students']:
                binary[day_idx, sid] = 1.0
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / training helpers  (mirrors run_hybrid.py)
# ─────────────────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, enriched, binary, seq_len):
        self.E = enriched
        self.B = binary
        self.seq_len = seq_len

    def __len__(self):
        return len(self.E) - self.seq_len

    def __getitem__(self, idx):
        X = self.E[idx: idx + self.seq_len]
        y = self.B[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


def train_model(model, train_loader, val_loader, device, epochs, patience, lr):
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_loss, best_state, stall = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}/{epochs}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, best_state, stall = val_loss, {
                k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stall += 1
            if stall >= patience:
                print(f"      early stop @ epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lstm_probs(model, enriched_seq, device):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(enriched_seq[-CFG['sequence_length']:]).unsqueeze(0).to(device)
        return torch.sigmoid(model(x)).squeeze().cpu().numpy()


def generate_groups(probs, n_groups, k, temperature):
    """Top-k with temperature-scaled sampling for n_groups candidates."""
    groups = []
    scaled = probs ** (1.0 / temperature)
    for _ in range(n_groups):
        p = scaled / scaled.sum()
        g = sorted(np.random.choice(55, k, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
    if not groups:
        groups.append(sorted(np.argsort(probs)[-k:].tolist()))
    return groups


def best_overlap(groups, actual_set):
    return max(len(set(g) & actual_set) for g in groups)


# ─────────────────────────────────────────────────────────────────────────────
# Run one condition (ORIGINAL or SHUFFLED)
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_condition(binary, label):
    """
    Full pipeline: build enriched features → train LSTM → fit Markov
    → simulate 100 test days → return list of per-day accuracy floats.
    """
    n_days   = len(binary)
    seq_len  = CFG['sequence_length']
    split    = int(n_days * CFG['train_ratio'])
    n_sim    = min(CFG['n_sim_days'], n_days - split)

    print(f"\n  Building enriched features for [{label}] ...")
    enriched = build_enriched_features(binary)          # (n_days, 220)

    # ── train / val split ────────────────────────────────────────────────────
    train_E = enriched[:split]
    train_B = binary[:split]
    val_split = int(split * 0.9)

    train_ds = SeqDataset(train_E[:val_split], train_B[:val_split], seq_len)
    val_ds   = SeqDataset(train_E[val_split:], train_B[val_split:], seq_len)

    train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)

    n_feat = enriched.shape[1]
    model = EnhancedLSTMPredictor(
        n_features=n_feat, hidden_size=128, num_layers=2,
        n_students=55, dropout=0.3
    ).to(DEVICE)

    print(f"  Training LSTM [{label}] ...")
    t0 = time.perf_counter()
    train_model(model, train_dl, val_dl, DEVICE,
                CFG['epochs'], CFG['early_stopping_patience'], CFG['lr'])
    print(f"  Training done in {time.perf_counter()-t0:.1f}s")

    # ── fit Markov on training portion ───────────────────────────────────────
    markov = FactoredMarkovChain(n_students=55)
    markov.fit(binary[:split])

    # ── online simulation ────────────────────────────────────────────────────
    accs = []
    running_binary   = binary[:split].copy()
    running_enriched = enriched[:split].copy()

    for i in range(n_sim):
        t_idx   = split + i
        actual  = set(np.where(binary[t_idx] == 1)[0].tolist())

        e_seq = running_enriched[-seq_len:]
        b_seq = running_binary[-seq_len:]

        lstm_p  = _lstm_probs(model, e_seq, DEVICE)
        markov_p = markov.predict_probabilities(b_seq)

        hybrid_p = CFG['alpha'] * lstm_p + (1 - CFG['alpha']) * markov_p
        groups   = generate_groups(hybrid_p, CFG['n_groups'], CFG['k'],
                                   CFG['temperature'])

        overlap = best_overlap(groups, actual)
        accs.append(overlap / CFG['k'])

        # update running context
        running_binary   = np.vstack([running_binary, binary[t_idx:t_idx+1]])
        new_e            = build_enriched_features(running_binary)
        running_enriched = new_e

    return accs


# ─────────────────────────────────────────────────────────────────────────────
# Random baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_random_baseline(binary):
    """No model — pick 6 random students every day."""
    n_days = len(binary)
    split  = int(n_days * CFG['train_ratio'])
    n_sim  = min(CFG['n_sim_days'], n_days - split)

    accs = []
    rng  = np.random.default_rng(SEED)
    for i in range(n_sim):
        actual  = set(np.where(binary[split + i] == 1)[0].tolist())
        pred    = set(rng.choice(55, 6, replace=False).tolist())
        accs.append(len(pred & actual) / CFG['k'])
    return accs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def summarise(accs, label):
    arr = np.array(accs) * 100
    avg = arr.mean()
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Avg accuracy : {avg:.2f}%")
    print(f"  Best day     : {arr.max():.2f}%")
    print(f"  Worst day    : {arr.min():.2f}%")
    print(f"  Std dev      : {arr.std():.4f}")
    dist = {k: int((arr == k/6*100).sum()) for k in range(7)}
    print(f"  Distribution : " +
          "  ".join(f"{k}/6={v}" for k, v in dist.items() if v > 0))
    return avg


if __name__ == '__main__':
    print("=" * 70)
    print("   RANDOMIZATION TEST — Is the ~26% accuracy statistically real?")
    print("=" * 70)

    # 1. Load original binary matrix
    print("\n[Step 1] Loading data ...")
    binary_orig = load_binary()
    print(f"  Loaded {len(binary_orig)} days × {binary_orig.shape[1]} students")

    # 2. Shuffled binary — randomly permute ALL rows
    binary_shuffled = binary_orig.copy()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(binary_shuffled))
    binary_shuffled = binary_shuffled[idx]
    print(f"  Shuffled version: {len(binary_shuffled)} rows in random order")

    # 3. Run random baseline (no model, no training needed)
    print("\n[Step 2] Running RANDOM baseline (no model) ...")
    accs_random = run_random_baseline(binary_orig)
    avg_random  = summarise(accs_random, "RANDOM BASELINE (pick 6 at random)")

    # 4. Run Hybrid on SHUFFLED data
    print("\n[Step 3] Running Hybrid on SHUFFLED data ...")
    accs_shuffled = run_hybrid_condition(binary_shuffled, "SHUFFLED")
    avg_shuffled  = summarise(accs_shuffled, "HYBRID — SHUFFLED data")

    # 5. Run Hybrid on ORIGINAL data
    print("\n[Step 4] Running Hybrid on ORIGINAL data ...")
    accs_orig = run_hybrid_condition(binary_orig, "ORIGINAL")
    avg_orig  = summarise(accs_orig, "HYBRID — ORIGINAL data")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("   VERDICT")
    print("=" * 70)
    print(f"\n  Random baseline     : {avg_random:.2f}%  ← theoretical floor")
    print(f"  Hybrid on SHUFFLED  : {avg_shuffled:.2f}%")
    print(f"  Hybrid on ORIGINAL  : {avg_orig:.2f}%")

    gap_signal  = avg_orig     - avg_random
    gap_shuffle = avg_shuffled - avg_random

    print(f"\n  Gain over random (ORIGINAL) : +{gap_signal:.2f}%")
    print(f"  Gain over random (SHUFFLED) : +{gap_shuffle:.2f}%")

    if avg_shuffled <= avg_random + 2.0:
        print("\n  ✓ SHUFFLED ≈ RANDOM  →  Models learn REAL temporal patterns.")
        print("    The ~26% benchmark is genuinely meaningful.")
    elif avg_shuffled >= avg_orig - 2.0:
        print("\n  ✗ SHUFFLED ≈ ORIGINAL  →  Models exploit BASE RATES only.")
        print("    The ~26% benchmark is mostly due to frequency bias,")
        print("    not true temporal/sequential learning.")
    else:
        pct_temporal = (avg_orig - avg_shuffled) / gap_signal * 100
        pct_baserate = (avg_shuffled - avg_random) / gap_signal * 100
        print(f"\n  ~ MIXED signal:")
        print(f"    Base-rate contribution  : ~{pct_baserate:.0f}% of the gain")
        print(f"    Temporal-pattern contribution: ~{pct_temporal:.0f}% of the gain")

    print("\n" + "=" * 70)
    print("   RANDOMIZATION TEST COMPLETE")
    print("=" * 70)
