"""
Hybrid LSTM+Markov — Loss Function Comparison
===============================================
Tests four alternative loss functions against the current FocalLoss baseline
on the Hybrid optimised architecture.  All use the same best-known hyperparams
discovered during optimize_hybrid.py: α=0.9, ng=10, T=2.0, iters=5, replay=10.

Loss functions tested
─────────────────────
1. FocalLoss          (current baseline, α=0.25, γ=2, ls=0.05)
   FL(p_t) = −α·(1−p_t)^γ·log(p_t)
   Down-weights easy negatives; good for class imbalance (6/55 ≈ 11% positive).

2. TopKBCELoss        (BCE weighted by prediction rank)
   Standard BCE but with a 4-tier weight matrix per element:
     TP in top-K         → weight = pos_weight  (5.0)
     FN (missed positive)→ weight = pos_weight  (5.0)  — same: push all positives up
     FP in top-K         → weight = fp_weight   (2.0)  — penalise wrong top-K
     TN outside top-K    → weight = tn_weight   (0.1)  — suppress easy negatives
   Concentrates gradient on the top-K decision boundary.

3. SoftmaxKHotLoss    (multinomial cross-entropy with k-hot target)
   Treats selection as a single multinomial draw over all N students.
   p_target  = y / K              (uniform distribution over K selected)
   p_pred    = softmax(logits)    (logits = logit(sigmoid_output))
   Loss      = −Σ p_target · log(p_pred)
   Forces the model to output a proper probability distribution and compete
   across ALL students rather than scoring each independently.

4. ListNetLoss        (listwise ranking — ListNet algorithm)
   Converts scores AND labels to probability distributions via softmax, then
   minimises KL divergence between them.
   p_pred   = softmax(logits / τ)           τ=1.0
   p_target = softmax(y_khot · label_scale) label_scale=10 → sharp on positives
   Loss     = −Σ p_target · log(p_pred)
   Listwise approach: considers the full ranking simultaneously.

5. LambdaRankLoss     (pairwise ranking with NDCG delta weighting)
   For each (positive i, negative j) pair:
     L_ij = log(1 + exp(−σ·(score_i − score_j))) · |ΔNDCG_ij|
   ΔNDCG_ij = |1/log₂(rank_i+1) − 1/log₂(rank_j+1)|  (rank-based discount)
   Fully vectorised: (B, N, N) pairwise matrices, no Python loops per sample.
   Optimises the rank ordering directly, weighted by how much each swap
   changes NDCG@6.

Architecture  (same for all losses):
  EnhancedLSTMPredictor: Input(n_feats)→Proj(128)→BiLSTM(256)→Attn→FC→Sigmoid
  For softmax-based losses (SoftmaxKHot, ListNet, LambdaRank):
    logits = torch.logit(sigmoid_output) is used inside the loss (invertible).

Best hyperparams used for ALL simulations (from optimize_hybrid.py):
  α=0.9, ng=10, T=2.0, iters=5, replay=10

Benchmarks:
  Hybrid optimised (FocalLoss, α=0.9, ng=10, T=2.0): 32.17%
  Order-3 Markov:                                      31.67%
  TD-HBB λ=0.99:                                       32.17%

Usage:
    cd dl_pipeline
    python hybrid_loss/run_hybrid_loss.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor
from markov_chain import FactoredMarkovChain

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'excel_path':             str(Path(__file__).resolve().parent.parent.parent / 'Database.xlsx'),
    'sequence_length':        14,
    'batch_size':             32,
    'epochs':                 50,
    'early_stopping_patience': 15,
    'train_ratio':            0.9,
    'n_simulation_days':      100,
    'n_students':             55,
    'k_students':             6,
    'random_seed':            42,
}

# Best hyperparams from optimize_hybrid.py
BEST = dict(alpha=0.9, n_groups=10, temperature=2.0,
            online_iters=5, replay_size=10)

BENCHMARKS = {
    'Hybrid optimised (FocalLoss)': 32.17,
    'TD-HBB λ=0.99             ': 32.17,
    'Order-3 Markov             ': 31.67,
    'DPM optimised              ': 30.83,
}


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Current baseline loss.
    FL(p_t) = −α·(1−p_t)^γ·log(p_t)  with optional label smoothing.
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = label_smoothing

    def forward(self, p, y):
        if self.ls > 0:
            y = y * (1 - self.ls) + self.ls * 0.5
        p  = p.clamp(1e-7, 1 - 1e-7)
        bce = -(y * p.log() + (1 - y) * (1 - p).log())
        pt  = torch.where(y > 0.5, p, 1 - p)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TopKBCELoss(nn.Module):
    """
    Weighted BCE focusing gradient on the top-K prediction boundary.

    4-tier weight matrix per (sample, student) element:
      actual=1              → pos_weight   (all positive students, whether in top-K or not)
      top-K predicted & =0  → fp_weight    (false positives: penalise wrong top-K)
      rest (TN, easy neg.)  → tn_weight    (suppress easy negatives)

    Effect: the model is told to maximise probability for actual selected students
    AND specifically penalise whichever K students it is currently ranking too high
    when they are not selected.  Easy true-negatives get nearly zero gradient.
    """
    def __init__(self, k=6, pos_weight=5.0, fp_weight=2.0, tn_weight=0.1,
                 label_smoothing=0.05):
        super().__init__()
        self.k = k; self.pw = pos_weight; self.fpw = fp_weight
        self.tnw = tn_weight; self.ls = label_smoothing

    def forward(self, p, y):
        if self.ls > 0:
            y_smooth = y * (1 - self.ls) + self.ls * 0.5
        else:
            y_smooth = y

        # Identify predicted top-K (hard, not differentiable; weight only)
        _, top_k_idx = torch.topk(p.detach(), self.k, dim=1)   # (B, K)
        top_k_mask   = torch.zeros_like(p, dtype=torch.bool)
        top_k_mask.scatter_(1, top_k_idx, True)

        actual_pos = y > 0.5   # (B, N)

        # Weight matrix
        w = torch.full_like(p, self.tnw)
        w[actual_pos]              = self.pw    # all positives: high weight
        w[top_k_mask & ~actual_pos]= self.fpw   # FP in top-K: penalise

        p_c = p.clamp(1e-7, 1 - 1e-7)
        bce = -(y_smooth * p_c.log() + (1 - y_smooth) * (1 - p_c).log())
        return (w * bce).mean()


class SoftmaxKHotLoss(nn.Module):
    """
    Multinomial cross-entropy: treats selection as one draw over N students.

    p_pred   = softmax(logits)          logits = logit(sigmoid_output)
    p_target = y / K  (normalised k-hot, optionally label-smoothed)
    Loss     = −Σ p_target · log(p_pred)

    The model must compete across ALL students simultaneously.  This is
    fundamentally different from per-student BCE: a student with higher score
    necessarily means another student has lower relative probability.
    """
    def __init__(self, temperature=1.0, label_smoothing=0.05):
        super().__init__()
        self.tau = temperature; self.ls = label_smoothing

    def forward(self, p, y):
        # Recover logits (invertible since sigmoid is monotone and bounded away from 0/1)
        logits = torch.logit(p.clamp(1e-6, 1 - 1e-6)) / self.tau

        # Normalised k-hot target distribution
        k       = y.sum(dim=1, keepdim=True).clamp(min=1)
        p_target = y / k   # (B, N)

        # Label smoothing over all N students
        if self.ls > 0:
            smooth   = self.ls / p.shape[1]
            p_target = p_target * (1 - self.ls) + smooth

        log_p_pred = F.log_softmax(logits, dim=-1)
        return -(p_target * log_p_pred).sum(dim=-1).mean()


class ListNetLoss(nn.Module):
    """
    ListNet listwise ranking loss.

    Both predicted scores and labels are converted to probability distributions
    via softmax, then we minimise their KL divergence (= cross-entropy).

    p_pred   = softmax(logits / τ)               τ=score_temp
    p_target = softmax(y_khot · label_scale)     label_scale makes positives sharp

    Setting label_scale=10: exp(10)≈22026 vs exp(0)=1 → positives dominate softmax
    → p_target effectively ≈ normalised k-hot.

    ListNet considers the ENTIRE ordering in one forward pass; unlike pairwise
    losses it does not decouple individual pairs.
    """
    def __init__(self, score_temp=1.0, label_scale=10.0):
        super().__init__()
        self.tau = score_temp; self.ls = label_scale

    def forward(self, p, y):
        logits   = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        p_target = F.softmax(y * self.ls, dim=-1)            # (B, N)
        log_pred = F.log_softmax(logits / self.tau, dim=-1)  # (B, N)
        return -(p_target * log_pred).sum(dim=-1).mean()


class LambdaRankLoss(nn.Module):
    """
    Simplified LambdaRank: pairwise ranking loss with NDCG-delta weighting.

    For each (positive i, negative j) pair in a sample:
      L_ij = log(1 + exp(−σ·(score_i − score_j))) · |ΔNDCG_ij|

    ΔNDCG_ij = |1/log₂(rank_i+1) − 1/log₂(rank_j+1)|
      Rank computed from current predicted scores (detached so only weights,
      not ranks themselves, flow gradients).

    Fully vectorised implementation:
      score_diff  (B, N, N) — score_i − score_j
      delta_ndcg  (B, N, N) — |gain_i − gain_j|
      pair_mask   (B, N, N) — 1 where (i=positive, j=negative)
    Total cost: 55×55×B float32 = ~96 K elements for B=32, negligible memory.
    """
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, p, y):
        scores = torch.logit(p.clamp(1e-6, 1 - 1e-6))   # (B, N)
        B, N   = scores.shape
        dev    = scores.device

        # ── Rank-based NDCG gains (detached: ranks only weight gradients) ──
        with torch.no_grad():
            _, sorted_idx = torch.sort(scores, dim=1, descending=True)  # (B, N)
            ranks         = torch.zeros_like(scores)
            ranks.scatter_(1, sorted_idx,
                           torch.arange(1, N + 1, dtype=scores.dtype,
                                        device=dev).unsqueeze(0).expand(B, -1))
        gains = 1.0 / torch.log2(ranks + 1.0)   # (B, N) NDCG position discount

        # ── Pairwise quantities (B, N, N) ──────────────────────────────────
        # [b, i, j] = score_b_i − score_b_j
        score_diff  = scores.unsqueeze(2) - scores.unsqueeze(1)   # (B, N, N)
        # |gain_i − gain_j|
        delta_ndcg  = (gains.unsqueeze(2) - gains.unsqueeze(1)).abs()  # (B, N, N)
        # pair_mask[b, i, j] = 1 if y[b,i]=1 and y[b,j]=0
        pos_mask    = (y > 0.5).float()  # (B, N)
        neg_mask    = (y < 0.5).float()  # (B, N)
        pair_mask   = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)  # (B, N, N)

        # ── LambdaRank pairwise logistic loss ──────────────────────────────
        # log(1 + exp(−σ · (score_i − score_j)))  for positive i, negative j
        pair_loss = torch.log1p(torch.exp(-self.sigma * score_diff)) \
                    * delta_ndcg * pair_mask

        n_pairs = pair_mask.sum().clamp(min=1)
        return pair_loss.sum() / n_pairs


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class HybridDataset(Dataset):
    def __init__(self, enriched, binary, seq_len=14):
        self.features = enriched; self.targets = binary; self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(model, train_loader, val_loader, criterion, device,
               lr=0.0005, epochs=50, patience=15) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf'); best_state = None; no_imp = 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss; best_state = model.state_dict(); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop @ epoch {epoch + 1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}  val_loss={val_loss:.5f}")

    model.load_state_dict(best_state)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(model_state, enriched, binary, split, device, n_features,
                   criterion, label,
                   alpha, n_groups, temperature, online_iters, replay_size
                   ) -> tuple[float, np.ndarray]:
    seq_len = CONFIG['sequence_length']
    N = CONFIG['n_students']; K = CONFIG['k_students']
    n_sim = min(CONFIG['n_simulation_days'], len(binary) - split)

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=N,
        hidden_size=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(model_state)

    markov = FactoredMarkovChain(n_students=N, smoothing=1.0)
    markov.fit(binary[:split])

    opt    = optim.Adam(model.parameters(), lr=0.0003)
    replay: list = []
    accs   = []
    t0     = time.perf_counter()

    for test_idx in range(n_sim):
        actual_idx = split + test_idx
        if actual_idx < seq_len:
            continue

        enriched_seq = enriched[actual_idx - seq_len: actual_idx]
        binary_seq   = binary  [actual_idx - seq_len: actual_idx]

        # LSTM probabilities
        model.eval()
        with torch.no_grad():
            X_t       = torch.FloatTensor(enriched_seq).unsqueeze(0).to(device)
            lstm_prob = model(X_t).cpu().numpy()[0]
        lstm_norm = lstm_prob / (lstm_prob.sum() + 1e-9)

        # Markov probabilities
        markov_prob = markov.predict_probabilities(binary_seq)

        # Blend
        hyb = (alpha * lstm_norm + (1 - alpha) * markov_prob
               if 0 < alpha < 1 else (lstm_norm if alpha == 1 else markov_prob))
        hyb = hyb / (hyb.sum() + 1e-9)

        # Temperature-scaled group generation
        p = np.clip(hyb, 1e-9, None) ** (1.0 / temperature)
        p /= p.sum()
        groups: list[list[int]] = []
        for _ in range(n_groups * 4):
            g = sorted(np.random.choice(N, K, replace=False, p=p).tolist())
            if g not in groups:
                groups.append(g)
            if len(groups) >= n_groups:
                break
        if not groups:
            groups.append(sorted(np.argsort(hyb)[-K:].tolist()))
        groups = groups[:n_groups]

        actual = set(np.where(binary[actual_idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K
        accs.append(best)

        # Online update
        replay.append((enriched_seq.copy(), binary[actual_idx].copy()))
        if actual_idx + 1 < len(binary) and actual_idx + 1 >= seq_len:
            replay.append((
                enriched[actual_idx + 1 - seq_len: actual_idx + 1].copy(),
                binary[actual_idx + 1].copy()
            ))
        if len(replay) >= 5:
            recent  = replay[-replay_size:]
            X_b = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            y_b = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            model.train()
            for _ in range(online_iters):
                opt.zero_grad()
                criterion(model(X_b), y_b).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if actual_idx >= 2:
            markov.update(binary[actual_idx],
                          binary[actual_idx - 1],
                          binary[actual_idx - 2])

    arr     = np.array(accs) * 100.0
    elapsed = time.perf_counter() - t0
    dist    = {k: int((arr == k / K * 100).sum()) for k in range(K + 1)}
    nz      = {k: v for k, v in dist.items() if v > 0}
    print(f"\n  -- {label} --")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))
    print(f"  Sim time: {elapsed:.2f}s")
    return float(arr.mean()), arr


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("HYBRID LSTM+MARKOV — LOSS FUNCTION COMPARISON".center(74))
    print("Focal  |  Top-k BCE  |  Softmax k-Hot CE  |  ListNet  |  LambdaRank".center(74))
    print("=" * 74)

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    print(f"  Fixed hyperparams: α={BEST['alpha']}, ng={BEST['n_groups']}, "
          f"T={BEST['temperature']}, iters={BEST['online_iters']}, "
          f"replay={BEST['replay_size']}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading data...")
    df     = pd.read_excel(CONFIG['excel_path'])
    df     = df.sort_values('Days').reset_index(drop=True)
    N      = CONFIG['n_students']
    binary = np.zeros((len(df), N), dtype=np.float32)
    for idx, row in df.iterrows():
        for col in [c for c in df.columns if c != 'Days']:
            sid = row[col]
            if pd.notna(sid) and 1 <= int(sid) <= N:
                binary[idx, int(sid) - 1] = 1.0

    enriched   = build_enriched_features(binary)
    n_features = enriched.shape[1]
    split      = int(len(binary) * CONFIG['train_ratio'])
    seq_len    = CONFIG['sequence_length']

    train_ds = HybridDataset(enriched[:split], binary[:split], seq_len)
    val_ds   = HybridDataset(enriched[split:], binary[split:], seq_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test"
          f" | n_features={n_features}\n")

    # ── Loss function definitions ─────────────────────────────────────────────
    loss_configs: list[tuple[str, nn.Module, str]] = [
        (
            'Focal (baseline)',
            FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05),
            'FL(p_t)=−α(1−p_t)^γ·log(p_t)  α=0.25 γ=2.0 ls=0.05'
        ),
        (
            'Top-k BCE',
            TopKBCELoss(k=6, pos_weight=5.0, fp_weight=2.0, tn_weight=0.1,
                        label_smoothing=0.05),
            'Weighted BCE: TP/FN→w=5  FP-in-top-K→w=2  TN→w=0.1  ls=0.05'
        ),
        (
            'Softmax k-Hot CE',
            SoftmaxKHotLoss(temperature=1.0, label_smoothing=0.05),
            'CE(softmax(logits), y/K)  multinomial over all 55 students'
        ),
        (
            'ListNet',
            ListNetLoss(score_temp=1.0, label_scale=10.0),
            'CE(softmax(logits), softmax(y·10))  listwise ranking'
        ),
        (
            'LambdaRank',
            LambdaRankLoss(sigma=1.0),
            'Σ log(1+exp(−σ·Δscore_ij))·|ΔNDCG_ij|  pairwise+NDCG weight'
        ),
    ]

    results: dict[str, np.ndarray] = {}
    prev_best = max(BENCHMARKS.values())

    for loss_name, criterion, description in loss_configs:
        print("=" * 74)
        print(f"  TRAINING WITH: {loss_name}")
        print(f"  Formula: {description}")
        print("=" * 74)

        # Fresh model for each loss
        model = EnhancedLSTMPredictor(
            n_features=n_features, n_students=N,
            hidden_size=128, num_layers=2, dropout=0.3).to(device)
        criterion = criterion.to(device)

        t_train = time.perf_counter()
        train_lstm(model, train_loader, val_loader, criterion, device,
                   lr=0.0005,
                   epochs=CONFIG['epochs'],
                   patience=CONFIG['early_stopping_patience'])
        print(f"  Training done in {time.perf_counter()-t_train:.1f}s")

        model_state = model.state_dict()

        # 100-day simulation with best hyperparams
        print(f"\n  Simulating 100 days (α={BEST['alpha']}, ng={BEST['n_groups']}, "
              f"T={BEST['temperature']}, iters={BEST['online_iters']}, "
              f"replay={BEST['replay_size']})...")
        avg, arr = run_simulation(
            model_state, enriched, binary, split, device, n_features,
            criterion=criterion,
            label=loss_name,
            **BEST
        )
        results[loss_name] = arr
        flag = '  ★ NEW BEST' if avg > prev_best else ''
        print(f"  → {avg:.2f}%{flag}")
        print()

    # ── Full leaderboard ──────────────────────────────────────────────────────
    print("=" * 74)
    print("  FULL LEADERBOARD")
    print("=" * 74)
    print(f"\n  {'Model':<32}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}  {'Std':>7}")
    print(f"  {'─'*32}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<32}  {v:>8.2f}%  {'—':>8}  {'—':>8}  {'—':>7}  [prev]")
    print(f"  {'─'*32}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for n, a in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        flag = '  ★ NEW BEST' if a.mean() > prev_best else ''
        print(f"  {n:<32}  {a.mean():>8.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    # ── Loss comparison table ─────────────────────────────────────────────────
    print(f"\n{'─'*74}")
    print("  LOSS COMPARISON  (relative to Focal baseline)")
    print("─" * 74)
    baseline_avg = results.get('Focal (baseline)', np.array([32.17])).mean()
    print(f"\n  {'Loss':<22}  {'Avg':>8}  {'ΔAvg':>8}  {'Std':>8}  "
          f"{'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for n, a in sorted(results.items(), key=lambda x: x[1].mean(), reverse=True):
        d = a.mean() - baseline_avg
        print(f"  {n:<22}  {a.mean():>8.2f}%  {d:>+8.2f}%  {a.std():>8.4f}  "
              f"{a.max():>7.2f}%  {a.min():>7.2f}%")

    # ── Design insights ───────────────────────────────────────────────────────
    print(f"\n{'─'*74}")
    print("  LOSS DESIGN NOTES")
    print("─" * 74)
    notes = {
        'Focal (baseline)': [
            'Per-student BCE with down-weighting of easy negatives.',
            'Good for 6/55 class imbalance; γ=2 focuses on hard examples.',
        ],
        'Top-k BCE': [
            'Concentrates gradient on the K-th rank boundary.',
            'Suppresses easy-TN gradient (tn_weight=0.1 vs fp_weight=2).',
            'Effective when model already ranks correctly; may hurt early training.',
        ],
        'Softmax k-Hot CE': [
            'Models selection as a competition (multinomial, not independent Bernoulli).',
            'Increasing one student\'s probability must decrease others → cleaner ranking.',
            'Sensitive to K being correct; label smoothing stabilises early training.',
        ],
        'ListNet': [
            'Listwise: considers the full ordering in one forward pass.',
            'label_scale=10 makes softmax(y·10) ≈ normalised k-hot.',
            'Gradient flows through all (pos, neg) rank positions simultaneously.',
        ],
        'LambdaRank': [
            'Pairwise: directly optimises NDCG@K rank ordering.',
            'ΔNDCG weights mean swaps near the K-cutoff get largest gradients.',
            'Fully vectorised: (B,N,N) tensors, no Python loops.',
        ],
    }
    for loss_name, points in notes.items():
        res_line = (f"  → {results[loss_name].mean():.2f}%  "
                    f"Std={results[loss_name].std():.2f}"
                    if loss_name in results else '')
        print(f"\n  {loss_name}{res_line}")
        for pt in points:
            print(f"    • {pt}")

    best_loss = max(results, key=lambda x: results[x].mean())
    print(f"\n{'='*74}")
    print(f"  BEST LOSS: {best_loss}  →  {results[best_loss].mean():.2f}%")
    print(f"{'='*74}\n")


if __name__ == '__main__':
    main()
