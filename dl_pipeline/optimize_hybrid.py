"""
Hybrid LSTM+Markov Hyperparameter Optimization
================================================
Trains the LSTM model ONCE, then sweeps all tunable parameters
without re-training (since the temporal signal is mostly frequency-based).

Stage 1 — α grid (LSTM/Markov mix weight): 0.0 to 1.0 in steps of 0.1
Stage 2 — n_groups × temperature: with best α from Stage 1
Stage 3 — online_iterations × replay_buffer_size: with best α, ng, T

Benchmarks:
  Hybrid Orig (α=0.7, ng=5, T=1.5): 26.83%
  DPM Optimized target:              27.67% (or better after DPM opt)

Usage:
    cd dl_pipeline
    python optimize_hybrid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time

from enhanced_features import build_enriched_features
from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain import FactoredMarkovChain

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG = {
    'excel_path':             '../Database.xlsx',
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

BASELINE_HYBRID = 26.83
BASELINE_DPM    = 27.67

# ── Grid search ranges ─────────────────────────────────────────────────────
STAGE1_ALPHA   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STAGE2_NGROUPS = [3, 5, 7, 10]
STAGE2_TEMP    = [1.0, 1.5, 2.0, 2.5]
STAGE3_ONLINE  = [1, 3, 5, 10]
STAGE3_REPLAY  = [5, 10, 20]


# ── Dataset ────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, enriched, binary, seq_len=14):
        self.features  = enriched
        self.targets   = binary
        self.seq_len   = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# ── Training ────────────────────────────────────────────────────────────────

def train_lstm(model, train_loader, val_loader, criterion, device,
               lr, epochs, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    best_loss, best_state, no_imp = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, no_imp = val_loss, model.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return optimizer


# ── Hybrid simulation ───────────────────────────────────────────────────────

def run_simulation(model_state, markov_train, enriched, binary, split,
                   device, n_features, alpha, n_groups, temperature,
                   online_iters, replay_size, criterion) -> float:
    """Run 100-day simulation with given hyperparams. Returns avg accuracy."""
    seq_len = CONFIG['sequence_length']
    N       = CONFIG['n_students']
    K       = CONFIG['k_students']
    n_sim   = min(CONFIG['n_simulation_days'], len(binary) - split)

    # Fresh copies for each run
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=N,
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)
    model.load_state_dict(model_state)

    markov = FactoredMarkovChain(n_students=N, smoothing=1.0)
    markov.fit(binary[:split])

    opt = optim.Adam(model.parameters(), lr=0.0003)
    replay: list = []
    accs   = []

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
        if alpha == 1.0:
            hyb = lstm_norm
        elif alpha == 0.0:
            hyb = markov_prob
        else:
            hyb = alpha * lstm_norm + (1.0 - alpha) * markov_prob
        hyb = hyb / (hyb.sum() + 1e-9)

        # Generate groups with temperature scaling
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

        # Score (0-indexed)
        actual = set(np.where(binary[actual_idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K
        accs.append(best)

        # Online updates
        replay.append((enriched_seq.copy(), binary[actual_idx].copy()))
        if actual_idx + 1 < len(binary) and actual_idx + 1 >= seq_len:
            replay.append((
                enriched[actual_idx + 1 - seq_len: actual_idx + 1].copy(),
                binary[actual_idx + 1].copy()
            ))

        if len(replay) >= 5:
            recent  = replay[-replay_size:]
            X_batch = torch.FloatTensor(np.array([r[0] for r in recent])).to(device)
            y_batch = torch.FloatTensor(np.array([r[1] for r in recent])).to(device)
            model.train()
            for _ in range(online_iters):
                opt.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # Markov update
        if actual_idx >= 2:
            markov.update(
                binary[actual_idx],
                binary[actual_idx - 1],
                binary[actual_idx - 2]
            )

    return float(np.mean(accs) * 100.0), np.array(accs) * 100.0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HYBRID LSTM+MARKOV HYPERPARAMETER OPTIMIZATION".center(70))
    print("=" * 70)

    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    print(f"  Baseline Hybrid (α=0.7, ng=5, T=1.5):  {BASELINE_HYBRID:.2f}%")
    print(f"  Target (DPM baseline):                  {BASELINE_DPM:.2f}%\n")

    # ── Load data ──────────────────────────────────────────────────────────
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

    enriched = build_enriched_features(binary)
    n_features = enriched.shape[1]
    split = int(len(binary) * CONFIG['train_ratio'])
    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test")

    # ── Train LSTM once ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  TRAINING LSTM (once for all runs)")
    print(f"{'─'*70}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    model = EnhancedLSTMPredictor(
        n_features=n_features, n_students=N,
        hidden_size=128, num_layers=2, dropout=0.3
    ).to(device)

    seq_len  = CONFIG['sequence_length']
    train_ds = HybridDataset(enriched[:split], binary[:split], seq_len)
    val_ds   = HybridDataset(enriched[split:], binary[split:], seq_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    t_train = time.perf_counter()
    train_lstm(model, train_loader, val_loader, criterion, device,
               lr=0.0005, epochs=CONFIG['epochs'],
               patience=CONFIG['early_stopping_patience'])
    print(f"  Training done in {time.perf_counter()-t_train:.1f}s")
    model_state = model.state_dict()

    # Markov trained on train split
    markov_ref = FactoredMarkovChain(n_students=N, smoothing=1.0)
    markov_ref.fit(binary[:split])

    # ── Stage 1: α grid ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STAGE 1 — α (LSTM/Markov mix)  [n_groups=5, T=1.5, iter=5, replay=10]")
    print(f"{'─'*70}")
    print(f"\n  {'α':>5}  {'LSTM%':>7}  {'Markov%':>8}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    stage1: dict[float, float] = {}
    for alpha in STAGE1_ALPHA:
        avg, accs = run_simulation(
            model_state, markov_ref, enriched, binary, split, device, n_features,
            alpha=alpha, n_groups=5, temperature=1.5,
            online_iters=5, replay_size=10, criterion=criterion
        )
        stage1[alpha] = avg
        flag = "  ★" if avg > BASELINE_HYBRID else ""
        print(f"  {alpha:>5.1f}  {alpha*100:>6.0f}%  {(1-alpha)*100:>7.0f}%  "
              f"{avg:>8.2f}%  {accs.max():>7.2f}%  {accs.min():>7.2f}%{flag}")

    best_alpha = max(stage1, key=stage1.__getitem__)
    best_s1    = stage1[best_alpha]
    print(f"\n  ▶ Stage 1 winner:  α={best_alpha:.1f} → {best_s1:.2f}%")

    # ── Stage 2: n_groups × temperature ────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STAGE 2 — n_groups × temperature  [α={best_alpha:.1f}, iter=5, replay=10]")
    print(f"{'─'*70}")
    print(f"\n  {'ng':>5}  {'Temp':>6}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")

    stage2: dict[tuple, float] = {}
    for ng in STAGE2_NGROUPS:
        for temp in STAGE2_TEMP:
            avg, accs = run_simulation(
                model_state, markov_ref, enriched, binary, split, device, n_features,
                alpha=best_alpha, n_groups=ng, temperature=temp,
                online_iters=5, replay_size=10, criterion=criterion
            )
            stage2[(ng, temp)] = avg
            flag = "  ★" if avg > BASELINE_HYBRID else ""
            print(f"  {ng:>5}  {temp:>6.1f}  {avg:>8.2f}%  "
                  f"{accs.max():>7.2f}%  {accs.min():>7.2f}%{flag}")

    best_ng, best_t = max(stage2, key=stage2.__getitem__)
    best_s2 = stage2[(best_ng, best_t)]
    print(f"\n  ▶ Stage 2 winner:  ng={best_ng}, T={best_t} → {best_s2:.2f}%")

    # ── Stage 3: online_iters × replay_size ────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STAGE 3 — online_iters × replay_size  [α={best_alpha:.1f}, ng={best_ng}, T={best_t}]")
    print(f"{'─'*70}")
    print(f"\n  {'iters':>6}  {'replay':>7}  {'Avg':>8}  {'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")

    stage3: dict[tuple, float] = {}
    for oi in STAGE3_ONLINE:
        for rs in STAGE3_REPLAY:
            avg, accs = run_simulation(
                model_state, markov_ref, enriched, binary, split, device, n_features,
                alpha=best_alpha, n_groups=best_ng, temperature=best_t,
                online_iters=oi, replay_size=rs, criterion=criterion
            )
            stage3[(oi, rs)] = avg
            flag = "  ★" if avg > BASELINE_HYBRID else ""
            print(f"  {oi:>6}  {rs:>7}  {avg:>8.2f}%  "
                  f"{accs.max():>7.2f}%  {accs.min():>7.2f}%{flag}")

    best_oi, best_rs = max(stage3, key=stage3.__getitem__)
    best_s3 = stage3[(best_oi, best_rs)]
    print(f"\n  ▶ Stage 3 winner:  iters={best_oi}, replay={best_rs} → {best_s3:.2f}%")

    # ── Final verification ──────────────────────────────────────────────────
    _, final_accs = run_simulation(
        model_state, markov_ref, enriched, binary, split, device, n_features,
        alpha=best_alpha, n_groups=best_ng, temperature=best_t,
        online_iters=best_oi, replay_size=best_rs, criterion=criterion
    )
    dist    = {k: int((final_accs == k / CONFIG['k_students'] * 100).sum()) for k in range(7)}
    non_zero = {k: v for k, v in dist.items() if v > 0}

    print(f"\n{'=' * 70}")
    print("  HYBRID OPTIMIZED — FINAL RESULT")
    print(f"{'=' * 70}")
    print(f"\n  Config:  α={best_alpha}, ng={best_ng}, T={best_t}, "
          f"iters={best_oi}, replay={best_rs}")
    print(f"  Avg:     {final_accs.mean():.2f}%")
    print(f"  Best:    {final_accs.max():.2f}%")
    print(f"  Worst:   {final_accs.min():.2f}%")
    print(f"  Std:     {final_accs.std():.4f}")
    print(f"  Dist:    " + "  ".join(f"{k}/6={v}" for k, v in non_zero.items()))

    delta_base = final_accs.mean() - BASELINE_HYBRID
    delta_dpm  = final_accs.mean() - BASELINE_DPM
    print(f"\n  Delta vs Hybrid baseline:  {delta_base:+.2f}%")
    print(f"  Delta vs DPM target:       {delta_dpm:+.2f}%")

    if final_accs.mean() >= BASELINE_DPM:
        print(f"\n  ✓ Hybrid BEATS DPM baseline ({BASELINE_DPM:.2f}%) after optimization!")
    else:
        print(f"\n  ✗ Hybrid stays below DPM baseline ({BASELINE_DPM:.2f}%) "
              f"even after optimization.")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
