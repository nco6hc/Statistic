"""
Advanced Models: CRF · Energy-Based Model · Graph Neural Network
================================================================

Three structurally distinct models, all predicting the 6-of-55 student
selection task.  Each targets a different weakness of the Hybrid M1/M2:

────────────────────────────────────────────────────────────────────────
MODEL 1 — Neural CRF  (Conditional Random Field)
────────────────────────────────────────────────────────────────────────
Motivation
  M1 predicts 55 independent Bernoulli probabilities and ignores the
  constraint that EXACTLY 6 students are chosen per day.  A CRF encodes
  a joint distribution over the whole 55-bit label vector.

Architecture
  Encoder : BiLSTM(220→128*2) + Attention   (same as M1)
  Unary   : Linear(256→55)                  per-student potential
  Pairwise: Learned 55×55 compatibility matrix W (symmetric, zero diag)
             captures "A and B tend to be chosen together / apart"
  Energy  : E(y) = −Σ_s unary_s · y_s − Σ_{s<t} W_st · y_s · y_t
  Inference: top-K beam search over the energy landscape to find
             the 6-element subset y* = argmin E(y|x)
  Training : Contrastive: E(y_true) + margin < E(y_neg) for random neg.

Output
  Probability proxy = normalised −unary scores (for group sampling oracle)
  Hard selection    = beam-search 6-element argmin (for accuracy metric)

────────────────────────────────────────────────────────────────────────
MODEL 2 — Energy-Based Model  (EBM)
────────────────────────────────────────────────────────────────────────
Motivation
  Rather than forcing a distribution, an EBM learns a scalar score
  E(x, y) such that low energy ↔ high compatibility with the observation.
  Training uses Noise-Contrastive Estimation (NCE): real selections must
  have lower energy than random corruptions.

Architecture
  Context encoder : BiLSTM(220→128) + mean-pool → context vector c (128)
  Selection encoder: Linear(55→64) → selection vector s (64)
  Joint scorer    : MLP([c ⊕ s] → 128 → 64 → 1) → scalar energy
  NCE training    : for each real y_t, sample 5 corruptions (random 6-of-55)
                    Loss = −log σ(E_neg − E_real)   [higher E_neg = better]
  Inference       : score all C(55,6)~202M candidates is intractable;
                    use greedy top-6 of per-student marginal energies
                    (fast proxy; exact for uncorrelated students)

────────────────────────────────────────────────────────────────────────
MODEL 3 — Graph Neural Network  (GNN)
────────────────────────────────────────────────────────────────────────
Motivation
  M1 treats each student independently (per-dimension sigmoid output).
  A GNN explicitly passes messages over the co-occurrence graph so that
  the probability of student A is informed by its graph-neighbours' state.

Architecture
  Graph   : Static undirected graph built from training co-occurrence.
            Node i — student i.  Edge (i,j) weight ∝ co-occurrence freq.
            Top-K edges per node (K=10) to keep it sparse.
  Temporal encoder: BiLSTM(220→128*2)+Attention → context c_t (256)
  Node init: h_i^(0) = Linear(256 + 4 × 1)(node_feat_i | c_t_slice_i)
            where node_feat_i = [freq7_i, freq14_i, recency_i, raw_i]
  GCN layers (2×):
    h_i^(l+1) = ReLU(BN( Σ_{j∈N(i)} a_ij · W^(l) · h_j^(l) + b^(l) ))
    a_ij = edge weight / sqrt(deg_i · deg_j)  (normalised Laplacian)
  Output head: h_i^(2) → Linear(64→1) → Sigmoid
  Training   : FocalLoss on all 55 student predictions

────────────────────────────────────────────────────────────────────────
Usage:
    cd dl_pipeline
    python advanced_models/run_crf_ebm_gnn.py
"""

from __future__ import annotations
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time, warnings
warnings.filterwarnings('ignore')

from enhanced_features import build_enriched_features
from enhanced_models    import FocalLoss

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = str(_DL_DIR.parent / 'Database.xlsx')
N_S         = 55
K_SEL       = 6
TRAIN_RATIO = 0.9
SIM_DAYS    = 100
SEQ_LEN     = 14
SEED        = 42
EPOCHS      = 60
PATIENCE    = 15
LR          = 5e-4
BATCH       = 32

# Oracle group sampling (same as Hybrid benchmark)
N_GROUPS    = 10
TEMPERATURE = 2.0

# GNN graph params
GNN_TOP_K_EDGES = 10     # keep only top-10 co-occurrence neighbours per node
GNN_HIDDEN      = 64
GNN_LAYERS      = 2

# CRF beam search width
BEAM_K = 20

# EBM noise samples per real example
EBM_NEG_SAMPLES = 5

BENCHMARKS = {
    'M1  Hybrid LSTM+Markov (α=0.9)': 32.17,
    'M2  TD-HBB λ=0.99             ': 32.17,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df      = pd.read_excel(EXCEL_PATH)
    df      = df.sort_values('Days').reset_index(drop=True)
    id_cols = [c for c in df.columns if c.startswith('Student ID')]
    T       = len(df)
    binary  = np.zeros((T, N_S), dtype=np.float32)
    avg_pos = np.zeros(N_S, dtype=np.float64)
    pos_cnt = np.zeros(N_S, dtype=np.float64)
    for row_idx, row in df.iterrows():
        for pi, col in enumerate(id_cols):
            sid = int(row[col]) - 1
            if 0 <= sid < N_S:
                binary[row_idx, sid] = 1.0
                avg_pos[sid] += pi + 1
                pos_cnt[sid] += 1.0
    avg_pos = np.where(pos_cnt > 0, avg_pos / pos_cnt, 3.5)
    return binary, avg_pos

# ─────────────────────────────────────────────────────────────────────────────
# Shared sequence dataset
# ─────────────────────────────────────────────────────────────────────────────

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, enriched, binary, seq_len=SEQ_LEN):
        self.enriched = enriched; self.binary = binary; self.sl = seq_len

    def __len__(self):  return len(self.enriched) - self.sl

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.enriched[idx: idx + self.sl]),
                torch.FloatTensor(self.binary  [idx + self.sl]))


# ─────────────────────────────────────────────────────────────────────────────
# Shared temporal encoder (BiLSTM + Attention) — used by all 3 models
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """BiLSTM + attention → fixed-size context vector."""

    def __init__(self, n_features=220, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                  nn.Dropout(0.15))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                            bidirectional=True,
                            dropout=0.3)
        self.attn_w = nn.Linear(hidden * 2, 1, bias=False)

    def forward(self, x):   # x: (B, T, F)
        x  = self.proj(x)   # (B, T, H)
        o, _ = self.lstm(x) # (B, T, 2H)
        a  = torch.softmax(self.attn_w(o), dim=1)   # (B, T, 1)
        c  = (o * a).sum(dim=1)                      # (B, 2H)
        return c             # (B, 256)

    @property
    def out_dim(self): return 256


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1  ─  Neural CRF
# ─────────────────────────────────────────────────────────────────────────────

class NeuralCRF(nn.Module):
    """
    Neural CRF for structured 6-of-55 subset prediction.

    Unary  potential: ψ_s(y_s=1) = encoder output projected to scalar
    Pairwise potential: φ_{st} = W[s,t] (learned 55×55 symmetric matrix)
    Energy: E(y|x) = -Σ_s ψ_s · y_s - Σ_{s<t} φ_st · y_s · y_t

    Training: max-margin with random negatives
      Loss = max(0, margin + E(y_real) - E(y_neg))

    Inference: beam-search — maintain top-BEAM_K partial subsets,
               extend greedily by the student that minimises marginal energy.
    """

    def __init__(self, n_features=220, hidden=128, n_students=N_S, beam_k=BEAM_K):
        super().__init__()
        self.N  = n_students
        self.bk = beam_k
        self.enc    = TemporalEncoder(n_features, hidden)
        self.unary  = nn.Linear(self.enc.out_dim, n_students)
        # Pairwise: symmetric, zero diagonal
        self._W_raw = nn.Parameter(torch.zeros(n_students, n_students))

    @property
    def W(self):
        W = (self._W_raw + self._W_raw.T) / 2.0
        return W - torch.diag(torch.diag(W))

    def energy(self, psi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        psi: (B, N_S) unary potentials
        y  : (B, N_S) binary labels (float)
        returns: (B,) energy scalar per example
        """
        unary_e   = -(psi * y).sum(dim=1)                        # (B,)
        pairwise_e = -(y @ self.W * y).sum(dim=1) * 0.5          # (B,)
        return unary_e + pairwise_e

    def forward(self, x):
        """Return unary potential logits (B, N_S)."""
        c = self.enc(x)
        return self.unary(c)  # raw scores before sigmoid

    def predict_probs(self, x):
        """Return normalised probability proxy for oracle sampling."""
        with torch.no_grad():
            psi = self.forward(x)
        probs = torch.sigmoid(psi).cpu().numpy()[0]
        return (probs / (probs.sum() + 1e-9)).astype(np.float32)

    def beam_predict(self, x):
        """
        Run beam search to find the 6-element subset with minimum energy.
        Returns: (N_S,) binary vector
        """
        with torch.no_grad():
            psi  = self.forward(x)[0]  # (N_S,)
            W    = self.W               # (N_S, N_S)

        N = self.N
        # Each beam state: (current_y_vec, current_energy, selected_set)
        # Start with empty selection
        beams = [(torch.zeros(N, device=psi.device), 0.0, set())]

        for step in range(K_SEL):
            new_beams = []
            for y_so_far, e_so_far, sel_set in beams:
                for s in range(N):
                    if s in sel_set:
                        continue
                    # Marginal energy decrease from adding student s
                    unary_contrib    = -psi[s].item()
                    pairwise_contrib = 0.0
                    if sel_set:
                        sel_idx = torch.tensor(list(sel_set), device=psi.device)
                        pairwise_contrib = -W[s, sel_idx].sum().item()
                    delta_e = unary_contrib + pairwise_contrib
                    new_beams.append((
                        None,
                        e_so_far + delta_e,
                        sel_set | {s}
                    ))
            # Keep top beam_k by lowest energy
            new_beams.sort(key=lambda b: b[1])
            beams = new_beams[:self.bk]
            # Rebuild y vectors lazily only at final step
            if step == K_SEL - 1:
                best_set = beams[0][2]
                y_out = np.zeros(N, dtype=np.float32)
                for s in best_set:
                    y_out[s] = 1.0
                return y_out

        return np.zeros(N, dtype=np.float32)


def train_crf(binary_train, enriched_train, device, tag='[CRF]'):
    n_features = enriched_train.shape[1]
    model = NeuralCRF(n_features=n_features, hidden=128, n_students=N_S).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)

    split_v = int(len(binary_train) * 0.9)
    tr_ds = SeqDataset(enriched_train[:split_v], binary_train[:split_v])
    vl_ds = SeqDataset(enriched_train[split_v:], binary_train[split_v:])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    rng   = np.random.default_rng(SEED)
    margin = 1.0
    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for X, y_real in tr_ld:
            X      = X.to(device)
            y_real = y_real.to(device)
            B      = y_real.shape[0]
            # Sample random negatives (random 6-of-55)
            neg_idx  = torch.stack([
                torch.tensor(rng.choice(N_S, K_SEL, replace=False),
                             dtype=torch.float32)
                for _ in range(B)
            ]).to(device)
            y_neg = torch.zeros(B, N_S, device=device)
            y_neg.scatter_(1, neg_idx.long(), 1.0)

            psi      = model(X)                             # (B, N_S)
            e_real   = model.energy(psi, y_real)            # (B,)
            e_neg    = model.energy(psi, y_neg)             # (B,)
            loss     = F.relu(margin + e_real - e_neg).mean()

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation: surrogate focal BCE on unary potentials
        model.eval()
        vl_loss = 0.0
        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
        with torch.no_grad():
            for X, y in vl_ld:
                logits = model(X.to(device))
                vl_loss += criterion(torch.sigmoid(logits), y.to(device)).item()
        vl_loss /= max(len(vl_ld), 1)
        sched.step(vl_loss)

        if vl_loss < best_loss:
            best_loss, best_st, no_imp = vl_loss, model.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    {tag} Early stop @ epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_loss={vl_loss:.5f}")

    print(f"    {tag} Training done in {time.perf_counter()-t0:.1f}s")
    model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2  ─  Energy-Based Model (EBM)
# ─────────────────────────────────────────────────────────────────────────────

class EnergyModel(nn.Module):
    """
    EBM: joint scorer E(context, selection) → scalar.

    context encoder: BiLSTM+Attention → 256-dim
    selection encoder: Linear(55→64) with ReLU
    joint MLP: [256 ⊕ 64] → 128 → 64 → 1

    Training: Noise-Contrastive Estimation (NCE).
      For each real selection y_real, draw EBM_NEG_SAMPLES random 6-of-55.
      Loss: -log σ(E_neg_k - E_real) summed over negatives.
      (Lower energy for real, higher for noise → correct ranking.)

    Inference: compute per-student marginal energy by forward-difference
      approximation — score the current top-6 unary candidates.
    """

    def __init__(self, n_features=220, hidden=128, sel_dim=64):
        super().__init__()
        self.enc      = TemporalEncoder(n_features, hidden)
        self.sel_enc  = nn.Sequential(
            nn.Linear(N_S, sel_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.scorer   = nn.Sequential(
            nn.Linear(self.enc.out_dim + sel_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def score(self, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """c: (B, 256), y: (B, N_S) → (B,) energy scalar (lower=better)."""
        s = self.sel_enc(y)           # (B, sel_dim)
        return self.scorer(torch.cat([c, s], dim=1)).squeeze(-1)  # (B,)

    def forward(self, x):
        """Returns context vector (B, 256)."""
        return self.enc(x)

    def predict_probs(self, x):
        """
        Greedy per-student marginal energy proxy.
        For each student s, score the selection {top-5 unary} ∪ {s}.
        Lower energy → higher selection probability.
        """
        with torch.no_grad():
            c = self.forward(x)   # (1, 256)
            # Unary energy proxy: score each singleton
            # To speed up: score e_i = E(c, e_i_vec) for one-hot e_i
            marginals = np.zeros(N_S, dtype=np.float64)
            for s in range(N_S):
                y_s = torch.zeros(1, N_S, device=c.device)
                y_s[0, s] = 1.0
                marginals[s] = -self.score(c, y_s).item()  # negate: lower E = better

        marginals = marginals - marginals.min()
        marginals = np.clip(marginals, 0, None)
        s = marginals.sum()
        if s < 1e-9:
            return np.ones(N_S, dtype=np.float32) / N_S
        return (marginals / s).astype(np.float32)


def train_ebm(binary_train, enriched_train, device, tag='[EBM]'):
    n_features = enriched_train.shape[1]
    model = EnergyModel(n_features=n_features).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)

    split_v = int(len(binary_train) * 0.9)
    tr_ds = SeqDataset(enriched_train[:split_v], binary_train[:split_v])
    vl_ds = SeqDataset(enriched_train[split_v:], binary_train[split_v:])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    rng = np.random.default_rng(SEED)
    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for X, y_real in tr_ld:
            X      = X.to(device)
            y_real = y_real.to(device)
            B      = y_real.shape[0]
            c      = model(X)          # (B, 256)

            e_real = model.score(c, y_real)   # (B,)

            # NCE: score EBM_NEG_SAMPLES negatives per example
            nce_loss = torch.zeros(B, device=device)
            for _ in range(EBM_NEG_SAMPLES):
                neg_idx = torch.tensor(
                    np.array([rng.choice(N_S, K_SEL, replace=False) for _ in range(B)]),
                    dtype=torch.long, device=device)
                y_neg = torch.zeros(B, N_S, device=device)
                y_neg.scatter_(1, neg_idx, 1.0)
                e_neg   = model.score(c, y_neg)   # (B,)
                # want e_real < e_neg  →  -log σ(e_neg - e_real)
                nce_loss = nce_loss + (-torch.log(torch.sigmoid(e_neg - e_real) + 1e-8))

            loss = nce_loss.mean() / EBM_NEG_SAMPLES
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation (same NCE objective)
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, y_real in vl_ld:
                X      = X.to(device)
                y_real = y_real.to(device)
                B      = y_real.shape[0]
                c      = model(X)
                e_real = model.score(c, y_real)
                batch_nce = torch.zeros(B, device=device)
                for _ in range(2):  # cheaper val
                    neg_idx = torch.tensor(
                        np.array([rng.choice(N_S, K_SEL, replace=False) for _ in range(B)]),
                        dtype=torch.long, device=device)
                    y_neg = torch.zeros(B, N_S, device=device)
                    y_neg.scatter_(1, neg_idx, 1.0)
                    e_neg = model.score(c, y_neg)
                    batch_nce += (-torch.log(torch.sigmoid(e_neg - e_real) + 1e-8))
                vl_loss += (batch_nce.mean() / 2).item()
        vl_loss /= max(len(vl_ld), 1)
        sched.step(vl_loss)

        if vl_loss < best_loss:
            best_loss, best_st, no_imp = vl_loss, model.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    {tag} Early stop @ epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_loss={vl_loss:.5f}")

    print(f"    {tag} Training done in {time.perf_counter()-t0:.1f}s")
    model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3  ─  Graph Neural Network (GNN)
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(binary_train: np.ndarray, top_k: int = GNN_TOP_K_EDGES,
                    device: str = 'cpu') -> torch.Tensor:
    """
    Build normalised adjacency matrix from co-occurrence on training data.
    Only keep top-K neighbours per node to stay sparse.
    Returns A: (N_S, N_S) normalised, on device.
    """
    cooc = np.zeros((N_S, N_S), dtype=np.float64)
    for t in range(len(binary_train)):
        sel = np.where(binary_train[t] == 1)[0]
        for i in sel:
            for j in sel:
                if i != j:
                    cooc[i, j] += 1.0

    # Zero out all but top-K neighbours
    A = cooc.copy()
    for i in range(N_S):
        row = A[i].copy()
        if row.max() < 1e-9:
            continue
        threshold = np.partition(row, -top_k)[-top_k] if top_k < N_S else 0.0
        A[i, row < threshold] = 0.0

    A = np.maximum(A, A.T)   # ensure symmetry after masking
    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    deg   = A.sum(axis=1)
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-9)), 0.0)
    A_norm = d_inv[:, None] * A * d_inv[None, :]
    # Add self-loops
    A_norm += np.eye(N_S) * 0.5
    return torch.FloatTensor(A_norm).to(device)


class GCNLayer(nn.Module):
    """Single GCN propagation layer: H' = σ(BN(A_norm · H · W))."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W  = nn.Linear(in_dim, out_dim, bias=True)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """H: (N_S, D), A: (N_S, N_S) → (N_S, out_dim)."""
        out = A @ self.W(H)           # (N_S, out_dim)
        out = self.bn(out)
        return F.relu(out)


class GNNPredictor(nn.Module):
    """
    Temporal encoder + GCN over student co-occurrence graph.

    Per timestep:
      1. Temporal context c ∈ R^256 from BiLSTM+Attention over enriched sequence
      2. Per-student node init: h_i^0 = [c_slice_i | node_raw_i] where
           c_slice_i = 4 enriched features for student i from the last timestep
           (raw, freq7, freq14, recency) → 4 dims
           global_ctx = c projected → N_S * node_h dims then reshape
      3. GCN_layers × message passing
      4. h_i^(L) → Linear(hidden→1) → Sigmoid

    Node feature per student: 4 scalar features (raw, freq7, freq14, recency)
    from the LAST timestep in the sequence window, plus a global context
    broadcast from the temporal encoder.
    """

    N_NODE_FEAT = 4   # raw, freq7, freq14, recency (one per student)

    def __init__(self, n_features=220, enc_hidden=128,
                 gcn_hidden=GNN_HIDDEN, n_layers=GNN_LAYERS):
        super().__init__()
        self.enc      = TemporalEncoder(n_features, enc_hidden)
        enc_out       = self.enc.out_dim   # 256

        # Project global context to per-node init features
        self.ctx_proj = nn.Linear(enc_out, gcn_hidden)  # (256 → gcn_hidden)

        # Per-node feature projection: [4 node feats + gcn_hidden ctx] → gcn_hidden
        self.node_proj = nn.Linear(self.N_NODE_FEAT + gcn_hidden, gcn_hidden)

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_hidden, gcn_hidden) for _ in range(n_layers)
        ])

        # Output
        self.out_head  = nn.Sequential(
            nn.Linear(gcn_hidden, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (B, SEQ_LEN, n_features)  — enriched feature sequence
        A : (N_S, N_S)                 — normalised adjacency (shared)
        returns: (B, N_S) probabilities
        """
        B = x.shape[0]
        # Global context from temporal encoder
        c = self.enc(x)                     # (B, 256)
        c_proj = self.ctx_proj(c)           # (B, gcn_hidden)

        # Per-student node features from last timestep
        # enriched layout: [raw(55), freq7(55), freq14(55), recency(55)]
        last = x[:, -1, :]                  # (B, n_features=220)
        # Split into 4 blocks of 55
        node_feats = last.view(B, 4, N_S).permute(0, 2, 1)  # (B, N_S, 4)

        # Broadcast global context to all nodes: (B, N_S, gcn_hidden)
        ctx_exp = c_proj.unsqueeze(1).expand(-1, N_S, -1)   # (B, N_S, gcn_hidden)

        # Concatenate: (B, N_S, 4 + gcn_hidden)
        h = torch.cat([node_feats, ctx_exp], dim=-1)
        h = F.relu(self.node_proj(h))   # (B, N_S, gcn_hidden)
        h = self.dropout(h)

        # GCN message passing — process each item in batch
        # A is shared across batch: (N_S, N_S)
        out_list = []
        for b in range(B):
            h_b = h[b]              # (N_S, gcn_hidden)
            for gcn in self.gcn_layers:
                h_b = gcn(h_b, A)   # (N_S, gcn_hidden)
            out_list.append(h_b)
        H_out = torch.stack(out_list, dim=0)   # (B, N_S, gcn_hidden)

        probs = self.out_head(H_out).squeeze(-1)  # (B, N_S)
        return probs


def train_gnn(binary_train, enriched_train, device, tag='[GNN]'):
    A          = build_adjacency(binary_train, top_k=GNN_TOP_K_EDGES, device=device)
    n_features = enriched_train.shape[1]
    model      = GNNPredictor(n_features=n_features).to(device)
    criterion  = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    opt        = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched      = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, patience=5)

    split_v = int(len(binary_train) * 0.9)
    tr_ds = SeqDataset(enriched_train[:split_v], binary_train[:split_v])
    vl_ds = SeqDataset(enriched_train[split_v:], binary_train[split_v:])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH, shuffle=False, drop_last=False)

    best_loss, best_st, no_imp = float('inf'), None, 0
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for X, y in tr_ld:
            X = X.to(device); y = y.to(device)
            preds = model(X, A)
            loss  = criterion(preds, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in vl_ld:
                vl += criterion(model(X.to(device), A), y.to(device)).item()
        vl /= max(len(vl_ld), 1)
        sched.step(vl)

        if vl < best_loss:
            best_loss, best_st, no_imp = vl, model.state_dict(), 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    {tag} Early stop @ epoch {epoch+1}")
                break
        if (epoch + 1) % 10 == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_loss={vl:.5f}")

    print(f"    {tag} Training done in {time.perf_counter()-t0:.1f}s")
    model.load_state_dict(best_st)
    return model, A


# ─────────────────────────────────────────────────────────────────────────────
# Oracle group sampling helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_groups(probs: np.ndarray, n_groups=N_GROUPS,
                  temperature=TEMPERATURE) -> list[list[int]]:
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1.0 / temperature)
    p  /= p.sum()
    groups: list[list[int]] = []
    for _ in range(n_groups * 6):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in groups:
            groups.append(g)
        if len(groups) >= n_groups:
            break
    if not groups:
        groups.append(sorted(np.argsort(probs)[-K_SEL:].tolist()))
    return groups[:n_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_crf(model: NeuralCRF, binary, enriched, split, device) -> np.ndarray:
    np.random.seed(SEED); n_sim = min(SIM_DAYS, len(binary) - split)
    accs = []
    model.eval()
    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN: accs.append(0.0); continue
        X = torch.FloatTensor(enriched[idx - SEQ_LEN: idx]).unsqueeze(0).to(device)
        probs  = model.predict_probs(X)
        groups = sample_groups(probs)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
    return np.array(accs) * 100.0


def simulate_ebm(model: EnergyModel, binary, enriched, split, device) -> np.ndarray:
    np.random.seed(SEED); n_sim = min(SIM_DAYS, len(binary) - split)
    accs = []
    model.eval()
    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN: accs.append(0.0); continue
        X = torch.FloatTensor(enriched[idx - SEQ_LEN: idx]).unsqueeze(0).to(device)
        probs  = model.predict_probs(X)
        groups = sample_groups(probs)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
    return np.array(accs) * 100.0


def simulate_gnn(model: GNNPredictor, A: torch.Tensor,
                 binary, enriched, split, device) -> np.ndarray:
    np.random.seed(SEED); n_sim = min(SIM_DAYS, len(binary) - split)
    accs = []
    model.eval()
    for i in range(n_sim):
        idx = split + i
        if idx < SEQ_LEN: accs.append(0.0); continue
        X = torch.FloatTensor(enriched[idx - SEQ_LEN: idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            probs_t = model(X, A).cpu().numpy()[0]
        probs  = np.clip(probs_t, 1e-9, None)
        probs /= probs.sum()
        groups = sample_groups(probs)
        actual = set(np.where(binary[idx] == 1)[0].tolist())
        best   = max(len(set(g) & actual) for g in groups) / K_SEL
        accs.append(best)
    return np.array(accs) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def print_result(name, arr):
    dist = {k: int((arr == k / K_SEL * 100).sum()) for k in range(K_SEL + 1)}
    nz   = {k: v for k, v in dist.items() if v > 0}
    print(f"\n  ── {name} ──")
    print(f"  Avg: {arr.mean():.2f}%  Best: {arr.max():.2f}%  "
          f"Worst: {arr.min():.2f}%  Std: {arr.std():.4f}")
    print(f"  Dist: " + "  ".join(f"{k}/6={v}" for k, v in nz.items()))


def main():
    print("=" * 76)
    print("ADVANCED MODELS: CRF · EBM · GNN".center(76))
    print("=" * 76)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n  Loading data...")
    binary, avg_pos = load_data()
    print("  Computing enriched features...")
    enriched   = build_enriched_features(binary)
    n_features = enriched.shape[1]
    split      = int(len(binary) * TRAIN_RATIO)
    print(f"  {len(binary)} days | {split} train | {len(binary)-split} test\n")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 1 — Neural CRF
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 76)
    print("  MODEL 1 — Neural CRF  (max-margin + beam search)")
    print("=" * 76)
    crf = train_crf(binary[:split], enriched[:split], device, tag='[CRF]')
    t0  = time.perf_counter()
    arr_crf = simulate_crf(crf, binary, enriched, split, device)
    print_result('Neural CRF', arr_crf)
    print(f"  Sim time: {time.perf_counter()-t0:.2f}s")
    results['Neural CRF'] = arr_crf

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 2 — Energy-Based Model
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  MODEL 2 — Energy-Based Model  (NCE training)")
    print("=" * 76)
    ebm = train_ebm(binary[:split], enriched[:split], device, tag='[EBM]')
    t0  = time.perf_counter()
    arr_ebm = simulate_ebm(ebm, binary, enriched, split, device)
    print_result('Energy-Based Model', arr_ebm)
    print(f"  Sim time: {time.perf_counter()-t0:.2f}s")
    results['Energy-Based Model'] = arr_ebm

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 3 — Graph Neural Network
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print(f"  MODEL 3 — GNN  (co-occurrence graph, {GNN_LAYERS}×GCN layers, "
          f"top-{GNN_TOP_K_EDGES} edges)")
    print("=" * 76)
    gnn, A = train_gnn(binary[:split], enriched[:split], device, tag='[GNN]')
    t0      = time.perf_counter()
    arr_gnn = simulate_gnn(gnn, A, binary, enriched, split, device)
    print_result('GNN (GCN)', arr_gnn)
    print(f"  Sim time: {time.perf_counter()-t0:.2f}s")
    results['GNN (GCN)'] = arr_gnn

    # ══════════════════════════════════════════════════════════════════════
    # LEADERBOARD
    # ══════════════════════════════════════════════════════════════════════
    prev_best = max(BENCHMARKS.values())
    print("\n" + "=" * 76)
    print("  FULL LEADERBOARD")
    print("=" * 76)
    print(f"\n  {'Model':<38}  {'Avg':>7}  {'Best':>7}  {'Worst':>7}  {'Std':>7}")
    print(f"  {'─'*38}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for n, v in BENCHMARKS.items():
        print(f"  {n:<38}  {v:>7.2f}%  {'—':>7}  {'—':>7}  {'—':>7}  [prev]")
    print(f"  {'─'*38}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    sorted_res = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    for n, a in sorted_res:
        flag = '  ★ NEW BEST' if a.mean() > prev_best else ''
        print(f"  {n:<38}  {a.mean():>7.2f}%  {a.max():>7.2f}%  "
              f"{a.min():>7.2f}%  {a.std():>7.4f}{flag}")

    print("\n" + "─" * 76)
    print("  NEW MODEL COMPARISON  (Δ vs M1 benchmark 32.17%)")
    print("─" * 76)
    print(f"\n  {'Model':<28}  {'Avg':>8}  {'ΔAvg':>7}  {'Std':>7}  "
          f"{'Best':>8}  {'Worst':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")
    for n, a in sorted_res:
        delta = a.mean() - prev_best
        print(f"  {n:<28}  {a.mean():>8.2f}%  {delta:>+7.2f}  "
              f"{a.std():>7.4f}  {a.max():>8.2f}%  {a.min():>8.2f}%")

    # ── Architecture summary ──────────────────────────────────────────────
    print("\n" + "─" * 76)
    print("  ARCHITECTURE NOTES")
    print("─" * 76)
    print(f"""
  Neural CRF
    • Encoder: BiLSTM(220→256) + Attention → unary potentials (55 scalars)
    • Pairwise: learned 55×55 symmetric compatibility matrix
    • Training: max-margin with random-negative contrasts (margin=1.0)
    • Inference: beam search (K={BEAM_K}) over 55-choose-6 energy landscape
    • Key idea: models "A+B co-selected" and "A−B mutually exclusive" jointly

  Energy-Based Model
    • Context: BiLSTM(220→256) + Attention → 256-dim context c
    • Selection encoder: 55 → 64 via MLP
    • Joint scorer: [c ⊕ s] → 128 → 64 → 1 scalar energy
    • Training: NCE with {EBM_NEG_SAMPLES} random negatives per real sample
    • Inference: per-student marginal energy (score each singleton, top-6)
    • Key idea: discriminative approach — learns "what makes a good group"

  GNN (Graph Convolutional Network)
    • Graph: co-occurrence matrix from training data,
             top-{GNN_TOP_K_EDGES} edges per node, symmetric normalised Laplacian
    • Node features: [raw_i, freq7_i, freq14_i, recency_i] from last window step
    • Global context: BiLSTM+Attention → broadcast to all nodes
    • {GNN_LAYERS}× GCN layers: H' = ReLU(BN(A·H·W))
    • Output: h_i^(L) → Linear → Sigmoid for each student
    • Training: FocalLoss(α=0.25, γ=2.0) on all 55 outputs
    • Key idea: student A's probability is directly informed by its
                graph-neighbours' state via message passing
""")
    best_name = sorted_res[0][0]
    best_val  = sorted_res[0][1].mean()
    print("=" * 76)
    print(f"  BEST NEW MODEL: {best_name}  →  {best_val:.2f}%")
    if best_val > prev_best:
        print(f"  ★ NEW OVERALL RECORD  (was {prev_best:.2f}%)")
    else:
        print(f"  Gap to record: {best_val - prev_best:+.2f}%  (record={prev_best:.2f}%)")
    print("=" * 76)


if __name__ == '__main__':
    main()
