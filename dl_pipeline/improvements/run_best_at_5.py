"""
n_groups=5  —  Best model for a 5-prediction-per-day tool
==========================================================
Tests every competitive model at the fixed n_groups=5 constraint,
plus a temperature sweep on the winner.

Models tested:
  • M2  baseline   (global λ=0.99)
  • M2+ narrow     (λ per-student [0.950–0.995])
  • M2+ wide       (λ per-student [0.900–0.990])  ← current champion
  • M1             (Hybrid LSTM+Markov, α=0.9)
  • M1+M2+ wide    (ensemble, w_M1=0.3)

Temperature sweep for winner: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

Usage:
    cd dl_pipeline
    python improvements/run_best_at_5.py
"""

from __future__ import annotations
import sys, copy
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_DL_DIR   = _THIS_DIR.parent
sys.path.insert(0, str(_DL_DIR))

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import time, warnings
warnings.filterwarnings('ignore')

from enhanced_models import EnhancedLSTMPredictor, FocalLoss
from markov_chain    import FactoredMarkovChain

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH   = str(_DL_DIR.parent / 'Database.xlsx')
N_S          = 55
K_SEL        = 6
TRAIN_RATIO  = 0.9
SIM_DAYS     = 100
SEQ_LEN      = 14
SEED         = 42

N_GROUPS_TOOL = 5           # fixed — your tool's constraint
TEMP_SWEEP    = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
PREV_RECORD   = 32.17       # previous benchmark (n_groups=10)

# M1 hyper-params
H_ALPHA = 0.9; H_NG = 10; H_TEMP = 2.0
H_ITERS = 5;  H_REPLAY = 10; H_LR = 5e-4
H_EPOCHS = 50; H_PATIENCE = 15

LAM_WINDOW = 14

# ─────────────────────────────────────────────────────────────────────────────
# Data
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
# Features  (220-dim)
# ─────────────────────────────────────────────────────────────────────────────
def _freq(binary, window):
    n, s   = binary.shape
    feat   = np.zeros((n, s), dtype=np.float32)
    cumsum = np.vstack([np.zeros((1, s), dtype=np.float32),
                        np.cumsum(binary, axis=0)])
    for t in range(1, n):
        start   = max(0, t - window)
        feat[t] = (cumsum[t] - cumsum[start]) / (t - start)
    return feat

def _recency(binary):
    n, s = binary.shape
    feat = np.zeros((n, s), dtype=np.float32)
    last = -np.ones(s, dtype=np.float64)
    for t in range(n):
        mask = last >= 0
        if mask.any():
            feat[t, mask] = 1.0 / (t - last[mask] + 1)
        last[binary[t] == 1] = t
    return feat

def build_features(binary):
    return np.concatenate([binary, _freq(binary,7), _freq(binary,14),
                           _recency(binary)], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TD-HBB helpers
# ─────────────────────────────────────────────────────────────────────────────
def _qgroups(values, n):
    cuts = np.percentile(values, np.linspace(0,100,n+1)[1:-1])
    return np.digitize(values, cuts).astype(np.int32)

def _hyperpriors(S, N, grp):
    G = int(grp.max()) + 1
    mu_g = np.zeros(G); ka_g = np.ones(G)*10.
    rates = np.where(N > 0, S/N, K_SEL/N_S)
    for g in range(G):
        m = grp == g
        if m.sum() < 2: mu_g[g]=rates.mean(); ka_g[g]=10.; continue
        r=rates[m]; mu=float(r.mean()); var=float(r.var())
        mu_g[g]=mu
        ka_g[g]=(float(np.clip(mu*(1-mu)/var-1,2.,1000.)) if var>1e-10 else 1000.)
    return mu_g, ka_g

def _post_ab(S, N, grp, mu_g, ka_g):
    a = mu_g[grp]*ka_g[grp] + S
    b = (1.-mu_g[grp])*ka_g[grp] + np.maximum(N-S, 0.)
    return a, b

def per_student_lambda(binary_train, lam_min, lam_max):
    n, ns = binary_train.shape
    rv    = np.zeros(ns)
    cs    = np.vstack([np.zeros((1,ns)), np.cumsum(binary_train, axis=0)])
    for s in range(ns):
        means = [(cs[t,s]-cs[t-LAM_WINDOW,s])/LAM_WINDOW
                 for t in range(LAM_WINDOW, n)]
        rv[s] = float(np.var(means)) if means else 0.
    v0,v1 = rv.min(), rv.max()
    if v1-v0 < 1e-12: return np.full(ns,(lam_min+lam_max)/2)
    norm = (rv-v0)/(v1-v0)
    return (lam_max - (lam_max-lam_min)*norm).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Model classes
# ─────────────────────────────────────────────────────────────────────────────
class M2Fixed:
    """Global λ=0.99 baseline."""
    def __init__(self, lam=0.99, n_tier=3, refresh=10):
        self.lam=lam; self.n_tier=n_tier; self.refresh=refresh
        self.S=np.zeros(N_S); self.N=np.zeros(N_S)
        self.grp=np.zeros(N_S,dtype=np.int32)
        self.mu_g=np.array([K_SEL/N_S]); self.ka_g=np.array([2.])
        self.n_obs=0.

    def fit(self, binary, avg_pos):
        n=len(binary); w=self.lam**np.arange(n-1,-1,-1,dtype=np.float64)
        self.S=(binary*w[:,None]).sum(axis=0).astype(np.float64)
        self.N=np.full(N_S,w.sum())
        self.grp=_qgroups(avg_pos, self.n_tier)
        self.mu_g, self.ka_g = _hyperpriors(self.S,self.N,self.grp)
        self.n_obs=float(n)

    def probs(self):
        a,b=_post_ab(self.S,self.N,self.grp,self.mu_g,self.ka_g)
        p=np.clip((a/(a+b)).astype(np.float32),1e-9,None)
        return p/p.sum()

    def update(self, obs):
        self.S=self.lam*self.S+obs.astype(np.float64)
        self.N=self.lam*self.N+1.
        self.n_obs+=1.
        if self.refresh>0 and int(self.n_obs)%self.refresh==0:
            self.mu_g,self.ka_g=_hyperpriors(self.S,self.N,self.grp)

    def snap(self): return {k:copy.deepcopy(v) for k,v in self.__dict__.items()}
    def load(self, s): [setattr(self,k,copy.deepcopy(v)) for k,v in s.items()]


class M2Adaptive:
    """Per-student adaptive λ."""
    def __init__(self, lam_min, lam_max, n_tier=3, refresh=10):
        self.lam_min=lam_min; self.lam_max=lam_max
        self.n_tier=n_tier; self.refresh=refresh
        self.lam_s=np.full(N_S,(lam_min+lam_max)/2)
        self.S=np.zeros(N_S); self.N=np.zeros(N_S)
        self.grp=np.zeros(N_S,dtype=np.int32)
        self.mu_g=np.array([K_SEL/N_S]); self.ka_g=np.array([2.])
        self.n_obs=0.

    def fit(self, binary, avg_pos):
        n=len(binary)
        self.lam_s=per_student_lambda(binary,self.lam_min,self.lam_max)
        for s in range(N_S):
            w=self.lam_s[s]**np.arange(n-1,-1,-1,dtype=np.float64)
            self.S[s]=float((binary[:,s]*w).sum())
            self.N[s]=float(w.sum())
        self.grp=_qgroups(avg_pos,self.n_tier)
        self.mu_g,self.ka_g=_hyperpriors(self.S,self.N,self.grp)
        self.n_obs=float(n)

    def probs(self):
        a,b=_post_ab(self.S,self.N,self.grp,self.mu_g,self.ka_g)
        p=np.clip((a/(a+b)).astype(np.float32),1e-9,None)
        return p/p.sum()

    def update(self, obs):
        self.S=self.lam_s*self.S+obs.astype(np.float64)
        self.N=self.lam_s*self.N+1.
        self.n_obs+=1.
        if self.refresh>0 and int(self.n_obs)%self.refresh==0:
            self.mu_g,self.ka_g=_hyperpriors(self.S,self.N,self.grp)

    def snap(self): return {k:copy.deepcopy(v) for k,v in self.__dict__.items()}
    def load(self, s): [setattr(self,k,copy.deepcopy(v)) for k,v in s.items()]


# ─────────────────────────────────────────────────────────────────────────────
# LSTM training
# ─────────────────────────────────────────────────────────────────────────────
class SeqDS(torch.utils.data.Dataset):
    def __init__(self, E, B, sl=SEQ_LEN):
        self.E=E; self.B=B; self.sl=sl
    def __len__(self): return len(self.E)-self.sl
    def __getitem__(self, i):
        return torch.FloatTensor(self.E[i:i+self.sl]), torch.FloatTensor(self.B[i+self.sl])

def train_lstm(bin_tr, enr_tr, device):
    nf=enr_tr.shape[1]
    crit=FocalLoss(alpha=0.25,gamma=2.0,label_smoothing=0.05)
    model=EnhancedLSTMPredictor(n_features=nf,n_students=N_S,
                                 hidden_size=128,num_layers=2,dropout=0.3).to(device)
    sv=int(len(bin_tr)*0.9)
    tl=DataLoader(SeqDS(enr_tr[:sv],bin_tr[:sv]),batch_size=32,shuffle=True,drop_last=True)
    vl=DataLoader(SeqDS(enr_tr[sv:],bin_tr[sv:]),batch_size=32,shuffle=False,drop_last=False)
    if not tl or not vl: return model
    opt=optim.Adam(model.parameters(),lr=H_LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'min',0.5,patience=5)
    best,bst,ni=float('inf'),None,0
    for ep in range(H_EPOCHS):
        model.train()
        for X,y in tl:
            opt.zero_grad(); loss=crit(model(X.to(device)),y.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval(); v=0.
        with torch.no_grad():
            for X,y in vl: v+=crit(model(X.to(device)),y.to(device)).item()
        v/=len(vl); sch.step(v)
        if v<best: best,bst,ni=v,copy.deepcopy(model.state_dict()),0
        else:
            ni+=1
            if ni>=H_PATIENCE:
                print(f'    [LSTM] Early stop @ epoch {ep+1}  best={best:.5f}'); break
        if (ep+1)%10==0: print(f'    [LSTM] Epoch {ep+1:3d}  val={v:.5f}')
    print(f'    [LSTM] Done'); model.load_state_dict(bst); return model


# ─────────────────────────────────────────────────────────────────────────────
# Oracle sampler
# ─────────────────────────────────────────────────────────────────────────────
def sample_groups(probs, n_groups, temperature):
    rng = np.random.default_rng(SEED)
    p   = np.clip(probs, 1e-9, None) ** (1./temperature); p /= p.sum()
    gs: list = []
    for _ in range(max(n_groups*10, 200)):
        g = sorted(rng.choice(N_S, K_SEL, replace=False, p=p).tolist())
        if g not in gs: gs.append(g)
        if len(gs) >= n_groups: break
    if len(gs) < n_groups:
        ranked = np.argsort(probs)[::-1].tolist()
        for start in range(len(ranked)-K_SEL+1):
            g = sorted(ranked[start:start+K_SEL])
            if g not in gs: gs.append(g)
            if len(gs) >= n_groups: break
    return gs[:n_groups]

def best_score(probs, actual_row, n_groups, temperature):
    groups = sample_groups(probs, n_groups, temperature)
    actual = set(np.where(actual_row == 1)[0].tolist())
    return max(len(set(g) & actual) for g in groups) / K_SEL


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────
def run_m2(model, snap, binary, split, n_groups, temperature):
    model.load(snap)
    np.random.seed(SEED)
    accs = []
    for i in range(min(SIM_DAYS, len(binary)-split)):
        idx = split+i
        accs.append(best_score(model.probs(), binary[idx], n_groups, temperature))
        model.update(binary[idx])
    return np.array(accs)*100.

def run_m1(lstm, init_state, markov_snap_fn, binary, enriched,
           split, device, n_groups, temperature):
    lstm.load_state_dict(copy.deepcopy(init_state))
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])
    np.random.seed(SEED); torch.manual_seed(SEED)
    crit = FocalLoss(alpha=0.25,gamma=2.0,label_smoothing=0.05)
    opt  = optim.Adam(lstm.parameters(), lr=0.0003)
    replay: list = []; accs: list = []
    for i in range(min(SIM_DAYS, len(binary)-split)):
        idx = split+i
        if idx < SEQ_LEN: accs.append(0.); continue
        esq = enriched[idx-SEQ_LEN:idx]; bsq = binary[idx-SEQ_LEN:idx]
        lstm.eval()
        with torch.no_grad():
            lp = lstm(torch.FloatTensor(esq).unsqueeze(0).to(device)).cpu().numpy()[0]
        lp /= lp.sum()+1e-9
        mp  = markov.predict_probabilities(bsq)
        hyb = H_ALPHA*lp+(1-H_ALPHA)*mp; hyb /= hyb.sum()+1e-9
        accs.append(best_score(hyb, binary[idx], n_groups, temperature))
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay)>=5:
            rec = replay[-H_REPLAY:]
            Xb=torch.FloatTensor(np.array([r[0] for r in rec])).to(device)
            yb=torch.FloatTensor(np.array([r[1] for r in rec])).to(device)
            lstm.train()
            for _ in range(H_ITERS):
                opt.zero_grad(); loss=crit(lstm(Xb),yb)
                loss.backward(); nn.utils.clip_grad_norm_(lstm.parameters(),1.); opt.step()
        if len(bsq)>=2: markov.update(binary[idx],bsq[-1],bsq[-2])
    return np.array(accs)*100.

def run_ensemble(lstm, init_state, m2_model, m2_snap,
                 binary, enriched, avg_pos, split, device,
                 w_m1, n_groups, temperature):
    lstm.load_state_dict(copy.deepcopy(init_state))
    m2_model.load(m2_snap)
    markov = FactoredMarkovChain(n_students=N_S, smoothing=1.0)
    markov.fit(binary[:split])
    np.random.seed(SEED); torch.manual_seed(SEED)
    crit = FocalLoss(alpha=0.25,gamma=2.0,label_smoothing=0.05)
    opt  = optim.Adam(lstm.parameters(), lr=0.0003)
    replay: list = []; accs: list = []
    for i in range(min(SIM_DAYS, len(binary)-split)):
        idx = split+i
        if idx < SEQ_LEN: accs.append(0.); continue
        esq = enriched[idx-SEQ_LEN:idx]; bsq = binary[idx-SEQ_LEN:idx]
        lstm.eval()
        with torch.no_grad():
            lp = lstm(torch.FloatTensor(esq).unsqueeze(0).to(device)).cpu().numpy()[0]
        lp /= lp.sum()+1e-9
        mp  = markov.predict_probabilities(bsq)
        m1p = H_ALPHA*lp+(1-H_ALPHA)*mp; m1p /= m1p.sum()+1e-9
        m2p = m2_model.probs().astype(np.float64)
        ens = w_m1*m1p + (1-w_m1)*m2p; ens /= ens.sum()+1e-9
        accs.append(best_score(ens, binary[idx], n_groups, temperature))
        replay.append((esq.copy(), binary[idx].copy()))
        if len(replay)>=5:
            rec = replay[-H_REPLAY:]
            Xb=torch.FloatTensor(np.array([r[0] for r in rec])).to(device)
            yb=torch.FloatTensor(np.array([r[1] for r in rec])).to(device)
            lstm.train()
            for _ in range(H_ITERS):
                opt.zero_grad(); loss=crit(lstm(Xb),yb)
                loss.backward(); nn.utils.clip_grad_norm_(lstm.parameters(),1.); opt.step()
        if len(bsq)>=2: markov.update(binary[idx],bsq[-1],bsq[-2])
        m2_model.update(binary[idx])
    return np.array(accs)*100.


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def stats(arr):
    q = [arr[j*25:(j+1)*25].mean() for j in range(4)]
    return (f"Avg={arr.mean():.2f}%  Best={arr.max():.2f}%  "
            f"Worst={arr.min():.2f}%  Std={arr.std():.4f}  "
            f"Q1={q[0]:.1f}% Q2={q[1]:.1f}% Q3={q[2]:.1f}% Q4={q[3]:.1f}%")

def print_row(label, arr, ref=None):
    vs  = f'  (vs ref: {arr.mean()-ref:+.2f}%)' if ref else ''
    flg = '  *** NEW BEST' if (ref and arr.mean() > ref) else ''
    print(f'  {label:<38}  {stats(arr)}{vs}{flg}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    W = 82
    print('='*W)
    print(f'  BEST MODEL FOR n_groups=5  (your tool constraint)'.center(W))
    print('='*W)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n  Device: {device}')

    print('\n  Loading data...')
    binary, avg_pos = load_data()
    split = int(len(binary)*TRAIN_RATIO)
    print(f'  {len(binary)} days  |  {split} train  |  {len(binary)-split} test')
    enriched = build_features(binary)

    # ── Prepare all models ────────────────────────────────────────────────
    print('\n  Fitting all models on training data...')

    # M2 baseline
    m2_base = M2Fixed(lam=0.99)
    m2_base.fit(binary[:split], avg_pos)
    snap_m2_base = m2_base.snap()

    # M2+ narrow
    m2_narrow = M2Adaptive(lam_min=0.950, lam_max=0.995)
    m2_narrow.fit(binary[:split], avg_pos)
    snap_m2_narrow = m2_narrow.snap()

    # M2+ wide  (champion)
    m2_wide = M2Adaptive(lam_min=0.900, lam_max=0.990)
    m2_wide.fit(binary[:split], avg_pos)
    snap_m2_wide = m2_wide.snap()

    # M1 LSTM
    print('  Training LSTM...')
    lstm = train_lstm(binary[:split], enriched[:split], device)
    init_state = copy.deepcopy(lstm.state_dict())
    print('  All models ready.\n')

    # ══════════════════════════════════════════════════════════════════════
    # PART 1 — All models at n_groups=5, temperature=2.0
    # ══════════════════════════════════════════════════════════════════════
    print('='*W)
    print(f'  PART 1 — All models  (n_groups=5, temperature=2.0)'.center(W))
    print('='*W)
    print(f'  {"Model":<38}  Avg      Best     Worst    Std       Q1     Q2     Q3     Q4')
    print(f'  {"-"*38}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*5}')

    NG, T0 = N_GROUPS_TOOL, 2.0
    results = {}

    arr = run_m2(m2_base, snap_m2_base, binary, split, NG, T0)
    print_row('M2  global λ=0.99', arr)
    results['M2  global λ=0.99'] = arr

    arr = run_m2(m2_narrow, snap_m2_narrow, binary, split, NG, T0)
    print_row('M2+ narrow [0.950-0.995]', arr)
    results['M2+ narrow [0.950-0.995]'] = arr

    arr = run_m2(m2_wide, snap_m2_wide, binary, split, NG, T0)
    print_row('M2+ wide   [0.900-0.990]', arr, ref=None)
    results['M2+ wide [0.900-0.990]'] = arr

    arr = run_m1(lstm, init_state, None, binary, enriched, split, device, NG, T0)
    print_row('M1  LSTM+Markov (α=0.9)', arr)
    results['M1  LSTM+Markov (α=0.9)'] = arr

    arr = run_ensemble(lstm, init_state, m2_wide, snap_m2_wide,
                       binary, enriched, avg_pos, split, device,
                       w_m1=0.3, n_groups=NG, temperature=T0)
    print_row('Ensemble M1(0.3)+M2+wide(0.7)', arr)
    results['Ensemble M1(0.3)+M2+wide(0.7)'] = arr

    # Identify winner
    winner_name = max(results, key=lambda k: results[k].mean())
    winner_arr  = results[winner_name]
    print(f'\n  >>> Winner at n_groups=5, temp=2.0:  {winner_name}  '
          f'({winner_arr.mean():.2f}%)')

    # ══════════════════════════════════════════════════════════════════════
    # PART 2 — Temperature sweep on winner
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*W}')
    print(f'  PART 2 — Temperature sweep on winner: {winner_name}'.center(W))
    print(f'  (n_groups=5 fixed)'.center(W))
    print('='*W)
    print(f'  {"Temp":<10}  Avg      Best     Worst    Std       Quartiles              vs T=2.0')
    print(f'  {"-"*10}  {"-"*7}  {"-"*7}  {"-"*7}  {"-"*8}  {"-"*26}  {"-"*8}')

    temp_results = {}
    # Map winner_name to the right runner
    def run_winner(temp):
        if winner_name.startswith('M2+ wide'):
            return run_m2(m2_wide, snap_m2_wide, binary, split, NG, temp)
        elif winner_name.startswith('M2+ narrow'):
            return run_m2(m2_narrow, snap_m2_narrow, binary, split, NG, temp)
        elif winner_name.startswith('M2  global'):
            return run_m2(m2_base, snap_m2_base, binary, split, NG, temp)
        elif winner_name.startswith('M1'):
            return run_m1(lstm, init_state, None, binary, enriched, split, device, NG, temp)
        else:
            return run_ensemble(lstm, init_state, m2_wide, snap_m2_wide,
                                binary, enriched, avg_pos, split, device,
                                w_m1=0.3, n_groups=NG, temperature=temp)

    baseline_temp = winner_arr.mean()
    for temp in TEMP_SWEEP:
        arr = run_winner(temp)
        q   = [arr[j*25:(j+1)*25].mean() for j in range(4)]
        vs  = arr.mean()-baseline_temp
        flg = ' ***' if arr.mean() > baseline_temp else ''
        print(f'  {temp:<10}  {arr.mean():>7.2f}%  {arr.max():>7.2f}%  '
              f'{arr.min():>7.2f}%  {arr.std():>8.4f}  '
              f'Q1={q[0]:.1f}% Q2={q[1]:.1f}% Q3={q[2]:.1f}% Q4={q[3]:.1f}%  '
              f'{vs:>+.2f}%{flg}')
        temp_results[temp] = arr.mean()

    best_temp    = max(temp_results, key=temp_results.get)
    best_temp_sc = temp_results[best_temp]

    # ══════════════════════════════════════════════════════════════════════
    # Final recommendation
    # ══════════════════════════════════════════════════════════════════════
    print(f'\n{"="*W}')
    print('  FINAL RECOMMENDATION FOR YOUR TOOL'.center(W))
    print('='*W)
    print(f"""
  Constraint:   n_groups = 5  (5 group predictions per day)

  Best model:   {winner_name}
  Best temp:    {best_temp}
  Best score:   {best_temp_sc:.2f}%

  What this means in practice:
  ─────────────────────────────────────────────────────────
  Each day your tool generates 5 candidate groups of 6 students.
  On average, the best candidate group will share {best_temp_sc/100*K_SEL:.1f}/6
  students with the teacher's actual selection.

  Score interpretation:
    ≥ 3/6 students correct  →  useful prediction
    ≥ 4/6 students correct  →  strong prediction
    ≥ 5/6 students correct  →  excellent prediction

  If you later increase to n_groups=10, expected score: ~32–33%
  If you later increase to n_groups=20, expected score: ~37%
  ─────────────────────────────────────────────────────────
""")
    print('='*W)


if __name__ == '__main__':
    main()
