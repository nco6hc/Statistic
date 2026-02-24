"""
Base Model Wrappers for the 4-Component Meta-Ensemble
======================================================

Each wrapper exposes a unified interface:
    fit(binary_history, ...)  → train / initialise from historical data
    predict_proba(...)        → return (n_students,) normalised probability vector
    update(binary_day, ...)   → online update after observing one day

Wrappers:
  1. MarkovWrapper   – FactoredMarkovChain with cooldown + co-occurrence
  2. XGBWrapper      – HistGradientBoosting (or XGBoost if installed) on 9 GB features
  3. LogisticWrapper – Logistic Regression on the same 9 GB features
  4. LSTMWrapper     – small LSTM (hidden=64) with mini online fine-tuning

MetaEnsemble:
  - Blends the 4 probability vectors with adaptive weights
  - Weights updated via EMA of per-model top-6 overlap accuracy
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_DL_DIR = Path(__file__).resolve().parent.parent   # dl_pipeline/
sys.path.insert(0, str(_DL_DIR))

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from markov_chain import FactoredMarkovChain
from gradient_boosting.gb_features import IncrementalGBState, build_gb_features
from models import LSTMStudentPredictor


# ============================================================
# Shared Utility
# ============================================================
def probs_to_groups(probs: np.ndarray, n_groups: int = 5, k: int = 6,
                    base_temp: float = 1.0, temp_step: float = 0.3) -> list:
    """
    Generate n_groups diverse candidate groups via temperature sampling.

    Later groups use higher temperature → more exploration.
    Within each group, students are sorted by descending posterior probability.
    """
    groups = []
    for g in range(n_groups):
        temp  = base_temp + temp_step * g
        adj   = np.power(np.maximum(probs, 1e-9), 1.0 / temp)
        adj  /= adj.sum()
        idx   = np.random.choice(len(adj), size=k, replace=False, p=adj)
        idx   = idx[np.argsort(-probs[idx])]
        groups.append((idx + 1).tolist())
    return groups


# ============================================================
# 1. Markov Chain Wrapper
# ============================================================
class MarkovWrapper:
    """
    Wraps FactoredMarkovChain.

    predict_proba() uses the last seq_len days of observed history.
    update() appends the new day and incrementally updates chain statistics.
    """
    name  = "Markov"
    color = "#e74c3c"

    def __init__(self, n_students: int = 55, seq_len: int = 14,
                 smoothing: float = 1.0):
        self.n_students = n_students
        self.seq_len    = seq_len
        self.mc         = FactoredMarkovChain(n_students=n_students, smoothing=smoothing)
        self._history   = []

    def fit(self, binary_history: np.ndarray) -> None:
        self.mc.fit(binary_history)
        # Seed history buffer with the tail of training data
        self._history = [binary_history[t].copy() for t in range(len(binary_history))]
        self._trim_history()

    def predict_proba(self) -> np.ndarray:
        recent = np.array(self._history[-self.seq_len:], dtype=np.float32)
        if len(recent) < 2:
            return np.ones(self.n_students, dtype=np.float32) / self.n_students
        probs = self.mc.predict_probabilities(recent).astype(np.float32)
        total = probs.sum()
        return probs / total if total > 1e-12 else probs

    def update(self, binary_day: np.ndarray) -> None:
        if len(self._history) >= 2:
            self.mc.update(binary_day,
                           yesterday=self._history[-1],
                           day_before=self._history[-2])
        self._history.append(binary_day.copy())
        self._trim_history()

    def _trim_history(self):
        # Keep only what we need to avoid unbounded memory growth
        if len(self._history) > self.seq_len + 5:
            self._history = self._history[-(self.seq_len + 5):]


# ============================================================
# 2. XGBoost / HistGradientBoosting Wrapper
# ============================================================
class XGBWrapper:
    """
    Gradient-boosted classifier on the 9-dimensional GB feature set.
    Uses XGBoost if installed, otherwise HistGradientBoostingClassifier.

    Features are computed incrementally via IncrementalGBState.
    """
    name  = "XGBoost" if _XGB_AVAILABLE else "HistGB"
    color = "#27ae60"

    def __init__(self, n_students: int = 55, pos_weight: float = 9.0):
        self.n_students = n_students
        self.pos_weight = pos_weight
        self._model     = None
        self._state     = None
        self._abs_day   = 0

    def fit(self, binary_history: np.ndarray, split_idx: int) -> None:
        print(f"  [{self.name}] Building features on {len(binary_history)} days...")
        X, y = build_gb_features(binary_history, n_students=self.n_students)
        sw   = np.where(y == 1, self.pos_weight, 1.0)

        if _XGB_AVAILABLE:
            self._model = _XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                min_child_weight=20, reg_lambda=1.0,
                scale_pos_weight=self.pos_weight,
                n_jobs=-1, random_state=42, verbosity=0,
                eval_metric='logloss',
            )
            self._model.fit(X, y.astype(int))
        else:
            self._model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=5,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=42,
            )
            self._model.fit(X, y, sample_weight=sw)

        self._state   = IncrementalGBState(binary_history, n_students=self.n_students)
        self._abs_day = split_idx

    def predict_proba(self, abs_day: int = None) -> np.ndarray:
        day   = abs_day if abs_day is not None else self._abs_day
        X_day = self._state.get_features_for_next_day(day)
        probs = self._model.predict_proba(X_day)[:, 1].astype(np.float32)
        total = probs.sum()
        return probs / total if total > 1e-12 else probs

    def update(self, binary_day: np.ndarray, abs_day: int = None) -> None:
        day = abs_day if abs_day is not None else self._abs_day
        self._state.update(binary_day, day)
        self._abs_day = day + 1


# ============================================================
# 3. Logistic Regression Wrapper
# ============================================================
class LogisticWrapper:
    """
    Global logistic regression classifier on the same 9 GB features.

    One shared model predicts P(student selected | features).
    Per-student probabilities are obtained by scoring 55 feature rows.
    """
    name  = "Logistic"
    color = "#3498db"

    def __init__(self, n_students: int = 55, pos_weight: float = 9.0,
                 C: float = 0.1):
        self.n_students = n_students
        self.pos_weight = pos_weight
        self._model     = LogisticRegression(
            solver='lbfgs', C=C, max_iter=1000,
            random_state=42, n_jobs=-1,
        )
        self._state   = None
        self._abs_day = 0

    def fit(self, binary_history: np.ndarray, split_idx: int) -> None:
        print(f"  [Logistic] Building features on {len(binary_history)} days...")
        X, y = build_gb_features(binary_history, n_students=self.n_students)
        sw   = np.where(y == 1, self.pos_weight, 1.0)
        self._model.fit(X, y, sample_weight=sw)

        self._state   = IncrementalGBState(binary_history, n_students=self.n_students)
        self._abs_day = split_idx

    def predict_proba(self, abs_day: int = None) -> np.ndarray:
        day   = abs_day if abs_day is not None else self._abs_day
        X_day = self._state.get_features_for_next_day(day)
        probs = self._model.predict_proba(X_day)[:, 1].astype(np.float32)
        total = probs.sum()
        return probs / total if total > 1e-12 else probs

    def update(self, binary_day: np.ndarray, abs_day: int = None) -> None:
        day = abs_day if abs_day is not None else self._abs_day
        self._state.update(binary_day, day)
        self._abs_day = day + 1


# ============================================================
# 4. LSTM Wrapper
# ============================================================
class LSTMWrapper:
    """
    Small single-layer LSTM (hidden=64) trained on 14-day sequences.

    Online adaptation:
      Every `finetune_every` update calls, run `finetune_epochs` gradient
      steps on the last `finetune_window` sequences. This keeps the model
      adapting to recent patterns without a full retrain.
    """
    name  = "LSTM"
    color = "#9b59b6"

    def __init__(self, n_students: int = 55, seq_len: int = 14,
                 hidden_size: int = 64, n_epochs: int = 20,
                 finetune_epochs: int = 3, finetune_every: int = 5,
                 finetune_window: int = 50):
        self.n_students     = n_students
        self.seq_len        = seq_len
        self.hidden_size    = hidden_size
        self.n_epochs       = n_epochs
        self.finetune_epochs = finetune_epochs
        self.finetune_every  = finetune_every
        self.finetune_window = finetune_window
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model         = None
        self._history       = None     # (T, n_students) float32 array
        self._update_count  = 0

    def fit(self, binary_history: np.ndarray) -> None:
        print(f"  [LSTM] Training on {len(binary_history)} days "
              f"(hidden={self.hidden_size}, epochs={self.n_epochs})...")

        # Build training sequences
        X_list, y_list = [], []
        for t in range(self.seq_len, len(binary_history)):
            X_list.append(binary_history[t - self.seq_len: t])
            y_list.append(binary_history[t])

        X = torch.FloatTensor(np.stack(X_list)).to(self.device)   # (N, seq, 55)
        y = torch.FloatTensor(np.stack(y_list)).to(self.device)   # (N, 55)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        self._model = LSTMStudentPredictor(
            n_students=self.n_students, hidden_size=self.hidden_size,
            num_layers=1, dropout=0.1,
        ).to(self.device)

        optimizer  = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion  = nn.BCELoss()
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        self._model.train()
        for epoch in range(self.n_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(bx), by)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"    epoch {epoch+1}/{self.n_epochs}")

        self._model.eval()
        self._history = np.array(binary_history, dtype=np.float32)

    def predict_proba(self) -> np.ndarray:
        self._model.eval()
        seq = self._history[-self.seq_len:]
        x   = torch.FloatTensor(seq[None]).to(self.device)   # (1, seq, 55)
        with torch.no_grad():
            probs = self._model(x).squeeze(0).cpu().numpy()  # (55,)
        total = probs.sum()
        return probs / total if total > 1e-12 else probs

    def update(self, binary_day: np.ndarray) -> None:
        self._history       = np.vstack([self._history, binary_day.astype(np.float32)])
        self._update_count += 1
        if self._update_count % self.finetune_every == 0:
            self._finetune()

    def _finetune(self) -> None:
        """Mini online fine-tune on recent sequences — a few gradient steps."""
        window = self._history[-self.finetune_window:]
        X_list, y_list = [], []
        for t in range(self.seq_len, len(window)):
            X_list.append(window[t - self.seq_len: t])
            y_list.append(window[t])
        if len(X_list) < 4:
            return

        X = torch.FloatTensor(np.stack(X_list)).to(self.device)
        y = torch.FloatTensor(np.stack(y_list)).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        criterion = nn.BCELoss()
        self._model.train()
        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            loss = criterion(self._model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()
        self._model.eval()


# ============================================================
# Meta-Ensemble: Adaptive Blending
# ============================================================
class MetaEnsemble:
    """
    Blends 4 base-model probability vectors with online-adapted weights.

    Algorithm:
        After each prediction, for each model i:
            score_i = (1 - α) * score_i  +  α * (overlap_i / k)

        Then re-compute weights via softmax with temperature τ:
            w_i = softmax(score_i / τ)

        Small τ → winner-take-all.   Large τ → nearly uniform.

    Parameters:
        ema_alpha   (α) : EMA decay for per-model accuracy tracking.
        temperature (τ) : Softmax temperature for weight computation.
    """

    def __init__(self, n_models: int = 4, temperature: float = 0.5,
                 ema_alpha: float = 0.15):
        self.n_models    = n_models
        self.temperature = temperature
        self.ema_alpha   = ema_alpha
        self._scores     = np.zeros(n_models, dtype=np.float64)
        self.weights     = np.ones(n_models, dtype=np.float64) / n_models
        self.weight_history: list = [self.weights.copy()]

    def blend(self, probs_list: list) -> np.ndarray:
        """Weighted sum of probability vectors, normalised."""
        blended = sum(w * p for w, p in zip(self.weights, probs_list))
        total   = blended.sum()
        return (blended / total).astype(np.float32) if total > 1e-12 else blended

    def update_weights(self, probs_list: list,
                       actual_binary: np.ndarray, k: int = 6) -> None:
        """EMA weight update based on each model's top-k accuracy."""
        actual = set(np.where(actual_binary == 1)[0])
        for i, probs in enumerate(probs_list):
            top_k   = set(np.argsort(-probs)[:k])
            overlap = len(top_k & actual) / k
            self._scores[i] = ((1 - self.ema_alpha) * self._scores[i]
                               + self.ema_alpha * overlap)

        # Softmax with temperature
        shifted = self._scores - self._scores.max()
        exp     = np.exp(shifted / max(self.temperature, 1e-6))
        self.weights = (exp / exp.sum()).astype(np.float64)
        self.weight_history.append(self.weights.copy())

    @property
    def current_weights_dict(self) -> dict:
        return {f"model_{i}": float(w) for i, w in enumerate(self.weights)}
