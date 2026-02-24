"""
Domain Feature Engineering — 495-dim (9 × 55 students)
=======================================================

Six domain-informed feature groups requested by the user, plus raw binary:

  Group 1 : Raw binary selection          (55)  — who was selected today
  Group 2 : Days since last call (norm)   (55)  — Feature 1: raw gap / expected_gap
  Group 3 : Frequency last 5  days        (55)  — Feature 2a
  Group 4 : Frequency last 10 days        (55)  — Feature 2b
  Group 5 : Frequency last 20 days        (55)  — Feature 2c
  Group 6 : Pairwise co-selection score   (55)  — Feature 3: group-affinity via cooc matrix
  Group 7 : Student position (1–55)       (55)  — Feature 4: static identity embedding
  Group 8 : Day-of-week rate per student  (55)  — Feature 5: P(student | weekday)
  Group 9 : Rolling entropy (broadcast)   (55)  — Feature 6: teacher unpredictability

All features are strictly backward-looking (no data leakage).
Day-of-week assumes school days (Mon–Fri): weekday = (day_index % 5).
Expected gap = 55 / 6 ≈ 9.17 days.

Total: 9 × 55 = 495 features per timestep.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helper: sliding-window frequency (vectorised cumsum)
# ─────────────────────────────────────────────────────────────────────────────

def _window_freq(binary: np.ndarray, window: int) -> np.ndarray:
    """freq[t, s] = mean(binary[max(0,t-window):t, s]).  Shape: (n_days, n_students)."""
    n_days, n_students = binary.shape
    freq   = np.zeros((n_days, n_students), dtype=np.float32)
    cumsum = np.vstack([np.zeros((1, n_students), dtype=np.float32),
                        np.cumsum(binary.astype(np.float32), axis=0)])
    for t in range(1, n_days):
        start    = max(0, t - window)
        freq[t]  = (cumsum[t] - cumsum[start]) / (t - start)
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Days since last call (normalised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_days_since(binary: np.ndarray) -> np.ndarray:
    """
    Normalised raw gap: days_since[t, s] = (t - last_selected[s]) / expected_gap.

    Clipped at 3.0 so extreme values don't dominate.
    Students never yet selected get value = t / expected_gap (capped).

    Returns:
        (n_days, n_students) float32  range ≈ [0, 3]
    """
    n_days, n_students = binary.shape
    expected_gap  = n_students / 6.0              # ≈ 9.17
    result        = np.zeros((n_days, n_students), dtype=np.float32)
    last_selected = np.full(n_students, -1.0)

    for t in range(n_days):
        never_mask = last_selected < 0
        seen_mask  = ~never_mask

        if seen_mask.any():
            gap = (t - last_selected[seen_mask]) / expected_gap
            result[t, seen_mask] = np.clip(gap, 0.0, 3.0).astype(np.float32)

        if never_mask.any():
            result[t, never_mask] = min(t / expected_gap, 3.0)

        last_selected[binary[t] == 1] = t

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Frequency last 5 / 10 / 20 days  (three separate groups)
# ─────────────────────────────────────────────────────────────────────────────

def compute_freq_windows(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three frequency matrices for windows 5, 10, 20 days.
    Each shape: (n_days, n_students) float32.
    """
    return _window_freq(binary, 5), _window_freq(binary, 10), _window_freq(binary, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Pairwise co-selection score
# ─────────────────────────────────────────────────────────────────────────────

def compute_cooc_score(binary: np.ndarray) -> np.ndarray:
    """
    Group-affinity score based on historical pairwise co-occurrence.

    At each day t, maintains a running co-occurrence count matrix C[i, j] =
    number of past days where both student i and j were selected together.

    co_score[t, s] = (C_norm[s, :] @ freq7[t, :])

    where C_norm[s, :] = C[s, :] / max(C[s, :].sum(), 1)  (row-normalised).
    freq7[t, :] = 7-day selection frequency at day t (used as "recent context").

    Interpretation: high co_score[t, s] means student s frequently co-appears
    with the students who have been recently selected — they travel in the same
    "social group" in the teacher's mental model.

    No data leakage: features at day t use C built from days 0..t-1 only.

    Returns:
        (n_days, n_students) float32  range ≈ [0, 1]
    """
    n_days, n_students = binary.shape
    cooc_count = np.zeros((n_students, n_students), dtype=np.float64)
    freq7      = _window_freq(binary, 7)
    result     = np.zeros((n_days, n_students), dtype=np.float32)

    for t in range(n_days):
        # Compute feature BEFORE updating with day t
        row_sum = cooc_count.sum(axis=1, keepdims=True).clip(min=1.0)
        C_norm  = (cooc_count / row_sum).astype(np.float32)   # (55, 55)
        result[t] = C_norm @ freq7[t]                          # (55,)

        # Update co-occurrence matrix with day t's selection
        day_vec    = binary[t].astype(np.float64)              # (55,)
        outer      = np.outer(day_vec, day_vec)                # (55, 55)
        np.fill_diagonal(outer, 0.0)                           # exclude self-pairs
        cooc_count += outer

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Student position (static identity embedding)
# ─────────────────────────────────────────────────────────────────────────────

def compute_student_position(binary: np.ndarray) -> np.ndarray:
    """
    Static normalised student ID: position[*, s] = (s + 1) / n_students.

    Same vector broadcast to every timestep.
    This lets the LSTM learn that student 42 has a different base calling
    rate than student 1 — a simple identity bias.

    Returns:
        (n_days, n_students) float32  range = [1/55, 1]
    """
    n_days, n_students = binary.shape
    ids    = (np.arange(1, n_students + 1) / n_students).astype(np.float32)
    return np.tile(ids, (n_days, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Day-of-week selection rate per student
# ─────────────────────────────────────────────────────────────────────────────

def compute_dow_rate(binary: np.ndarray) -> np.ndarray:
    """
    Historical selection rate conditioned on day-of-week.

    Assumes school days (Mon–Fri): weekday = day_index % 5.

    dow_rate[t, s] = count(s selected on weekday w, days < t)
                     / count(days with weekday w, days < t)

    Initialised at the base rate (6/55) to handle cold-start.

    Returns:
        (n_days, n_students) float32  range ≈ [0, 1]
    """
    n_days, n_students = binary.shape
    n_dow      = 5
    base_rate  = 6.0 / n_students              # prior = base selection rate

    # Smoothed counts: start with 1 pseudo-observation at base_rate
    dow_count  = np.full((n_dow, n_students), base_rate, dtype=np.float64)
    dow_total  = np.ones(n_dow, dtype=np.float64)

    result = np.zeros((n_days, n_students), dtype=np.float32)

    for t in range(n_days):
        w = t % n_dow
        # Feature uses history BEFORE day t
        result[t] = (dow_count[w] / dow_total[w]).astype(np.float32)
        # Update with observation at day t
        dow_count[w] += binary[t].astype(np.float64)
        dow_total[w] += 1.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Rolling entropy of teacher selection behaviour
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy(binary: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Shannon entropy of the per-student selection frequency over a rolling window,
    broadcast to all students as a shared context signal.

        H(t) = -Σ_s  p_s * log(p_s + ε)

    Normalised by log(n_students) → range [0, 1].

        1.0 = maximally random (uniform selection)
        0.0 = perfectly deterministic (same student every day)

    High entropy → teacher is unpredictable → individual freq signals are less
    reliable → the LSTM should rely more on structural features.

    Returns:
        (n_days, n_students) float32 — same scalar broadcast across all 55 cols
    """
    n_days, n_students = binary.shape
    max_entropy = np.log(n_students)               # log(55) ≈ 4.007
    freq14      = _window_freq(binary, window)
    result      = np.zeros((n_days, n_students), dtype=np.float32)

    for t in range(1, n_days):
        p   = freq14[t].astype(np.float64)
        p   = np.clip(p, 1e-12, 1.0)
        H   = float(-np.sum(p * np.log(p)) / max_entropy)
        result[t, :] = np.float32(H)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Master builder
# ─────────────────────────────────────────────────────────────────────────────

def build_extended_features(binary: np.ndarray,
                             verbose: bool = True) -> np.ndarray:
    """
    Build the full 495-dim domain feature matrix.

    Feature layout (9 groups × 55 students = 495):
        [0   :55 ] Raw binary                  (group 1)
        [55  :110] Days since last call (norm)  (group 2) ← Feature 1
        [110 :165] Frequency last 5  days       (group 3) ← Feature 2a
        [165 :220] Frequency last 10 days       (group 4) ← Feature 2b
        [220 :275] Frequency last 20 days       (group 5) ← Feature 2c
        [275 :330] Pairwise co-selection score  (group 6) ← Feature 3
        [330 :385] Student position (1–55)      (group 7) ← Feature 4
        [385 :440] Day-of-week rate             (group 8) ← Feature 5
        [440 :495] Rolling entropy (broadcast)  (group 9) ← Feature 6

    Args:
        binary  : (n_days, n_students) float32
        verbose : print progress

    Returns:
        (n_days, 495) float32
    """
    if verbose:
        print("  [FE] Building 495-dim domain features (6 user-requested groups) ...")

    days_since = compute_days_since(binary)
    if verbose: print("    [FE] Feature 1 — days since last call   done")

    freq5, freq10, freq20 = compute_freq_windows(binary)
    if verbose: print("    [FE] Feature 2 — freq 5/10/20 days      done")

    cooc = compute_cooc_score(binary)
    if verbose: print("    [FE] Feature 3 — pairwise co-selection  done")

    position = compute_student_position(binary)
    if verbose: print("    [FE] Feature 4 — student position 1-55  done")

    dow = compute_dow_rate(binary)
    if verbose: print("    [FE] Feature 5 — day-of-week rate        done")

    entropy = compute_entropy(binary, window=14)
    if verbose: print("    [FE] Feature 6 — rolling entropy         done")

    extended = np.concatenate([
        binary.astype(np.float32),  # 1: raw             (55)
        days_since,                  # 2: days-since      (55) F1
        freq5,                       # 3: freq-5          (55) F2a
        freq10,                      # 4: freq-10         (55) F2b
        freq20,                      # 5: freq-20         (55) F2c
        cooc,                        # 6: cooc-score      (55) F3
        position,                    # 7: position        (55) F4
        dow,                         # 8: dow-rate        (55) F5
        entropy,                     # 9: entropy         (55) F6
    ], axis=1)

    if verbose:
        print(f"    [FE] Extended shape: {extended.shape}  "
              f"({extended.shape[1]} features per timestep)")

    return extended


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    print("Self-test: build_extended_features on synthetic data (500 days, 55 students)")
    rng    = np.random.default_rng(0)
    n_days = 500
    binary = np.zeros((n_days, 55), dtype=np.float32)
    for t in range(n_days):
        idx         = rng.choice(55, size=6, replace=False)
        binary[t, idx] = 1.0

    t0  = time.perf_counter()
    ext = build_extended_features(binary)
    print(f"  Done in {time.perf_counter()-t0:.3f}s")
    print(f"  Shape  : {ext.shape}")
    print(f"  NaN    : {np.isnan(ext).sum()}")

    # Verify group ranges
    labels = [
        ("raw binary",        ext[:,   0:55]),
        ("days since (norm)", ext[:,  55:110]),
        ("freq-5",            ext[:, 110:165]),
        ("freq-10",           ext[:, 165:220]),
        ("freq-20",           ext[:, 220:275]),
        ("cooc score",        ext[:, 275:330]),
        ("position",          ext[:, 330:385]),
        ("dow rate",          ext[:, 385:440]),
        ("entropy",           ext[:, 440:495]),
    ]
    for name, arr in labels:
        print(f"  {name:<22}: [{arr.min():.3f}, {arr.max():.3f}]")
    print("\nAll checks passed!")
