"""
Gradient Boosting Feature Engineering

6 rich feature groups for student selection prediction.
Each row represents one (day, student) pair.

Feature Groups:
  1. Days since last call          - rotation/cooldown signal
  2. Frequency last 5/10/20 days   - short/medium/long-term selection rate
  3. Pairwise co-selection score   - group pattern signal
  4. Position in class (1-55)      - student identity
  5. Day-of-week pattern           - weekly selection rhythm
  6. Entropy of teacher behavior   - predictability of teacher's choices

Total: 9 scalar features per (day, student) sample.

Feature layout:
  Index  Feature
  -----  -------
  0      days_since_last (log-scaled)
  1      freq_5   (selection rate in last 5 days)
  2      freq_10  (selection rate in last 10 days)
  3      freq_20  (selection rate in last 20 days)
  4      pairwise_coselection_score
  5      student_position (normalized 0-1)
  6      dow_freq (this student's frequency on this day-of-week)
  7      entropy_7d  (teacher entropy over last 7 days)
  8      entropy_20d (teacher entropy over last 20 days)
"""

import numpy as np
from typing import List, Tuple


# ============================================================
# Feature 1: Days Since Last Call
# ============================================================
def compute_days_since_last(binary: np.ndarray) -> np.ndarray:
    """
    For each (day t, student s): number of days since student s was
    last selected BEFORE day t. Only uses past data — no leakage.

    Students never selected: get value = t+1 (maximum days).
    Result is log-scaled to reduce skewness.

    Args:
        binary: (n_days, n_students) selection matrix

    Returns:
        (n_days, n_students) float32 array, log-scaled
    """
    n_days, n_students = binary.shape
    days_since = np.full((n_days, n_students), fill_value=0.0, dtype=np.float32)
    last_seen = -np.ones(n_students, dtype=np.int32)

    for t in range(n_days):
        for s in range(n_students):
            if last_seen[s] >= 0:
                days_since[t, s] = float(t - last_seen[s])
            else:
                days_since[t, s] = float(t + 1)  # Never selected yet
        # Update AFTER computing features for day t
        for s in np.where(binary[t] == 1)[0]:
            last_seen[s] = t

    return np.log1p(days_since)


# ============================================================
# Feature 2: Selection Frequency Over Windows
# ============================================================
def compute_frequency_windows(binary: np.ndarray,
                               windows: List[int] = (5, 10, 20)) -> List[np.ndarray]:
    """
    Selection frequency for each student over multiple sliding windows.

    freq[t, s] = mean(binary[max(0, t-w):t, s])  — past only.

    Args:
        binary: (n_days, n_students)
        windows: list of window sizes

    Returns:
        List of (n_days, n_students) arrays, one per window
    """
    n_days, n_students = binary.shape

    # Cumulative sum for O(1) window queries
    cumsum = np.vstack([
        np.zeros((1, n_students), dtype=np.float32),
        np.cumsum(binary.astype(np.float32), axis=0)
    ])

    results = []
    for w in windows:
        freq = np.zeros((n_days, n_students), dtype=np.float32)
        for t in range(1, n_days):
            start = max(0, t - w)
            count = t - start
            freq[t] = (cumsum[t] - cumsum[start]) / count
        results.append(freq)

    return results


# ============================================================
# Feature 3: Pairwise Co-Selection Score
# ============================================================
def compute_pairwise_coselection(binary: np.ndarray,
                                  recent_window: int = 3) -> np.ndarray:
    """
    For each (day t, student s): weighted score capturing how often
    student s co-appears with students who were recently selected.

    pairwise_score[t, s] = sum_j(cooc_rate[s,j] × recent[j])

    where cooc_rate[s,j] = P(j selected | s selected)
    and   recent[j] = 1 if j was selected in last `recent_window` days.

    Args:
        binary: (n_days, n_students)
        recent_window: look-back window for "recently selected"

    Returns:
        (n_days, n_students) float32 array
    """
    n_days, n_students = binary.shape
    pairwise_score = np.zeros((n_days, n_students), dtype=np.float32)

    # Running co-occurrence and individual counts
    cooc = np.zeros((n_students, n_students), dtype=np.float64)
    ind_counts = np.zeros(n_students, dtype=np.float64)

    for t in range(n_days):
        # Who was recently selected (last `recent_window` days before t)?
        if t > 0:
            start = max(0, t - recent_window)
            recent = binary[start:t].sum(axis=0) > 0  # (n_students,)

            if ind_counts.max() > 0 and recent.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    # cooc_rate[s, j] = cooc[s,j] / ind_counts[s]
                    cooc_rate = np.where(
                        ind_counts[:, None] > 0,
                        cooc / ind_counts[:, None],
                        0.0
                    )
                # Score for each student: dot product with recent indicator
                pairwise_score[t] = (cooc_rate * recent[None, :]).sum(axis=1)

        # Update co-occurrence with today's data (AFTER scoring)
        selected = np.where(binary[t] == 1)[0]
        ind_counts[selected] += 1
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                cooc[selected[i], selected[j]] += 1
                cooc[selected[j], selected[i]] += 1

    return pairwise_score


# ============================================================
# Feature 4: Student Position in Class
# ============================================================
def compute_student_position(n_students: int = 55) -> np.ndarray:
    """
    Normalized student ID: student_id / n_students ∈ (0, 1].

    Static per-student feature representing ordinal position in class.

    Returns: (n_students,) float32 array
    """
    return np.arange(1, n_students + 1, dtype=np.float32) / float(n_students)


# ============================================================
# Feature 5: Day-of-Week Pattern
# ============================================================
def compute_dow_pattern(binary: np.ndarray, n_dow: int = 5,
                         start_day: int = 0) -> np.ndarray:
    """
    Historical frequency of each student on each day-of-week.

    Uses `(start_day + t) % n_dow` as the DOW proxy (5-day school week).
    Only uses data before day t — no leakage.

    Args:
        binary: (n_days, n_students)
        n_dow: Number of DOW buckets (5 = Mon-Fri)
        start_day: Absolute day index of binary[0] (for consistent DOW mapping)

    Returns:
        (n_days, n_students) float32 array
    """
    n_days, n_students = binary.shape
    dow_freq = np.zeros((n_days, n_students), dtype=np.float32)

    dow_sel_count = np.zeros((n_dow, n_students), dtype=np.float64)
    dow_total_count = np.zeros(n_dow, dtype=np.float64)
    base_rate = 6.0 / n_students  # Prior: overall selection rate

    for t in range(n_days):
        dow = (start_day + t) % n_dow

        if dow_total_count[dow] > 0:
            dow_freq[t] = (dow_sel_count[dow] / dow_total_count[dow]).astype(np.float32)
        else:
            # No history for this DOW yet: use base rate
            dow_freq[t] = float(base_rate)

        # Update AFTER computing features for day t
        dow_sel_count[dow] += binary[t]
        dow_total_count[dow] += 1

    return dow_freq


# ============================================================
# Feature 6: Entropy of Teacher Behavior
# ============================================================
def compute_teacher_entropy(binary: np.ndarray,
                             windows: List[int] = (7, 20)) -> List[np.ndarray]:
    """
    Shannon entropy of the teacher's selection distribution over sliding windows.

    H(t) = -sum_s [ p_s * log(p_s) ]

    where p_s = frequency of student s in the last `w` days before t.

    - High H: teacher selects widely (unpredictable)
    - Low H: teacher concentrates on a few students (predictable)

    This is a DAY-level feature (same value for all students on day t).

    Args:
        binary: (n_days, n_students)
        windows: list of window sizes to compute entropy over

    Returns:
        List of (n_days,) float32 arrays, one per window
    """
    n_days, n_students = binary.shape
    results = []

    cumsum = np.vstack([
        np.zeros((1, n_students), dtype=np.float64),
        np.cumsum(binary.astype(np.float64), axis=0)
    ])

    for w in windows:
        entropy = np.zeros(n_days, dtype=np.float32)
        for t in range(1, n_days):
            start = max(0, t - w)
            window_counts = cumsum[t] - cumsum[start]
            total = window_counts.sum()
            if total > 0:
                probs = window_counts / total
                probs = probs[probs > 1e-12]
                entropy[t] = float(-np.sum(probs * np.log(probs)))
        results.append(entropy)

    return results


# ============================================================
# Build Full Feature Matrix
# ============================================================
def build_gb_features(binary: np.ndarray, n_students: int = 55,
                       start_day: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the complete (n_days * n_students, 9) feature matrix for training.

    For each day t and each student s, creates one row with all features.
    The target y = 1 if student s was selected on day t.

    Args:
        binary: (n_days, n_students) selection matrix
        n_students: number of students
        start_day: absolute day index of binary[0] (for DOW consistency)

    Returns:
        X: (n_days * n_students, 9) float32 feature matrix
        y: (n_days * n_students,)   float32 target vector
    """
    n_days = len(binary)

    print("  Building Gradient Boosting features...")

    print("    [1/6] Days since last call...")
    days_since = compute_days_since_last(binary)                  # (n_days, n_students)

    print("    [2/6] Frequency windows (5/10/20 days)...")
    freq_5, freq_10, freq_20 = compute_frequency_windows(binary, windows=[5, 10, 20])

    print("    [3/6] Pairwise co-selection scores...")
    pairwise = compute_pairwise_coselection(binary, recent_window=3)

    print("    [4/6] Student position (1-55)...")
    position = compute_student_position(n_students)               # (n_students,)
    position_mat = np.tile(position, (n_days, 1))                 # (n_days, n_students)

    print("    [5/6] Day-of-week pattern...")
    dow = compute_dow_pattern(binary, n_dow=5, start_day=start_day)  # (n_days, n_students)

    print("    [6/6] Teacher entropy (7-day and 20-day)...")
    entropy_7, entropy_20 = compute_teacher_entropy(binary, windows=[7, 20])
    entropy_7_mat  = np.tile(entropy_7[:, None],  (1, n_students))  # (n_days, n_students)
    entropy_20_mat = np.tile(entropy_20[:, None], (1, n_students))

    # Stack all features: (n_days, n_students, 9)
    X_3d = np.stack([
        days_since,      # 0: days since last call
        freq_5,          # 1: freq last 5 days
        freq_10,         # 2: freq last 10 days
        freq_20,         # 3: freq last 20 days
        pairwise,        # 4: pairwise co-selection
        position_mat,    # 5: student position
        dow,             # 6: DOW frequency
        entropy_7_mat,   # 7: teacher entropy 7d
        entropy_20_mat,  # 8: teacher entropy 20d
    ], axis=2)

    X = X_3d.reshape(n_days * n_students, 9).astype(np.float32)
    y = binary.reshape(n_days * n_students).astype(np.float32)

    pos_rate = y.mean()
    print(f"    Feature matrix: {X.shape} | Target: {y.shape} | "
          f"Positive rate: {pos_rate:.4f} (expected ~{6/n_students:.4f})")

    return X, y


# ============================================================
# Incremental State for Online Simulation
# ============================================================
class IncrementalGBState:
    """
    Maintains running statistics for efficient per-day feature computation
    during the online simulation — avoids full recomputation each step.
    """

    FEATURE_NAMES = [
        'days_since_last', 'freq_5', 'freq_10', 'freq_20',
        'pairwise_score', 'student_position',
        'dow_freq', 'entropy_7d', 'entropy_20d'
    ]

    def __init__(self, binary_history: np.ndarray, n_students: int = 55,
                 start_day: int = 0):
        """
        Initialize from historical binary data.

        Args:
            binary_history: (n_days, n_students) historical data
            n_students: number of students
            start_day: absolute index of binary_history[0] for DOW
        """
        self.n_students = n_students
        self.start_day = start_day
        self.binary_history = binary_history.copy()

        # --- Running statistics ---
        # Co-occurrence
        self.cooc = np.zeros((n_students, n_students), dtype=np.float64)
        self.ind_counts = np.zeros(n_students, dtype=np.float64)

        # DOW
        self.dow_sel = np.zeros((5, n_students), dtype=np.float64)
        self.dow_total = np.zeros(5, dtype=np.float64)

        # Last seen day for each student
        self.last_seen = -np.ones(n_students, dtype=np.int32)

        # Populate from history
        for t, row in enumerate(binary_history):
            abs_t = start_day + t
            selected = np.where(row == 1)[0]

            # Co-occurrence update
            self.ind_counts[selected] += 1
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    self.cooc[selected[i], selected[j]] += 1
                    self.cooc[selected[j], selected[i]] += 1

            # DOW update
            dow = abs_t % 5
            self.dow_sel[dow] += row
            self.dow_total[dow] += 1

            # Last seen
            for s in selected:
                self.last_seen[s] = abs_t

    def get_features_for_next_day(self, next_abs_day: int) -> np.ndarray:
        """
        Build (n_students, 9) feature rows for the NEXT prediction day.

        Args:
            next_abs_day: Absolute day index of the day to predict

        Returns:
            (n_students, 9) float32 feature array
        """
        n = self.n_students
        hist = self.binary_history

        # [0] Days since last call
        days_since = np.array([
            float(next_abs_day - self.last_seen[s]) if self.last_seen[s] >= 0
            else float(next_abs_day + 1)
            for s in range(n)
        ], dtype=np.float32)
        days_since_norm = np.log1p(days_since)

        # [1-3] Frequency windows
        def freq_window(w):
            window = hist[-w:] if len(hist) >= w else hist
            return window.mean(axis=0).astype(np.float32)

        freq_5  = freq_window(5)
        freq_10 = freq_window(10)
        freq_20 = freq_window(20)

        # [4] Pairwise co-selection
        recent = hist[-3:].sum(axis=0) > 0 if len(hist) >= 3 else np.zeros(n, dtype=bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            cooc_rate = np.where(
                self.ind_counts[:, None] > 0,
                self.cooc / self.ind_counts[:, None],
                0.0
            )
        pairwise = (cooc_rate * recent[None, :]).sum(axis=1).astype(np.float32)

        # [5] Student position
        position = (np.arange(1, n + 1, dtype=np.float32) / n)

        # [6] DOW frequency
        dow = next_abs_day % 5
        if self.dow_total[dow] > 0:
            dow_freq = (self.dow_sel[dow] / self.dow_total[dow]).astype(np.float32)
        else:
            dow_freq = np.full(n, 6.0 / n, dtype=np.float32)

        # [7-8] Teacher entropy
        def entropy_window(w):
            window = hist[-w:] if len(hist) >= w else hist
            counts = window.sum(axis=0)
            total = counts.sum()
            if total > 0:
                probs = counts / total
                probs = probs[probs > 1e-12]
                return float(-np.sum(probs * np.log(probs)))
            return 0.0

        ent_7  = entropy_window(7)
        ent_20 = entropy_window(20)

        X_day = np.column_stack([
            days_since_norm,
            freq_5, freq_10, freq_20,
            pairwise,
            position,
            dow_freq,
            np.full(n, ent_7, dtype=np.float32),
            np.full(n, ent_20, dtype=np.float32),
        ])

        return X_day.astype(np.float32)

    def update(self, binary_day: np.ndarray, abs_day: int):
        """
        Update running statistics with one new day's data.

        Args:
            binary_day: (n_students,) binary selection for today
            abs_day: absolute day index for today
        """
        self.binary_history = np.vstack([
            self.binary_history, binary_day[None, :]
        ])

        selected = np.where(binary_day == 1)[0]

        # Update co-occurrence
        self.ind_counts[selected] += 1
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                self.cooc[selected[i], selected[j]] += 1
                self.cooc[selected[j], selected[i]] += 1

        # Update DOW
        dow = abs_day % 5
        self.dow_sel[dow] += binary_day
        self.dow_total[dow] += 1

        # Update last seen
        for s in selected:
            self.last_seen[s] = abs_day
