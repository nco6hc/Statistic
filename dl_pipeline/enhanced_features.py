"""
Feature Engineering for Enhanced LSTM Model

Computes enriched features from raw binary selection vectors:
- Raw binary selection (55-dim)
- 7-day selection frequency (55-dim)
- 14-day selection frequency (55-dim)
- Recency signal (55-dim)

Total: 220 features per timestep (4 x 55 students)

All features only look BACKWARD in time (no data leakage).
"""

import numpy as np
from typing import Tuple


def compute_frequency_features(binary_vectors: np.ndarray, window: int) -> np.ndarray:
    """
    Compute selection frequency over a sliding window.
    
    For each day t, freq[t, s] = mean of binary_vectors[max(0,t-window):t, s]
    Only uses past data (no data leakage).
    
    Args:
        binary_vectors: Binary matrix (n_days, n_students)
        window: Number of past days to average over
        
    Returns:
        Frequency features (n_days, n_students)
    """
    n_days, n_students = binary_vectors.shape
    freq = np.zeros((n_days, n_students), dtype=np.float32)
    
    # Use cumulative sum for efficient computation
    cumsum = np.vstack([np.zeros((1, n_students), dtype=np.float32),
                        np.cumsum(binary_vectors, axis=0)])
    
    for t in range(1, n_days):
        start = max(0, t - window)
        count = t - start
        freq[t] = (cumsum[t] - cumsum[start]) / count
    
    return freq


def compute_recency_features(binary_vectors: np.ndarray) -> np.ndarray:
    """
    Compute recency signal: 1 / (days_since_last_selected + 1)
    
    Higher value = more recently selected.
    Only uses past data (no data leakage).
    
    Args:
        binary_vectors: Binary matrix (n_days, n_students)
        
    Returns:
        Recency features (n_days, n_students)
    """
    n_days, n_students = binary_vectors.shape
    recency = np.zeros((n_days, n_students), dtype=np.float32)
    
    # Track last selected day for each student
    last_selected = -np.ones(n_students, dtype=np.float64)
    
    for t in range(n_days):
        # Compute recency based on last selection BEFORE day t
        mask = last_selected >= 0
        if np.any(mask):
            days_since = t - last_selected[mask]
            recency[t, mask] = 1.0 / (days_since + 1)
        
        # Update last_selected AFTER computing features for day t
        selected = binary_vectors[t] == 1
        last_selected[selected] = t
    
    return recency


def build_enriched_features(binary_vectors: np.ndarray) -> np.ndarray:
    """
    Build complete enriched feature set from raw binary vectors.
    
    Features per timestep (all 55-dim):
    1. Raw binary selection - who was selected today
    2. 7-day frequency - short-term selection rate
    3. 14-day frequency - medium-term selection rate
    4. Recency signal - how recently each student was selected
    
    Total: 220 features per timestep (4 x 55 students)
    
    Args:
        binary_vectors: Binary matrix (n_days, n_students)
        
    Returns:
        Enriched features (n_days, 220)
    """
    print("  Computing enriched features...")
    
    freq_7 = compute_frequency_features(binary_vectors, window=7)
    print("    - 7-day frequency features computed")
    
    freq_14 = compute_frequency_features(binary_vectors, window=14)
    print("    - 14-day frequency features computed")
    
    recency = compute_recency_features(binary_vectors)
    print("    - Recency features computed")
    
    # Stack all features along feature dimension
    enriched = np.concatenate([
        binary_vectors,   # Raw selection (55)
        freq_7,           # 7-day frequency (55)
        freq_14,          # 14-day frequency (55)
        recency,          # Recency signal (55)
    ], axis=1)
    
    print(f"    Enriched shape: {enriched.shape} "
          f"({enriched.shape[1]} features per timestep)")
    
    return enriched
