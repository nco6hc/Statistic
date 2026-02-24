"""
Factored Markov Chain for Student Selection Prediction

Since the full state space C(55,6) ~ 202 million is too large,
we use a FACTORED approach with 3 components:

1. Individual Transition: P(student_i today | student_i yesterday, day before, ...)
   - Captures rotation patterns: selected yesterday -> less likely today
   - Uses order-2 (looks at last 2 days)

2. Pairwise Co-occurrence: P(student_i and student_j selected together)
   - Captures group patterns: "A and B often appear together"
   - Normalized by individual frequencies

3. Cooldown Signal: Days since last selected for each student
   - Students selected very recently may be less likely
   - Students not selected for a long time may be more likely

The combined Markov probability is blended with LSTM probabilities
for the hybrid prediction.
"""

import numpy as np
from typing import Tuple


class FactoredMarkovChain:
    """
    Factored Markov Chain for student selection prediction.
    
    Instead of modeling the full joint distribution over all C(55,6)
    possible selections, we factor it into:
    - Per-student transition probabilities (order-2)
    - Pairwise co-occurrence scores
    - Cooldown-based probability adjustments
    """
    
    def __init__(self, n_students: int = 55, smoothing: float = 1.0):
        """
        Args:
            n_students: Number of students
            smoothing: Laplace smoothing parameter to avoid zero probabilities
        """
        self.n_students = n_students
        self.smoothing = smoothing
        
        # Order-2 transition counts: P(selected_today | state_yesterday, state_day_before)
        # state = 0 (not selected) or 1 (selected)
        # Shape: (n_students, 2, 2) -> [student][yesterday_state][day_before_state] -> count_selected
        self.transition_counts = np.zeros((n_students, 2, 2), dtype=np.float64)
        self.transition_totals = np.zeros((n_students, 2, 2), dtype=np.float64)
        
        # Pairwise co-occurrence: how often student i and j are selected together
        # Shape: (n_students, n_students)
        self.cooccurrence = np.zeros((n_students, n_students), dtype=np.float64)
        self.individual_counts = np.zeros(n_students, dtype=np.float64)
        
        # Total days observed
        self.total_days = 0
    
    def fit(self, binary_vectors: np.ndarray):
        """
        Learn transition and co-occurrence statistics from historical data.
        
        Args:
            binary_vectors: Binary matrix (n_days, n_students)
        """
        n_days = len(binary_vectors)
        self.total_days = n_days
        
        print(f"  Fitting Markov Chain on {n_days} days...")
        
        # Individual frequency counts
        self.individual_counts = binary_vectors.sum(axis=0)
        
        # Order-2 transition counts
        for t in range(2, n_days):
            today = binary_vectors[t]
            yesterday = binary_vectors[t - 1]
            day_before = binary_vectors[t - 2]
            
            for s in range(self.n_students):
                y_state = int(yesterday[s])
                db_state = int(day_before[s])
                
                self.transition_totals[s, y_state, db_state] += 1
                if today[s] == 1:
                    self.transition_counts[s, y_state, db_state] += 1
        
        # Pairwise co-occurrence
        for t in range(n_days):
            selected = np.where(binary_vectors[t] == 1)[0]
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    self.cooccurrence[selected[i], selected[j]] += 1
                    self.cooccurrence[selected[j], selected[i]] += 1
        
        print(f"    - Order-2 transitions computed")
        print(f"    - Pairwise co-occurrence matrix computed")
        print(f"    - Average selection rate: {self.individual_counts.mean() / n_days:.4f}")
    
    def predict_probabilities(self, recent_days: np.ndarray) -> np.ndarray:
        """
        Predict probability for each student using Markov Chain.
        
        Args:
            recent_days: Binary vectors for recent days, shape (n_recent, n_students)
                        At least 2 days needed for order-2 transitions.
        
        Returns:
            Probability vector (n_students,)
        """
        n_students = self.n_students
        
        # --- Component 1: Transition probabilities ---
        yesterday = recent_days[-1]
        day_before = recent_days[-2] if len(recent_days) >= 2 else np.zeros(n_students)
        
        transition_probs = np.zeros(n_students, dtype=np.float64)
        for s in range(n_students):
            y_state = int(yesterday[s])
            db_state = int(day_before[s])
            
            count = self.transition_counts[s, y_state, db_state]
            total = self.transition_totals[s, y_state, db_state]
            
            # Laplace-smoothed probability
            transition_probs[s] = (count + self.smoothing) / (total + 2 * self.smoothing)
        
        # --- Component 2: Co-occurrence boost ---
        # Based on who was selected yesterday, which students tend to co-occur?
        yesterday_selected = np.where(yesterday == 1)[0]
        cooccurrence_scores = np.zeros(n_students, dtype=np.float64)
        
        if len(yesterday_selected) > 0:
            for sel in yesterday_selected:
                cooccurrence_scores += self.cooccurrence[sel]
            
            # Normalize by number of yesterday's selections and total days
            cooccurrence_scores /= (len(yesterday_selected) * max(self.total_days, 1))
            
            # Boost: students who co-occur with yesterday's selections
            # But scale it relative to baseline frequency
            baseline_freq = self.individual_counts / max(self.total_days, 1)
            # Ratio of co-occurrence to baseline (>1 means more likely with this group)
            with np.errstate(divide='ignore', invalid='ignore'):
                cooccurrence_ratio = np.where(
                    baseline_freq > 0,
                    cooccurrence_scores / baseline_freq,
                    1.0
                )
            cooccurrence_ratio = np.clip(cooccurrence_ratio, 0.5, 2.0)
        else:
            cooccurrence_ratio = np.ones(n_students, dtype=np.float64)
        
        # --- Component 3: Cooldown signal ---
        # How many days since last selected? More recent = potentially less likely (rotation)
        days_since = np.full(n_students, len(recent_days), dtype=np.float64)
        for t in range(len(recent_days) - 1, -1, -1):
            selected_now = np.where(recent_days[t] == 1)[0]
            for s in selected_now:
                if days_since[s] == len(recent_days):
                    days_since[s] = len(recent_days) - 1 - t
        
        # Cooldown factor: very recently selected -> slight penalty,
        # not selected for a while -> slight boost
        # Using a mild sigmoid-like curve centered at ~3 days
        cooldown_factor = 1.0 / (1.0 + np.exp(-(days_since - 3.0) / 2.0))
        # Normalize so mean is ~1
        cooldown_factor = cooldown_factor / cooldown_factor.mean()
        
        # --- Combine components ---
        # Transition is the primary signal, modulated by co-occurrence and cooldown
        markov_probs = transition_probs * cooccurrence_ratio * cooldown_factor
        
        # Normalize to valid probability distribution
        total = markov_probs.sum()
        if total > 0:
            markov_probs = markov_probs / total
        else:
            markov_probs = np.ones(n_students) / n_students
        
        return markov_probs
    
    def update(self, new_day: np.ndarray, yesterday: np.ndarray, 
               day_before: np.ndarray):
        """
        Online update: add one new observation to the Markov Chain.
        
        Args:
            new_day: Binary vector for today (n_students,)
            yesterday: Binary vector for yesterday (n_students,)
            day_before: Binary vector for day before yesterday (n_students,)
        """
        self.total_days += 1
        
        # Update individual counts
        self.individual_counts += new_day
        
        # Update transitions
        for s in range(self.n_students):
            y_state = int(yesterday[s])
            db_state = int(day_before[s])
            self.transition_totals[s, y_state, db_state] += 1
            if new_day[s] == 1:
                self.transition_counts[s, y_state, db_state] += 1
        
        # Update co-occurrence
        selected = np.where(new_day == 1)[0]
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                self.cooccurrence[selected[i], selected[j]] += 1
                self.cooccurrence[selected[j], selected[i]] += 1


if __name__ == "__main__":
    """Quick test of Markov Chain."""
    print("Testing Factored Markov Chain...")
    
    np.random.seed(42)
    n_days, n_students = 100, 55
    
    # Generate random binary data (6 selected per day)
    binary = np.zeros((n_days, n_students), dtype=np.float32)
    for t in range(n_days):
        selected = np.random.choice(n_students, size=6, replace=False)
        binary[t, selected] = 1
    
    mc = FactoredMarkovChain(n_students=n_students)
    mc.fit(binary)
    
    probs = mc.predict_probabilities(binary[-14:])
    print(f"  Probability shape: {probs.shape}")
    print(f"  Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Sum: {probs.sum():.4f}")
    print("  Test passed!")
