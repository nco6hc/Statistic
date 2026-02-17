"""
Feature Engineering for Student Selection Prediction

This module creates features for predicting which students will be selected next
based on historical selection patterns.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


class StudentSelectionFeatureEngineer:
    """Creates features for predicting student selections."""
    
    def __init__(self, binary_vectors: np.ndarray, days: np.ndarray, total_students: int = 55):
        """
        Initialize the feature engineer.
        
        Args:
            binary_vectors: Binary matrix (n_samples, n_students)
            days: Array of day numbers corresponding to each sample
            total_students: Total number of students
        """
        self.binary_vectors = binary_vectors
        self.days = days
        self.total_students = total_students
        self.n_samples = len(binary_vectors)
        
    def compute_frequency_features(self, lookback_windows: List[int] = [5, 10, 20, 30]) -> np.ndarray:
        """
        Compute frequency of each student in last N days.
        
        Args:
            lookback_windows: List of window sizes to compute frequencies
            
        Returns:
            Feature matrix of shape (n_samples, n_students * len(lookback_windows))
        """
        print(f"Computing frequency features for windows: {lookback_windows}...")
        
        features = []
        
        for window in lookback_windows:
            window_features = np.zeros((self.n_samples, self.total_students))
            
            for i in range(self.n_samples):
                start_idx = max(0, i - window)
                if i > 0:
                    # Sum selections in the window and normalize by window size
                    window_selections = self.binary_vectors[start_idx:i].sum(axis=0)
                    window_features[i] = window_selections / min(window, i)
            
            features.append(window_features)
        
        return np.hstack(features)
    
    def compute_days_since_selected(self) -> np.ndarray:
        """
        Compute days since each student was last selected.
        
        Returns:
            Feature matrix of shape (n_samples, n_students)
        """
        print("Computing days since last selected...")
        
        days_since = np.zeros((self.n_samples, self.total_students))
        last_selected_day = np.full(self.total_students, -np.inf)
        
        for i in range(self.n_samples):
            current_day = self.days[i]
            
            # Update days since for all students
            days_since[i] = current_day - last_selected_day
            
            # Cap at a maximum value to avoid huge numbers
            days_since[i] = np.minimum(days_since[i], 1000)
            
            # Update last selected day for students selected today
            if i > 0:  # Don't use current day's selections for features
                selected_students = np.where(self.binary_vectors[i-1] == 1)[0]
                last_selected_day[selected_students] = self.days[i-1]
        
        return days_since
    
    def compute_pairwise_cooccurrence(self, lookback_window: int = 30) -> np.ndarray:
        """
        Compute pairwise co-occurrence frequencies.
        For each student, compute how often they appear with other students.
        
        Args:
            lookback_window: Window size for computing co-occurrences
            
        Returns:
            Feature matrix of shape (n_samples, n_students * n_students)
            Flattened co-occurrence matrices for each sample
        """
        print(f"Computing pairwise co-occurrence (window={lookback_window})...")
        
        features = np.zeros((self.n_samples, self.total_students * self.total_students))
        
        for i in range(self.n_samples):
            start_idx = max(0, i - lookback_window)
            
            if i > 0:
                # Get historical data
                window_data = self.binary_vectors[start_idx:i]
                
                # Compute co-occurrence matrix
                # cooccur[j, k] = how many times student j and k were selected together
                cooccur = np.dot(window_data.T, window_data)
                
                # Normalize by window size
                cooccur = cooccur / min(lookback_window, i)
                
                # Flatten to feature vector
                features[i] = cooccur.flatten()
        
        return features
    
    def compute_rolling_statistics(self, windows: List[int] = [5, 10, 20]) -> np.ndarray:
        """
        Compute rolling window statistics for each student.
        
        Args:
            windows: List of window sizes
            
        Returns:
            Feature matrix with statistics (mean, std, min, max) for each window
        """
        print(f"Computing rolling statistics for windows: {windows}...")
        
        all_features = []
        
        for window in windows:
            # Mean frequency
            mean_features = np.zeros((self.n_samples, self.total_students))
            # Std of frequency
            std_features = np.zeros((self.n_samples, self.total_students))
            # Min (0 or 1 basically)
            min_features = np.zeros((self.n_samples, self.total_students))
            # Max (0 or 1 basically)
            max_features = np.zeros((self.n_samples, self.total_students))
            
            for i in range(self.n_samples):
                start_idx = max(0, i - window)
                
                if i > 0:
                    window_data = self.binary_vectors[start_idx:i]
                    
                    if len(window_data) > 0:
                        mean_features[i] = window_data.mean(axis=0)
                        std_features[i] = window_data.std(axis=0)
                        min_features[i] = window_data.min(axis=0)
                        max_features[i] = window_data.max(axis=0)
            
            all_features.extend([mean_features, std_features, min_features, max_features])
        
        return np.hstack(all_features)
    
    def compute_selection_patterns(self) -> np.ndarray:
        """
        Compute additional pattern-based features.
        
        Returns:
            Feature matrix with pattern features
        """
        print("Computing selection pattern features...")
        
        features = []
        
        # 1. Total selections in last N days (overall activity)
        for window in [5, 10, 20]:
            total_selections = np.zeros((self.n_samples, 1))
            for i in range(self.n_samples):
                start_idx = max(0, i - window)
                if i > 0:
                    total_selections[i] = self.binary_vectors[start_idx:i].sum()
            features.append(total_selections)
        
        # 2. Consecutive selection streak for each student
        streak_features = np.zeros((self.n_samples, self.total_students))
        current_streak = np.zeros(self.total_students)
        
        for i in range(1, self.n_samples):
            # Update streaks based on previous day
            selected = self.binary_vectors[i-1] == 1
            current_streak[selected] += 1
            current_streak[~selected] = 0
            streak_features[i] = current_streak.copy()
        
        features.append(streak_features)
        
        # 3. Day of sequence (normalized)
        day_feature = (self.days - self.days.min()) / (self.days.max() - self.days.min())
        features.append(day_feature.reshape(-1, 1))
        
        return np.hstack(features)
    
    def create_feature_matrix(self, 
                            include_frequency: bool = True,
                            include_days_since: bool = True,
                            include_cooccurrence: bool = True,
                            include_rolling_stats: bool = True,
                            include_patterns: bool = True,
                            frequency_windows: List[int] = [5, 10, 20, 30],
                            rolling_windows: List[int] = [5, 10, 20],
                            cooccurrence_window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create complete feature matrix and label matrix.
        
        Args:
            include_frequency: Include frequency features
            include_days_since: Include days since selected features
            include_cooccurrence: Include pairwise co-occurrence features
            include_rolling_stats: Include rolling statistics
            include_patterns: Include pattern-based features
            frequency_windows: Windows for frequency features
            rolling_windows: Windows for rolling statistics
            cooccurrence_window: Window for co-occurrence
            
        Returns:
            Tuple of (X, Y) where:
                X: Feature matrix (n_samples, n_features)
                Y: Label matrix (n_samples, n_students) - binary indicators
        """
        print("="*60)
        print("CREATING FEATURE MATRIX")
        print("="*60)
        
        feature_components = []
        feature_names = []
        
        if include_frequency:
            freq_features = self.compute_frequency_features(frequency_windows)
            feature_components.append(freq_features)
            for window in frequency_windows:
                feature_names.extend([f'freq_{window}d_student_{i+1}' for i in range(self.total_students)])
            print(f"✓ Frequency features: {freq_features.shape}")
        
        if include_days_since:
            days_features = self.compute_days_since_selected()
            feature_components.append(days_features)
            feature_names.extend([f'days_since_student_{i+1}' for i in range(self.total_students)])
            print(f"✓ Days since features: {days_features.shape}")
        
        if include_cooccurrence:
            cooccur_features = self.compute_pairwise_cooccurrence(cooccurrence_window)
            feature_components.append(cooccur_features)
            for i in range(self.total_students):
                for j in range(self.total_students):
                    feature_names.append(f'cooccur_s{i+1}_s{j+1}')
            print(f"✓ Co-occurrence features: {cooccur_features.shape}")
        
        if include_rolling_stats:
            rolling_features = self.compute_rolling_statistics(rolling_windows)
            feature_components.append(rolling_features)
            for window in rolling_windows:
                for stat in ['mean', 'std', 'min', 'max']:
                    feature_names.extend([f'rolling_{window}d_{stat}_student_{i+1}' for i in range(self.total_students)])
            print(f"✓ Rolling statistics: {rolling_features.shape}")
        
        if include_patterns:
            pattern_features = self.compute_selection_patterns()
            feature_components.append(pattern_features)
            print(f"✓ Pattern features: {pattern_features.shape}")
        
        # Combine all features
        X = np.hstack(feature_components)
        
        # Labels are the binary vectors (what we want to predict)
        Y = self.binary_vectors.copy()
        
        print("="*60)
        print(f"FINAL FEATURE MATRIX: {X.shape}")
        print(f"FINAL LABEL MATRIX: {Y.shape}")
        print("="*60)
        
        return X, Y, feature_names
    
    def prepare_train_test(self, X: np.ndarray, Y: np.ndarray, 
                          test_size: float = 0.1,
                          min_train_samples: int = 100) -> Tuple:
        """
        Prepare train/test split for time series data.
        
        Args:
            X: Feature matrix
            Y: Label matrix
            test_size: Proportion for test set
            min_train_samples: Minimum samples to use for training
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test)
        """
        print(f"\nPreparing train/test split (test_size={test_size})...")
        
        # Remove samples with insufficient historical data
        # (e.g., first few samples where features can't be computed properly)
        valid_start = min_train_samples
        X = X[valid_start:]
        Y = Y[valid_start:]
        
        # Split chronologically
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        Y_train = Y[:split_idx]
        Y_test = Y[split_idx:]
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features per sample: {X_train.shape[1]}")
        print(f"Labels per sample: {Y_train.shape[1]}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("⚠ Warning: NaN or Inf values detected in training features")
        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            print("⚠ Warning: NaN or Inf values detected in test features")
        
        return X_train, X_test, Y_train, Y_test


def main():
    """Main execution function."""
    print("Loading processed data...")
    
    # Load the processed data from previous step
    X_train_raw = np.load('X_train.npy')
    X_test_raw = np.load('X_test.npy')
    y_train_days = np.load('y_train.npy')
    y_test_days = np.load('y_test.npy')
    
    # Combine for feature engineering (we'll resplit later)
    binary_vectors = np.vstack([X_train_raw, X_test_raw])
    days = np.hstack([y_train_days, y_test_days])
    
    print(f"Total samples: {len(binary_vectors)}")
    print(f"Days range: {days.min()} to {days.max()}")
    
    # Initialize feature engineer
    engineer = StudentSelectionFeatureEngineer(binary_vectors, days, total_students=55)
    
    # Create feature matrix with all features
    X, Y, feature_names = engineer.create_feature_matrix(
        include_frequency=True,
        include_days_since=True,
        include_cooccurrence=True,
        include_rolling_stats=True,
        include_patterns=True,
        frequency_windows=[5, 10, 20, 30],
        rolling_windows=[5, 10, 20],
        cooccurrence_window=30
    )
    
    # Prepare train/test split
    X_train, X_test, Y_train, Y_test = engineer.prepare_train_test(
        X, Y, test_size=0.1, min_train_samples=50
    )
    
    # Save feature matrices
    print("\nSaving feature matrices...")
    np.save('X_train_features.npy', X_train)
    np.save('X_test_features.npy', X_test)
    np.save('Y_train_labels.npy', Y_train)
    np.save('Y_test_labels.npy', Y_test)
    
    # Save feature names for interpretability
    with open('feature_names.txt', 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i}: {name}\n")
    
    print("✓ Saved: X_train_features.npy, X_test_features.npy")
    print("✓ Saved: Y_train_labels.npy, Y_test_labels.npy")
    print("✓ Saved: feature_names.txt")
    
    # Display feature matrix statistics
    print("\n" + "="*60)
    print("FEATURE MATRIX STATISTICS")
    print("="*60)
    print(f"Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Feature mean: {X_train.mean():.4f}")
    print(f"Feature std: {X_train.std():.4f}")
    print(f"Sparsity: {(X_train == 0).sum() / X_train.size * 100:.2f}% zeros")
    
    print("\n" + "="*60)
    print("LABEL MATRIX STATISTICS")
    print("="*60)
    print(f"Average students per day (train): {Y_train.sum(axis=1).mean():.2f}")
    print(f"Average students per day (test): {Y_test.sum(axis=1).mean():.2f}")
    print(f"Label distribution: {Y_train.sum() / Y_train.size * 100:.2f}% selected")
    
    return X_train, X_test, Y_train, Y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test, feature_names = main()
