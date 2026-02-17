"""
Monte Carlo Student Selection Model
Uses historical probability distributions to predict student selections
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict
import json


class MonteCarloStudentPredictor:
    """
    Monte Carlo simulation model for student selection prediction.
    Uses historical selection frequencies and patterns to generate probabilistic predictions.
    """
    
    def __init__(self, num_students: int = 55, selections_per_day: int = 6):
        self.num_students = num_students
        self.selections_per_day = selections_per_day
        
        # Probability distribution for each student
        self.student_probabilities = np.ones(num_students) / num_students
        
        # Co-occurrence matrix: how often students are selected together
        self.co_occurrence = np.zeros((num_students, num_students))
        
        # Recency weights: more recent selections weighted higher
        self.recency_decay = 0.95
        
        # Selection history
        self.selection_history = []
        
    def fit(self, historical_data: np.ndarray, use_recency: bool = True):
        """
        Fit the model on historical selection data.
        
        Args:
            historical_data: Array of shape (n_days, selections_per_day)
                            Each row contains student IDs selected that day
            use_recency: Whether to apply recency weighting (recent days weighted more)
        """
        n_days = len(historical_data)
        
        # Calculate base selection frequencies
        selection_counts = np.zeros(self.num_students)
        
        # Build co-occurrence matrix and calculate weighted frequencies
        for day_idx, selections in enumerate(historical_data):
            # Calculate recency weight (more recent = higher weight)
            if use_recency:
                weight = self.recency_decay ** (n_days - day_idx - 1)
            else:
                weight = 1.0
            
            # Update selection counts with recency weight
            for student_id in selections:
                if 0 <= student_id < self.num_students:
                    selection_counts[student_id] += weight
            
            # Update co-occurrence matrix
            for i, student_i in enumerate(selections):
                for j, student_j in enumerate(selections):
                    if i != j and 0 <= student_i < self.num_students and 0 <= student_j < self.num_students:
                        self.co_occurrence[student_i, student_j] += weight
        
        # Normalize to get probabilities
        if selection_counts.sum() > 0:
            self.student_probabilities = selection_counts / selection_counts.sum()
        
        # Normalize co-occurrence matrix
        for i in range(self.num_students):
            if self.co_occurrence[i].sum() > 0:
                self.co_occurrence[i] = self.co_occurrence[i] / self.co_occurrence[i].sum()
        
        # Store history for pattern analysis
        self.selection_history = historical_data.tolist()
        
    def predict_monte_carlo(self, n_simulations: int = 5000, temperature: float = 1.0) -> List[List[int]]:
        """
        Generate predictions using Monte Carlo simulation.
        
        Args:
            n_simulations: Number of Monte Carlo simulations to run
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
        
        Returns:
            List of 5 groups of predicted students (each group has 6 students)
        """
        # Run Monte Carlo simulations
        simulation_results = []
        
        for _ in range(n_simulations):
            selected = self._simulate_selection(temperature)
            simulation_results.append(tuple(sorted(selected)))
        
        # Count most common selection combinations
        combination_counts = Counter(simulation_results)
        
        # Get top 5 most likely combinations
        top_combinations = combination_counts.most_common(5)
        
        # If we don't have 5 unique combinations, generate more with variation
        predictions = []
        for combo, _ in top_combinations:
            predictions.append(list(combo))
        
        # Fill remaining slots with probability-weighted sampling
        while len(predictions) < 5:
            selected = self._simulate_selection(temperature * 1.5)  # Higher temp for diversity
            if sorted(selected) not in [sorted(p) for p in predictions]:
                predictions.append(selected)
        
        return predictions[:5]
    
    def _simulate_selection(self, temperature: float = 1.0) -> List[int]:
        """
        Simulate a single selection using probability-weighted sampling.
        
        Args:
            temperature: Controls randomness (higher = more random)
        
        Returns:
            List of selected student IDs
        """
        selected = []
        available_students = list(range(self.num_students))
        probs = self.student_probabilities.copy()
        
        # Apply temperature to probabilities
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / probs.sum()
        
        for _ in range(self.selections_per_day):
            if len(available_students) == 0:
                break
            
            # Get probabilities for available students
            available_probs = probs[available_students]
            
            # Normalize probabilities (handle edge cases)
            prob_sum = available_probs.sum()
            if prob_sum > 0:
                available_probs = available_probs / prob_sum
            else:
                # If all probabilities are zero, use uniform distribution
                available_probs = np.ones(len(available_students)) / len(available_students)
            
            # Sample one student
            selected_idx = np.random.choice(len(available_students), p=available_probs)
            student_id = available_students[selected_idx]
            selected.append(student_id)
            
            # Remove selected student from available pool
            available_students.remove(student_id)
            
            # Boost probabilities of students who often appear with this student
            if len(selected) < self.selections_per_day:
                for idx in available_students:
                    probs[idx] *= (1 + self.co_occurrence[student_id, idx])
                
                # Re-normalize
                probs = probs / probs.sum()
        
        return selected
    
    def update_online(self, new_selections: List[int]):
        """
        Update the model with new selection data (online learning).
        
        Args:
            new_selections: List of student IDs selected
        """
        # Add to history
        self.selection_history.append(new_selections)
        
        # Refit with updated history (using recency weighting)
        historical_data = np.array(self.selection_history)
        self.fit(historical_data, use_recency=True)


def calculate_accuracy(predicted_groups: List[List[int]], actual: List[int]) -> Tuple[int, float, List[int]]:
    """
    Calculate accuracy of predictions.
    
    Args:
        predicted_groups: List of 5 predicted groups
        actual: Actual selected students
    
    Returns:
        (best_overlap, best_accuracy, overlap_counts)
    """
    actual_set = set(actual)
    overlap_counts = []
    
    for group in predicted_groups:
        overlap = len(set(group) & actual_set)
        overlap_counts.append(overlap)
    
    best_overlap = max(overlap_counts)
    best_accuracy = (best_overlap / len(actual)) * 100
    
    return best_overlap, best_accuracy, overlap_counts


def load_data(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load student selection data from Excel file.
    
    Returns:
        (data, num_students) where data is array of shape (n_days, 6)
    """
    df = pd.read_excel(file_path)
    
    # Get student ID columns (all columns except 'Days')
    student_cols = [col for col in df.columns if col.startswith('Student ID')]
    
    # Extract student selections
    data = df[student_cols].values
    
    # Determine number of unique students
    unique_students = set()
    for row in data:
        unique_students.update(row)
    num_students = max(unique_students) + 1
    
    return data, num_students


def print_separator(title: str = ""):
    """Print a section separator."""
    print("\n" + "=" * 70)
    if title:
        print(title.center(70))
        print("=" * 70)
    else:
        print("=" * 70)


if __name__ == "__main__":
    print("Monte Carlo Student Selection Predictor")
    print("=" * 70)
    
    # Load data
    data_path = "../Database.xlsx"
    print(f"\nLoading data from {data_path}...")
    data, num_students = load_data(data_path)
    print(f"Loaded {len(data)} days of data with {num_students} unique students")
    
    # Split data: 90% train, 10% test
    train_ratio = 0.9
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} days")
    print(f"  Testing: {len(test_data)} days")
    
    # Initialize and train model
    print_separator("TRAINING MONTE CARLO MODEL")
    model = MonteCarloStudentPredictor(num_students=num_students)
    model.fit(train_data, use_recency=True)
    
    print("\nModel trained successfully!")
    print(f"Student probability distribution:")
    top_10_students = np.argsort(model.student_probabilities)[-10:][::-1]
    for student_id in top_10_students:
        prob = model.student_probabilities[student_id] * 100
        print(f"  Student {student_id}: {prob:.2f}%")
    
    # Evaluate on test set with online learning
    print_separator("ONLINE LEARNING EVALUATION")
    print(f"\nRunning online learning simulation with {len(test_data)} test days...")
    print("Predictions made every 2 days, model updated after each prediction.\n")
    
    accuracies = []
    predictions_log = []
    
    for day_idx in range(0, len(test_data), 2):
        if day_idx >= len(test_data):
            break
        
        actual_day = split_idx + day_idx + 1
        
        print_separator(f"DAY {day_idx + 1}: MAKING PREDICTION")
        
        # Make prediction using Monte Carlo
        predicted_groups = model.predict_monte_carlo(n_simulations=5000, temperature=1.2)
        
        print(f"\nPredicted 5 groups of students:")
        for i, group in enumerate(predicted_groups, 1):
            print(f"  Group {i}: {group}")
        
        # Get actual selection
        actual = test_data[day_idx].tolist()
        
        # Calculate accuracy
        best_overlap, best_accuracy, overlap_counts = calculate_accuracy(predicted_groups, actual)
        accuracies.append(best_accuracy)
        
        # Display feedback
        print(f"\nDay {day_idx + 1} Feedback:")
        print(f"  Predicted Groups ({len(predicted_groups)} total):")
        for i, (group, overlap) in enumerate(zip(predicted_groups, overlap_counts), 1):
            marker = " (BEST)" if overlap == best_overlap else ""
            print(f"    Group {i}: {group} (overlap: {overlap}/6){marker}")
        print(f"  Actual: {actual}")
        print(f"  Best Overlap: {best_overlap} / 6")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        
        # Log prediction
        predictions_log.append({
            'day': day_idx + 1,
            'actual_day': actual_day,
            'predicted_groups': predicted_groups,
            'actual': actual,
            'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
            'overlap_counts': overlap_counts
        })
        
        # Update model with actual selection (online learning)
        model.update_online(actual)
        
        # Update with the next day too if available
        if day_idx + 1 < len(test_data):
            next_actual = test_data[day_idx + 1].tolist()
            model.update_online(next_actual)
    
    # Calculate statistics
    print_separator("ONLINE LEARNING STATISTICS")
    
    avg_accuracy = np.mean(accuracies)
    best_accuracy = np.max(accuracies)
    worst_accuracy = np.min(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"\nTotal predictions: {len(accuracies)}")
    print(f"Average accuracy: {avg_accuracy:.2f}%")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Worst accuracy: {worst_accuracy:.2f}%")
    print(f"Std deviation: {std_accuracy:.4f}")
    
    # Accuracy distribution
    print(f"\nAccuracy distribution:")
    for correct in range(7):
        accuracy_val = (correct / 6) * 100
        count = sum(1 for acc in accuracies if abs(acc - accuracy_val) < 1.0)
        percentage = (count / len(accuracies)) * 100
        print(f"  {correct}/6 correct: {count:3d} ({percentage:5.1f}%)")
    
    # Trend analysis
    if len(accuracies) >= 20:
        first_10 = np.mean(accuracies[:10])
        last_10 = np.mean(accuracies[-10:])
        improvement = last_10 - first_10
        print(f"\nTrend analysis:")
        print(f"  First 10 predictions: {first_10:.2f}%")
        print(f"  Last 10 predictions: {last_10:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
    
    # Save results
    import os
    os.makedirs("mc_results", exist_ok=True)
    
    # Save predictions log
    with open("mc_results/predictions_log.json", "w") as f:
        json.dump(predictions_log, f, indent=2)
    print(f"\nPredictions log saved: mc_results/predictions_log.json")
    
    # Save summary statistics
    summary = {
        'model_type': 'Monte Carlo Simulation',
        'num_students': int(num_students),
        'train_days': int(len(train_data)),
        'test_days': int(len(test_data)),
        'total_predictions': int(len(accuracies)),
        'average_accuracy': float(avg_accuracy),
        'best_accuracy': float(best_accuracy),
        'worst_accuracy': float(worst_accuracy),
        'std_deviation': float(std_accuracy),
        'top_10_students': {
            int(student_id): float(model.student_probabilities[student_id] * 100)
            for student_id in top_10_students
        }
    }
    
    with open("mc_results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved: mc_results/summary.json")
    
    print_separator("MONTE CARLO EVALUATION COMPLETE")
    print("\nThe Monte Carlo model is ready for comparison with LSTM!")
