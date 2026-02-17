"""
Markov Chain Model for Student Selection Prediction

This module implements a first-order Markov Chain where:
- State = binary vector of selected students
- Transitions = probability of moving from one state to another
- Prediction = probability distribution over students based on current state
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from collections import defaultdict, Counter
import pickle


class StudentSelectionMarkovChain:
    """First-order Markov Chain for student selection prediction."""
    
    def __init__(self, total_students: int = 55, alpha: float = 1.0):
        """
        Initialize the Markov Chain model.
        
        Args:
            total_students: Total number of students
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.total_students = total_students
        self.alpha = alpha
        
        # Store transitions as dictionary: state_tuple -> Counter of next states
        self.transitions = defaultdict(Counter)
        
        # Store state counts for normalization
        self.state_counts = Counter()
        
        # Set of all observed states
        self.observed_states = set()
        
        # Number of transitions observed
        self.n_transitions = 0
        
    def _binary_to_tuple(self, binary_vector: np.ndarray) -> tuple:
        """
        Convert binary vector to tuple for use as dictionary key.
        
        Args:
            binary_vector: Binary vector of shape (n_students,)
            
        Returns:
            Tuple representation of the binary vector
        """
        return tuple(binary_vector.astype(int))
    
    def _tuple_to_binary(self, state_tuple: tuple) -> np.ndarray:
        """Convert state tuple back to binary vector."""
        return np.array(state_tuple, dtype=int)
    
    def fit(self, binary_vectors: np.ndarray, verbose: bool = True) -> 'StudentSelectionMarkovChain':
        """
        Fit the Markov Chain model on historical data.
        
        Args:
            binary_vectors: Binary matrix (n_samples, n_students)
            verbose: Print progress information
            
        Returns:
            Self for method chaining
        """
        if verbose:
            print("="*60)
            print("FITTING MARKOV CHAIN MODEL")
            print("="*60)
            print(f"Training samples: {len(binary_vectors)}")
            print(f"Alpha (Laplace smoothing): {self.alpha}")
        
        # Count transitions
        for i in range(len(binary_vectors) - 1):
            current_state = self._binary_to_tuple(binary_vectors[i])
            next_state = self._binary_to_tuple(binary_vectors[i + 1])
            
            # Record transition
            self.transitions[current_state][next_state] += 1
            self.state_counts[current_state] += 1
            
            # Track observed states
            self.observed_states.add(current_state)
            self.observed_states.add(next_state)
            
            self.n_transitions += 1
        
        if verbose:
            print(f"Total transitions: {self.n_transitions}")
            print(f"Unique states observed: {len(self.observed_states)}")
            print(f"Unique source states: {len(self.transitions)}")
            
            # Analyze state diversity
            avg_next_states = np.mean([len(next_states) for next_states in self.transitions.values()])
            print(f"Average next states per source: {avg_next_states:.2f}")
            
            # Most common states
            print("\nMost common states:")
            for state, count in self.state_counts.most_common(5):
                selected_students = [i+1 for i, val in enumerate(state) if val == 1]
                print(f"  State {selected_students}: {count} times")
        
        return self
    
    def _get_transition_probability(self, current_state: tuple, next_state: tuple) -> float:
        """
        Get transition probability with Laplace smoothing.
        
        P(next_state | current_state) = (count + alpha) / (total + alpha * |S|)
        where |S| is the number of possible next states
        
        Args:
            current_state: Current state tuple
            next_state: Next state tuple
            
        Returns:
            Transition probability
        """
        count = self.transitions[current_state][next_state]
        total = self.state_counts[current_state]
        
        # Number of possible states (all observed states as potential next states)
        n_possible_states = len(self.observed_states)
        
        # Laplace smoothing
        probability = (count + self.alpha) / (total + self.alpha * n_possible_states)
        
        return probability
    
    def predict_next_state_distribution(self, current_state: np.ndarray, 
                                       top_k: int = 10) -> List[Tuple[tuple, float]]:
        """
        Predict distribution over next states given current state.
        
        Args:
            current_state: Binary vector of current state
            top_k: Return top-k most likely next states
            
        Returns:
            List of (state, probability) tuples, sorted by probability
        """
        current_state_tuple = self._binary_to_tuple(current_state)
        
        # If we've seen this state before, use observed transitions
        if current_state_tuple in self.transitions:
            # Get all next states and their probabilities
            next_state_probs = []
            for next_state in self.observed_states:
                prob = self._get_transition_probability(current_state_tuple, next_state)
                next_state_probs.append((next_state, prob))
        else:
            # Unseen state - use uniform distribution over all observed states
            uniform_prob = 1.0 / len(self.observed_states)
            next_state_probs = [(state, uniform_prob) for state in self.observed_states]
        
        # Sort by probability and return top-k
        next_state_probs.sort(key=lambda x: x[1], reverse=True)
        return next_state_probs[:top_k]
    
    def predict_student_probabilities(self, current_state: np.ndarray, 
                                     method: str = 'weighted') -> np.ndarray:
        """
        Predict probability of each student being selected next.
        
        Args:
            current_state: Binary vector of current state
            method: 'weighted' (weight by state probability) or 
                   'binary' (count states where student appears)
            
        Returns:
            Probability distribution over students (shape: n_students)
        """
        current_state_tuple = self._binary_to_tuple(current_state)
        
        student_probs = np.zeros(self.total_students)
        
        if current_state_tuple in self.transitions:
            # Get all possible next states with their probabilities
            for next_state in self.observed_states:
                trans_prob = self._get_transition_probability(current_state_tuple, next_state)
                
                # Add probability mass to students in this next state
                next_state_binary = self._tuple_to_binary(next_state)
                
                if method == 'weighted':
                    # Weighted by transition probability
                    student_probs += next_state_binary * trans_prob
                elif method == 'binary':
                    # Binary count (1 if student appears in any next state)
                    student_probs += next_state_binary * (trans_prob > 0)
        else:
            # Unseen state - use marginal probabilities from all observed states
            for state in self.observed_states:
                state_binary = self._tuple_to_binary(state)
                student_probs += state_binary
            student_probs /= len(self.observed_states)
        
        # Normalize to ensure it's a valid probability distribution
        if student_probs.sum() > 0:
            student_probs /= student_probs.sum()
        else:
            # Fallback to uniform distribution
            student_probs = np.ones(self.total_students) / self.total_students
        
        return student_probs
    
    def predict_top_k_students(self, current_state: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict top-k most likely students to be selected next.
        
        Args:
            current_state: Binary vector of current state
            k: Number of students to predict (default: 6)
            
        Returns:
            Tuple of (student_ids, probabilities)
        """
        student_probs = self.predict_student_probabilities(current_state)
        
        # Get top-k students
        top_k_indices = np.argsort(student_probs)[::-1][:k]
        top_k_probs = student_probs[top_k_indices]
        
        # Convert to 1-indexed student IDs
        top_k_student_ids = top_k_indices + 1
        
        return top_k_student_ids, top_k_probs
    
    def evaluate(self, test_vectors: np.ndarray, metrics: List[str] = ['accuracy', 'precision', 'recall']) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_vectors: Binary matrix of test samples
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        print("\n" + "="*60)
        print("EVALUATING MARKOV CHAIN MODEL")
        print("="*60)
        
        results = {}
        predictions = []
        actuals = []
        
        # Make predictions for each test sample
        for i in range(len(test_vectors) - 1):
            current_state = test_vectors[i]
            actual_next = test_vectors[i + 1]
            
            # Predict next state
            predicted_probs = self.predict_student_probabilities(current_state)
            predicted_students, _ = self.predict_top_k_students(current_state, k=6)
            
            # Convert to binary vector
            predicted_binary = np.zeros(self.total_students)
            predicted_binary[predicted_students - 1] = 1
            
            predictions.append(predicted_binary)
            actuals.append(actual_next)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Compute metrics
        if 'accuracy' in metrics:
            # Exact match accuracy (all 6 students correct)
            exact_match = np.all(predictions == actuals, axis=1).mean()
            results['exact_match_accuracy'] = exact_match
            
            # Element-wise accuracy
            element_accuracy = (predictions == actuals).mean()
            results['element_accuracy'] = element_accuracy
        
        if 'precision' in metrics:
            # Precision: of predicted selections, how many were correct
            true_positives = (predictions * actuals).sum(axis=1)
            predicted_positives = predictions.sum(axis=1)
            precision = (true_positives / predicted_positives).mean()
            results['precision'] = precision
        
        if 'recall' in metrics:
            # Recall: of actual selections, how many were predicted
            true_positives = (predictions * actuals).sum(axis=1)
            actual_positives = actuals.sum(axis=1)
            recall = (true_positives / actual_positives).mean()
            results['recall'] = recall
        
        # F1 score
        if 'precision' in results and 'recall' in results:
            if results['precision'] + results['recall'] > 0:
                results['f1_score'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'])
            else:
                results['f1_score'] = 0.0
        
        # Overlap metric (how many students overlap)
        overlap = (predictions * actuals).sum(axis=1)
        results['avg_overlap'] = overlap.mean()
        results['overlap_distribution'] = {
            f'{i}_students': (overlap == i).sum() / len(overlap)
            for i in range(7)
        }
        
        # Print results
        print(f"\nTest samples: {len(predictions)}")
        print(f"\nPerformance Metrics:")
        print(f"  Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.4f}")
        print(f"  Element Accuracy: {results.get('element_accuracy', 0):.4f}")
        print(f"  Precision: {results.get('precision', 0):.4f}")
        print(f"  Recall: {results.get('recall', 0):.4f}")
        print(f"  F1 Score: {results.get('f1_score', 0):.4f}")
        print(f"  Average Overlap: {results.get('avg_overlap', 0):.2f} / 6 students")
        
        print(f"\nOverlap Distribution:")
        for i in range(7):
            pct = results['overlap_distribution'][f'{i}_students'] * 100
            print(f"  {i} students correct: {pct:.2f}%")
        
        return results
    
    def save(self, filepath: str):
        """Save the model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'total_students': self.total_students,
                'alpha': self.alpha,
                'transitions': dict(self.transitions),
                'state_counts': dict(self.state_counts),
                'observed_states': self.observed_states,
                'n_transitions': self.n_transitions
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.total_students = data['total_students']
        self.alpha = data['alpha']
        self.transitions = defaultdict(Counter, {k: Counter(v) for k, v in data['transitions'].items()})
        self.state_counts = Counter(data['state_counts'])
        self.observed_states = data['observed_states']
        self.n_transitions = data['n_transitions']
        
        print(f"Model loaded from {filepath}")
        return self


def main():
    """Main execution function."""
    print("Loading data...")
    
    # Load the binary vectors
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize and fit Markov Chain
    print("\nTraining Markov Chain...")
    markov_model = StudentSelectionMarkovChain(total_students=55, alpha=1.0)
    markov_model.fit(X_train, verbose=True)
    
    # Evaluate on test set
    results = markov_model.evaluate(X_test)
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    # Use last training state to predict first test state
    current_state = X_train[-1]
    actual_next = X_test[0]
    
    print(f"\nCurrent state (last training day):")
    current_selected = [i+1 for i, val in enumerate(current_state) if val == 1]
    print(f"  Selected students: {current_selected}")
    
    print(f"\nActual next state (first test day):")
    actual_selected = [i+1 for i, val in enumerate(actual_next) if val == 1]
    print(f"  Selected students: {actual_selected}")
    
    print(f"\nPredicted next state:")
    predicted_students, predicted_probs = markov_model.predict_top_k_students(current_state, k=6)
    for student_id, prob in zip(predicted_students, predicted_probs):
        match = "✓" if student_id in actual_selected else "✗"
        print(f"  {match} Student {student_id}: {prob:.4f}")
    
    # Show top-10 student probabilities
    print(f"\nTop-10 student probabilities:")
    all_probs = markov_model.predict_student_probabilities(current_state)
    top_10_indices = np.argsort(all_probs)[::-1][:10]
    for idx in top_10_indices:
        student_id = idx + 1
        match = "✓" if student_id in actual_selected else "✗"
        print(f"  {match} Student {student_id}: {all_probs[idx]:.4f}")
    
    # Save model
    print("\n" + "="*60)
    markov_model.save('markov_model.pkl')
    
    # Test different alpha values
    print("\n" + "="*60)
    print("TESTING DIFFERENT ALPHA VALUES")
    print("="*60)
    
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    alpha_results = []
    
    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}")
        model = StudentSelectionMarkovChain(total_students=55, alpha=alpha)
        model.fit(X_train, verbose=False)
        results = model.evaluate(X_test, metrics=['precision', 'recall'])
        alpha_results.append({
            'alpha': alpha,
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'avg_overlap': results['avg_overlap']
        })
    
    print("\n" + "="*60)
    print("ALPHA COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Alpha':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Avg Overlap':<12}")
    print("-" * 60)
    for res in alpha_results:
        print(f"{res['alpha']:<10.1f} {res['precision']:<12.4f} {res['recall']:<12.4f} "
              f"{res['f1_score']:<12.4f} {res['avg_overlap']:<12.2f}")
    
    return markov_model, results


if __name__ == "__main__":
    markov_model, results = main()
