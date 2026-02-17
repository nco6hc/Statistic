"""
Online Learning Pipeline with Prediction Tracking

Implements the complete end-to-end pipeline for online learning
and prediction every 2 days with feedback integration.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json
from collections import deque


class OnlineLearningPipeline:
    """Pipeline for online learning with periodic predictions and feedback."""
    
    def __init__(self, model, trainer, data_manager, sequence_length: int = 14,
                 prediction_interval: int = 2, checkpoint_dir: str = 'checkpoints'):
        """
        Initialize online learning pipeline.
        
        Args:
            model: PyTorch model
            trainer: Trainer instance
            data_manager: DataManager instance
            sequence_length: Sequence length for predictions
            prediction_interval: Days between predictions
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.trainer = trainer
        self.data_manager = data_manager
        self.sequence_length = sequence_length
        self.prediction_interval = prediction_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Prediction tracking
        self.predictions_log = []
        self.accuracy_history = []
        
        # Replay buffer for online learning
        self.replay_buffer = deque(maxlen=100)
        
        # Current day counter
        self.current_day = 0
        
    def make_prediction(self, n_groups: int = 5) -> Tuple[List[List[int]], np.ndarray]:
        """
        Make prediction for multiple groups of 6 students.
        
        Args:
            n_groups: Number of different groups to generate
        
        Returns:
            Tuple of (list of predicted_groups, full probabilities)
            - predicted_groups: list of n_groups, each with 6 student IDs
            - probabilities: full probability distribution for all 55 students
        """
        # Get recent sequence
        recent_sequence = self.data_manager.get_recent_sequence(self.sequence_length)
        
        # Convert to tensor
        X = torch.FloatTensor(recent_sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction with multiple groups
        all_batch_groups, predicted_probs = self.trainer.predict(X, k=6, n_groups=n_groups)
        
        # Return groups for first (and only) batch item
        return all_batch_groups[0], predicted_probs[0]
    
    def receive_feedback(self, actual_students: List[int], 
                        predicted_groups: List[List[int]],
                        predicted_probs: np.ndarray) -> Dict:
        """
        Receive actual results and calculate accuracy.
        
        Args:
            actual_students: List of actually selected student IDs (1-indexed)
            predicted_groups: List of n_groups predicted student ID groups
            predicted_probs: Full prediction probabilities for all students
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Calculate overlap for each group
        actual_set = set(actual_students)
        overlaps = []
        accuracies = []
        
        for group in predicted_groups:
            overlap = len(actual_set & set(group))
            accuracy = overlap / 6.0
            overlaps.append(overlap)
            accuracies.append(accuracy)
        
        # Use best group accuracy for tracking
        best_overlap = max(overlaps)
        best_accuracy = max(accuracies)
        best_group_idx = accuracies.index(best_accuracy)
        
        # Log prediction
        prediction_log = {
            'day': self.current_day,
            'timestamp': datetime.now().isoformat(),
            'predicted_groups': predicted_groups,
            'best_group': predicted_groups[best_group_idx],
            'best_group_index': best_group_idx,
            'actual': actual_students,
            'overlaps': overlaps,
            'accuracies': accuracies,
            'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
            'top_6_probs': predicted_probs[np.argsort(predicted_probs)[-6:][::-1]].tolist()
        }
        
        self.predictions_log.append(prediction_log)
        self.accuracy_history.append(best_accuracy)
        
        print(f"\nDay {self.current_day} Feedback:")
        print(f"  Predicted Groups ({len(predicted_groups)} total):")
        for i, group in enumerate(predicted_groups):
            marker = " ⭐ BEST" if i == best_group_idx else ""
            print(f"    Group {i+1}: {group} (overlap: {overlaps[i]}/6){marker}")
        print(f"  Actual: {actual_students}")
        print(f"  Best Overlap: {best_overlap} / 6")
        print(f"  Best Accuracy: {best_accuracy:.2%}")
        
        return prediction_log
    
    def update_model(self, actual_students: List[int]):
        """
        Update model with new data using online learning.
        
        Args:
            actual_students: List of actually selected student IDs
        """
        # Add new selection to data manager
        self.data_manager.append_new_selection(actual_students)
        
        # Get recent sequence as input
        recent_data = self.data_manager.get_recent_sequence(self.sequence_length + 1)
        X = recent_data[:-1]  # Input sequence
        y = recent_data[-1]   # Target (latest selection)
        
        # Add to replay buffer
        self.replay_buffer.append((X, y))
        
        # Online update with replay buffer
        if len(self.replay_buffer) >= 5:
            # Sample from replay buffer
            batch_X = []
            batch_y = []
            for x, y in list(self.replay_buffer)[-10:]:  # Use last 10 samples
                batch_X.append(x)
                batch_y.append(y)
            
            batch_X = torch.FloatTensor(np.array(batch_X))
            batch_y = torch.FloatTensor(np.array(batch_y))
            
            # Perform online update
            self.trainer.online_update(batch_X, batch_y, n_iterations=5)
            
            print(f"Model updated with {len(batch_X)} samples")
    
    def run_simulation(self, test_data: np.ndarray, n_days: int = None):
        """
        Run simulation of online learning process.
        
        Args:
            test_data: Test data to simulate (n_samples, n_students)
            n_days: Number of days to simulate (None = all test data)
        """
        print("="*70)
        print("STARTING ONLINE LEARNING SIMULATION")
        print("="*70)
        
        if n_days is None:
            n_days = len(test_data)
        
        n_days = min(n_days, len(test_data))
        
        for day in range(n_days):
            self.current_day = day + 1
            
            # Make prediction every prediction_interval days
            if day % self.prediction_interval == 0:
                print(f"\n{'='*70}")
                print(f"DAY {self.current_day}: MAKING PREDICTION")
                print(f"{'='*70}")
                
                predicted_groups, predicted_probs = self.make_prediction(n_groups=5)
                
                print(f"\nPredicted 5 groups of students:")
                for i, group in enumerate(predicted_groups):
                    print(f"  Group {i+1}: {group}")
                
                # Get actual selection from test data
                actual_selection = test_data[day]
                actual_students = [i+1 for i, val in enumerate(actual_selection) if val == 1]
                
                # Receive feedback
                self.receive_feedback(actual_students, predicted_groups, predicted_probs)
                
                # Update model
                self.update_model(actual_students)
                
                # Save checkpoint every 10 predictions
                if len(self.predictions_log) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_day_{self.current_day}.pth')
            else:
                # Just update with actual data (no prediction)
                actual_selection = test_data[day]
                actual_students = [i+1 for i, val in enumerate(actual_selection) if val == 1]
                self.update_model(actual_students)
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print overall statistics."""
        print("\n" + "="*70)
        print("ONLINE LEARNING STATISTICS")
        print("="*70)
        
        if not self.accuracy_history:
            print("No predictions made yet")
            return
        
        accuracies = np.array(self.accuracy_history)
        
        print(f"\nTotal predictions: {len(self.predictions_log)}")
        print(f"Average accuracy: {accuracies.mean():.2%}")
        print(f"Best accuracy: {accuracies.max():.2%}")
        print(f"Worst accuracy: {accuracies.min():.2%}")
        print(f"Std deviation: {accuracies.std():.4f}")
        
        # Accuracy distribution
        print(f"\nAccuracy distribution:")
        for overlap in range(7):
            count = (accuracies == overlap/6).sum()
            pct = count / len(accuracies) * 100
            print(f"  {overlap}/6 correct: {count:3d} ({pct:5.1f}%)")
        
        # Trend analysis
        if len(accuracies) >= 10:
            recent_10 = accuracies[-10:].mean()
            first_10 = accuracies[:10].mean()
            print(f"\nTrend analysis:")
            print(f"  First 10 predictions: {first_10:.2%}")
            print(f"  Last 10 predictions: {recent_10:.2%}")
            print(f"  Improvement: {(recent_10 - first_10):.2%}")
    
    def save_checkpoint(self, filename: str):
        """Save pipeline checkpoint."""
        self.trainer.save_checkpoint(filename)
        
        # Save pipeline state
        pipeline_state = {
            'current_day': self.current_day,
            'predictions_log': self.predictions_log,
            'accuracy_history': self.accuracy_history,
            'sequence_length': self.sequence_length,
            'prediction_interval': self.prediction_interval
        }
        
        state_path = self.checkpoint_dir / f'pipeline_state_{filename}.json'
        with open(state_path, 'w') as f:
            json.dump(pipeline_state, f, indent=2)
        
        print(f"Pipeline state saved: {state_path}")
    
    def load_checkpoint(self, filename: str):
        """Load pipeline checkpoint."""
        self.trainer.load_checkpoint(filename)
        
        # Load pipeline state
        state_path = self.checkpoint_dir / f'pipeline_state_{filename}.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                pipeline_state = json.load(f)
            
            self.current_day = pipeline_state['current_day']
            self.predictions_log = pipeline_state['predictions_log']
            self.accuracy_history = pipeline_state['accuracy_history']
            
            print(f"Pipeline state loaded: {state_path}")
    
    def save_predictions_log(self, filename: str = 'predictions_log.csv'):
        """Save predictions log to CSV."""
        if not self.predictions_log:
            print("No predictions to save")
            return
        
        df = pd.DataFrame(self.predictions_log)
        output_path = self.checkpoint_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Predictions log saved: {output_path}")
    
    def plot_accuracy_over_time(self, filename: str = 'accuracy_plot.png'):
        """Plot accuracy over time."""
        if not self.accuracy_history:
            print("No accuracy history to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot accuracy
            ax.plot(self.accuracy_history, marker='o', linewidth=2, markersize=4)
            
            # Add moving average
            if len(self.accuracy_history) >= 5:
                window = 5
                moving_avg = pd.Series(self.accuracy_history).rolling(window=window).mean()
                ax.plot(moving_avg, linewidth=2, linestyle='--', 
                       label=f'{window}-prediction moving average')
            
            ax.set_xlabel('Prediction Number')
            ax.set_ylabel('Accuracy (Overlap / 6)')
            ax.set_title('Online Learning: Prediction Accuracy Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 1])
            
            output_path = self.checkpoint_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Accuracy plot saved: {output_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    print("Online learning pipeline module loaded successfully")
