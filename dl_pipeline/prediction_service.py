"""
Prediction Service - Separate Prediction and Correction APIs

This module provides a service for making predictions ahead of time
and submitting corrections later, with proper day tracking.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from .data_loader import DataManager
from .trainer import Trainer
from .enhanced_models import EnhancedLSTMPredictor
from .enhanced_features import build_enriched_features


class PredictionService:
    """
    Service for managing predictions and corrections separately.
    
    Workflow:
    1. predict_for_day(day_number) - Make prediction before class
    2. submit_correction(day_number, actual_students) - Submit results after class
    3. update_model() - Update model with collected corrections
    """
    
    def __init__(self, model: EnhancedLSTMPredictor, trainer: Trainer, 
                 data_manager: DataManager, checkpoint_dir: str = 'predictions'):
        """
        Initialize prediction service.
        
        Args:
            model: Trained Enhanced LSTM prediction model
            trainer: Trainer instance
            data_manager: Data manager with historical data
            checkpoint_dir: Directory to store predictions and corrections
        """
        self.model = model
        self.trainer = trainer
        self.data_manager = data_manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Build enriched features from all historical data
        self._rebuild_enriched_features()
        
        # Tracking structures
        self.pending_predictions = {}  # day -> prediction_data
        self.completed_predictions = {}  # day -> (prediction_data, actual_students)
        
        # Load existing predictions if available
        self._load_state()
    
    def _rebuild_enriched_features(self):
        """Rebuild enriched features (220-dim) from binary data."""
        if self.data_manager.binary_vectors is not None:
            self.enriched_features = build_enriched_features(
                self.data_manager.binary_vectors
            )
        else:
            self.enriched_features = None
    
    def predict_for_day(self, day_number: int, n_groups: int = 5, 
                       save: bool = True) -> Dict:
        """
        Make prediction for a specific day (before class happens).
        
        Args:
            day_number: Day number for prediction
            n_groups: Number of groups to generate
            save: Whether to save prediction to disk
            
        Returns:
            Dictionary with prediction details
        """
        print(f"\n{'='*70}")
        print(f"MAKING PREDICTION FOR DAY {day_number}")
        print(f"{'='*70}")
        
        # Check if already predicted
        if day_number in self.pending_predictions:
            print(f"⚠ Day {day_number} already has a pending prediction")
            return self.pending_predictions[day_number]
        
        if day_number in self.completed_predictions:
            print(f"⚠ Day {day_number} already completed")
            return self.completed_predictions[day_number][0]
        
        # Build enriched features from full binary history and get last 14 days
        self._rebuild_enriched_features()
        recent_enriched = self.enriched_features[-14:]
        
        # Convert to tensor (220 features per timestep)
        X = torch.FloatTensor(recent_enriched).unsqueeze(0)
        
        # Make prediction
        all_groups, probabilities = self.trainer.predict(X, k=6, n_groups=n_groups)
        predicted_groups = all_groups[0]  # First batch
        probs = probabilities[0]  # First batch
        
        # Create prediction record
        prediction_data = {
            'day': day_number,
            'timestamp': datetime.now().isoformat(),
            'predicted_groups': predicted_groups,
            'probabilities': probs.tolist(),
            'top_students_per_group': predicted_groups,
            'status': 'pending',
            'n_groups': n_groups
        }
        
        # Store in pending
        self.pending_predictions[day_number] = prediction_data
        
        # Display prediction
        print(f"\n✓ Generated {n_groups} groups for Day {day_number}:")
        for i, group in enumerate(predicted_groups):
            group_probs = [probs[s-1] for s in group]
            avg_prob = np.mean(group_probs)
            print(f"  Group {i+1}: {group} (avg prob: {avg_prob:.1%})")
        
        # Save to disk
        if save:
            self._save_prediction(day_number, prediction_data)
        
        print(f"\n💾 Prediction saved. Status: PENDING correction")
        print(f"   Use submit_correction({day_number}, actual_students) after class")
        
        return prediction_data
    
    def submit_correction(self, day_number: int, actual_students: List[int],
                         auto_update: bool = False) -> Dict:
        """
        Submit actual students selected for a specific day (after class).
        
        Args:
            day_number: Day number for correction
            actual_students: List of 6 student IDs that were actually selected
            auto_update: Whether to immediately update model
            
        Returns:
            Dictionary with correction results and accuracy metrics
        """
        print(f"\n{'='*70}")
        print(f"SUBMITTING CORRECTION FOR DAY {day_number}")
        print(f"{'='*70}")
        
        # Validate
        if len(actual_students) != 6:
            raise ValueError(f"Expected 6 students, got {len(actual_students)}")
        
        if not all(1 <= s <= 55 for s in actual_students):
            raise ValueError("Student IDs must be between 1 and 55")
        
        # Check if prediction exists
        if day_number not in self.pending_predictions:
            if day_number in self.completed_predictions:
                print(f"⚠ Day {day_number} already has a correction")
                return self.completed_predictions[day_number][0]
            else:
                raise ValueError(f"No prediction found for day {day_number}. "
                               f"Please make a prediction first.")
        
        # Get prediction
        prediction_data = self.pending_predictions[day_number]
        predicted_groups = prediction_data['predicted_groups']
        
        # Calculate accuracy for each group
        actual_set = set(actual_students)
        overlaps = []
        accuracies = []
        
        for group in predicted_groups:
            overlap = len(actual_set & set(group))
            accuracy = overlap / 6.0
            overlaps.append(overlap)
            accuracies.append(accuracy)
        
        # Find best group
        best_overlap = max(overlaps)
        best_accuracy = max(accuracies)
        best_group_idx = accuracies.index(best_accuracy)
        best_group = predicted_groups[best_group_idx]
        
        # Create correction record
        correction_data = {
            'day': day_number,
            'timestamp': datetime.now().isoformat(),
            'actual_students': actual_students,
            'predicted_groups': predicted_groups,
            'overlaps': overlaps,
            'accuracies': accuracies,
            'best_group': best_group,
            'best_group_index': best_group_idx,
            'best_overlap': best_overlap,
            'best_accuracy': best_accuracy,
            'status': 'corrected'
        }
        
        # Move from pending to completed
        del self.pending_predictions[day_number]
        self.completed_predictions[day_number] = (prediction_data, correction_data)
        
        # Display results
        print(f"\n✓ Correction received for Day {day_number}")
        print(f"   Actual students: {actual_students}")
        print(f"\n📊 Group Performance:")
        for i, (group, overlap) in enumerate(zip(predicted_groups, overlaps)):
            marker = " ⭐ BEST" if i == best_group_idx else ""
            print(f"   Group {i+1}: {group}")
            print(f"            Overlap: {overlap}/6 ({accuracies[i]:.1%}){marker}")
        
        print(f"\n🎯 Best Performance:")
        print(f"   Group {best_group_idx + 1} matched {best_overlap}/6 students")
        print(f"   Accuracy: {best_accuracy:.1%}")
        
        # Save correction
        self._save_correction(day_number, correction_data)
        
        # Add to data manager for future predictions
        self.data_manager.append_new_selection(actual_students, day_number=day_number)
        
        # Rebuild enriched features with the new data point
        self._rebuild_enriched_features()
        
        # Save back to Excel database
        try:
            self.data_manager.save_to_excel()
            print(f"\n💾 ✅ Database updated with Day {day_number} actual results.")
            print(f"   Excel file: {self.data_manager.excel_path}")
        except PermissionError:
            print(f"\n💾 ⚠️  Cannot update Database.xlsx - file is open in Excel")
            print(f"   Please close Excel and run this command to update:")
            print(f"   python prediction_api.py update-db {day_number}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not update database: {e}")
        
        print(f"   Data ready for future predictions.")
        
        # Auto-update model if requested
        if auto_update:
            print(f"\n🔄 Auto-updating model...")
            self._update_model_with_correction(day_number, actual_students)
        else:
            print(f"\n💡 Tip: Call update_model() to train on collected corrections")
        
        return correction_data
    
    def update_model(self, days: Optional[List[int]] = None, 
                    n_iterations: int = 10) -> Dict:
        """
        Update model with collected corrections (batch training).
        
        Args:
            days: Specific days to train on (None = all completed)
            n_iterations: Number of training iterations
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*70}")
        print(f"UPDATING MODEL WITH CORRECTIONS")
        print(f"{'='*70}")
        
        # Get days to train on
        if days is None:
            days = sorted(self.completed_predictions.keys())
        
        if not days:
            print("⚠ No corrections available for training")
            return {'status': 'no_data'}
        
        print(f"\nTraining on {len(days)} corrections:")
        print(f"  Days: {days}")
        
        # Rebuild enriched features from full history
        self._rebuild_enriched_features()
        
        # Prepare training data from corrections
        sequences = []
        targets = []
        
        for day in days:
            _, correction_data = self.completed_predictions[day]
            actual_students = correction_data['actual_students']
            
            # Get enriched sequence (last 14 days)
            recent_enriched = self.enriched_features[-14:]
            
            # Create binary target
            target = np.zeros(55, dtype=np.float32)
            for student_id in actual_students:
                target[student_id - 1] = 1
            
            sequences.append(recent_enriched)
            targets.append(target)
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))
        
        # Update model with FocalLoss (same as training)
        from .enhanced_models import FocalLoss
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
        
        losses = []
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration+1}/{n_iterations}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(losses)
        print(f"\n✓ Model updated")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Trained on {len(days)} corrections")
        
        # Save updated model
        self.trainer.save_checkpoint('model_after_corrections.pth')
        print(f"   Model saved: model_after_corrections.pth")
        
        return {
            'status': 'success',
            'n_corrections': len(days),
            'avg_loss': avg_loss,
            'iterations': n_iterations
        }
    
    def get_pending_predictions(self) -> List[int]:
        """Get list of days with pending predictions (awaiting correction)."""
        return sorted(self.pending_predictions.keys())
    
    def get_completed_predictions(self) -> List[int]:
        """Get list of days with completed predictions (already corrected)."""
        return sorted(self.completed_predictions.keys())
    
    def get_prediction_status(self, day_number: int) -> str:
        """
        Get status of a specific day's prediction.
        
        Returns:
            'pending', 'completed', or 'not_found'
        """
        if day_number in self.pending_predictions:
            return 'pending'
        elif day_number in self.completed_predictions:
            return 'completed'
        else:
            return 'not_found'
    
    def get_statistics(self) -> Dict:
        """Get overall statistics about predictions and corrections."""
        if not self.completed_predictions:
            return {
                'total_predictions': len(self.pending_predictions),
                'pending_corrections': len(self.pending_predictions),
                'completed_corrections': 0,
                'avg_best_accuracy': 0,
                'completion_rate': 0
            }
        
        # Calculate statistics
        best_accuracies = [
            corr['best_accuracy'] 
            for _, corr in self.completed_predictions.values()
        ]
        
        return {
            'total_predictions': len(self.pending_predictions) + len(self.completed_predictions),
            'pending_corrections': len(self.pending_predictions),
            'completed_corrections': len(self.completed_predictions),
            'avg_best_accuracy': np.mean(best_accuracies),
            'max_accuracy': max(best_accuracies),
            'min_accuracy': min(best_accuracies),
            'completion_rate': len(self.completed_predictions) / 
                             (len(self.pending_predictions) + len(self.completed_predictions))
        }
    
    def _save_prediction(self, day_number: int, prediction_data: Dict):
        """Save prediction to disk."""
        filepath = self.checkpoint_dir / f'prediction_day_{day_number}.json'
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = prediction_data.copy()
            if isinstance(data_to_save.get('probabilities'), np.ndarray):
                data_to_save['probabilities'] = data_to_save['probabilities'].tolist()
            json.dump(data_to_save, f, indent=2)
    
    def _save_correction(self, day_number: int, correction_data: Dict):
        """Save correction to disk."""
        filepath = self.checkpoint_dir / f'correction_day_{day_number}.json'
        with open(filepath, 'w') as f:
            json.dump(correction_data, f, indent=2)
        
        # Also update summary CSV
        self._update_summary_csv()
    
    def _update_summary_csv(self):
        """Update summary CSV with all completed predictions."""
        if not self.completed_predictions:
            return
        
        records = []
        for day, (pred, corr) in sorted(self.completed_predictions.items()):
            records.append({
                'day': day,
                'prediction_timestamp': pred['timestamp'],
                'correction_timestamp': corr['timestamp'],
                'predicted_groups': str(corr['predicted_groups']),
                'best_group': str(corr['best_group']),
                'best_group_index': corr['best_group_index'],
                'actual_students': str(corr['actual_students']),
                'overlaps': str(corr['overlaps']),
                'best_overlap': corr['best_overlap'],
                'best_accuracy': corr['best_accuracy']
            })
        
        df = pd.DataFrame(records)
        csv_path = self.checkpoint_dir / 'predictions_summary.csv'
        df.to_csv(csv_path, index=False)
    
    def _load_state(self):
        """Load existing predictions and corrections from disk."""
        # Load predictions
        for pred_file in self.checkpoint_dir.glob('prediction_day_*.json'):
            day = int(pred_file.stem.split('_')[-1])
            with open(pred_file, 'r') as f:
                prediction_data = json.load(f)
            
            # Check if correction exists
            corr_file = self.checkpoint_dir / f'correction_day_{day}.json'
            if corr_file.exists():
                with open(corr_file, 'r') as f:
                    correction_data = json.load(f)
                self.completed_predictions[day] = (prediction_data, correction_data)
            else:
                self.pending_predictions[day] = prediction_data
    
    def _update_model_with_correction(self, day_number: int, actual_students: List[int]):
        """Update model with a single correction."""
        self.update_model(days=[day_number], n_iterations=5)
