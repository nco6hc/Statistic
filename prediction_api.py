"""
Prediction API - Easy-to-use interface for making predictions and submitting corrections

Usage:
1. Make prediction: python prediction_api.py predict 1309
2. Submit correction: python prediction_api.py correct 1309 5,10,15,20,25,30
3. View status: python prediction_api.py status
4. Update model: python prediction_api.py update
"""

import sys
import argparse
from pathlib import Path

# Add dl_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / 'dl_pipeline'))

import torch
from dl_pipeline.data_loader import DataManager
from dl_pipeline.models import LSTMStudentPredictor
from dl_pipeline.trainer import Trainer
from dl_pipeline.prediction_service import PredictionService


class PredictionAPI:
    """High-level API for prediction service"""
    
    def __init__(self, model_path: str = 'dl_pipeline/dl_checkpoints/final_online_model.pth',
                 data_path: str = 'Database.xlsx',
                 predictions_dir: str = 'predictions'):
        """
        Initialize the prediction API.
        
        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to Excel database
            predictions_dir: Directory to store predictions and corrections
        """
        print("="*70)
        print("INITIALIZING PREDICTION API")
        print("="*70)
        
        # Load data
        print("\n1. Loading data...")
        self.data_manager = DataManager(excel_path=data_path)
        self.data_manager.data = self.data_manager.load_data()
        self.data_manager.convert_to_binary()
        self.data_manager.sort_by_days()
        print(f"   ✓ Loaded {len(self.data_manager.binary_vectors)} days of data")
        
        # Load model
        print("\n2. Loading model...")
        self.model = LSTMStudentPredictor(n_students=55)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Model loaded from {model_path}")
        else:
            print(f"   ⚠ No checkpoint found at {model_path}")
            print(f"   Using untrained model")
        
        # Create trainer and service
        print("\n3. Initializing service...")
        self.trainer = Trainer(self.model, checkpoint_dir='dl_checkpoints')
        self.service = PredictionService(
            model=self.model,
            trainer=self.trainer,
            data_manager=self.data_manager,
            checkpoint_dir=predictions_dir
        )
        print(f"   ✓ Service ready")
        
        # Load existing state
        pending = self.service.get_pending_predictions()
        completed = self.service.get_completed_predictions()
        print(f"\n📊 Current Status:")
        print(f"   Pending predictions: {len(pending)}")
        if pending:
            print(f"   Days: {pending}")
        print(f"   Completed predictions: {len(completed)}")
        if completed:
            print(f"   Days: {completed[-10:] if len(completed) > 10 else completed}")
        
        print("\n" + "="*70)
        print("✅ API READY")
        print("="*70)
    
    def predict(self, day_number: int, n_groups: int = 5):
        """
        Make prediction for a specific day.
        
        Args:
            day_number: Day number (e.g., 1309)
            n_groups: Number of groups to generate (default: 5)
        """
        return self.service.predict_for_day(day_number, n_groups=n_groups)
    
    def correct(self, day_number: int, actual_students: list, auto_update: bool = False):
        """
        Submit correction for a specific day.
        
        Args:
            day_number: Day number (e.g., 1309)
            actual_students: List of 6 student IDs
            auto_update: Whether to immediately update model
        """
        return self.service.submit_correction(day_number, actual_students, auto_update=auto_update)
    
    def update_model(self, days: list = None):
        """
        Update model with all collected corrections.
        
        Args:
            days: Specific days to train on (None = all)
        """
        return self.service.update_model(days=days)
    
    def status(self):
        """Display current status of predictions and corrections."""
        stats = self.service.get_statistics()
        
        print("\n" + "="*70)
        print("PREDICTION SERVICE STATUS")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total predictions made: {stats['total_predictions']}")
        print(f"   Pending corrections: {stats['pending_corrections']}")
        print(f"   Completed corrections: {stats['completed_corrections']}")
        
        if stats['completed_corrections'] > 0:
            print(f"\n🎯 Accuracy Metrics:")
            print(f"   Average best group accuracy: {stats['avg_best_accuracy']:.1%}")
            print(f"   Best prediction: {stats['max_accuracy']:.1%}")
            print(f"   Worst prediction: {stats['min_accuracy']:.1%}")
            print(f"   Completion rate: {stats['completion_rate']:.1%}")
        
        pending = self.service.get_pending_predictions()
        if pending:
            print(f"\n⏳ Pending Predictions (awaiting correction):")
            for day in pending:
                print(f"   Day {day}")
        
        completed = self.service.get_completed_predictions()
        if completed:
            print(f"\n✅ Completed Predictions (last 10):")
            for day in completed[-10:]:
                _, corr = self.service.completed_predictions[day]
                print(f"   Day {day}: {corr['best_overlap']}/6 correct ({corr['best_accuracy']:.1%})")
        
        print("\n" + "="*70)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Student Selection Prediction API')
    parser.add_argument('command', choices=['predict', 'correct', 'status', 'update'],
                       help='Command to execute')
    parser.add_argument('day', nargs='?', type=int,
                       help='Day number (for predict/correct commands)')
    parser.add_argument('students', nargs='?', type=str,
                       help='Comma-separated student IDs (for correct command)')
    parser.add_argument('--groups', type=int, default=5,
                       help='Number of groups to generate (default: 5)')
    parser.add_argument('--auto-update', action='store_true',
                       help='Auto-update model after correction')
    
    args = parser.parse_args()
    
    # Initialize API
    api = PredictionAPI()
    
    # Execute command
    if args.command == 'predict':
        if args.day is None:
            print("❌ Error: Day number required for predict command")
            print("   Usage: python prediction_api.py predict 1309")
            sys.exit(1)
        
        api.predict(args.day, n_groups=args.groups)
    
    elif args.command == 'correct':
        if args.day is None or args.students is None:
            print("❌ Error: Day number and student IDs required for correct command")
            print("   Usage: python prediction_api.py correct 1309 5,10,15,20,25,30")
            sys.exit(1)
        
        try:
            students = [int(s.strip()) for s in args.students.split(',')]
            api.correct(args.day, students, auto_update=args.auto_update)
        except ValueError as e:
            print(f"❌ Error parsing student IDs: {e}")
            sys.exit(1)
    
    elif args.command == 'status':
        api.status()
    
    elif args.command == 'update':
        api.update_model()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("="*70)
        print("PREDICTION API - Usage Examples")
        print("="*70)
        print("\n1. Make prediction for day 1309:")
        print("   python prediction_api.py predict 1309")
        print("\n2. Submit correction for day 1309:")
        print("   python prediction_api.py correct 1309 5,10,15,20,25,30")
        print("\n3. View status:")
        print("   python prediction_api.py status")
        print("\n4. Update model with all corrections:")
        print("   python prediction_api.py update")
        print("\n" + "="*70)
        sys.exit(0)
    
    main()
