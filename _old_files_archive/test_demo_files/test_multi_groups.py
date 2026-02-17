"""
Test script for multiple group predictions
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add dl_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / 'dl_pipeline'))

from data_loader import DataManager
from models import LSTMStudentPredictor
from trainer import Trainer
from online_pipeline import OnlineLearningPipeline


def test_multi_group_prediction():
    """Test the multi-group prediction functionality"""
    print("="*70)
    print("TESTING MULTI-GROUP PREDICTION")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    data_manager = DataManager(excel_path='Database.xlsx')
    data_manager.data = data_manager.load_data()
    data_manager.convert_to_binary()
    data_manager.sort_by_days()
    train_dataset, val_dataset = data_manager.create_datasets(sequence_length=14)
    
    # Create model
    print("\n2. Creating model...")
    model = LSTMStudentPredictor(n_students=55)
    
    # Try to load trained model
    checkpoint_path = 'dl_pipeline/dl_checkpoints/final_online_model.pth'
    if Path(checkpoint_path).exists():
        print(f"   Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ Model loaded successfully")
    else:
        print("   ⚠ No checkpoint found, using untrained model for demo")
    
    # Create trainer
    trainer = Trainer(model, checkpoint_dir='dl_checkpoints')
    
    # Create pipeline
    print("\n3. Creating online learning pipeline...")
    pipeline = OnlineLearningPipeline(
        model=model,
        trainer=trainer,
        data_manager=data_manager,
        sequence_length=14,
        prediction_interval=2
    )
    
    # Make a prediction
    print("\n4. Making prediction with 5 groups...")
    print("-" * 70)
    
    predicted_groups, probabilities = pipeline.make_prediction(n_groups=5)
    
    print(f"\n✓ Generated {len(predicted_groups)} groups:")
    print()
    for i, group in enumerate(predicted_groups):
        # Get probabilities for this group
        group_probs = [probabilities[student_id - 1] for student_id in group]
        avg_prob = np.mean(group_probs)
        
        print(f"Group {i+1}:")
        print(f"  Students: {group}")
        print(f"  Avg probability: {avg_prob:.2%}")
        print(f"  Individual probs: {[f'{p:.2%}' for p in group_probs]}")
        print()
    
    # Check for uniqueness within groups
    print("\n5. Validation:")
    for i, group in enumerate(predicted_groups):
        if len(set(group)) != len(group):
            print(f"  ⚠ Group {i+1} has duplicates!")
        else:
            print(f"  ✓ Group {i+1}: All students unique within group")
    
    # Check diversity across groups
    all_students = set()
    for group in predicted_groups:
        all_students.update(group)
    
    total_selections = sum(len(group) for group in predicted_groups)
    print(f"\n  Total student selections: {total_selections} (5 groups × 6 students)")
    print(f"  Unique students across all groups: {len(all_students)}")
    print(f"  Diversity ratio: {len(all_students) / total_selections:.2%}")
    
    # Test with actual feedback
    print("\n6. Testing feedback mechanism...")
    print("-" * 70)
    
    # Simulate actual selection
    np.random.seed(42)
    actual_students = list(np.random.choice(55, 6, replace=False) + 1)
    
    print(f"\nSimulated actual selection: {actual_students}")
    
    feedback = pipeline.receive_feedback(actual_students, predicted_groups, probabilities)
    
    print(f"\n✓ Feedback processed")
    print(f"  Best group overlap: {feedback['best_overlap']}/6")
    print(f"  Best group accuracy: {feedback['best_accuracy']:.2%}")
    print(f"  Best group index: {feedback['best_group_index'] + 1}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    test_multi_group_prediction()
