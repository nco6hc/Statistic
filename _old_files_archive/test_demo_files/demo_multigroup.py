"""
Quick demo of multi-group predictions with online learning
"""

import torch
from pathlib import Path

from dl_pipeline.data_loader import DataManager
from dl_pipeline.models import LSTMStudentPredictor
from dl_pipeline.trainer import Trainer
from dl_pipeline.online_pipeline import OnlineLearningPipeline


def run_demo():
    """Run a quick demo of the updated pipeline"""
    print("="*70)
    print("MULTI-GROUP PREDICTION DEMO")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_manager = DataManager(excel_path='Database.xlsx')
    data_manager.data = data_manager.load_data()
    data_manager.convert_to_binary()
    data_manager.sort_by_days()
    train_dataset, val_dataset = data_manager.create_datasets(sequence_length=14)
    
    # Load trained model
    print("\nLoading trained model...")
    model = LSTMStudentPredictor(n_students=55)
    
    checkpoint_path = 'dl_pipeline/dl_checkpoints/final_online_model.pth'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded")
    else:
        print("⚠ No checkpoint found")
    
    # Create trainer and pipeline
    trainer = Trainer(model, checkpoint_dir='demo_checkpoints')
    pipeline = OnlineLearningPipeline(
        model=model,
        trainer=trainer,
        data_manager=data_manager,
        sequence_length=14,
        prediction_interval=2,
        checkpoint_dir='demo_checkpoints'
    )
    
    # Run simulation for 20 days (10 predictions)
    print("\nRunning 20-day simulation (10 predictions)...")
    print("="*70)
    
    total_data = data_manager.binary_vectors
    split_idx = int(len(total_data) * 0.9)
    test_data = total_data[split_idx:split_idx+20]  # Just 20 days
    
    pipeline.run_simulation(test_data, n_days=20)
    
    # Save results
    print("\nSaving results...")
    pipeline.save_predictions_log('demo_predictions_multigroup.csv')
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nCheck demo_checkpoints/demo_predictions_multigroup.csv for results")
    print("Each prediction now includes 5 different groups of 6 students")


if __name__ == "__main__":
    run_demo()
