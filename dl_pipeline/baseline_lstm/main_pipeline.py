"""
Main End-to-End Deep Learning Pipeline

Complete pipeline for training and deploying online learning system
for student selection prediction.
"""

import sys
from pathlib import Path

# Resolve paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent         # baseline_lstm/
_DL_DIR   = _THIS_DIR.parent                        # dl_pipeline/
_ROOT_DIR  = _DL_DIR.parent                         # Statistic/
_CKPT_DIR  = str(_DL_DIR / 'dl_checkpoints' / 'baseline_lstm')

sys.path.insert(0, str(_DL_DIR))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data_loader import DataManager, prepare_data_pipeline
from models import create_model
from trainer import Trainer
from online_pipeline import OnlineLearningPipeline


def train_initial_model(excel_path: str = 'Database.xlsx',
                        model_type: str = 'lstm',
                        sequence_length: int = 14,
                        epochs: int = 50,
                        batch_size: int = 32,
                        device: str = 'cpu'):
    """
    Train initial model on historical data.
    
    Args:
        excel_path: Path to Excel data file
        model_type: Type of model ('lstm', 'transformer', 'cnn_lstm')
        sequence_length: Sequence length for inputs
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        Tuple of (model, trainer, data_manager)
    """
    print("="*70)
    print("PHASE 1: INITIAL MODEL TRAINING")
    print("="*70)
    
    # Prepare data
    print("\nPreparing data...")
    data_manager, train_dataset, val_dataset = prepare_data_pipeline(
        excel_path=excel_path,
        sequence_length=sequence_length,
        train_ratio=0.9
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, n_students=55)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        checkpoint_dir=_CKPT_DIR
    )
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=15
    )
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    trainer.save_history('training_history.json')
    
    return model, trainer, data_manager


def run_online_learning_simulation(model, trainer, data_manager,
                                   sequence_length: int = 14,
                                   prediction_interval: int = 2,
                                   n_simulation_days: int = 100):
    """
    Run online learning simulation.
    
    Args:
        model: Trained model
        trainer: Trainer instance
        data_manager: DataManager instance
        sequence_length: Sequence length
        prediction_interval: Days between predictions
        n_simulation_days: Number of days to simulate
    """
    print("\n" + "="*70)
    print("PHASE 2: ONLINE LEARNING SIMULATION")
    print("="*70)
    
    # Get test data (last 10% of data)
    total_data = data_manager.binary_vectors
    split_idx = int(len(total_data) * 0.9)
    test_data = total_data[split_idx:]
    
    print(f"\nSimulation setup:")
    print(f"  Test data available: {len(test_data)} days")
    print(f"  Simulation days: {min(n_simulation_days, len(test_data))}")
    print(f"  Prediction interval: Every {prediction_interval} days")
    print(f"  Expected predictions: {min(n_simulation_days, len(test_data)) // prediction_interval}")
    
    # Create online learning pipeline
    pipeline = OnlineLearningPipeline(
        model=model,
        trainer=trainer,
        data_manager=data_manager,
        sequence_length=sequence_length,
        prediction_interval=prediction_interval,
        checkpoint_dir=_CKPT_DIR
    )
    
    # Run simulation
    pipeline.run_simulation(test_data, n_days=n_simulation_days)
    
    # Save results
    pipeline.save_predictions_log('online_predictions_log.csv')
    pipeline.save_checkpoint('final_online_model.pth')
    
    return pipeline


def deploy_for_production(checkpoint_path: str = 'dl_checkpoints/best_model.pth',
                          excel_path: str = 'Database.xlsx',
                          model_type: str = 'lstm',
                          sequence_length: int = 14):
    """
    Deploy model for production use.
    
    Args:
        checkpoint_path: Path to model checkpoint
        excel_path: Path to data file
        model_type: Model type
        sequence_length: Sequence length
        
    Returns:
        Tuple of (model, trainer, data_manager, pipeline)
    """
    print("="*70)
    print("DEPLOYING MODEL FOR PRODUCTION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    data_manager = DataManager(excel_path)
    data_manager.load_data()
    data_manager.convert_to_binary()
    data_manager.sort_by_days()
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, n_students=55)
    
    # Create trainer and load checkpoint
    trainer = Trainer(model=model, checkpoint_dir='dl_checkpoints')
    
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(Path(checkpoint_path).name)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found")
    
    # Create pipeline
    pipeline = OnlineLearningPipeline(
        model=model,
        trainer=trainer,
        data_manager=data_manager,
        sequence_length=sequence_length,
        prediction_interval=2
    )
    
    print("\n✓ Model deployed and ready for predictions")
    
    return model, trainer, data_manager, pipeline


def make_next_prediction(pipeline: OnlineLearningPipeline) -> dict:
    """
    Make prediction for next 6 students.
    
    Args:
        pipeline: OnlineLearningPipeline instance
        
    Returns:
        Dictionary with prediction results
    """
    print("\n" + "="*70)
    print("MAKING NEXT PREDICTION")
    print("="*70)
    
    predicted_students, predicted_probs = pipeline.make_prediction()
    
    result = {
        'predicted_students': predicted_students.tolist(),
        'probabilities': predicted_probs.tolist(),
        'timestamp': pipeline.predictions_log[-1]['timestamp'] if pipeline.predictions_log else None
    }
    
    print(f"\nPredicted next 6 students:")
    for i, (student, prob) in enumerate(zip(predicted_students, predicted_probs), 1):
        print(f"  {i}. Student {student:2d} - Probability: {prob:.4f}")
    
    return result


def provide_feedback_and_update(pipeline: OnlineLearningPipeline, 
                               actual_students: list):
    """
    Provide feedback with actual results and update model.
    
    Args:
        pipeline: OnlineLearningPipeline instance
        actual_students: List of actually selected student IDs
    """
    print("\n" + "="*70)
    print("RECEIVING FEEDBACK AND UPDATING MODEL")
    print("="*70)
    
    # Get last prediction
    if not pipeline.predictions_log:
        print("No predictions have been made yet")
        return
    
    last_prediction = pipeline.predictions_log[-1]
    predicted_students = np.array(last_prediction['predicted'])
    predicted_probs = np.array(last_prediction['probabilities'])
    
    # Receive feedback
    pipeline.receive_feedback(actual_students, predicted_students, predicted_probs)
    
    # Update model
    pipeline.update_model(actual_students)
    
    # Save checkpoint
    pipeline.save_checkpoint(f'update_{len(pipeline.predictions_log)}.pth')


def main():
    """Main execution function."""
    print("="*70)
    print("STUDENT SELECTION PREDICTION - DEEP LEARNING PIPELINE")
    print("="*70)
    print("\nThis pipeline demonstrates:")
    print("  1. Initial model training on historical data")
    print("  2. Online learning with periodic predictions")
    print("  3. Feedback integration and model updates")
    print("  4. Accuracy tracking over time")
    print("  5. Model checkpointing")
    
    # Configuration
    CONFIG = {
        'excel_path': str(_ROOT_DIR / 'Database.xlsx'),
        'model_type': 'lstm',  # Options: 'lstm', 'transformer', 'cnn_lstm'
        'sequence_length': 14,
        'epochs': 30,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'prediction_interval': 2,
        'n_simulation_days': 100
    }
    
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Phase 1: Train initial model
    model, trainer, data_manager = train_initial_model(
        excel_path=CONFIG['excel_path'],
        model_type=CONFIG['model_type'],
        sequence_length=CONFIG['sequence_length'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device']
    )
    
    # Phase 2: Run online learning simulation
    pipeline = run_online_learning_simulation(
        model=model,
        trainer=trainer,
        data_manager=data_manager,
        sequence_length=CONFIG['sequence_length'],
        prediction_interval=CONFIG['prediction_interval'],
        n_simulation_days=CONFIG['n_simulation_days']
    )
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  • dl_checkpoints/best_model.pth - Best model from initial training")
    print("  • dl_checkpoints/final_model.pth - Final trained model")
    print("  • dl_checkpoints/final_online_model.pth - Model after online learning")
    print("  • dl_checkpoints/training_history.json - Training history")
    print("  • dl_checkpoints/online_predictions_log.csv - All predictions and results")
    print("\nThe model is now ready for production deployment!")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
