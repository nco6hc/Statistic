"""
Deep Learning Pipeline for Student Selection Prediction

A complete end-to-end pipeline with online learning capabilities.
"""

__version__ = "1.0.0"

from .data_loader import DataManager, StudentSelectionDataset, prepare_data_pipeline
from .models import LSTMStudentPredictor, TransformerStudentPredictor, CNNLSTMStudentPredictor, create_model
from .trainer import Trainer
from .online_pipeline import OnlineLearningPipeline

__all__ = [
    'DataManager',
    'StudentSelectionDataset',
    'prepare_data_pipeline',
    'LSTMStudentPredictor',
    'TransformerStudentPredictor',
    'CNNLSTMStudentPredictor',
    'create_model',
    'Trainer',
    'OnlineLearningPipeline'
]
