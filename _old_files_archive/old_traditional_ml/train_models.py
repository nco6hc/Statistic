"""
Multi-Model Training for Student Selection Prediction

This module trains and compares multiple machine learning models:
1. Logistic Regression (multi-label)
2. Random Forest
3. Gradient Boosting (XGBoost)
4. Neural Network (Keras)

Uses cross-validation and outputs probability predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")


class StudentSelectionModelTrainer:
    """Trains and evaluates multiple models for student selection prediction."""
    
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, 
                 X_test: np.ndarray, Y_test: np.ndarray):
        """
        Initialize the model trainer.
        
        Args:
            X_train: Training features (n_samples, n_features)
            Y_train: Training labels (n_samples, n_students)
            X_test: Test features (n_samples, n_features)
            Y_test: Test labels (n_samples, n_students)
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.n_students = Y_train.shape[1]
        self.n_features = X_train.shape[1]
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Store trained models
        self.models = {}
        self.results = {}
        
        print(f"Initialized trainer:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {self.n_features}")
        print(f"  Students: {self.n_students}")
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None, model_name: str = "") -> Dict:
        """
        Evaluate predictions with multiple metrics.
        
        Args:
            y_true: True labels (n_samples, n_students)
            y_pred: Predicted labels (n_samples, n_students)
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Multi-label metrics
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        results['f1_samples'] = f1_score(y_true, y_pred, average='samples')
        results['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        results['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        
        # Exact match accuracy
        results['exact_match'] = accuracy_score(y_true, y_pred)
        
        # Element-wise accuracy
        results['element_accuracy'] = (y_pred == y_true).mean()
        
        # Overlap metrics (how many students overlap)
        overlap = (y_pred * y_true).sum(axis=1)
        results['avg_overlap'] = overlap.mean()
        results['max_overlap'] = overlap.max()
        results['min_overlap'] = overlap.min()
        
        return results
    
    def train_logistic_regression(self, C: float = 1.0) -> Dict:
        """
        Train multi-label Logistic Regression.
        
        Args:
            C: Regularization parameter
            
        Returns:
            Evaluation results
        """
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        # Use OneVsRestClassifier for multi-label
        model = OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1),
            n_jobs=-1
        )
        
        print("Training...")
        model.fit(self.X_train_scaled, self.Y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(self.X_test_scaled)
        
        # Convert probabilities to binary predictions (top-6 students)
        y_pred = self._proba_to_top_k(y_pred_proba, k=6)
        
        # Evaluate
        results = self.evaluate_predictions(self.Y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        self._print_results("Logistic Regression", results)
        
        return results
    
    def train_random_forest(self, n_estimators: int = 100, max_depth: int = 20) -> Dict:
        """
        Train multi-label Random Forest.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            
        Returns:
            Evaluation results
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
                min_samples_split=10,
                min_samples_leaf=5
            ),
            n_jobs=-1
        )
        
        print(f"Training with {n_estimators} trees, max_depth={max_depth}...")
        model.fit(self.X_train_scaled, self.Y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(self.X_test_scaled)
        
        # Convert to binary predictions
        y_pred = self._proba_to_top_k(y_pred_proba, k=6)
        
        # Evaluate
        results = self.evaluate_predictions(self.Y_test, y_pred, y_pred_proba, "Random Forest")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        self._print_results("Random Forest", results)
        
        return results
    
    def train_xgboost(self, n_estimators: int = 100, max_depth: int = 6, 
                     learning_rate: float = 0.1) -> Dict:
        """
        Train multi-label XGBoost.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            
        Returns:
            Evaluation results
        """
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost not available, skipping...")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        model = OneVsRestClassifier(
            xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            n_jobs=-1
        )
        
        print(f"Training with {n_estimators} rounds, max_depth={max_depth}, lr={learning_rate}...")
        model.fit(self.X_train_scaled, self.Y_train)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(self.X_test_scaled)
        
        # Convert to binary predictions
        y_pred = self._proba_to_top_k(y_pred_proba, k=6)
        
        # Evaluate
        results = self.evaluate_predictions(self.Y_test, y_pred, y_pred_proba, "XGBoost")
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        self._print_results("XGBoost", results)
        
        return results
    
    def train_neural_network(self, epochs: int = 50, batch_size: int = 32,
                           hidden_layers: List[int] = [256, 128, 64]) -> Dict:
        """
        Train multi-label Neural Network with PyTorch.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            hidden_layers: List of hidden layer sizes
            
        Returns:
            Evaluation results
        """
        if not PYTORCH_AVAILABLE:
            print("\nPyTorch not available, skipping...")
            return {}
        
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK (PyTorch)")
        print("="*60)
        
        # Define PyTorch model
        class MultiLabelNN(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size):
                super(MultiLabelNN, self).__init__()
                layers_list = []
                
                # Input to first hidden layer
                prev_size = input_size
                for units in hidden_layers:
                    layers_list.append(nn.Linear(prev_size, units))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.BatchNorm1d(units))
                    layers_list.append(nn.Dropout(0.3))
                    prev_size = units
                
                # Output layer
                layers_list.append(nn.Linear(prev_size, output_size))
                layers_list.append(nn.Sigmoid())
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, x):
                return self.network(x)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiLabelNN(self.n_features, hidden_layers, self.n_students).to(device)
        
        print(f"\nModel architecture:")
        print(model)
        print(f"Device: {device}")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(device)
        Y_train_tensor = torch.FloatTensor(self.Y_train).to(device)
        X_test_tensor = torch.FloatTensor(self.X_test_scaled).to(device)
        
        # Split for validation
        val_split = int(0.8 * len(X_train_tensor))
        X_train_t, X_val_t = X_train_tensor[:val_split], X_train_tensor[val_split:]
        Y_train_t, Y_val_t = Y_train_tensor[:val_split], Y_train_tensor[val_split:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        
        # Training loop
        print("\nTraining...")
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, Y_val_t).item()
                val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        # Predict on test set
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test_tensor).cpu().numpy()
        
        # Convert to binary predictions
        y_pred = self._proba_to_top_k(y_pred_proba, k=6)
        
        # Evaluate
        results = self.evaluate_predictions(self.Y_test, y_pred, y_pred_proba, "Neural Network")
        results['train_losses'] = train_losses
        results['val_losses'] = val_losses
        
        self.models['neural_network'] = model
        self.results['neural_network'] = results
        
        self._print_results("Neural Network", results)
        
        return results
    
    def _proba_to_top_k(self, probabilities: np.ndarray, k: int = 6) -> np.ndarray:
        """
        Convert probabilities to binary predictions by selecting top-k.
        
        Args:
            probabilities: Probability matrix (n_samples, n_students)
            k: Number of students to select
            
        Returns:
            Binary prediction matrix
        """
        predictions = np.zeros_like(probabilities)
        
        for i in range(len(probabilities)):
            top_k_indices = np.argsort(probabilities[i])[-k:]
            predictions[i, top_k_indices] = 1
        
        return predictions
    
    def _print_results(self, model_name: str, results: Dict):
        """Print evaluation results."""
        print(f"\n{model_name} Results:")
        print(f"  F1 Score (micro): {results['f1_micro']:.4f}")
        print(f"  F1 Score (macro): {results['f1_macro']:.4f}")
        print(f"  F1 Score (samples): {results['f1_samples']:.4f}")
        print(f"  Precision: {results['precision_micro']:.4f}")
        print(f"  Recall: {results['recall_micro']:.4f}")
        print(f"  Exact Match: {results['exact_match']:.4f}")
        print(f"  Average Overlap: {results['avg_overlap']:.2f} / 6 students")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'F1 (micro)': results['f1_micro'],
                'F1 (macro)': results['f1_macro'],
                'F1 (samples)': results['f1_samples'],
                'Precision': results['precision_micro'],
                'Recall': results['recall_micro'],
                'Avg Overlap': results['avg_overlap']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 (samples)', ascending=False)
        
        print("\n", df.to_string(index=False))
        
        return df
    
    def predict_next_day(self, current_features: np.ndarray, 
                        model_name: str = 'neural_network') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next day's selections using specified model.
        
        Args:
            current_features: Feature vector for current day
            model_name: Name of model to use
            
        Returns:
            Tuple of (student_ids, probabilities)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        # Scale features
        features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        
        # Predict probabilities
        if model_name == 'neural_network' and PYTORCH_AVAILABLE:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(device)
                probabilities = model(features_tensor).cpu().numpy()[0]
        else:
            probabilities = model.predict_proba(features_scaled)[0]
        
        # Get top-6 students
        top_6_indices = np.argsort(probabilities)[-6:][::-1]
        top_6_student_ids = top_6_indices + 1  # Convert to 1-indexed
        top_6_probs = probabilities[top_6_indices]
        
        return top_6_student_ids, top_6_probs
    
    def save_models(self, prefix: str = "model"):
        """Save all trained models."""
        print("\nSaving models...")
        
        # Save scikit-learn models
        for name in ['logistic_regression', 'random_forest', 'xgboost']:
            if name in self.models:
                filename = f"{prefix}_{name}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(self.models[name], f)
                print(f"  Saved: {filename}")
        
        # Save PyTorch model
        if 'neural_network' in self.models and PYTORCH_AVAILABLE:
            filename = f"{prefix}_neural_network.pth"
            torch.save(self.models['neural_network'].state_dict(), filename)
            print(f"  Saved: {filename}")
        
        # Save scaler
        with open(f"{prefix}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved: {prefix}_scaler.pkl")
        
        # Save results
        with open(f"{prefix}_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        print(f"  Saved: {prefix}_results.pkl")


def main():
    """Main execution function."""
    print("="*60)
    print("STUDENT SELECTION MODEL TRAINING")
    print("="*60)
    
    # Load feature matrices
    print("\nLoading feature matrices...")
    X_train = np.load('X_train_features.npy')
    X_test = np.load('X_test_features.npy')
    Y_train = np.load('Y_train_labels.npy')
    Y_test = np.load('Y_test_labels.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize trainer
    trainer = StudentSelectionModelTrainer(X_train, Y_train, X_test, Y_test)
    
    # Train models
    trainer.train_logistic_regression(C=1.0)
    trainer.train_random_forest(n_estimators=100, max_depth=20)
    
    if XGBOOST_AVAILABLE:
        trainer.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
    
    if PYTORCH_AVAILABLE:
        trainer.train_neural_network(
            epochs=50,
            batch_size=32,
            hidden_layers=[256, 128, 64]
        )
    
    # Compare models
    comparison_df = trainer.compare_models()
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nSaved: model_comparison.csv")
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    # Use last test sample
    test_features = X_test[-1]
    actual_selection = Y_test[-1]
    actual_students = [i+1 for i, val in enumerate(actual_selection) if val == 1]
    
    print(f"\nActual selected students: {actual_students}")
    
    # Predict with best model (if available)
    for model_name in ['neural_network', 'xgboost', 'random_forest', 'logistic_regression']:
        if model_name in trainer.models:
            print(f"\n{model_name.replace('_', ' ').title()} Predictions:")
            pred_students, pred_probs = trainer.predict_next_day(test_features, model_name)
            
            for student_id, prob in zip(pred_students, pred_probs):
                match = "✓" if student_id in actual_students else "✗"
                print(f"  {match} Student {student_id}: {prob:.4f}")
            
            overlap = len(set(pred_students) & set(actual_students))
            print(f"  Overlap: {overlap} / 6 students")
    
    # Save models
    trainer.save_models(prefix="trained_model")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
