"""
Training Module with Online Learning Support

Handles model training, validation, and incremental updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class Trainer:
    """Trainer class for student selection prediction models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 learning_rate: float = 0.001, checkpoint_dir: str = 'checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.epoch = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (top-6 prediction)
            pred_top6 = self._get_top_k_predictions(outputs, k=6)
            actual_top6 = self._get_top_k_predictions(batch_y, k=6)
            correct_predictions += self._calculate_overlap(pred_top6, actual_top6)
            total_predictions += len(batch_X)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / (total_predictions * 6)  # Normalize by 6 students
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_top6 = self._get_top_k_predictions(outputs, k=6)
                actual_top6 = self._get_top_k_predictions(batch_y, k=6)
                correct_predictions += self._calculate_overlap(pred_top6, actual_top6)
                total_predictions += len(batch_X)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / (total_predictions * 6)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, early_stopping_patience: int = 10) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        print("="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def online_update(self, new_X: torch.Tensor, new_y: torch.Tensor, 
                     n_iterations: int = 10) -> float:
        """
        Perform online learning update with new data.
        
        Args:
            new_X: New input data (batch, sequence_length, n_students)
            new_y: New target data (batch, n_students)
            n_iterations: Number of update iterations
            
        Returns:
            Average loss over iterations
        """
        self.model.train()
        total_loss = 0.0
        
        new_X = new_X.to(self.device)
        new_y = new_y.to(self.device)
        
        for _ in range(n_iterations):
            self.optimizer.zero_grad()
            outputs = self.model(new_X)
            loss = self.criterion(outputs, new_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / n_iterations
        print(f"Online update: avg loss = {avg_loss:.4f}")
        
        return avg_loss
    
    def predict(self, X: torch.Tensor, k: int = 6, n_groups: int = 5, 
                temperature: float = 1.5) -> Tuple[List[List[List[int]]], np.ndarray]:
        """
        Make predictions with multiple diverse groups.
        
        Args:
            X: Input tensor (batch, sequence_length, n_students)
            k: Number of students to select per group
            n_groups: Number of different groups to generate
            temperature: Sampling temperature (higher = more diverse)
            
        Returns:
            Tuple of (list of predicted groups, full probabilities for all students)
            - predicted_groups: [batch][group][student_ids] - shape (batch_size, n_groups, k)
            - probabilities: full probability distribution (batch_size, n_students)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            probabilities = outputs.cpu().numpy()
        
        all_batch_groups = []
        
        # Generate groups for each batch
        for batch_idx in range(probabilities.shape[0]):
            batch_probs = probabilities[batch_idx]
            batch_groups = []
            
            # Generate n_groups different selections
            for group_idx in range(n_groups):
                # Apply temperature scaling for diversity
                # Different temperature for each group ensures variety
                group_temp = temperature * (0.7 + 0.15 * group_idx)
                adjusted_probs = np.power(batch_probs, 1.0 / group_temp)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                # Sample k students based on probabilities (no repeats within group)
                selected_indices = np.random.choice(
                    len(adjusted_probs),
                    size=k,
                    replace=False,
                    p=adjusted_probs
                )
                
                # Sort by original probability (highest first)
                selected_indices = selected_indices[np.argsort(-batch_probs[selected_indices])]
                
                # Convert to 1-indexed student IDs
                selected_students = (selected_indices + 1).tolist()
                batch_groups.append(selected_students)
            
            all_batch_groups.append(batch_groups)
        
        return all_batch_groups, probabilities
    
    def _get_top_k_predictions(self, outputs: torch.Tensor, k: int = 6) -> torch.Tensor:
        """Get top-k predictions from output probabilities."""
        _, top_k_indices = torch.topk(outputs, k, dim=1)
        return top_k_indices
    
    def _calculate_overlap(self, pred: torch.Tensor, actual: torch.Tensor) -> int:
        """Calculate overlap between predicted and actual selections."""
        overlap = 0
        for i in range(len(pred)):
            pred_set = set(pred[i].cpu().numpy())
            actual_set = set(actual[i].cpu().numpy())
            overlap += len(pred_set & actual_set)
        return overlap
    
    def save_checkpoint(self, filename: str, save_optimizer: bool = True):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            save_optimizer: Whether to save optimizer state
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str, load_optimizer: bool = True):
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
            load_optimizer: Whether to load optimizer state
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {self.epoch}, Best val loss: {self.best_val_loss:.4f}")
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / filename
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")


if __name__ == "__main__":
    print("Trainer module loaded successfully")
