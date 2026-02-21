"""
Enhanced LSTM Model with Attention Mechanism and Focal Loss

Improvements over baseline LSTMStudentPredictor:
1. Bidirectional LSTM - sees context from both directions
2. Attention mechanism - learns which past days matter most
3. Focal Loss - focuses on harder-to-predict students
4. Label smoothing - prevents overconfident predictions
5. Input projection - handles richer 220-dim feature input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    
    Down-weights easy negatives and focuses learning on harder examples.
    With 55 students and only 6 selected per day, most labels are 0 (negative).
    Standard BCELoss treats all equally; Focal Loss addresses this imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor (default 0.25)
        gamma: Focusing parameter - higher = more focus on hard examples (default 2.0)
        label_smoothing: Smooth hard labels to prevent overconfidence (default 0.05)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing: 1 -> 0.975, 0 -> 0.025 (with default 0.05)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy (element-wise)
        bce = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # Focal weight: (1 - p_t)^gamma
        pt = torch.where(targets > 0.5, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine
        loss = self.alpha * focal_weight * bce
        
        return loss.mean()


class AttentionLayer(nn.Module):
    """
    Attention mechanism over LSTM sequence outputs.
    
    Instead of just using the last hidden state, attention learns
    which timesteps in the sequence are most important for prediction.
    
    Args:
        hidden_size: Size of LSTM hidden state (doubled for bidirectional)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> tuple:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            
        Returns:
            context: (batch, hidden_size) - attention-weighted context vector
            weights: (batch, seq_len, 1) - attention weights for interpretability
        """
        # Compute attention scores
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_size)
        
        return context, attn_weights


class EnhancedLSTMPredictor(nn.Module):
    """
    Enhanced LSTM with Bidirectional processing + Attention mechanism.
    
    Architecture:
        Input (220 features) -> Projection (128) -> BiLSTM (128*2=256)
        -> Attention -> FC(256->256->128->55) -> Sigmoid
    
    Key improvements:
    - Input projection: compresses 220 enriched features to hidden_size
    - Bidirectional LSTM: reads sequence forwards AND backwards
    - Attention: weighted combination of all timesteps (not just last)
    - Deeper FC head: 3 layers with BatchNorm and Dropout
    
    Args:
        n_features: Number of input features per timestep (220 with enrichment)
        n_students: Number of students to predict (55)
        hidden_size: LSTM hidden dimension (128)
        num_layers: Number of LSTM layers (2)
        dropout: Dropout rate (0.3)
    """
    
    def __init__(self, n_features: int = 220, n_students: int = 55,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_students = n_students
        self.hidden_size = hidden_size
        
        # Input projection: compress rich features to hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention over sequence (input is hidden_size*2 due to bidirectional)
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_students),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            Probabilities (batch, n_students)
        """
        # Project input features
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention-weighted context
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_size*2)
        
        # Output prediction
        out = self.fc(context)  # (batch, n_students)
        
        return out


if __name__ == "__main__":
    """Test the enhanced model."""
    print("=" * 60)
    print("TESTING ENHANCED MODEL")
    print("=" * 60)
    
    batch_size = 8
    seq_len = 14
    n_features = 220
    n_students = 55
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Test model
    model = EnhancedLSTMPredictor(
        n_features=n_features,
        n_students=n_students,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    output = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {n_params:,}")
    
    # Test focal loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    target = torch.zeros(batch_size, n_students)
    for i in range(batch_size):
        idx = np.random.choice(n_students, size=6, replace=False)
        target[i, idx] = 1.0
    
    loss = criterion(output, target)
    print(f"Focal Loss:   {loss.item():.4f}")
    
    print("\nAll tests passed!")
