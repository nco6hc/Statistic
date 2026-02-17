"""
Deep Learning Models for Student Selection Prediction

Implements LSTM and Transformer-based sequence models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMStudentPredictor(nn.Module):
    """LSTM-based model for predicting student selections."""
    
    def __init__(self, n_students: int = 55, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize LSTM model.
        
        Args:
            n_students: Number of students (input/output dimension)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMStudentPredictor, self).__init__()
        
        self.n_students = n_students
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_students,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, n_students)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, n_students)
            
        Returns:
            Output tensor of shape (batch, n_students) with probabilities
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        out = F.relu(self.fc1(last_hidden))
        out = self.batch_norm1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.fc2(out))
        out = self.batch_norm2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = torch.sigmoid(out)  # Output probabilities
        
        return out


class TransformerStudentPredictor(nn.Module):
    """Transformer-based model for predicting student selections."""
    
    def __init__(self, n_students: int = 55, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 256, dropout: float = 0.3):
        """
        Initialize Transformer model.
        
        Args:
            n_students: Number of students
            d_model: Dimension of model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerStudentPredictor, self).__init__()
        
        self.n_students = n_students
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(n_students, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_students)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, n_students)
            
        Returns:
            Output tensor of shape (batch, n_students) with probabilities
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last position
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x


class CNNLSTMStudentPredictor(nn.Module):
    """CNN-LSTM hybrid model for pattern extraction + temporal modeling."""
    
    def __init__(self, n_students: int = 55, cnn_channels: int = 64,
                 lstm_hidden: int = 128, dropout: float = 0.3):
        """
        Initialize CNN-LSTM model.
        
        Args:
            n_students: Number of students
            cnn_channels: Number of CNN channels
            lstm_hidden: LSTM hidden size
            dropout: Dropout rate
        """
        super(CNNLSTMStudentPredictor, self).__init__()
        
        self.n_students = n_students
        
        # CNN for pattern extraction
        self.conv1 = nn.Conv1d(n_students, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_hidden, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, n_students)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, n_students)
            
        Returns:
            Output tensor of shape (batch, n_students)
        """
        # Transpose for CNN: (batch, n_students, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout_cnn(x)
        
        # Transpose back for LSTM: (batch, sequence_length, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        
        # Output
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x


def create_model(model_type: str = 'lstm', n_students: int = 55, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('lstm', 'transformer', 'cnn_lstm')
        n_students: Number of students
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return LSTMStudentPredictor(n_students=n_students, **kwargs)
    elif model_type == 'transformer':
        return TransformerStudentPredictor(n_students=n_students, **kwargs)
    elif model_type == 'cnn_lstm':
        return CNNLSTMStudentPredictor(n_students=n_students, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("="*60)
    print("TESTING MODELS")
    print("="*60)
    
    batch_size = 8
    sequence_length = 14
    n_students = 55
    
    # Create dummy input
    x = torch.randn(batch_size, sequence_length, n_students)
    
    # Test LSTM
    print("\nTesting LSTM model...")
    lstm_model = create_model('lstm', n_students=n_students)
    output = lstm_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Test Transformer
    print("\nTesting Transformer model...")
    transformer_model = create_model('transformer', n_students=n_students)
    output = transformer_model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # Test CNN-LSTM
    print("\nTesting CNN-LSTM model...")
    cnn_lstm_model = create_model('cnn_lstm', n_students=n_students)
    output = cnn_lstm_model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in cnn_lstm_model.parameters()):,}")
    
    print("\n✓ Model tests complete")
