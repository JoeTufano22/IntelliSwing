"""
Lightweight autoencoder for learning professional golf swing spatial patterns.
Focuses on spatial features (body positions, angles) independent of tempo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonAutoencoder(nn.Module):
    """
    Autoencoder for learning professional golf swing spatial patterns.
    
    Architecture:
    - Encoder: 1D CNN for spatial features + LSTM for sequence modeling
    - Decoder: LSTM + 1D CNN to reconstruct skeleton sequences
    
    Input: [batch_size, seq_length, 132] (132 = 33 landmarks × 4 values)
    Output: [batch_size, seq_length, 132] (reconstructed skeletons)
    """
    
    def __init__(self, 
                 input_dim=132,  # 33 landmarks × 4 values
                 hidden_dim=128,
                 latent_dim=64,
                 num_lstm_layers=2,
                 dropout=0.2):
        """
        Args:
            input_dim: Input feature dimension (132 for MediaPipe skeletons)
            hidden_dim: Hidden dimension for CNN and LSTM
            latent_dim: Latent representation dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(SkeletonAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: Spatial feature extraction + temporal modeling
        # 1D CNN for spatial feature extraction from landmarks
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # LSTM for temporal sequence modeling
        self.encoder_lstm = nn.LSTM(
            hidden_dim,
            latent_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Decoder: Reconstruct temporal sequence
        self.decoder_lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # 1D CNN for spatial reconstruction
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        
        Returns:
            reconstructed: Reconstructed tensor of shape [batch_size, seq_length, input_dim]
            latent: Latent representation (optional, for analysis)
        """
        batch_size, seq_length, input_dim = x.size()
        
        # Encoder
        # Reshape for CNN: [batch, seq, features] -> [batch, features, seq]
        x_reshaped = x.transpose(1, 2)  # [batch, input_dim, seq_length]
        
        # CNN for spatial features
        cnn_out = self.encoder_cnn(x_reshaped)  # [batch, hidden_dim, seq_length]
        cnn_out = self.dropout(cnn_out)
        
        # Reshape back for LSTM: [batch, hidden_dim, seq] -> [batch, seq, hidden_dim]
        cnn_out = cnn_out.transpose(1, 2)  # [batch, seq_length, hidden_dim]
        
        # LSTM for temporal modeling
        lstm_out, (hidden, cell) = self.encoder_lstm(cnn_out)
        # lstm_out: [batch, seq_length, latent_dim]
        
        # Decoder
        # Use LSTM hidden state to reconstruct sequence
        # Initialize decoder with encoder's last hidden state
        decoder_input = lstm_out  # Use encoder output as decoder input
        
        # Decoder LSTM
        decoder_lstm_out, _ = self.decoder_lstm(decoder_input)
        # decoder_lstm_out: [batch, seq_length, hidden_dim]
        
        # Reshape for CNN: [batch, seq, hidden_dim] -> [batch, hidden_dim, seq]
        decoder_lstm_out = decoder_lstm_out.transpose(1, 2)
        
        # CNN for spatial reconstruction
        reconstructed = self.decoder_cnn(decoder_lstm_out)
        # reconstructed: [batch, input_dim, seq_length]
        
        # Reshape back: [batch, input_dim, seq] -> [batch, seq, input_dim]
        reconstructed = reconstructed.transpose(1, 2)
        
        return reconstructed
    
    def encode(self, x):
        """
        Encode input to latent representation (for analysis/debugging).
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
        
        Returns:
            latent: Latent representation [batch_size, seq_length, latent_dim]
        """
        batch_size, seq_length, input_dim = x.size()
        
        # Encoder
        x_reshaped = x.transpose(1, 2)
        cnn_out = self.encoder_cnn(x_reshaped)
        cnn_out = cnn_out.transpose(1, 2)
        
        lstm_out, _ = self.encoder_lstm(cnn_out)
        
        return lstm_out
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

