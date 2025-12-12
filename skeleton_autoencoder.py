"""Lightweight autoencoder for golf swing spatial patterns."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletonAutoencoder(nn.Module):
    """Autoencoder for spatial patterns."""
    
    def __init__(self, 
                 input_dim=132,
                 hidden_dim=128,
                 latent_dim=64,
                 num_lstm_layers=2,
                 dropout=0.2):
        super(SkeletonAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.encoder_lstm = nn.LSTM(
            hidden_dim,
            latent_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.decoder_lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        
        x_reshaped = x.transpose(1, 2)
        cnn_out = self.encoder_cnn(x_reshaped)
        cnn_out = self.dropout(cnn_out)
        
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, (hidden, cell) = self.encoder_lstm(cnn_out)
        
        decoder_input = lstm_out
        decoder_lstm_out, _ = self.decoder_lstm(decoder_input)

        decoder_lstm_out = decoder_lstm_out.transpose(1, 2)
        
        reconstructed = self.decoder_cnn(decoder_lstm_out)
        reconstructed = reconstructed.transpose(1, 2)
        
        return reconstructed
    
    def encode(self, x):
        batch_size, seq_length, input_dim = x.size()
        
        x_reshaped = x.transpose(1, 2)
        cnn_out = self.encoder_cnn(x_reshaped)
        cnn_out = cnn_out.transpose(1, 2)
        
        lstm_out, _ = self.encoder_lstm(cnn_out)
        
        return lstm_out
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


