"""Train skeleton autoencoder to learn professional golf swing spatial patterns."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skeleton_dataloader import SkeletonDataset
from skeleton_autoencoder import SkeletonAutoencoder


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    
    for batch_idx, skeleton_seq in enumerate(dataloader):
        skeleton_seq = skeleton_seq.to(device)
        
        reconstructed = model(skeleton_seq)
        loss = criterion(reconstructed, skeleton_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), skeleton_seq.size(0))
    
    return losses.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for skeleton_seq in dataloader:
            skeleton_seq = skeleton_seq.to(device)
            
            reconstructed = model(skeleton_seq)
            loss = criterion(reconstructed, skeleton_seq)
            
            losses.update(loss.item(), skeleton_seq.size(0))
    
    return losses.avg


def main():
    split = 1
    epochs = 50
    save_every = 10
    n_cpu = 2
    seq_length = 64
    batch_size = 8
    learning_rate = 0.001
    
    hidden_dim = 128
    latent_dim = 64
    num_lstm_layers = 2
    dropout = 0.2
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple Silicon (MPS)')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    skeletons_file = 'data/skeletons.pkl'
    if not os.path.exists(skeletons_file):
        print(f"Error: {skeletons_file} not found.")
        print("Please run extract_skeletons.py first to extract skeletons from videos.")
        return
    
    model = SkeletonAutoencoder(
        input_dim=132,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    model.to(device)
    model.train()
    
    train_dataset = SkeletonDataset(
        data_file=f'data/train_split_{split}.pkl',
        skeletons_file=skeletons_file,
        seq_length=seq_length,
        train=True
    )
    
    val_dataset = SkeletonDataset(
        data_file=f'data/val_split_{split}.pkl',
        skeletons_file=skeletons_file,
        seq_length=seq_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    os.makedirs('models', exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        if epoch % save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': {
                    'hidden_dim': hidden_dim,
                    'latent_dim': latent_dim,
                    'num_lstm_layers': num_lstm_layers,
                    'dropout': dropout
                }
            }
            
            checkpoint_path = f'models/skeleton_autoencoder_epoch_{epoch}.pth.tar'
            torch.save(checkpoint, checkpoint_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = 'models/skeleton_autoencoder_best.pth.tar'
                torch.save(checkpoint, best_path)
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")
    
    print("=" * 80)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()


