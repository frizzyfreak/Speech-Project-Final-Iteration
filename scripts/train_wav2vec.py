import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model_wav2vec import Wav2VecIntent
from scripts.wav2vec_dataset import Wav2VecIntentDataset
from scripts.utils.wav2vec_utils import process_batch_for_wav2vec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def custom_collate(batch):
    """Custom collate function for processing wav2vec batches"""
    waveforms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # Pad waveforms
    waveforms_padded, attention_mask = process_batch_for_wav2vec(waveforms)
    
    return waveforms_padded, attention_mask, labels

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    logger.info(f"Found {num_classes} classes in label map")
    
    # Create datasets
    train_dataset = Wav2VecIntentDataset(args.train_csv, label_map, is_training=True)
    valid_dataset = Wav2VecIntentDataset(args.valid_csv, label_map, is_training=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    # Create model
    model = Wav2VecIntent(num_classes=num_classes, pretrained_model=args.pretrained_model)
    model.to(device)
    
    # Freeze wav2vec feature extractor
    if args.freeze_feature_extractor:
        logger.info("Freezing feature extractor")
        model.wav2vec.feature_extractor._freeze_parameters()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize for early stopping
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    patience_counter = 0
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Training")
        for waveforms, attention_mask, labels in pbar:
            # Move data to device
            waveforms = waveforms.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(waveforms, attention_mask)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward pass
                outputs = model(waveforms, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Update weights
                optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total if train_total > 0 else 0
            })
        
        # Calculate average training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Validating")
            for waveforms, attention_mask, labels in pbar:
                # Move data to device
                waveforms = waveforms.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(waveforms, attention_mask)
                loss = criterion(outputs, labels)
                
                # Track statistics
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': valid_loss / (pbar.n + 1),
                    'acc': 100. * valid_correct / valid_total if valid_total > 0 else 0
                })
        
        # Calculate average validation metrics
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100. * valid_correct / valid_total if valid_total > 0 else 0
        
        # Log results
        logger.info(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, "
                   f"Valid loss: {valid_loss:.4f}, Valid acc: {valid_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Save checkpoint
        checkpoint_path = output_dir / f"wav2vec_checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }, checkpoint_path)
        
        # Check for improvement
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / "wav2vec_best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with accuracy: {best_valid_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_valid_acc:.2f}%")
    return best_valid_acc

def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec intent recognition model")
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default="data/processed2/new_train_data.csv",
                        help='Path to training CSV file')
    parser.add_argument('--valid_csv', type=str, default="synthetic/filtered_csv_file.csv",
                        help='Path to validation CSV file')
    parser.add_argument('--label_map', type=str, default="data/processed2/label_map.json",
                        help='Path to label map JSON file')
    
    # Model arguments
    parser.add_argument('--pretrained_model', type=str, default="facebook/wav2vec2-large",
                        help='Name or path to pretrained Wav2Vec model')
    parser.add_argument('--freeze_feature_extractor', action='store_true',
                        help='Freeze feature extractor part of Wav2Vec')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Hardware arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="checkpoints10/wav2vec",
                        help='Output directory for model checkpoints')
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()