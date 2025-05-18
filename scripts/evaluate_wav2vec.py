import os
import sys
import torch
import json
import argparse
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model_wav2vec import Wav2VecIntent
from scripts.wav2vec_dataset import Wav2VecIntentDataset
from scripts.utils.wav2vec_utils import process_batch_for_wav2vec

# Define setup_logging function
def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Set up logger
logger = setup_logging()

def evaluate(model, dataloader, label_map, device):
    """Evaluate the model on a test dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, attention_mask, labels in dataloader:
            # Move data to device
            waveforms = waveforms.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(waveforms, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    inv_label_map = {v: k for k, v in label_map.items()}
    all_preds_labels = [inv_label_map[p] for p in all_preds]
    all_labels_labels = [inv_label_map[l] for l in all_labels]
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(label_map.keys()))
    
    return accuracy, report

# Define a custom collate function
def custom_collate(batch):
    """Custom collate function to process batches for Wav2Vec."""
    waveforms = [item[0] for item in batch]  # Extract waveforms
    labels = torch.tensor([item[1] for item in batch])  # Extract labels
    
    # Process waveforms for Wav2Vec
    waveforms_padded, attention_mask = process_batch_for_wav2vec(waveforms)
    
    return waveforms_padded, attention_mask, labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec intent recognition model")
    
    # Model arguments
    parser.add_argument('--model', type=str, default="checkpoints11/wav2vec/wav2vec_best_model.pt", 
                        help='Path to model checkpoint')
    parser.add_argument('--label_map', type=str, default="data/processed/label_map.json", 
                        help='Path to label map JSON')
    
    # Dataset arguments
    parser.add_argument('--test_csv', type=str, default="data/processed/test_data.csv",
                        help='Path to test CSV file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    
    # Hardware arguments
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = Wav2VecIntent(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    # Load test dataset
    logger.info(f"Loading test dataset from {args.test_csv}")
    test_dataset = Wav2VecIntentDataset(args.test_csv, label_map, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
        pin_memory=True
    )
    # Evaluate the model
    logger.info("Evaluating the model...")
    accuracy, report = evaluate(model, test_loader, label_map, device)
    
    # Log results
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info("\nClassification Report:\n")
    logger.info(report)

if __name__ == "__main__":
    main()