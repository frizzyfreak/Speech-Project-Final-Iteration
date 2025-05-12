import os
import sys
import torch
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model_wav2vec import Wav2VecIntent
from scripts.utils.wav2vec_utils import load_audio_for_wav2vec
from scripts.test_model import setup_logging  # Reuse your existing logging setup

# Set up logger
logger = setup_logging()

def predict(model, audio_path, label_map, device):
    """Make a prediction on a single audio file"""
    try:
        # Load and preprocess audio
        waveform = load_audio_for_wav2vec(audio_path)
        
        if waveform is None:
            return None
            
        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(device)
        
        # Create attention mask (all 1s for single sample)
        attention_mask = torch.ones_like(waveform)
        
        # Get prediction
        with torch.no_grad():
            logits = model(waveform, attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Get label from class index
        inv_label_map = {v: k for k, v in label_map.items()}
        predicted_label = inv_label_map.get(pred_class, "Unknown")
        
        # Get top predictions
        top_k = min(3, len(inv_label_map))
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        top_predictions = [
            {
                "label": inv_label_map.get(idx.item(), "Unknown"),
                "probability": prob.item()
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        result = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_predictions": top_predictions
        }
        
        # Print results
        logger.info(f"\n----- PREDICTION RESULTS -----")
        logger.info(f"Predicted intent: {predicted_label}")
        logger.info(f"Confidence: {confidence*100:.2f}%")
        
        logger.info("\nTop predictions:")
        for i, pred in enumerate(top_predictions):
            logger.info(f"  {i+1}. {pred['label']} ({pred['probability']*100:.2f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def batch_test(model, audio_dir, label_map, device):
    """Process all audio files in a directory"""
    logger.info(f"Batch processing audio files in {audio_dir}")
    
    # Get all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return
    
    # Process each file
    for audio_path in audio_files:
        logger.info(f"\nProcessing: {audio_path.name}")
        predict(model, str(audio_path), label_map, device)

def interactive_test(model, label_map, device):
    """Interactive mode for testing audio files"""
    logger.info("Interactive testing mode. Enter path to audio file (or 'q' to quit):")
    
    while True:
        audio_path = input("\nEnter audio path (or 'q' to quit): ").strip()
        
        if audio_path.lower() == 'q':
            logger.info("Exiting interactive mode")
            break
        
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            continue
        
        predict(model, audio_path, label_map, device)

def main():
    parser = argparse.ArgumentParser(description="Test Wav2Vec intent recognition model")
    
    # Model arguments
    parser.add_argument('--model', type=str, default="new_checkpoints/wav2vec/wav2vec_best_model.pt", 
                        help='Path to model checkpoint')
    parser.add_argument('--label_map', type=str, default="data/processed/label_map.json", 
                        help='Path to label map JSON')
    
    # Test mode arguments
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    parser.add_argument('--audio', type=str, help='Path to audio file or directory')
    
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
    model.eval()
    
    # Run test mode
    if args.interactive:
        interactive_test(model, label_map, device)
    elif args.audio:
        if os.path.isdir(args.audio):
            batch_test(model, args.audio, label_map, device)
        elif os.path.isfile(args.audio):
            predict(model, args.audio, label_map, device)
        else:
            logger.error(f"Invalid path: {args.audio}")
    else:
        logger.error("Please specify --interactive or --audio")

if __name__ == "__main__":
    main()