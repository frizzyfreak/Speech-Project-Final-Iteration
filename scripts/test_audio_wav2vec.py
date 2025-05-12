import os
import sys
import torch
import json
import argparse
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Add this import at the top of your script

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model_wav2vec import Wav2VecIntent
from scripts.utils.wav2vec_utils import load_audio_for_wav2vec, process_batch_for_wav2vec

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

def predict_single_file(model, audio_path, label_map, device):
    """Make a prediction on a single audio file."""
    try:
        # Load and preprocess audio
        waveform = load_audio_for_wav2vec(audio_path)
        if waveform is None:
            logger.error(f"Failed to load audio file: {audio_path}")
            return None

        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(device)
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

        logger.info(f"Predicted intent: {predicted_label} (Confidence: {confidence * 100:.2f}%)")
        return {"label": predicted_label, "confidence": confidence, "class_index": pred_class}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def predict_batch(model, audio_dir, label_map, device):
    """Make predictions on all audio files in a directory."""
    logger.info(f"Batch processing audio files in {audio_dir}")

    # Load the details.csv file to create the numeric-to-descriptive mapping
    details_path = Path("data/processed2/details_unique.csv")
    if not details_path.exists():
        logger.error(f"Details file not found at {details_path}")
        return

    # Create a mapping from numeric prefixes to descriptive labels
    details_df = pd.read_csv(details_path)
    numeric_to_descriptive = {
        row['filename'].split("_")[0]: row['class'] for _, row in details_df.iterrows()
    }

    # Get all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    results = []
    all_preds = []
    all_labels = []

    for audio_path in audio_files:
        logger.info(f"Processing: {audio_path.name}")
        # Extract numeric prefix and map to descriptive label
        ground_truth_label_numeric = audio_path.stem.split("_")[0]
        ground_truth_label = numeric_to_descriptive.get(ground_truth_label_numeric)

        if not ground_truth_label or ground_truth_label not in label_map:
            logger.error(f"Ground truth label '{ground_truth_label_numeric}' not found in label map for file: {audio_path.name}")
            continue
        ground_truth_index = label_map[ground_truth_label]

        result = predict_single_file(model, audio_path, label_map, device)
        if result:
            results.append({"file": audio_path.name, **result})
            all_preds.append(result["class_index"])
            all_labels.append(ground_truth_index)

            # Log predictions and ground truth
            logger.info(f"File: {audio_path.name}, Ground Truth: {ground_truth_label}, Predicted: {result['label']}")

    # Calculate accuracy
    if all_labels and all_preds:
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Batch Accuracy: {accuracy * 100:.2f}%")
    else:
        logger.warning("No valid predictions or ground truth labels found. Accuracy cannot be calculated.")
        accuracy = 0.0

    # Generate confusion matrix if there are valid predictions
    if all_labels and all_preds:
        cm = confusion_matrix(all_labels, all_preds)
        inv_label_map = {v: k for k, v in label_map.items()}
        labels = [inv_label_map[i] for i in range(len(label_map))]

        # Create a folder to save the confusion matrix
        output_dir = Path("scripts/confusion_matrices7")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "confusion_matrix.png"

        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(output_file)
        logger.info(f"Confusion matrix saved to {output_file}")
        plt.close()

    return results

def main():
    parser = argparse.ArgumentParser(description="Test Wav2Vec intent recognition model on audio files")
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--label_map', type=str, required=True, 
                        help='Path to label map JSON')
    
    # Input arguments
    parser.add_argument('--audio', type=str, required=True, 
                        help='Path to an audio file or directory containing audio files')
    
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
    
    # Check if input is a file or directory
    audio_path = Path(args.audio)
    if audio_path.is_file():
        logger.info(f"Testing single audio file: {audio_path}")
        predict_single_file(model, audio_path, label_map, device)
    elif audio_path.is_dir():
        logger.info(f"Testing batch of audio files in directory: {audio_path}")
        results = predict_batch(model, audio_path, label_map, device)
        if results:
            logger.info("\nBatch Results:")
            for result in results:
                logger.info(f"File: {result['file']}, Predicted Intent: {result['label']}, Confidence: {result['confidence'] * 100:.2f}%")
    else:
        logger.error(f"Invalid audio path: {audio_path}")

if __name__ == "__main__":
    main()