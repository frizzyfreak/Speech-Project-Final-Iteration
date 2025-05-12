import optuna
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from tqdm import tqdm
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.model_wav2vec import Wav2VecIntent
from scripts.wav2vec_dataset import Wav2VecIntentDataset
from scripts.utils.wav2vec_utils import process_batch_for_wav2vec
from transformers import Wav2Vec2Model

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

def objective(trial):
    logger.info(f"\n==================== Starting Trial {trial.number} ====================")

    # Get hyperparameters to optimize
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    freeze_feature_extractor = trial.suggest_categorical('freeze_feature_extractor', [True, False])

    logger.info(f"Trial {trial.number} - Hyperparameters: "
                f"batch_size={batch_size}, lr={lr:.6f}, weight_decay={weight_decay:.6f}, "
                f"dropout={dropout:.3f}, hidden_size={hidden_size}, freeze_feature_extractor={freeze_feature_extractor}")

    # Define paths
    train_csv = "data/processed/train_data.csv"
    valid_csv = "data/processed/valid_data.csv"
    label_map_path = "data/processed/label_map.json"

    # Load label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Trial {trial.number}: Using device {device}")

    # Create datasets
    train_dataset = Wav2VecIntentDataset(train_csv, label_map, is_training=True)
    valid_dataset = Wav2VecIntentDataset(valid_csv, label_map, is_training=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate,
        pin_memory=True
    )

    class CustomWav2VecIntent(nn.Module):
        def __init__(self, num_classes, hidden_size, dropout_rate=0.1):
            super().__init__()
            self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            wav2vec_dim = self.wav2vec.config.hidden_size
            self.attention = nn.Linear(wav2vec_dim, 1)
            self.projection = nn.Linear(wav2vec_dim, hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, input_values, attention_mask=None):
            outputs = self.wav2vec(input_values, attention_mask=attention_mask, return_dict=True)
            hidden_states = outputs.last_hidden_state
            attn_weights = torch.nn.functional.softmax(self.attention(hidden_states), dim=1)
            x = torch.sum(hidden_states * attn_weights, dim=1)
            x = self.projection(x)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    try:
        model = CustomWav2VecIntent(num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout).to(device)

        if freeze_feature_extractor:
            for param in model.wav2vec.feature_extractor.parameters():
                param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_epochs = 3
        best_val_acc = 0.0

        logger.info(f"Trial {trial.number}: Starting training for {num_epochs} epochs")

        try:
            for epoch in range(num_epochs):
                logger.info(f"Trial {trial.number} - Epoch [{epoch+1}/{num_epochs}] - Training start")

                model.train()
                train_loss = 0.0

                train_pbar = tqdm(train_loader, desc=f'Trial {trial.number} | Epoch [{epoch+1}/{num_epochs}] | Train', leave=False)
                for batch_idx, (waveforms, attention_mask, labels) in enumerate(train_pbar):
                    try:
                        waveforms = waveforms.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(waveforms, attention_mask)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_pbar.set_postfix(loss=f'{loss.item():.4f}')

                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logger.warning(f"CUDA OOM in training batch {batch_idx+1}. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e

                train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

                logger.info(f"Trial {trial.number} - Epoch [{epoch+1}/{num_epochs}] - Validation start")

                model.eval()
                valid_correct = 0
                valid_total = 0
                valid_loss = 0.0

                valid_pbar = tqdm(valid_loader, desc=f'Trial {trial.number} | Epoch [{epoch+1}/{num_epochs}] | Val', leave=False)
                with torch.no_grad():
                    for batch_idx, (waveforms, attention_mask, labels) in enumerate(valid_pbar):
                        try:
                            waveforms = waveforms.to(device)
                            attention_mask = attention_mask.to(device)
                            labels = labels.to(device)

                            outputs = model(waveforms, attention_mask)
                            loss = criterion(outputs, labels)

                            valid_loss += loss.item()
                            _, predicted = outputs.max(1)
                            valid_total += labels.size(0)
                            valid_correct += predicted.eq(labels).sum().item()

                            valid_pbar.set_postfix(loss=f'{loss.item():.4f}')

                            torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if 'out of memory' in str(e):
                                logger.warning(f"CUDA OOM in validation batch {batch_idx+1}. Skipping batch.")
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e

                val_acc = valid_correct / valid_total if valid_total > 0 else 0
                val_loss = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0

                logger.info(f"Trial {trial.number} - Epoch [{epoch+1}/{num_epochs}] - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        except Exception as e:
            logger.error(f"Trial {trial.number} - Error during training: {str(e)}")
            return 0.0

        logger.info(f"==================== End of Trial {trial.number} | Best Val Acc: {best_val_acc:.4f} ====================\n")
        return best_val_acc

    except Exception as e:
        logger.error(f"Trial {trial.number} - Error initializing model: {str(e)}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for Wav2Vec intent model")
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL')
    parser.add_argument('--study_name', type=str, default="wav2vec_intent_tuning", help='Optuna study name')
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )

    try:
        logger.info(f"Optuna Study '{args.study_name}' started with {args.n_trials} trials")
        study.optimize(objective, n_trials=args.n_trials)

        logger.info("\n\n========== Optimization Results ==========")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        try:
            import matplotlib.pyplot as plt
            import optuna.visualization as vis

            plots_dir = Path("results/optuna_plots")
            plots_dir.mkdir(parents=True, exist_ok=True)

            history_fig = vis.plot_optimization_history(study)
            history_fig.write_image(str(plots_dir / "optimization_history.png"))

            param_importances_fig = vis.plot_param_importances(study)
            param_importances_fig.write_image(str(plots_dir / "param_importances.png"))

            parallel_coord_fig = vis.plot_parallel_coordinate(study)
            parallel_coord_fig.write_image(str(plots_dir / "parallel_coordinate.png"))

            logger.info(f"\nOptuna visualization plots saved to {plots_dir}")
        except Exception as e:
            logger.warning(f"Could not generate Optuna plots: {str(e)}")

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")

if __name__ == "__main__":
    main()
