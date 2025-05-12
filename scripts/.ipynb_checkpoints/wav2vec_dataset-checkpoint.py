import torch
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

class Wav2VecIntentDataset(Dataset):
    """
    Dataset for Wav2Vec intent recognition
    Loads audio files as raw waveforms (not mel spectrograms)
    
    Adapts to the FSC dataset format:
    Index,path,speaker,transcription,action,object,location,intent
    """
    
    def __init__(self, csv_path, label_map=None, is_training=True, sample_rate=16000):
        self.is_training = is_training
        self.sample_rate = sample_rate
        
        # Load CSV file
        self.df = pd.read_csv(csv_path)
        
        # Check if we need to map column names
        if 'intent' not in self.df.columns and 'path' in self.df.columns:
            logger.info("Using FSC dataset format with 'intent' in the last column")
            
            # Extract audio paths and labels
            self.samples = []
            for _, row in self.df.iterrows():
                audio_path = row['path']
                
                # In your FSC dataset, intent is the last column
                intent_label = row.iloc[-1]  # Get the last column value
                
                # Map label to index if label_map provided
                if label_map is not None:
                    if intent_label in label_map:
                        label_idx = label_map[intent_label]
                    else:
                        logger.warning(f"Label '{intent_label}' not found in label map")
                        continue
                else:
                    label_idx = intent_label
                    
                self.samples.append((audio_path, label_idx))
        else:
            # Standard format with explicit 'intent' column
            logger.error("CSV file must have a 'path' column and an intent column")
            raise ValueError("Invalid CSV format")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            # Load audio with torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Remove channel dimension
            waveform = waveform.squeeze(0)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Audio augmentation for training
            if self.is_training:
                # Add small amount of noise
                noise = torch.randn_like(waveform) * 0.001
                waveform = waveform + noise
                
                # Random gain adjustment
                gain = 0.8 + 0.4 * torch.rand(1)
                waveform = waveform * gain
            
            # Return waveform and label
            return waveform, label
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            # Return empty tensor in case of error
            return torch.zeros(16000), label