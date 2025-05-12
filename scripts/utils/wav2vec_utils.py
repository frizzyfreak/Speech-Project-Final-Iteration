import torch
import torchaudio
import os
import logging

logger = logging.getLogger(__name__)

def load_audio_for_wav2vec(audio_path):
    """
    Load and preprocess audio file for wav2vec model
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        waveform: Processed audio as tensor
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return None
            
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed (wav2vec expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Remove channel dimension
        waveform = waveform.squeeze(0)
        
        return waveform
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

def process_batch_for_wav2vec(batch):
    """
    Process a batch of audio for wav2vec
    Pads sequences to same length and creates attention mask
    
    Args:
        batch: List of waveform tensors
        
    Returns:
        padded_batch: Padded batch tensor
        attention_mask: Attention mask tensor
    """
    # Get lengths of each item
    lengths = [x.size(0) for x in batch]
    max_length = max(lengths)
    
    # Create padded batch and attention mask
    padded_batch = torch.zeros(len(batch), max_length)
    attention_mask = torch.zeros(len(batch), max_length)
    
    # Fill in the tensors
    for i, (waveform, length) in enumerate(zip(batch, lengths)):
        padded_batch[i, :length] = waveform
        attention_mask[i, :length] = 1.0
        
    return padded_batch, attention_mask