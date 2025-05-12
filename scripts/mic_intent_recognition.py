import torch
import torchaudio
from transformers import Wav2Vec2Processor
from models.model_wav2vec import Wav2VecIntent
import json

def save_dummy_audio(filename='recorded.wav', sample_rate=16000, duration=5):
    """
    Simulate a recorded audio by asking user to provide a .wav file
    OR record audio externally and save it as 'recorded.wav'.
    (No live recording since sounddevice is unavailable)

    Args:
        filename (str): File name to save the audio.
        sample_rate (int): Target sample rate.
        duration (int): Target duration in seconds (optional).
    """
    print(f"⚠️ Live recording disabled (no PortAudio).")
    print(f"👉 Please manually record a {duration}-second .wav file (16kHz, mono)")
    print(f"and save it as: {filename}")
    input("Press Enter once your 'recorded.wav' file is ready...")

def load_audio(file_path, sample_rate=16000):
    """
    Load a .wav audio file as waveform.

    Args:
        file_path (str): Path to .wav file.
        sample_rate (int): Target sample rate.

    Returns:
        torch.Tensor: Audio waveform (1D tensor)
    """
    waveform, orig_sample_rate = torchaudio.load(file_path)

    # Resample if needed
    if orig_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform.squeeze()  # [T]

def predict_intent(model, processor, waveform, label_map):
    """
    Predict the intent from the audio waveform.
    """
    # Preprocess the waveform
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Predict
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
        predicted_idx = torch.argmax(logits, dim=1).item()

    # Map the predicted index to the intent
    intent = label_map.get(predicted_idx, "Unknown")
    return intent

if __name__ == "__main__":
    # Load the trained model
    model_path = "checkpoints7/wav2vec/wav2vec_best_model.pt"
    model = Wav2VecIntent(num_classes=31, pretrained_model="facebook/wav2vec2-large")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    model.to("cpu")  # Use CPU for inference

    # Load the processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large")

    # Load the label map
    label_map_path = "data/processed2/label_map.json"
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}  # Reverse the label map

    # Save dummy / ask user to prepare audio
    save_dummy_audio(filename='recorded.wav', duration=5)

    # Load the recorded audio
    waveform = load_audio(file_path='recorded.wav', sample_rate=16000)

    # Predict the intent
    intent = predict_intent(model, processor, waveform, inv_label_map)
    print(f"\n🎯 Predicted Intent: {intent}")
