import torch
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
from models.model_wav2vec import Wav2VecIntent  # Import your custom model class

# Load the model
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedded label map
label_map = {
    "activate_lamp": 0,
    "activate_lights": 1,
    "activate_lights_bedroom": 2,
    "activate_lights_kitchen": 3,
    "activate_lights_washroom": 4,
    "activate_music": 5,
    "bring_juice": 6,
    "bring_newspaper": 7,
    "bring_shoes": 8,
    "bring_socks": 9,
    "change language_Chinese": 10,
    "change language_English": 11,
    "change language_German": 12,
    "change language_Korean": 13,
    "change language_none": 14,
    "deactivate_lamp": 15,
    "deactivate_lights": 16,
    "deactivate_lights_bedroom": 17,
    "deactivate_lights_kitchen": 18,
    "deactivate_lights_washroom": 19,
    "deactivate_music": 20,
    "decrease_heat": 21,
    "decrease_heat_bedroom": 22,
    "decrease_heat_kitchen": 23,
    "decrease_heat_washroom": 24,
    "decrease_volume": 25,
    "increase_heat": 26,
    "increase_heat_bedroom": 27,
    "increase_heat_kitchen": 28,
    "increase_heat_washroom": 29,
    "increase_volume": 30
}
# Reverse the label map to map indices back to labels
index_to_label = {v: k for k, v in label_map.items()}

print("Loading model...")
# Initialize the custom Wav2VecIntent model
num_classes = 31  # Update this to match the number of classes in your training
pretrained_model = "facebook/wav2vec2-large"  # Use the same pretrained model as in training
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)

# Load the state dictionary into the model
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()
print("Model loaded successfully!")

# Function to record audio from the microphone
def record_audio(duration=5, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return np.squeeze(audio)

# Function to preprocess audio for the model
def preprocess_audio(audio, sample_rate=16000):
    print("Preprocessing audio...")
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return waveform.to(device)

# Function to test the model
def test_model(audio_waveform):
    print("Testing the model...")
    with torch.no_grad():
        output = model(audio_waveform)
    print("Model testing complete.")
    
    # Convert logits to probabilities
    probabilities = F.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    
    # Get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    print("Predicted Class Index:", predicted_class)
    
    # Map class index to class name
    predicted_class_name = index_to_label.get(predicted_class, "Unknown Class")
    print("Predicted Class Name:", predicted_class_name)
    
    return predicted_class_name

# Main function
if __name__ == "__main__":
    # Record audio from the microphone
    duration = 5  # Duration in seconds
    sample_rate = 16000  # Sample rate in Hz
    audio = record_audio(duration=duration, sample_rate=sample_rate)

    # Preprocess the audio
    audio_waveform = preprocess_audio(audio, sample_rate=sample_rate)

    # Test the model
    predicted_label = test_model(audio_waveform)

    # Print the final predicted label
    print("Final Predicted Label:", predicted_label)