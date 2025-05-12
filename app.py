import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import torch
import torch.nn.functional as F
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
index_to_label = {v: k for k, v in label_map.items()}

# Initialize the model
num_classes = 31
pretrained_model = "facebook/wav2vec2-large"
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

st.title("Speech Intent Recognition")
st.write("Speak into your microphone to predict the command.")

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    st.info(f"Listening for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    st.success("Recording complete!")
    return np.squeeze(audio), sample_rate

# Function to preprocess audio
def preprocess_audio(audio, sample_rate=16000):
    if sample_rate != 16000:
        raise ValueError("Audio must have a sample rate of 16kHz.")
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return waveform.to(device)

# Function to predict intent
def predict_intent(audio_waveform):
    with torch.no_grad():
        output = model(audio_waveform)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = index_to_label.get(predicted_class, "Unknown Class")
    return predicted_label

# Record button
if st.button("Listen"):
    audio, sample_rate = record_audio()

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, sample_rate)
        st.audio(temp_audio_file.name, format="audio/wav")

        # Preprocess the audio and predict intent
        try:
            audio_waveform = preprocess_audio(audio, sample_rate)
            prediction = predict_intent(audio_waveform)
            st.success(f"Predicted Command: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")