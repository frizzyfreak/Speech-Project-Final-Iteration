import streamlit as st
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import torch
import torch.nn.functional as F
import os
import gdown
from models.model_wav2vec import Wav2VecIntent  # Import your custom model class

# Google Drive file ID for the model
FILE_ID = "1vBjvOY9Ko1aJiWxjj8fCHwJAh2X5gs5n"
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"

# Function to download the model from Google Drive
def download_model_from_drive(file_id, destination):
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info("Downloading model from Google Drive...")
        gdown.download(url, destination, quiet=False)
        st.success("Model downloaded successfully!")

# Download the model if it doesn't exist
download_model_from_drive(FILE_ID, MODEL_PATH)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 31
pretrained_model = "facebook/wav2vec2-large"
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

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

st.title("Speech Intent Recognition")
st.write("Upload an audio file to predict the command.")

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

# Upload button
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(uploaded_file.read())
        temp_audio_file_path = temp_audio_file.name

    # Load the audio file
    audio = AudioSegment.from_file(temp_audio_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz mono audio
    audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Preprocess the audio and predict intent
    try:
        audio_waveform = preprocess_audio(audio_samples, sample_rate=16000)
        prediction = predict_intent(audio_waveform)
        st.success(f"Predicted Command: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")