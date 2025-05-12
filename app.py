# app.py

import streamlit as st
import numpy as np
import tempfile
import torch
import torchaudio
import os
import requests
from models.model_wav2vec import Wav2VecIntent  # Your custom model class

# Resolve torchaudio backend warning by explicitly setting
torchaudio.set_audio_backend("sox_io")

# === Label Map ===
label_map = {
    "activate_lamp": 0, "activate_lights": 1, "activate_lights_bedroom": 2, "activate_lights_kitchen": 3,
    "activate_lights_washroom": 4, "activate_music": 5, "bring_juice": 6, "bring_newspaper": 7,
    "bring_shoes": 8, "bring_socks": 9, "change_language_Chinese": 10, "change_language_English": 11,
    "change_language_German": 12, "change_language_Korean": 13, "change_language_none": 14,
    "deactivate_lamp": 15, "deactivate_lights": 16, "deactivate_lights_bedroom": 17, "deactivate_lights_kitchen": 18,
    "deactivate_lights_washroom": 19, "deactivate_music": 20, "decrease_heat": 21, "decrease_heat_bedroom": 22,
    "decrease_heat_kitchen": 23, "decrease_heat_washroom": 24, "decrease_volume": 25, "increase_heat": 26,
    "increase_heat_bedroom": 27, "increase_heat_kitchen": 28, "increase_heat_washroom": 29, "increase_volume": 30
}
index_to_label = {v: k for k, v in label_map.items()}

# === OneDrive Model Download ===
ONEDRIVE_DIRECT_LINK = "https://onedrive.live.com/download?resid=758381408C57EFA8%211507&authkey=!ADdJbNHuCkH-JMo"
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"

def download_model_from_onedrive(url, destination):
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        st.info("Downloading model from OneDrive...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download model. Please check the link or permissions.")
            st.stop()

download_model_from_onedrive(ONEDRIVE_DIRECT_LINK, MODEL_PATH)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 31
pretrained_model = "facebook/wav2vec2-large"
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    st.error(f"Failed to load the model: {str(e)}")
    st.stop()

# === Streamlit App ===
st.title("🎙️ Speech Intent Recognition (Microphone + Upload)")

# === Preprocess audio ===
def preprocess_audio(audio_waveform, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_waveform = resampler(audio_waveform)
    # Mono
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform.to(device)

# === Predict intent ===
def predict_intent(audio_waveform):
    with torch.no_grad():
        output = model(audio_waveform)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = index_to_label.get(predicted_class, "Unknown Class")
    return predicted_label

# === Upload or Record ===
option = st.radio("Choose Input Method:", ["Upload audio file", "Record from microphone"])

if option == "Upload audio file":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_file_path = temp_audio_file.name

        try:
            audio_waveform, sample_rate = torchaudio.load(temp_audio_file_path)
            audio_waveform = preprocess_audio(audio_waveform, sample_rate)

            prediction = predict_intent(audio_waveform)
            st.success(f"Predicted Command: **{prediction}**")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:
    st.info("🎙️ Click the button to record audio (5 seconds).")
    if st.button("Record Audio"):
        import sounddevice as sd
        import scipy.io.wavfile

        fs = 16000  # 16kHz
        duration = 5  # seconds

        st.info("Recording...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        st.success("Recording complete!")

        temp_audio_file_path = "temp_recorded.wav"
        scipy.io.wavfile.write(temp_audio_file_path, fs, audio)

        try:
            audio_waveform, sample_rate = torchaudio.load(temp_audio_file_path)
            audio_waveform = preprocess_audio(audio_waveform, sample_rate)

            prediction = predict_intent(audio_waveform)
            st.success(f"Predicted Command: **{prediction}**")
        except Exception as e:
            st.error(f"Error processing recording: {str(e)}")
