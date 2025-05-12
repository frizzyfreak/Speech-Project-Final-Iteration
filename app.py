# app.py

import streamlit as st
import numpy as np
import tempfile
import torch
import torchaudio
import os
import requests
from models.model_wav2vec import Wav2VecIntent  # Your custom model class
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
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

# === Model Configuration ===
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"
ONEDRIVE_URL = "https://1drv.ms/u/c/758381408c57efa8/Efpm5WOByIBEreOl02sgnhcBWf9AMXrryl4a1DudWnSSgQ?e=upGwOb"

# Function to download the model from OneDrive
def download_model_from_onedrive(url, destination):
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    st.info("Downloading model from OneDrive...")
    
    try:
        # Convert OneDrive link to direct download link
        if "1drv.ms" in url:
            response = requests.get(url, allow_redirects=True)
            file_url = response.url.replace("redir", "download")
        else:
            file_url = url
        
        # Download the file
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate the file is a PyTorch model
            try:
                # Try to load the first few bytes to check if it's a valid file
                with open(destination, 'rb') as f:
                    header = f.read(10)
                    if b'PK\x03\x04' not in header:  # PyTorch models are zip files
                        raise ValueError("Downloaded file is not a valid PyTorch model")
                
                st.success(f"Model downloaded successfully to: {destination}")
            except Exception as file_error:
                os.remove(destination)
                raise ValueError(f"Downloaded file is not a valid PyTorch model: {str(file_error)}")
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")
            raise FileNotFoundError(f"HTTP error: {response.status_code}")
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        raise

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    try:
        download_model_from_onedrive(ONEDRIVE_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download the model: {str(e)}")
        st.stop()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 31
pretrained_model = "facebook/wav2vec2-large"
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)

try:
    try:
        # Standard loading
        state_dict = torch.load(MODEL_PATH, map_location=device)
        if isinstance(state_dict, dict):
            # Clean any corrupted keys
            clean_state_dict = {k.strip('\r\n') if isinstance(k, str) else k: v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            st.success("Model loaded successfully!")
        else:
            # If state_dict is not a dictionary, it might be the full model
            model = state_dict
            st.success("Full model loaded successfully!")
    except Exception as e:
        st.warning(f"Standard loading failed: {str(e)}")
        try:
            # Try JIT loading
            st.info("Trying alternative loading method...")
            jit_model = torch.jit.load(MODEL_PATH, map_location=device)
            model = jit_model
            st.success("Model loaded with JIT!")
        except Exception as e2:
            st.warning(f"JIT loading failed: {str(e2)}")
            st.info("Using base pre-trained model")
    
    # Ensure model is in eval mode
    model.eval()
except Exception as e:
    st.error(f"Failed to initialize the model: {str(e)}")
    st.stop()

# === Streamlit App ===
st.title("🎙️ Speech Intent Recognition (Microphone + Upload)")

# === Preprocess audio ===
def preprocess_audio(audio_waveform, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_waveform = resampler(audio_waveform)
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform.to(device)

# === Predict intent ===
def predict_intent(audio_waveform):
    with torch.no_grad():
        try:
            output = model(audio_waveform)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = index_to_label.get(predicted_class, "Unknown Class")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            predicted_label = "Error in prediction"
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
            audio_waveform, sample_rate = torchaudio.load(temp_audio_file_path, backend="default")
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
            audio_waveform, sample_rate = torchaudio.load(temp_audio_file_path, backend="default")
            audio_waveform = preprocess_audio(audio_waveform, sample_rate)
            prediction = predict_intent(audio_waveform)
            st.success(f"Predicted Command: **{prediction}**")
        except Exception as e:
            st.error(f"Error processing recording: {str(e)}")
