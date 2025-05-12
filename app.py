import streamlit as st
import torch
import torchaudio
import os
import requests
from models.model_wav2vec import Wav2VecIntent  # Your custom model

# ----- CONFIG -----
ONEDRIVE_LINK = "https://1drv.ms/u/c/758381408c57efa8/Efpm5WOByIBEreOl02sgnhcBWf9AMXrryl4a1DudWnSSgQ?e=c7JuQy"
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"

# Embedded label map
label_map = {
    "activate_lamp": 0, "activate_lights": 1, "activate_lights_bedroom": 2, "activate_lights_kitchen": 3,
    "activate_lights_washroom": 4, "activate_music": 5, "bring_juice": 6, "bring_newspaper": 7,
    "bring_shoes": 8, "bring_socks": 9, "change language_Chinese": 10, "change language_English": 11,
    "change language_German": 12, "change language_Korean": 13, "change language_none": 14, "deactivate_lamp": 15,
    "deactivate_lights": 16, "deactivate_lights_bedroom": 17, "deactivate_lights_kitchen": 18,
    "deactivate_lights_washroom": 19, "deactivate_music": 20, "decrease_heat": 21, "decrease_heat_bedroom": 22,
    "decrease_heat_kitchen": 23, "decrease_heat_washroom": 24, "decrease_volume": 25, "increase_heat": 26,
    "increase_heat_bedroom": 27, "increase_heat_kitchen": 28, "increase_heat_washroom": 29, "increase_volume": 30
}
index_to_label = {v: k for k, v in label_map.items()}


# ----- FUNCTIONS -----
def download_model_from_onedrive(url, destination):
    """Download model from OneDrive shared link"""
    if not os.path.exists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        st.info("Downloading model from OneDrive... (may take a few minutes)")
        try:
            response = requests.get(url, allow_redirects=True, stream=True)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download the model: {str(e)}")
            raise FileNotFoundError("Model file could not be downloaded. Please check the OneDrive link.")


def preprocess_audio(audio_waveform, sample_rate, device):
    """Resample and mono-channel the audio"""
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_waveform = resampler(audio_waveform)
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform.to(device)


def predict_intent(audio_waveform, model, index_to_label):
    """Run model inference"""
    with torch.no_grad():
        output = model(audio_waveform)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = index_to_label.get(predicted_class, "Unknown Class")
    return predicted_label


# ----- STREAMLIT APP -----
st.set_page_config(page_title="Speech Intent Recognition", page_icon="🎙️")
st.title("🎙️ Speech Intent Recognition from Microphone")

# Step 1: Download model (if needed)
try:
    download_model_from_onedrive(ONEDRIVE_LINK, MODEL_PATH)
except FileNotFoundError:
    st.stop()

# Step 2: Load model
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

# Step 3: Microphone recorder
st.info("Record your command (5-10 sec recommended). Click 'Start Recording' and 'Stop'.")

audio_bytes = st.audio_recorder("🎙️ Record Audio", format="audio/wav")

if audio_bytes is not None:
    st.success("Audio recorded! Processing...")

    try:
        # Save to temp file and load
        with open("temp_recording.wav", "wb") as f:
            f.write(audio_bytes)

        audio_waveform, sample_rate = torchaudio.load("temp_recording.wav")
        audio_waveform = preprocess_audio(audio_waveform, sample_rate, device)

        prediction = predict_intent(audio_waveform, model, index_to_label)
        st.success(f"✅ Predicted Command: **{prediction}**")

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")