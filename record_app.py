import streamlit as st
import numpy as np
import torch
import torchaudio
import tempfile
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Set page configuration and styling
st.set_page_config(page_title="Speech Intent Recognition", layout="wide")

# Add custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #EEFBEE;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🎙️ Speech Intent Recognition</h1>', unsafe_allow_html=True)

# Define model info and labels
REPO_ID = "avi292423/speech-intent-recognition-project"
MODEL_FILENAME = "wav2vec_best_model.pt"
LABELS = [
    "activate_lamp", "activate_lights", "activate_lights_bedroom", "activate_lights_kitchen",
    "activate_lights_washroom", "activate_music", "bring_juice", "bring_newspaper",
    "bring_shoes", "bring_socks", "change_language_Chinese", "change_language_English",
    "change_language_German", "change_language_Korean", "change_language_none",
    "deactivate_lamp", "deactivate_lights", "deactivate_lights_bedroom", "deactivate_lights_kitchen",
    "deactivate_lights_washroom", "deactivate_music", "decrease_heat", "decrease_heat_bedroom",
    "decrease_heat_kitchen", "decrease_heat_washroom", "decrease_volume", "increase_heat",
    "increase_heat_bedroom", "increase_heat_kitchen", "increase_heat_washroom", "increase_volume"
]

# Import your exact model architecture
class Wav2VecIntent(torch.nn.Module):
    def __init__(self, num_classes=31, pretrained_model="facebook/wav2vec2-large"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        hidden_size = self.wav2vec.config.hidden_size
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.layer_norm(hidden_states)
        x = torch.mean(hidden_states, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

@st.cache_resource
def load_model():
    """Download and load the model"""
    with st.spinner("Downloading model from Hugging Face Hub..."):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            model = Wav2VecIntent(num_classes=len(LABELS), pretrained_model="facebook/wav2vec2-large").to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")
            return model, device, feature_extractor
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None, None, None

# Load the model
model, device, feature_extractor = load_model()

if not model or not feature_extractor:
    st.error("⚠️ Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Function to preprocess and predict
def process_audio(audio_file_path):
    """Process audio and return prediction"""
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path, backend="soundfile")
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, padding=True, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            predicted_class = torch.argmax(outputs, dim=1).item()
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted_class].item() * 100
        return LABELS[predicted_class], confidence
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None

# Audio recording using streamlit-webrtc
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        audio_data = frame.to_ndarray()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            torchaudio.save(temp_audio.name, torch.tensor(audio_data).unsqueeze(0), 16000)
            st.audio(temp_audio.name, format="audio/wav")
            with st.spinner("Processing audio..."):
                intent, confidence = process_audio(temp_audio.name)
            if intent:
                st.markdown(f"""
                <div class="success-box">
                    <h3>🎯 Predicted Intent:</h3>
                    <h2 style="color:#1E88E5">{intent}</h2>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

webrtc_streamer(key="audio", audio_processor_factory=AudioProcessor)