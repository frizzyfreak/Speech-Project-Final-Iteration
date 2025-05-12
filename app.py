# app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Model
from audiorecorder import audiorecorder
import io

# ================================
# Define the Wav2Vec2 + Classifier model
# ================================

class Wav2VecIntent(nn.Module):
    def __init__(self, num_classes=31, pretrained_model="facebook/wav2vec2-large"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        hidden_size = self.wav2vec.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.layer_norm(hidden_states)
        attn_weights = F.softmax(self.attention(hidden_states), dim=1)
        x = torch.sum(hidden_states * attn_weights, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ================================
# Streamlit App UI
# ================================

st.set_page_config(page_title="Speech Intent Classifier", page_icon="🎙️")
st.title("🎙️ Speech Command Recognition (Microphone Demo)")
st.write("Record your voice and predict the intent of your command.")

# ================================
# Load pre-trained model weights
# ================================

@st.cache_resource
def load_model(checkpoint_path):
    model = Wav2VecIntent()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Specify your model checkpoint file
checkpoint_path = "model.pt"  # replace with your actual filename (model trained weights)
model = load_model(checkpoint_path)

# Dummy label names (replace with your actual intent labels)
labels = [f"intent_{i+1}" for i in range(31)]

# ================================
# Microphone Recording
# ================================

st.subheader("🎤 Record your command")

audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    st.audio(audio.tobytes(), format="audio/wav")

    # Convert to waveform tensor
    audio_bytes = io.BytesIO(audio.tobytes())
    waveform, sample_rate = torchaudio.load(audio_bytes)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    input_values = waveform.squeeze().unsqueeze(0)  # shape: [1, time]

    st.write("✅ Audio recorded! Making prediction...")

    with st.spinner('Predicting...'):
        with torch.no_grad():
            logits = model(input_values)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predicted_label = labels[predicted_class_id]
            confidence = torch.softmax(logits, dim=1)[0, predicted_class_id].item()

    st.success(f"### 🏷️ Predicted Intent: **{predicted_label}**")
    st.progress(confidence)
    st.write(f"**Confidence:** {confidence:.2f}")

