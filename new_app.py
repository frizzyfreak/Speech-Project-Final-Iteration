import streamlit as st
import numpy as np
import torch
import torchaudio
import tempfile
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

# Set page configuration and styling
st.set_page_config(page_title="Speech Intent Recognition", layout="wide")

# Add some custom CSS for better appearance
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
class Wav2VecIntent(nn.Module):
    def __init__(self, num_classes=31, pretrained_model="facebook/wav2vec2-large"):
        super().__init__()
        # Load pretrained wav2vec model
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        # Get hidden size from model config
        hidden_size = self.wav2vec.config.hidden_size
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Add attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Add dropout for regularization   
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_values, attention_mask=None):
        # Get wav2vec features
        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state  # [batch, sequence, hidden]
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(hidden_states), dim=1)
        x = torch.sum(hidden_states * attn_weights, dim=1)  # Weighted sum
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification
        x = self.fc(x)
        return x

@st.cache_resource
def load_model():
    """Download and load the model"""
    with st.spinner("Downloading model from Hugging Face Hub..."):
        try:
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"Using device: {device}")
            
            # Download model file
            try:
                model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
                st.success(f"Downloaded model file from {REPO_ID}")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.warning("Using base model without custom weights")
                # Initialize the model with the base model
                model = Wav2VecIntent(num_classes=len(LABELS), pretrained_model="facebook/wav2vec2-large").to(device)
                model.eval()
                return model, device, None
            
            # Initialize the model with wav2vec2-base (we'll replace with the actual weights)
            model = Wav2VecIntent(num_classes=len(LABELS), pretrained_model="facebook/wav2vec2-large").to(device)
            
            # Load the state dict
            try:
                state_dict = torch.load(model_path, map_location=device)
                
                # Try loading with flexible matching
                try:
                    model.load_state_dict(state_dict, strict=False)
                    st.success("Model weights loaded successfully!")
                except Exception as load_error:
                    st.warning(f"Flexible loading with some missing keys: {str(load_error)}")
            
                model.eval()
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")
                return model, device, feature_extractor
                
            except Exception as e:
                st.warning(f"Failed to load state dict: {str(e)}")
                st.info("Using base model without custom weights")
                model.eval()
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")
                return model, device, feature_extractor
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None, None, None

# Load our model
model, device, feature_extractor = load_model()

if model and feature_extractor:
    st.markdown('<p class="info-box">✅ Model and feature extractor loaded successfully!</p>', 
                unsafe_allow_html=True)
else:
    st.error("⚠️ Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Function to preprocess and predict
def process_audio(audio_file_path):
    """Process audio and return prediction"""
    try:
        # Load and resample audio if needed
        waveform, sample_rate = torchaudio.load(audio_file_path, backend="soundfile")
        
        # Resample if the audio is not 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000
            
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process the audio to match the model's input format
        inputs = feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=sample_rate, 
            padding=True, 
            return_tensors="pt"
        ).input_values.to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(inputs)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
        # Get prediction confidence
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted_class].item() * 100
            
        return LABELS[predicted_class], confidence
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h2 class="sub-header">Option 1: Upload Audio</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(uploaded_file.read())
            
            # Display the audio player
            st.audio(temp_audio.name, format="audio/wav")
            
            # Process the audio
            with st.spinner("Processing audio..."):
                intent, confidence = process_audio(temp_audio.name)
            
            if intent:
                # Display result with nice formatting
                st.markdown(f"""
                <div class="success-box">
                    <h3>🎯 Predicted Intent:</h3>
                    <h2 style="color:#1E88E5">{intent}</h2>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="sub-header">Option 2: Record Audio (Coming Soon)</h2>', unsafe_allow_html=True)
    st.info("Audio recording directly in the browser is not yet supported. Please use the upload option.")

# Display information about the model and available commands
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app recognizes speech commands using a Wav2Vec2 model fine-tuned for speech intent recognition.
    
    ### Available Commands:
    
    The model can recognize these 31 different commands:
    
    | Home Automation | Object Fetching | Language | Media Control | Temperature |
    | --- | --- | --- | --- | --- |
    | activate_lamp | bring_juice | change_language_Chinese | activate_music | increase_heat |
    | activate_lights | bring_newspaper | change_language_English | deactivate_music | increase_heat_bedroom |
    | activate_lights_bedroom | bring_shoes | change_language_German | decrease_volume | increase_heat_kitchen |
    | activate_lights_kitchen | bring_socks | change_language_Korean | increase_volume | increase_heat_washroom |
    | activate_lights_washroom | | change_language_none | | decrease_heat |
    | deactivate_lamp | | | | decrease_heat_bedroom |
    | deactivate_lights | | | | decrease_heat_kitchen |
    | deactivate_lights_bedroom | | | | decrease_heat_washroom |
    | deactivate_lights_kitchen | | | | |
    | deactivate_lights_washroom | | | | |
    
    ### Technical Details:
    
    This model is based on the Wav2Vec2 architecture fine-tuned on a custom dataset of speech commands.
    The model file (`wav2vec_best_model.pt`) is loaded directly from the Hugging Face Hub.
    """)

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Hugging Face")