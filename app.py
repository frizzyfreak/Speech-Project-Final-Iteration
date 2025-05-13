import streamlit as st
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile

st.title("Speech Intent Recognition")
st.write("Speak into your microphone to predict the command.")

# Function to record audio
def record_audio(duration=10, sample_rate=16000):
    st.info(f"Listening for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    st.success("Recording complete!")
    return np.squeeze(audio), sample_rate

# Record button
if st.button("Listen"):
    audio, sample_rate = record_audio()

    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, sample_rate)
        st.audio(temp_audio_file.name, format="audio/wav")

        # Send the audio file to the backend API
        with open(temp_audio_file.name, "rb") as f:
            files = {"file": f}
            response = requests.post("http://127.0.0.1:5000/predict", files=files)

        # Display the prediction result
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Unknown")
            st.success(f"Predicted Command: {prediction}")