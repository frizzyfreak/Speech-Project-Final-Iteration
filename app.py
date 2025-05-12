import streamlit as st
import requests

st.title("Audio Command Prediction")
st.write("Upload an audio file to predict the command.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display the audio file
    st.audio(uploaded_file, format="audio/wav")

    # Send the file to the backend API
    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)

        if response.status_code == 200:
            prediction = response.json().get("prediction", "Unknown")
            st.success(f"Predicted Command: {prediction}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")