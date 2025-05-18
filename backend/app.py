from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from models.model_wav2vec import Wav2VecIntent
from huggingface_hub import hf_hub_download
import torch
import soundfile as sf

app = FastAPI()

# Download model from Hugging Face
MODEL_PATH = hf_hub_download(repo_id="avi292423/speech-intent-recognition-project", filename="wav2vec_best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

num_classes = 31
pretrained_model = "facebook/wav2vec2-base"  # Use base for less RAM, or keep large if needed
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    audio, sample_rate = sf.read("temp.wav")
    if sample_rate != 16000:
        return JSONResponse({"error": "Audio must have a sample rate of 16kHz."}, status_code=400)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(waveform)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = index_to_label.get(predicted_class, "Unknown Class")
    return {"prediction": predicted_label}