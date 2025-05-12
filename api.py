# api.py (FastAPI backend)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchaudio
from io import BytesIO
from models.model_wav2vec import Wav2VecIntent

app = FastAPI()

# Load model at startup
num_classes = 31
pretrained_model = "facebook/wav2vec2-large"
MODEL_PATH = "checkpoints11/wav2vec/wav2vec_best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

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

def preprocess_audio(audio_waveform, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_waveform = resampler(audio_waveform)
    if audio_waveform.shape[0] > 1:
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    return audio_waveform.to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        audio_waveform, sample_rate = torchaudio.load(BytesIO(content))
        audio_waveform = preprocess_audio(audio_waveform, sample_rate)

        with torch.no_grad():
            output = model(audio_waveform)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = index_to_label.get(predicted_class, "Unknown Class")

        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
