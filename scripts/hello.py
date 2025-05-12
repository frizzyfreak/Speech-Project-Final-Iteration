# quantize_model.py
import torch
from models.model_wav2vec import Wav2VecIntent

num_classes = 31
pretrained_model = "facebook/wav2vec2-large"

# Load your model
model = Wav2VecIntent(num_classes=num_classes, pretrained_model=pretrained_model)
state_dict = torch.load("checkpoints11/wav2vec/wav2vec_best_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model (~50% smaller)
torch.save(quantized_model.state_dict(), "wav2vec_best_model_quantized.pt")
