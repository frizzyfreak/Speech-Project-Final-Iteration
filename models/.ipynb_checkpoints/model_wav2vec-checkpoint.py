import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

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