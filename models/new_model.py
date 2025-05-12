import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class Wav2VecIntent(nn.Module):
    def __init__(self, 
                 num_classes=31, 
                 pretrained_model="facebook/wav2vec2-large",
                 use_attention=True, 
                 freeze_wav2vec=True, 
                 classifier_depth=2):
        super().__init__()
        
        # Load pretrained Wav2Vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        # Optionally freeze Wav2Vec2 parameters
        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        # Hidden size from model config
        hidden_size = self.wav2vec.config.hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Attention mechanism
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Linear(hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head — deeper version (optional)
        if classifier_depth == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        elif classifier_depth == 1:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError("classifier_depth must be 1 or 2")

    def forward(self, input_values, attention_mask=None):
        # Extract features from Wav2Vec2
        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Sequence pooling
        if self.use_attention:
            attn_logits = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
            
            # We remove manual masking here (let Wav2Vec2 handle padding internally)
            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
            x = torch.sum(hidden_states * attn_weights, dim=1)  # [batch, hidden]
        else:
            # Mean pooling (ignore padding if mask provided)
            if attention_mask is not None:
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                lengths = attention_mask.sum(dim=1, keepdim=True)
                x = masked_hidden.sum(dim=1) / lengths.clamp(min=1e-9)
            else:
                x = hidden_states.mean(dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification head
        if hasattr(self, 'fc1'):  # deeper classifier
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        else:  # shallow classifier
            x = self.fc(x)
        
        return x
