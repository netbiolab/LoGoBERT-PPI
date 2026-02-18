import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import os



class ESM2_MLP(nn.Module):
    def __init__(
        self,
        model_name: str = 'facebook/esm2_t33_650M_UR50D',
        embedding_dim: int = 512,
        dropout: float = 0.1,
        pos_weight: float = 10.0
    ):
        super().__init__() 
        self.encoder = AutoModel.from_pretrained(model_name) # esm2
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim) 

        self.classifier = nn.Sequential( 
            nn.Linear(2 * embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        self.pos_weight = pos_weight
        
        
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  
        token_emb = outputs.last_hidden_state 
        token_emb = self.dropout(self.projection(token_emb))  
        return token_emb
    def mean_pooling(self, emb, mask): 
            mask = mask.unsqueeze(-1).float() 
            summed = torch.sum(emb * mask, dim=1)
            counted = mask.sum(dim=1).clamp(min=1e-9) 
            return summed / counted

    def forward(self, input_a, input_b, labels=None):
        emb_a = self.encode(input_a['input_ids'], input_a['attention_mask'])
        emb_b = self.encode(input_b['input_ids'], input_b['attention_mask'])

        pooled_a = self.mean_pooling(emb_a, input_a['attention_mask'])
        pooled_b = self.mean_pooling(emb_b, input_b['attention_mask'])

        concat = torch.cat([pooled_a, pooled_b], dim = 1)

        logits = self.classifier(concat).squeeze(-1)


        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=logits.device))
            loss = loss_fn(logits, labels.float())
            return loss, logits
        else:
            return torch.sigmoid(logits)
