import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import os



class LoGo_BERT(nn.Module):
    def __init__(
        self,
        model_name: str = 'facebook/esm2_t33_650M_UR50D', 
        embedding_dim: int = 512, 
        dropout: float = 0.1, 
        pos_weight: float = 10.0, 
        use_ln_g1: bool = True,  
        score_norm: str = 'none', 
        hidden_mult: int = 1, 
        act: str = 'relu', 
        score_fn: str = 'dot', 
        use_maxsim: bool = True,
    ):
        super().__init__() 
        
        self.encoder = AutoModel.from_pretrained(model_name) 
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU() if act.lower() == 'silu' else nn.ReLU()
        input_dim = 3 * embedding_dim + 1 
        h = embedding_dim * hidden_mult
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, h),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(h,1)
        )
        
        self.sbert_weight = nn.Parameter(torch.ones(3*embedding_dim)) 
        self.maxsim_weight = nn.Parameter(torch.ones(1))
        
        self.use_ln_g1 = use_ln_g1
        self.use_maxsim = use_maxsim
        if self.use_ln_g1:
            self.ln_g1 = nn.LayerNorm(3 * embedding_dim)
        self.score_norm = score_norm.lower()
        self.pos_weight = pos_weight
        self.score_fn = score_fn
        
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask = attention_mask)
        last_hidden_state = outputs.last_hidden_state
        token_embed = self.dropout(self.projection(last_hidden_state))
        return token_embed
    
    def mean_pooling(self, embed, mask):
        mask = mask.unsqueeze(-1).float()
        summed = torch.sum(embed*mask, dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        result = summed/counted
        return result
    
    
    def maxsim_dot(self, emb_a, mask_a, emb_b, mask_b, score_fn='dot', return_matrix: bool=False):
        B, L1, D = emb_a.size() 
        L2 = emb_b.size(1) 
        if score_fn == 'dot':
            sim_matrix = torch.bmm(emb_a, emb_b.transpose(1, 2))  
        elif score_fn == 'cosine':
            a = F.normalize(emb_a, dim=-1)
            b = F.normalize(emb_b, dim=-1)
            sim_matrix = torch.bmm(a, b.transpose(1, 2))
        else:
            raise ValueError(f"Invalid mode: {score_fn}. Choose 'dot' or 'cosine'.")
        neg_inf = torch.finfo(sim_matrix.dtype).min 
        
        sim_matrix = sim_matrix.masked_fill(~mask_a[:, :, None].bool(), neg_inf)
        sim_matrix = sim_matrix.masked_fill(~mask_b[:, None, :].bool(), neg_inf)
        max_per_query = sim_matrix.max(dim=2).values  
        has_valid_key = mask_b.any(dim=1, keepdim=True)      
        max_per_query = torch.where(has_valid_key, max_per_query, torch.zeros_like(max_per_query)) 
        
        mask_a_float = mask_a.float()
        max_per_query = max_per_query * mask_a_float
        
        summed = torch.sum(max_per_query, dim=1, keepdim=True) 
        valid_len = mask_a_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        len_normalized = summed / valid_len
        if return_matrix:
           
            return len_normalized, sim_matrix

        return len_normalized
    
    
    def forward(self, input_a, input_b, labels=None, return_extras: bool = False):
        emb_a = self.encode(input_a['input_ids'], input_a['attention_mask'])
        emb_b = self.encode(input_b['input_ids'], input_b['attention_mask'])

        pooled_a = self.mean_pooling(emb_a, input_a['attention_mask'])
        pooled_b = self.mean_pooling(emb_b, input_b['attention_mask'])
        abs_diff = torch.abs(pooled_a - pooled_b)

        group1 = torch.cat([pooled_a, pooled_b, abs_diff], dim=1)
        if self.use_ln_g1:
            group1 = self.ln_g1(group1)
        weighted_group1 = group1 * self.sbert_weight  

        if self.use_maxsim:
            score = self.maxsim_dot(
                emb_a, input_a['attention_mask'],
                emb_b, input_b['attention_mask'],
                score_fn=self.score_fn
            )  
            if self.score_norm == "tanh":
                score = torch.tanh(score)
            weighted_group2 = score * self.maxsim_weight  
        else:
            weighted_group2 = pooled_a.new_zeros(pooled_a.size(0), 1)

        concat = torch.cat([weighted_group1, weighted_group2], dim=1)  
        logits = self.classifier(concat).squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=logits.device))
            loss = loss_fn(logits, labels.float())
            if return_extras:
                D = pooled_a.size(1)  
                return loss, logits, {"concat": concat, "D": D}
            return loss, logits
        else:
            probs = torch.sigmoid(logits)
            if return_extras:
                D = pooled_a.size(1)  
                return probs, {"concat": concat, "D": D}
            return probs
    
    def extract_features(self, input_a, input_b):
        emb_a = self.encode(input_a['input_ids'], input_a['attention_mask'])
        emb_b = self.encode(input_b['input_ids'], input_b['attention_mask'])
        pooled_a = self.mean_pooling(emb_a, input_a['attention_mask'])
        pooled_b = self.mean_pooling(emb_b, input_b['attention_mask'])
        if self.use_maxsim:
            
            maxsim_score = self.maxsim_dot(emb_a, input_a['attention_mask'], emb_b, input_b['attention_mask'], score_fn=self.score_fn)
        else:
            maxsim_score = pooled_a.new_zeros(pooled_a.size(0), 1)
        return pooled_a, pooled_b, maxsim_score
    

    @torch.no_grad()
    def predict_from_embeds(self, emb_a, mask_a, emb_b, mask_b, return_logits=False,
                        return_matrix: bool=False):
 
        pooled_a = self.mean_pooling(emb_a, mask_a)
        pooled_b = self.mean_pooling(emb_b, mask_b)
        abs_diff = torch.abs(pooled_a - pooled_b)

        group1 = torch.cat([pooled_a, pooled_b, abs_diff], dim=1)
        if self.use_ln_g1:
            group1 = self.ln_g1(group1)
        weighted_group1 = group1 * self.sbert_weight
        
        if self.use_maxsim:
            

            if return_matrix:
                score, sim_matrix = self.maxsim_dot(
                    emb_a, mask_a, emb_b, mask_b,
                    score_fn=self.score_fn,
                    return_matrix=True
                )
            else:
                score = self.maxsim_dot(
                    emb_a, mask_a, emb_b, mask_b,
                    score_fn=self.score_fn
                )
                sim_matrix = None
            
                
            if self.score_norm == "tanh":
                score = torch.tanh(score)
        else:
            score = pooled_a.new_zeros(pooled_a.size(0), 1)
            sim_matrix = None
  
        weighted_group2 = score * self.maxsim_weight

        concat = torch.cat([weighted_group1, weighted_group2], dim=1)  # (B, 3D+1)


        logits = self.classifier(concat).squeeze(-1)
        if return_logits and return_matrix:
            return logits, sim_matrix
        if return_logits:
            return logits
        probs = torch.sigmoid(logits)
        if return_matrix:
            return probs, sim_matrix
        return probs