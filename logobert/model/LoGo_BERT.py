# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin


class LoGo_BERT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        embedding_dim: int = 512,
        dropout: float = 0.1,
        pos_weight: float = 10.0,
        use_ln_g1: bool = True,
        score_norm: str = "none",
        hidden_mult: int = 1,
        act: str = "relu",
        score_fn: str = "dot",
        use_maxsim: bool = True,
    ):
        super().__init__()

        self._init_args = dict(
            model_name=model_name,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pos_weight=float(pos_weight),
            use_ln_g1=bool(use_ln_g1),
            score_norm=str(score_norm),
            hidden_mult=int(hidden_mult),
            act=str(act),
            score_fn=str(score_fn),
            use_maxsim=bool(use_maxsim),
        )

        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU() if act.lower() == "silu" else nn.ReLU()

        input_dim = 3 * embedding_dim + 1
        h = embedding_dim * hidden_mult
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, h),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

        self.sbert_weight = nn.Parameter(torch.ones(3 * embedding_dim))
        self.maxsim_weight = nn.Parameter(torch.ones(1))

        self.use_ln_g1 = use_ln_g1
        self.use_maxsim = use_maxsim
        if self.use_ln_g1:
            self.ln_g1 = nn.LayerNorm(3 * embedding_dim)

        self.score_norm = score_norm.lower()
        self.score_fn = score_fn

        
        self.register_buffer("pos_weight_buf", torch.tensor([float(pos_weight)]))

    @property
    def config(self):
        return self._init_args

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embed = self.dropout(self.projection(outputs.last_hidden_state))
        return token_embed

    def mean_pooling(self, embed, mask):
        mask = mask.unsqueeze(-1).float()
        summed = torch.sum(embed * mask, dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counted

    def maxsim_dot(self, emb_a, mask_a, emb_b, mask_b, score_fn="dot"):
        if score_fn == "dot":
            sim_matrix = torch.bmm(emb_a, emb_b.transpose(1, 2))
        elif score_fn == "cosine":
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
        return summed / valid_len

    def forward(self, input_a, input_b, labels=None):
        emb_a = self.encode(input_a["input_ids"], input_a["attention_mask"])
        emb_b = self.encode(input_b["input_ids"], input_b["attention_mask"])

        pooled_a = self.mean_pooling(emb_a, input_a["attention_mask"])
        pooled_b = self.mean_pooling(emb_b, input_b["attention_mask"])
        abs_diff = torch.abs(pooled_a - pooled_b)

        group1 = torch.cat([pooled_a, pooled_b, abs_diff], dim=1)
        if self.use_ln_g1:
            group1 = self.ln_g1(group1)
        weighted_group1 = group1 * self.sbert_weight

        if self.use_maxsim:
            score = self.maxsim_dot(
                emb_a, input_a["attention_mask"],
                emb_b, input_b["attention_mask"],
                score_fn=self.score_fn,
            )
            if self.score_norm == "tanh":
                score = torch.tanh(score)
            weighted_group2 = score * self.maxsim_weight
        else:
            weighted_group2 = pooled_a.new_zeros(pooled_a.size(0), 1)

        concat = torch.cat([weighted_group1, weighted_group2], dim=1)
        logits = self.classifier(concat).squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_buf.to(logits.device))
            loss = loss_fn(logits, labels.float())
            return loss, logits

        return torch.sigmoid(logits)


    @torch.inference_mode()
    def predict_from_embeds(self, emb_a, mask_a, emb_b, mask_b):

        pooled_a = self.mean_pooling(emb_a, mask_a)
        pooled_b = self.mean_pooling(emb_b, mask_b)
        abs_diff = torch.abs(pooled_a - pooled_b)

        group1 = torch.cat([pooled_a, pooled_b, abs_diff], dim=1)
        if self.use_ln_g1:
            group1 = self.ln_g1(group1)
        weighted_group1 = group1 * self.sbert_weight
        
        if self.use_maxsim:
        
            score = self.maxsim_dot(
                emb_a, mask_a, emb_b, mask_b,
                score_fn=self.score_fn
            )
                
            if self.score_norm == "tanh":
                score = torch.tanh(score)
        else:
            score = pooled_a.new_zeros(pooled_a.size(0), 1)


        weighted_group2 = score * self.maxsim_weight

        concat = torch.cat([weighted_group1, weighted_group2], dim=1)  # (B, 3D+1)


        logits = self.classifier(concat).squeeze(-1)

        probs = torch.sigmoid(logits)

        return probs
