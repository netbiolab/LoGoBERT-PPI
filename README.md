# LoGoBERT-PPI

LoGoBERT-PPI is a protein–protein interaction (PPI) prediction framework based on protein language model embeddings and late-interaction scoring.  
The model enables scalable all-by-all inference across large proteomes by decoupling protein embedding from interaction scoring.

This repository contains the training and inference code used in the LoGoBERT-PPI study.

---

## Overview

LoGoBERT-PPI follows a two-stage pipeline designed for scalable interaction inference:

1. **Protein embedding**

   Protein sequences are encoded using a pretrained protein language model (ESM2).  
   Embeddings are computed once and cached, allowing reuse across multiple inference tasks.

2. **Pairwise interaction inference**

   Interaction scores are computed using late interaction (MaxSim-style) scoring between token-level embeddings.  
   Since embeddings are precomputed, interaction scoring can be performed efficiently in large batches.

By separating embedding and interaction scoring, LoGoBERT-PPI enables scalable inference compared to cross-encoder architectures that jointly encode protein pairs.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/LoGoBERT-PPI.git
cd LoGoBERT-PPI
```
Conda Environment (recommended)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate logobert
```

Insatll this repository as an editable package:
```bash
pip install -e .
```
---
## Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- transformers
- huggingface_hub
- numpy
- pandas
- tqdm
- biopython
- scikit-learn
- wandb (optional, for training)
Exact versions are pinned in environment.yml for reproducibility.

---

## Model Weights
Pretrained models are available on Hugging Face:

👉 https://huggingface.co/hbeen/LoGoBERT-PPI-Eukaryote

Models can be loaded directly using:

```python
from logobert.model.LoGo_BERT import LoGo_BERT

model = LoGo_BERT.from_pretrained(
    "hbeen/LoGoBERT-PPI-Eukaryote"
)
```
The tokenizer is loaded from the base protein language model (ESM2).
---
## Training
Example multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \
scripts/train_ddp.py \
  --train_path data/train.tsv \
  --val_path data/val.tsv \
  --model_name facebook/esm2_t33_650M_UR50D \
  --embedding_dim 512 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 20 \
  --save_path checkpoints/run1
```
---

## Input Format
**Pair file (CSV)**
A CSV file containing protein identifier pairs:
```text
query,text
P12345,Q99999
P12345,Q88888
```
**FASTA file**
Protein identifiers must match IDs used in the pair file.
---
## Inference
**Compute embeddings and infer interactions:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
scripts/infer_pairs_ddp.py \
  --pair_csv pairs.csv \
  --fasta_path proteins.fasta \
  --hf_repo hbeen/LoGoBERT-PPI-Eukaryote \
  --model_name facebook/esm2_t33_650M_UR50D \
  --embedding_save_path embeddings.pt \
  --output_path pair_scores.csv
```
**Use cached embeddings**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
scripts/infer_pairs_ddp.py \
  --pair_csv pairs.csv \
  --embeddings_path embeddings.pt \
  --hf_repo hbeen/LoGoBERT-PPI-Eukaryote \
  --output_path pair_scores.csv
```
Embeddings are computed once and reused during interaction scoring to enable efficient large-scale inference.
---
## Acknowledgements
Parts of the batching utilities were adapted from PLM-interact (MIT License, Dan Liu, 2024).
See the NOTICE file for details.
---
## License

This project is released under the Apache License 2.0.

Some components were adapted from PLM-interact (MIT License, Dan Liu, 2024).
See the NOTICE file for details.
