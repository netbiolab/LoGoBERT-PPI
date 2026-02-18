# utils/collate.py

import torch

def smart_batching_collate(batch, tokenizer, max_length, device):
    """
    Collate function for protein-pair inputs.

    Note:
        This function was adapted from the batching/collate logic used in PLM-interact
        (e.g., the training script commonly referred to as `train_mlm.py`), then simplified
        for LoGoBERT-PPI's pairwise classification setting.

        Please see the repository's NOTICE/README for attribution details.
    """
    input_a_texts, input_b_texts = [], []
    has_labels = hasattr(batch[0], "label")
    labels = []

    for example in batch:
        input_a_texts.append(example.texts[0])
        input_b_texts.append(example.texts[1])
        if has_labels:
            labels.append(example.label)

    input_a = tokenizer(input_a_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    input_b = tokenizer(input_b_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")

    input_a = {key: val.to(device) for key, val in input_a.items()}
    input_b = {key: val.to(device) for key, val in input_b.items()}

    if has_labels:
        labels = torch.tensor(labels, dtype=torch.float).to(device)
        return input_a, input_b, labels
    else:
        return input_a, input_b

