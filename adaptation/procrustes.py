""" Procrustes method to find a orthogonal linear mapping between
source and target embeddings.
"""
import os
import torch
from pathlib import Path


PATH = os.path.join(Path(__file__).parents[1], "assets/datasets")
TRANSFORMED_DATASETS = [
    os.path.join(PATH, "wos_v_1_transformed_BERT_hidden_0.pt"),
    os.path.join(PATH, "wos_v_1_transformed_SCIBERT_hidden_0.pt"),
    os.path.join(PATH, "wos_v_1_transformed_DISTILBERT_hidden_0.pt"),
]


if __name__ == "__main__":
    source_embed, _, _ = torch.load(TRANSFORMED_DATASETS[0])
    target_embed, _, _ = torch.load(TRANSFORMED_DATASETS[1])
