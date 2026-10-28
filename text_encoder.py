"""Text encoder: K text queries per image -> [B, K, d] embeddings."""

from typing import List
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Converts K class-name queries per image into a [B, K, d] embedding.

    In production replace with CLIP / BERT text encoder; here a learned lookup.
    """

    def __init__(self, vocab: List[str], d: int = 512):
        super().__init__()
        self.vocab = {w: i for i, w in enumerate(vocab)}
        self.emb = nn.Embedding(len(vocab), d)

    def forward(self, queries: List[List[str]]) -> torch.Tensor:
        """queries: list of length B of lists of length K of class strings.

        Returns: tensor of shape [B, K, d].
        """
        idx = torch.tensor(
            [[self.vocab[q] for q in qs] for qs in queries],
            device=self.emb.weight.device,
        )  # [B, K]
        return self.emb(idx)  # [B, K, d]
