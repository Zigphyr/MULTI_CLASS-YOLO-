"""Toy dataset for end-to-end sanity training.

Replace with the real YOLO-Count counting dataset. Each sample must yield:
    image           -- tensor [3, H, W]
    queries         -- list of K class strings
    ground truth    -- tensor [K] of per-class counts
"""

import random
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VOCAB: List[str] = ["car", "person", "dog", "chair", "cat", "bike"]


class ToyCountingDataset(Dataset):
    """Random images + random per-class counts.

    Each sample yields K queries (mix of present + absent classes).
    30% of ground-truth entries are forced to zero so the model learns to
    say "none" and does not hallucinate counts (Step 5: absent classes).
    """

    def __init__(self, n: int = 256, img_size: int = 640,
                 k_range=(1, 4), vocab=VOCAB):
        self.n, self.img_size = n, img_size
        self.k_range, self.vocab = k_range, vocab

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        img = torch.rand(3, self.img_size, self.img_size)
        K = random.randint(*self.k_range)
        qs = random.sample(self.vocab, K)
        gt = torch.randint(0, 10, (K,)).float()
        gt = torch.where(torch.rand(K) < 0.3, torch.zeros(K), gt)
        return img, qs, gt


def collate(batch):
    """Pad per-sample K to the max K in the batch using the first vocab token."""
    imgs = torch.stack([b[0] for b in batch])
    qs = [b[1] for b in batch]
    Kmax = max(len(q) for q in qs)
    padded_q, mask = [], []
    for q, (_, _, _) in zip(qs, batch):
        pad = Kmax - len(q)
        padded_q.append(q + [VOCAB[0]] * pad)
        mask.append([1.0] * len(q) + [0.0] * pad)
    gts = torch.stack([F.pad(b[2], (0, Kmax - len(b[1]))) for b in batch])
    mask = torch.tensor(mask)
    return imgs, padded_q, gts, mask
