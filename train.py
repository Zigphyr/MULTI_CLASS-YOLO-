"""Training loops.

Step 5: train Model B.
Step 9: train Model C in stages -- load B, freeze backbone + PAFPN
        (and probably the existing projection), train QVI, then optionally
        unfreeze everything at a 10-100x lower learning rate.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .text_encoder import TextEncoder
from .models import ModelB, ModelC
from .dataset import ToyCountingDataset, collate, VOCAB


def train_model_B(epochs: int = 2, bs: int = 4, lr: float = 1e-3,
                  cfg: Config = Config()) -> ModelB:
    """Train the multi-class extension (Model B)."""
    enc = TextEncoder(VOCAB, d=cfg.embed_dim).to(cfg.device)
    model = ModelB(enc, cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        ToyCountingDataset(n=64, img_size=cfg.img_size),
        batch_size=bs, shuffle=True, collate_fn=collate,
    )

    for ep in range(epochs):
        model.train()
        running = 0.0
        for img, qs, gt, mask in loader:
            img = img.to(cfg.device)
            gt = gt.to(cfg.device)
            mask = mask.to(cfg.device)
            counts, _, _ = model(img, qs)                 # [B, K]
            loss_per = F.l1_loss(counts, gt, reduction="none")
            # Average over valid K only (padded entries ignored).
            loss = (loss_per * mask).sum() / mask.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[B] epoch {ep + 1}  loss={running / len(loader):.4f}")
    return model


def train_model_C(model_B: ModelB, epochs: int = 2, bs: int = 4,
                  lr: float = 1e-4, cfg: Config = Config()) -> ModelC:
    """Load Model B, freeze backbone + PAFPN + projections, train QVI."""
    cfg_C = Config(**{**cfg.__dict__, "use_qvi": True, "qvi_type": "modulation"})
    model = ModelC(model_B.text_enc, cfg_C).to(cfg.device)

    # Copy weights from Model B (same tensor names for shared parts).
    missing, _ = model.load_state_dict(model_B.state_dict(), strict=False)
    print("transferred from B; QVI params fresh:", len(missing))

    # Freeze everything except QVI.
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("qvi.")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    loader = DataLoader(
        ToyCountingDataset(n=64, img_size=cfg.img_size),
        batch_size=bs, shuffle=True, collate_fn=collate,
    )

    for ep in range(epochs):
        model.train()
        running = 0.0
        for img, qs, gt, mask in loader:
            img = img.to(cfg.device)
            gt = gt.to(cfg.device)
            mask = mask.to(cfg.device)
            counts, _, _ = model(img, qs)
            loss = (
                F.l1_loss(counts, gt, reduction="none") * mask
            ).sum() / mask.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[C] epoch {ep + 1}  loss={running / len(loader):.4f}")
    return model
