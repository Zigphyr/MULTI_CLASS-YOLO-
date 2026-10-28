"""Full models A, B, C.

- Model A: original YOLO-Count (one query per forward pass).
- Model B: multi-class extension (K queries in one pass, no QVI).
- Model C: Model B + QVI (query-conditioned visual features).
"""

from typing import List
import torch
import torch.nn as nn

from .config import Config
from .text_encoder import TextEncoder
from .backbone import YoloBackbonePAFPN
from .head import MultiClassHead
from .qvi import QVI


class ModelA(nn.Module):
    """Original YOLO-Count: ONE query per forward pass."""

    def __init__(self, text_enc: TextEncoder, cfg: Config = Config()):
        super().__init__()
        self.text_enc = text_enc
        self.backbone = YoloBackbonePAFPN(out_ch=cfg.backbone_channels)
        self.head = MultiClassHead(c_in=cfg.backbone_channels, d=cfg.embed_dim)

    def forward(self, img: torch.Tensor, queries: List[List[str]]):
        assert all(len(q) == 1 for q in queries), "Model A is single-query"
        feat = self.backbone(img)
        txt = self.text_enc(queries)                              # [B, 1, 512]
        den, cls, _ = self.head(feat, txt)                        # [B, 1, H, W]
        counts = (den * cls.sigmoid()).sum(dim=(-1, -2))          # [B, 1]
        return counts, den, cls


class ModelB(nn.Module):
    """Multi-class extension: K queries in ONE forward pass."""

    def __init__(self, text_enc: TextEncoder, cfg: Config = Config()):
        super().__init__()
        self.text_enc = text_enc
        self.backbone = YoloBackbonePAFPN(out_ch=cfg.backbone_channels)
        self.head = MultiClassHead(c_in=cfg.backbone_channels, d=cfg.embed_dim)

    def forward(self, img: torch.Tensor, queries: List[List[str]]):
        feat = self.backbone(img)                                 # [B, C, H, W] shared
        txt = self.text_enc(queries)                              # [B, K, 512]
        den_sim, cls_sim, _ = self.head(feat, txt)                # [B, K, H, W] each
        counts = (den_sim * cls_sim.sigmoid()).sum(dim=(-1, -2))  # [B, K]
        return counts, den_sim, cls_sim


class ModelC(nn.Module):
    """Model B + QVI. Config switch keeps backward compatibility."""

    def __init__(self, text_enc: TextEncoder, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.text_enc = text_enc
        self.backbone = YoloBackbonePAFPN(out_ch=cfg.backbone_channels)
        self.head = MultiClassHead(c_in=cfg.backbone_channels, d=cfg.embed_dim)
        self.qvi = (
            QVI(c_in=cfg.embed_dim, d_txt=cfg.embed_dim,
                c=cfg.qvi_channels, mode=cfg.qvi_type)
            if cfg.use_qvi
            else None
        )

    def forward(self, img: torch.Tensor, queries: List[List[str]]):
        feat = self.backbone(img)
        txt = self.text_enc(queries)
        den_sim, cls_sim, vis_den = self.head(feat, txt)

        if self.qvi is not None:
            d_den, d_cls = self.qvi(vis_den, txt)                 # [B, K, H, W] each
            den_maps = den_sim + d_den                            # residual add
            cls_maps = cls_sim + d_cls
        else:
            den_maps, cls_maps = den_sim, cls_sim

        counts = (den_maps * cls_maps.sigmoid()).sum(dim=(-1, -2))  # [B, K]
        return counts, den_maps, cls_maps
