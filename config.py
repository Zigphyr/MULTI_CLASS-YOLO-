"""Global configuration for the YOLO-Count multi-class + QVI pipeline."""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    img_size: int = 640
    backbone_channels: int = 256       # C out of PAFPN
    embed_dim: int = 512               # 512-D shared embedding space (per report)
    qvi_channels: int = 64             # c in QVI (keeps [B*K,c,H,W] small)
    use_qvi: bool = False              # False => exactly Model B
    qvi_type: str = "modulation"       # none | gated | modulation | attention
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
