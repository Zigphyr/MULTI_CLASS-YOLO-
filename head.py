"""Multi-class head (Step 1 of the implementation guide).

Image features are computed ONCE. The class dimension K is only introduced
at the text-visual similarity (einsum) step. The full feature map is never
replicated K times.

Pipeline:
    density_head_new : shared spatial feature extractor      [B, C', H, W]
    density_proj     : projects to 512-D embedding space     [B, 512, H, W]
    cls_proj         : parallel projection for classification [B, 512, H, W]
    einsum           : dot product with text embeddings       [B, K, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassHead(nn.Module):
    """Model B head: shared visual features + text-visual similarity by einsum."""

    def __init__(self, c_in: int = 256, c_hidden: int = 256, d: int = 512):
        super().__init__()
        self.density_head_new = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(c_hidden, c_hidden, 3, 1, 1), nn.SiLU(),
        )
        self.density_proj = nn.Conv2d(c_hidden, d, 1)   # density stream
        self.cls_proj = nn.Conv2d(c_hidden, d, 1)       # classification stream
        self.log_scale = nn.Parameter(torch.tensor(0.0))  # learnable temperature

    def forward(self, feat: torch.Tensor, txt: torch.Tensor):
        """Compute class-wise density and classification maps.

        Args:
            feat: [B, C, H, W]  shared visual features (computed once).
            txt:  [B, K, 512]   K text embeddings per image.

        Returns:
            den_sim:  [B, K, H, W]  density similarity map per class.
            cls_sim:  [B, K, H, W]  classification similarity map per class.
            vis_den:  [B, 512, H, W] visual embedding (used by QVI).
        """
        x = self.density_head_new(feat)        # [B, C', H, W]
        vis_den = self.density_proj(x)         # [B, 512, H, W]
        vis_cls = self.cls_proj(x)             # [B, 512, H, W]

        # L2 normalise so dot product is a cosine similarity.
        vis_den_n = F.normalize(vis_den, dim=1)
        vis_cls_n = F.normalize(vis_cls, dim=1)
        txt_n = F.normalize(txt, dim=-1)

        scale = self.log_scale.exp()
        den_sim = torch.einsum("bchw,bkc->bkhw", vis_den_n, txt_n) * scale
        cls_sim = torch.einsum("bchw,bkc->bkhw", vis_cls_n, txt_n) * scale
        return den_sim, cls_sim, vis_den
