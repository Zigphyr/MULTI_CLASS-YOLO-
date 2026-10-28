"""Query-Visual Interaction (QVI) module.

Step 7 of the implementation guide: modulation with residual, zero-initialised
output convolutions so Model C == Model B exactly at initialisation.

A nonlinearity (GELU after the refine conv) is required to avoid the algebra
collapse:  <g*v + b, t> = <v, g*t> + <b, t>  --- without the nonlinearity QVI
would simply rename the existing dot product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QVI(nn.Module):
    """Query-Visual Interaction producing per-class corrections.

    Output convs are zero-initialised, so at init the corrections are 0 and
    Model C reproduces Model B exactly.
    """

    def __init__(
        self,
        c_in: int = 512,
        d_txt: int = 512,
        c: int = 64,
        mode: str = "modulation",
    ):
        super().__init__()
        self.mode = mode
        self.reduce = nn.Conv2d(c_in, c, 1)  # 1x1 reduction (memory!)
        self.mlp = nn.Sequential(
            nn.Linear(d_txt, d_txt // 2),
            nn.ReLU(),
            nn.Linear(d_txt // 2, 2 * c),    # produces (gamma, beta)
        )
        self.refine = nn.Conv2d(c, c, 1)
        self.out_den = nn.Conv2d(c, 1, 1)    # density correction
        self.out_cls = nn.Conv2d(c, 1, 1)    # classification correction

        # Start as "do nothing" -- see Step 7 note on zero-init.
        for m in (self.out_den, self.out_cls):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, vis: torch.Tensor, txt: torch.Tensor):
        """Compute per-class corrections.

        Args:
            vis: [B, 512, H, W] visual embedding.
            txt: [B, K, 512]    K text embeddings per image.

        Returns:
            d_den: [B, K, H, W] density correction.
            d_cls: [B, K, H, W] classification correction.
        """
        B, K = txt.shape[:2]
        v = self.reduce(vis)                                    # [B, c, H, W]

        if self.mode == "gated":  # Option B: gated feature fusion
            g = torch.sigmoid(self.mlp(txt)[..., : v.shape[1]])  # [B, K, c]
            vk = v.unsqueeze(1) * g[..., None, None]
        else:  # Option C: query-conditioned modulation
            g, b = self.mlp(txt).chunk(2, -1)                   # [B, K, c] each
            vk = v.unsqueeze(1) * (1 + g[..., None, None]) + b[..., None, None]

        flat = vk.flatten(0, 1)                                 # [B*K, c, H, W]
        flat = flat + F.gelu(self.refine(flat))                 # nonlinear refinement
        d_den = self.out_den(flat).view(B, K, *flat.shape[-2:])
        d_cls = self.out_cls(flat).view(B, K, *flat.shape[-2:])
        return d_den, d_cls
