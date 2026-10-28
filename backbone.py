"""Stand-in YOLO backbone + PAFPN producing shared visual features F.

Replace with the real backbone + PAFPN from the YOLO-Count repository.
This stub only exists so the rest of the pipeline is runnable end-to-end.
"""

import torch
import torch.nn as nn


class YoloBackbonePAFPN(nn.Module):
    """Produces shared visual features F in R^[B, C, H, W].

    Runs ONCE per image regardless of the number of text queries.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, out_ch, 3, 2, 1), nn.SiLU(),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.SiLU(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, 3, H0, W0] -> [B, C, H, W]."""
        return self.neck(self.stem(img))
