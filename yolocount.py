"""Step 2 of the implementation guide: make yolocount.py class-agnostic.

Key changes vs. the single-class version:
    * mask broadcast  [B, 1, H, W] -> [B, K, H, W]
    * predictions kept as [B, K]
    * no hard-coded class index (e.g. pred[:, 0])
    * .tolist() in visualisation (plotting code expects list, not scalar)
    * epsilon in divisions (NaN safety for empty classes)
"""

from typing import List
import torch


def apply_mask_broadcast(density: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the same spatial mask to all K classes.

    density: [B, K, H, W]
    mask:    [B, 1, H, W]
    returns: [B, K, H, W]
    """
    assert density.ndim == 4 and mask.ndim == 4 and mask.shape[1] == 1
    return density * mask  # broadcasts across K


def aggregate_counts(
    density: torch.Tensor,
    cls_logits: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sum density * confidence over H, W to get a per-class count.

    density, cls_logits: [B, K, H, W]
    returns:             [B, K]
    """
    conf = cls_logits.sigmoid()
    q = (density * conf).sum(dim=(-1, -2))
    # epsilon-safe normalisation keeps empty classes from blowing up on division
    return q / (1.0 + eps)


def predictions_to_list(counts: torch.Tensor) -> List[List[float]]:
    """Replace .item() / squeeze() paths that break for K > 1."""
    return counts.detach().cpu().tolist()
