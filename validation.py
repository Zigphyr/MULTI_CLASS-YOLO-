"""Step 3 of the implementation guide: per-class + overall MAE / RMSE.

Why per-class matters: if 9 classes are perfect and one is badly wrong, the
overall average looks fine. Per-class scores are only meaningful if the same
class sits at the same index; otherwise group by class name.
"""

from typing import Dict
import torch


@torch.no_grad()
def evaluate(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """pred, gt both [N, K] where N accumulates across batches."""
    err = (pred - gt).abs()                                 # [N, K]
    mae_per_class = err.mean(0)                             # [K]
    rmse_per_class = ((pred - gt) ** 2).mean(0).sqrt()      # [K]
    mae_overall = err.mean()
    rmse_overall = ((pred - gt) ** 2).mean().sqrt()
    return dict(
        mae_per_class=mae_per_class,
        rmse_per_class=rmse_per_class,
        mae_overall=mae_overall,
        rmse_overall=rmse_overall,
    )
