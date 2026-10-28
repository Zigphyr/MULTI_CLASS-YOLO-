"""Step 11: ablation harness.

Variants:   B  |  +QVI (Gated)  |  +QVI (Modulation)   [+ Attention if warranted]
Use at least 3 seeds per variant and report mean and standard deviation.

Key comparison: B vs C. If the gap is smaller than the seed-to-seed spread,
there is no evidence that QVI helps (publishable negative result).
"""

import random
import torch
from torch.utils.data import DataLoader

from .config import Config
from .train import train_model_B, train_model_C
from .dataset import ToyCountingDataset, collate


@torch.no_grad()
def _evaluate_on_toy(model, cfg: Config) -> float:
    """Mask-weighted MAE across batches with variable K."""
    model.eval()
    loader = DataLoader(
        ToyCountingDataset(n=32, img_size=cfg.img_size),
        batch_size=4, collate_fn=collate,
    )
    total_err, total_n = 0.0, 0
    for img, qs, gt, mask in loader:
        c, *_ = model(img.to(cfg.device), qs)
        err = (c.cpu() - gt).abs() * mask                 # pad entries zeroed
        total_err += err.sum().item()
        total_n += mask.sum().item()
    return total_err / max(total_n, 1.0)


def run_ablation(seeds=(0, 1, 2), cfg: Config = Config()):
    results = {k: [] for k in ("B", "QVI-gated", "QVI-mod")}
    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)

        B_model = train_model_B(epochs=1, cfg=cfg)
        results["B"].append(_evaluate_on_toy(B_model, cfg))

        for mode, key in (("gated", "QVI-gated"), ("modulation", "QVI-mod")):
            cfg_v = Config(**{**cfg.__dict__, "use_qvi": True, "qvi_type": mode})
            C = train_model_C(B_model, epochs=1, cfg=cfg_v)
            results[key].append(_evaluate_on_toy(C, cfg_v))

    for k, vals in results.items():
        t = torch.tensor(vals)
        print(f"{k:<12}  MAE  mean={t.mean():.3f}  std={t.std():.3f}")
    return results
