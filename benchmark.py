"""Step 10: benchmark Models A / B / C at K in {1, 2, 5, 10, 20}.

Warm-up and torch.cuda.synchronize() are required because CUDA runs
asynchronously; timing without them measures nothing useful.
"""

import time
from typing import List, Tuple
import torch

from .config import Config
from .text_encoder import TextEncoder
from .models import ModelA, ModelB, ModelC
from .dataset import VOCAB


@torch.no_grad()
def benchmark(model, text_enc: TextEncoder, Ks=(1, 2, 5, 10, 20),
              img_size: int = 640, device: str = "cuda") -> List[Tuple]:
    rows = []
    vocab_size = len(text_enc.vocab)
    for K in Ks:
        K = min(K, vocab_size)                      # cap at vocab size
        q = [list(text_enc.vocab.keys())[:K]]
        img = torch.randn(1, 3, img_size, img_size, device=device)

        for _ in range(20):  # warm-up
            model(img, q)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model(img, q)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / 100 * 1000

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model(img, q)
            mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            mem_mb = float("nan")
        rows.append((K, latency_ms, mem_mb))
    return rows


def benchmark_A_B_C(cfg: Config = Config()):
    enc = TextEncoder(VOCAB, d=cfg.embed_dim).to(cfg.device)
    A = ModelA(enc, cfg).to(cfg.device).eval()
    B = ModelB(enc, cfg).to(cfg.device).eval()
    C = ModelC(
        enc, Config(**{**cfg.__dict__, "use_qvi": True}),
    ).to(cfg.device).eval()

    print(f"{'Model':<6}{'K':>4}{'Latency (ms)':>16}{'Peak Mem (MB)':>16}")

    # Model A: K passes through the full model.
    vocab_size = len(enc.vocab)
    for K in (1, 2, 5, 10, 20):
        K = min(K, vocab_size)
        img = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
        qs = list(enc.vocab.keys())[:K]
        for _ in range(10):
            [A(img, [[q]]) for q in qs]
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            [A(img, [[q]]) for q in qs]
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        print(f"{'A':<6}{K:>4}{(time.time() - t0) / 20 * 1000:>16.2f}{'-':>16}")

    for name, m in (("B", B), ("C", C)):
        rows = benchmark(m, enc, device=cfg.device, img_size=cfg.img_size)
        for K, lat, mem in rows:
            mem_str = f"{mem:.1f}" if mem == mem else "-"  # NaN check
            print(f"{name:<6}{K:>4}{lat:>16.2f}{mem_str:>16}")
