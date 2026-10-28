"""End-to-end entry point: sanity tests -> train B -> train C -> benchmark."""

import torch

from yolocount_mc.config import Config
from yolocount_mc.text_encoder import TextEncoder
from yolocount_mc.models import ModelB, ModelC
from yolocount_mc.dataset import VOCAB
from yolocount_mc.train import train_model_B, train_model_C
from yolocount_mc.benchmark import benchmark_A_B_C
from yolocount_mc.ablation import run_ablation


def test_shapes(model, text_enc, cfg):
    img = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    vocab_size = len(text_enc.vocab)
    for K in (1, 2, 5, 20):
        K = min(K, vocab_size)                       # cap at vocab size
        qs = [list(text_enc.vocab.keys())[:K] for _ in range(2)]
        counts, den, cls = model(img, qs)
        assert counts.shape == (2, K)
        assert den.shape[:2] == (2, K)
        assert cls.shape[:2] == (2, K)
    print("test_shapes: OK")


def test_consistency(model, text_enc, cfg, atol=1e-4):
    model.eval()
    img = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    q = list(text_enc.vocab.keys())[:3]
    with torch.no_grad():
        out_K3, *_ = model(img, [q])
        out_sep = torch.stack(
            [model(img, [[qi]])[0][:, 0] for qi in q], dim=1,
        )
    assert torch.allclose(out_K3, out_sep, atol=atol)
    print("test_consistency: OK")


def test_qvi_zero_init(cfg):
    enc = TextEncoder(VOCAB, d=cfg.embed_dim).to(cfg.device)
    cfg_B = Config(**{**cfg.__dict__, "use_qvi": False})
    cfg_C = Config(**{**cfg.__dict__, "use_qvi": True, "qvi_type": "modulation"})
    B = ModelC(enc, cfg_B).to(cfg.device).eval()
    C = ModelC(enc, cfg_C).to(cfg.device).eval()
    C.load_state_dict(B.state_dict(), strict=False)

    img = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    q = [list(enc.vocab.keys())[:2]]
    with torch.no_grad():
        cB = B(img, q)[0]
        cC = C(img, q)[0]
    assert torch.allclose(cB, cC, atol=1e-5)
    print("test_qvi_zero_init_equals_B: OK")


def main():
    cfg = Config(img_size=128, backbone_channels=64, embed_dim=128, qvi_channels=16)

    # 1. Sanity BEFORE training (minutes vs days of debugging).
    enc = TextEncoder(VOCAB, d=cfg.embed_dim).to(cfg.device)
    B_dry = ModelB(enc, cfg).to(cfg.device).eval()
    test_shapes(B_dry, enc, cfg)
    test_consistency(B_dry, enc, cfg)
    test_qvi_zero_init(cfg)

    # 2. Train B, then C (stage-wise).
    B_trained = train_model_B(epochs=2, cfg=cfg)
    _ = train_model_C(B_trained, epochs=2, cfg=cfg)

    # 3. Benchmark + ablation.
    benchmark_A_B_C(cfg)
    run_ablation(cfg=cfg)


if __name__ == "__main__":
    main()
