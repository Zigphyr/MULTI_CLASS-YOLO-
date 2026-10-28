"""Step 4 of the implementation guide: WRITE TESTS BEFORE TRAINING.

Finding a shape or consistency bug here costs minutes. Finding one after
training costs days.
"""

import pytest
import torch

from yolocount_mc.config import Config
from yolocount_mc.text_encoder import TextEncoder
from yolocount_mc.models import ModelB, ModelC
# 24 words is enough for the K=20 shape test.
TEST_VOCAB = [f"c{i}" for i in range(24)]


@pytest.fixture
def cfg():
    # Small image for fast CI; CPU-friendly.
    return Config(img_size=64, backbone_channels=32, embed_dim=64, qvi_channels=16,
                  device="cpu")


@pytest.fixture
def enc(cfg):
    return TextEncoder(TEST_VOCAB, d=cfg.embed_dim).to(cfg.device)


def test_shapes(cfg, enc):
    """Test 3: shapes for K in {1, 2, 5, 20}."""
    model = ModelB(enc, cfg).to(cfg.device).eval()
    img = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    for K in (1, 2, 5, 20):
        qs = [list(enc.vocab.keys())[:K] for _ in range(2)]
        counts, den, cls = model(img, qs)
        assert counts.shape == (2, K), f"counts {counts.shape}"
        assert den.shape[:2] == (2, K), f"den {den.shape}"
        assert cls.shape[:2] == (2, K), f"cls {cls.shape}"


def test_consistency_B_eq_separate_runs(cfg, enc):
    """Test 2 (strongest): K classes together == K separate single-class runs.

    If this fails, something leaks across classes (e.g. a softmax or normalisation
    taken over the K dimension).
    """
    model = ModelB(enc, cfg).to(cfg.device).eval()
    img = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    q = list(enc.vocab.keys())[:3]
    with torch.no_grad():
        out_K3, *_ = model(img, [q])                            # [1, 3]
        out_sep = torch.stack(
            [model(img, [[qi]])[0][:, 0] for qi in q], dim=1,
        )                                                       # [1, 3]
    assert torch.allclose(out_K3, out_sep, atol=1e-4), (
        f"consistency broken: max diff "
        f"{(out_K3 - out_sep).abs().max().item():.2e}"
    )


def test_qvi_zero_init_equals_B(cfg, enc):
    """Fresh Model C with use_qvi=True must reproduce Model B exactly."""
    cfg_B = Config(**{**cfg.__dict__, "use_qvi": False})
    cfg_C = Config(**{**cfg.__dict__, "use_qvi": True, "qvi_type": "modulation"})

    B = ModelC(enc, cfg_B).to(cfg.device).eval()
    C = ModelC(enc, cfg_C).to(cfg.device).eval()
    # Share non-QVI weights.
    C.load_state_dict(B.state_dict(), strict=False)

    img = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
    q = [list(enc.vocab.keys())[:2]]
    with torch.no_grad():
        cB = B(img, q)[0]
        cC = C(img, q)[0]
    assert torch.allclose(cB, cC, atol=1e-5), (cB - cC).abs().max()
