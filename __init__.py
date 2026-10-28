"""YOLO-Count Multi-Class + QVI package.

Three models:
    * Model A: original YOLO-Count   (one query per forward pass)
    * Model B: multi-class extension (K classes in one pass, no QVI)
    * Model C: Model B + QVI         (Query-Visual Interaction)
"""

from .config import Config
from .text_encoder import TextEncoder
from .backbone import YoloBackbonePAFPN
from .head import MultiClassHead
from .qvi import QVI
from .models import ModelA, ModelB, ModelC

__all__ = [
    "Config",
    "TextEncoder",
    "YoloBackbonePAFPN",
    "MultiClassHead",
    "QVI",
    "ModelA",
    "ModelB",
    "ModelC",
]
