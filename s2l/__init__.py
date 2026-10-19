"""Copy the Same, Distill the Difference: linear-attention ViTs, Softmax-to-linear initializations,
attention distillation. Importing this package registers the ViT entrypoints with timm."""
import argparse
import warnings

import torch

# timm 0.9.16 checkpoints store the training args as an argparse.Namespace and are read with torch.load()
# without weights_only=False; PyTorch >= 2.6 defaults to weights_only=True and rejects that global, which
# breaks --resume and --initial-checkpoint. Allow-listing the Namespace keeps the safe loader working.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([argparse.Namespace])


with warnings.catch_warnings():
    # our entrypoints intentionally replace timm's vit_{tiny,small,base}_patch16_224
    warnings.filterwarnings('ignore', message='Overwriting .* in registry')
    from . import vision_transformer  # noqa: F401

from .vision_transformer import VisionTransformer  # noqa: F401
from .dual_vit import DualVisionTransformer  # noqa: F401
from .datasets import create_dataset  # noqa: F401
