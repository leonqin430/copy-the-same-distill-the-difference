"""Copy the Same, Distill the Difference: linear-attention ViTs, Softmax-to-linear initializations,
attention distillation. Importing this package registers the ViT entrypoints with timm."""
import warnings

with warnings.catch_warnings():
    # our entrypoints intentionally replace timm's vit_{tiny,small,base}_patch16_224
    warnings.filterwarnings('ignore', message='Overwriting .* in registry')
    from . import vision_transformer  # noqa: F401

from .vision_transformer import VisionTransformer  # noqa: F401
from .dual_vit import DualVisionTransformer  # noqa: F401
from .datasets import create_dataset  # noqa: F401
