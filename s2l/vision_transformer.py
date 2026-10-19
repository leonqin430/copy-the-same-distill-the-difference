""" Vision Transformer with linear-attention operators and Softmax-to-linear initializations.

Copy the Same, Distill the Difference: Initializing Linear Vision Transformers (arXiv:2609.35745).

This module is a trimmed copy of ``timm/models/vision_transformer.py`` (timm 0.9.16, Ross Wightman,
Apache-2.0) as modified by Structured Initialization (Jianqiao Zheng et al., MIT; Mimetic / Impulse
initializations, sinusoidal position embedding), extended with:

* the linear attention operators used in the paper: ``attn_type`` in {'elu1', 'relu', 'inline', 'mhla', 'ttt'}
  (plus 'softmax' for the Softmax ViT and the 'performer' / 'hedgehog' / 'qtvit' variants kept for reference);
* the weight-initialization settings: ``weight_init`` in {'skip' (Random), 'mimetic0.7_0.7', 'impulse3',
  'copy' (Copy: attention projections), 'ffncopy' (MLPCopy: MLP blocks), 'fullcopy' (FullCopy: all matching
  weights)}, the copy modes reading the Softmax checkpoint given by ``copy_ckpt_path``.

Only the three entrypoints used in the paper are registered (``vit_tiny_patch16_224``, ``vit_small_patch16_224``,
``vit_base_patch16_224``); importing this module replaces timm's entrypoints of the same names.
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType,  build_sincos2d_pos_embed
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class InLineAttention(nn.Module):
    """InLine Attention (NeurIPS 2024, "Bridging the Divide").

    Restores injectivity of linear attention via subtraction normalization.
    Uses identity kernel (no feature map). Attention-only version without local conv residual.
    Same Q,K,V,proj weights as standard Attention — enables direct weight copying.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # InLine attention: subtraction normalization restores injectivity
        # Equivalent to Eq. 5 in the paper (identity kernel, scale folded in)
        kv = (k.transpose(-2, -1) * (self.scale / N) ** 0.5) @ (v * (self.scale / N) ** 0.5)
        x = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QTViTAttention(nn.Module):
    """QT-ViT Attention (NeurIPS 2024).

    Quadratic Taylor expansion of softmax: L2-normalize Q,K, square them,
    append learnable ones, then standard linear attention.
    Same Q,K,V,proj weights as standard Attention — enables direct weight copying.
    Extra params: one learnable scalar (ones_scale).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-15

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # QT-ViT: learnable scalar for the appended ones
        self.ones_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Quadratic Taylor: L2-normalize, square, re-normalize
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        q = q ** 2
        k = k ** 2
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps)

        # Append learnable ones for bias term
        ones = torch.ones(B, self.num_heads, N, 1, device=q.device, dtype=q.dtype) * self.ones_scale
        q = torch.cat([q, ones], dim=-1)      # [B, H, N, d+1]
        k = torch.cat([k, ones], dim=-1)

        # Pad v with 1s for normalizer
        v = F.pad(v, (0, 1), mode="constant", value=1)  # [B, H, N, d+1]

        # Linear attention
        kv = k.transpose(-2, -1) @ v           # [B, H, d+1, d+1]
        out = q @ kv                             # [B, H, N, d+1]
        x = out[..., :-1] / (out[..., -1:] + self.eps)  # normalize

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HedgehogAttention(nn.Module):
    """Hedgehog Attention (ICLR 2024).

    Learnable MLP feature map: φ(x) = [exp(Wx), exp(-Wx)].
    Doubles the feature dim (d → 2d) for the kernel, then standard linear attention.
    Same Q,K,V,proj weights as standard Attention — enables direct weight copying.
    Extra params: one linear layer per Q and K (head_dim → head_dim).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Hedgehog: learnable feature maps (initialized to identity)
        self.mlp_q = nn.Linear(self.head_dim, self.head_dim)
        self.mlp_k = nn.Linear(self.head_dim, self.head_dim)
        nn.init.eye_(self.mlp_q.weight)
        nn.init.zeros_(self.mlp_q.bias)
        nn.init.eye_(self.mlp_k.weight)
        nn.init.zeros_(self.mlp_k.bias)

    def _phi(self, x: torch.Tensor, mlp: nn.Linear) -> torch.Tensor:
        # φ(x) = [exp(Wx), exp(-Wx)] — always non-negative, doubles dim
        # Clamp to prevent exp() overflow during from-scratch training
        t = mlp(x).clamp(-15, 15)
        return torch.cat([torch.exp(t), torch.exp(-t)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply Hedgehog feature maps: [B, H, N, d] -> [B, H, N, 2d]
        q = self._phi(q, self.mlp_q)
        k = self._phi(k, self.mlp_k)

        # Linear attention with doubled feature dim
        kv = k.transpose(-2, -1) @ v           # [B, H, 2d, d]
        num = q @ kv                             # [B, H, N, d]
        denom = q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True)  # [B, H, N, 1]
        x = num / (denom + 1e-6)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    """Linear attention with configurable kernel function.

    Replaces softmax(QK^T)V with phi(Q)(phi(K)^T V) for O(Nd^2) complexity.
    Supports: 'elu1' (ELU+1), 'relu', 'performer'.
    Same Q,K,V,proj weights as standard Attention — enables direct weight copying.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            kernel: str = 'elu1',
            performer_nb_features: int = 256,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel = kernel

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if kernel == 'performer':
            self.register_buffer(
                'random_features',
                torch.randn(self.head_dim, performer_nb_features) / (self.head_dim ** 0.5),
            )

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == 'elu1':
            return F.elu(x) + 1.0
        elif self.kernel == 'relu':
            return F.relu(x)
        elif self.kernel == 'performer':
            # FAVOR+ random feature map: exp(Wx - ||x||^2 / 2)
            x_proj = x @ self.random_features  # [B, H, N, nb_features]
            x_norm = (x ** 2).sum(dim=-1, keepdim=True) / 2.0
            return torch.exp(x_proj - x_norm) + 1e-6
        else:
            raise ValueError(f'Unknown kernel: {self.kernel}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply kernel function
        q = self._phi(q)  # [B, H, N, d] or [B, H, N, nb_features]
        k = self._phi(k)

        # Efficient linear attention: O(N * d^2) instead of O(N^2 * d)
        kv = k.transpose(-2, -1) @ v           # [B, H, d, d]
        num = q @ kv                             # [B, H, N, d]
        denom = q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True)  # [B, H, N, 1]
        x = num / (denom + 1e-6)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MHLAttention(nn.Module):
    """Multi-Head Linear Attention (MHLA, ICLR 2026).

    Correct implementation following the official repo (DAGroup-PKU/MHLA).
    Uses windowed linear attention with cross-window mixing via BlockDistanceConv.
    Same Q,K,V,proj weights as softmax — enables direct weight copying.
    Extra params: piece_attn (1x1 conv for cross-window mixing), lepe (5x5 DWConv).

    Input is [B, N, C] — internally reshapes into [B, num_pieces, window_size, C].
    Requires N to be divisible by window_size.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            window_size: int = 49,
            embed_len: int = 196,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-6

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Window parameters
        self.window_size = window_size
        self.window_len = int(window_size ** 0.5)
        self.embed_len = embed_len
        self.num_pieces = embed_len // window_size
        self.pieces_len = int(self.num_pieces ** 0.5)

        # LEPE: local positional encoding via 5x5 DWConv on V
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        # Cross-window mixing via distance-based 1x1 conv
        self.piece_attn = self._build_block_distance_conv(
            num_patches_per_side=int(embed_len ** 0.5),
            patch_group_size=window_size,
        )

    def _build_block_distance_conv(self, num_patches_per_side, patch_group_size):
        """Build a 1x1 conv with cosine-distance-based initialization for cross-window mixing."""
        patches_per_block_side = int(patch_group_size ** 0.5)
        blocks_per_side = num_patches_per_side // patches_per_block_side
        total_blocks = blocks_per_side ** 2

        # Compute block center distances
        centers = []
        for i in range(blocks_per_side):
            for j in range(blocks_per_side):
                centers.append([i + 0.5, j + 0.5])
        centers = torch.tensor(centers, dtype=torch.float32)

        dist = torch.zeros(total_blocks, total_blocks)
        for i in range(total_blocks):
            for j in range(total_blocks):
                dist[i, j] = torch.norm(centers[i] - centers[j], p=2)

        # Cosine transform (following official default)
        import math
        max_dist = dist.max()
        normalized = dist / max_dist * math.pi / 4
        mat = torch.cos(normalized)
        mat = mat / mat.sum(dim=0, keepdim=True)

        conv = nn.Conv2d(total_blocks, total_blocks, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.data = mat.unsqueeze(-1).unsqueeze(-1)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        pl = self.pieces_len
        wl = self.window_len

        # Reshape to windowed: [B, N, C] -> [B, num_pieces, window_size, C]
        x_win = x.reshape(B, pl, wl, pl, wl, C).permute(0, 1, 3, 2, 4, 5).reshape(
            B, self.num_pieces, self.window_size, C)

        # QKV projection
        q, k, v = self.qkv(x_win).chunk(3, dim=-1)

        # LEPE on V
        v_2d = v.reshape(B, pl, pl, wl, wl, C).permute(0, 5, 1, 3, 2, 4).reshape(
            B, C, pl * wl, pl * wl)
        lepe = self.lepe(v_2d)
        lepe = lepe.reshape(B, C, pl, wl, pl, wl).permute(0, 2, 4, 3, 5, 1).reshape(
            B, self.num_pieces, self.window_size, C)

        # Normalize and apply ReLU kernel
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to multi-head: [B, num_pieces, window_size, C] -> [(B*H), num_pieces, window_size, D]
        q, k, v = [t.reshape(B, self.num_pieces, self.window_size, H, D).permute(
            0, 3, 1, 2, 4).reshape(B * H, self.num_pieces, self.window_size, D)
            for t in (q, k, v)]

        q = torch.relu(q) + self.eps
        k = torch.relu(k) + self.eps

        k_t = k.transpose(-2, -1)  # [B*H, num_pieces, D, window_size]

        # Local KV summaries + cross-window mixing
        kv = torch.matmul(k_t, v)            # [B*H, num_pieces, D, D]
        kv = self.piece_attn(kv)             # cross-window mixing

        # Normalizer with cross-window mixing
        k_sum = k_t.sum(dim=-1, keepdim=True)  # [B*H, num_pieces, D, 1]
        normalizer = self.piece_attn(torch.matmul(q, k_sum)) + self.eps  # [B*H, num_pieces, window_size, 1]

        out = torch.matmul(q, kv) / normalizer  # [B*H, num_pieces, window_size, D]

        # Reshape back: [(B*H), num_pieces, window_size, D] -> [B, num_pieces, window_size, C]
        out = out.reshape(B, H, self.num_pieces, self.window_size, D).permute(
            0, 2, 3, 1, 4).reshape(B, self.num_pieces, self.window_size, C)
        out = out + lepe

        # Project
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back to [B, N, C]
        out = out.reshape(B, pl, pl, wl, wl, C).permute(0, 1, 3, 2, 4, 5).reshape(B, N, C)
        return out


class TTTAttention(nn.Module):
    """Test-Time Training Attention (ViT^3, CVPR 2026).

    Two parallel branches: simplified SwiGLU inner module + 3x3 DWC inner module.
    Uses hand-derived closed-form gradients for inner training.

    The first dim*3 of qkv maps to standard Q,K,V — enables partial weight copying.
    Extra params: w1, w2 (inner SwiGLU), w3 (inner DWC), q2/k2/v2 portion of qkv, proj.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # QKV: first dim*3 = standard Q1,K1,V1; last head_dim*3 = Q2,K2,V2 for DWC branch
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)

        # Inner module weights
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)

        # Output projection: dim + head_dim -> dim (concatenates both branches)
        self.proj = nn.Linear(dim + head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5

    def _inner_train_swiglu(self, k, v, w1, w2):
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - g1, w2 - g2

    def _inner_train_dwc(self, k, v, w):
        B, C, H, W = k.shape
        e = -v / float(H * W) * self.scale
        k = F.pad(k, (1, 1, 1, 1))
        outs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys, xs = 1 + dy, 1 + dx
                dot = (k[:, :, ys:ys + H, xs:xs + W] * e).sum(dim=(-2, -1))
                outs.append(dot)
        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        return w.repeat(B, 1, 1, 1) - g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        d = self.head_dim
        h = w = int(n ** 0.5)
        assert h * w == n, f"TTT requires square token grid, got N={n}"

        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training
        w1, w2 = self._inner_train_swiglu(k1, v1, self.w1, self.w2)
        w3 = self._inner_train_dwc(k2, v2, self.w3)

        # Apply to queries
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            attn_type: str = 'softmax',
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_kwargs = dict(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        if attn_type == 'softmax':
            self.attn = Attention(**attn_kwargs)
        elif attn_type in ('elu1', 'relu', 'performer'):
            self.attn = LinearAttention(**attn_kwargs, kernel=attn_type)
        elif attn_type == 'inline':
            self.attn = InLineAttention(**attn_kwargs)
        elif attn_type == 'qtvit':
            self.attn = QTViTAttention(**attn_kwargs)
        elif attn_type == 'hedgehog':
            self.attn = HedgehogAttention(**attn_kwargs)
        elif attn_type == 'mhla':
            self.attn = MHLAttention(**attn_kwargs)
        elif attn_type == 'ttt':
            self.attn = TTTAttention(**attn_kwargs)
        else:
            raise ValueError(f'Unknown attn_type: {attn_type}')
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """ Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_parallel: int = 2,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', mlp_layer(
                    dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', '', 'mimetic', 'impulse', 'copy', 'fullcopy', 'ffncopy'] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            sin_pe: bool = False,
            pseudo_input1: Literal['P','U','G','A','B'] = 'P',
            pseudo_input2: Literal['P','U','G','A','B'] = 'P',
            init_once: bool = True,
            attn_type: str = 'softmax',
            copy_ckpt_path: str = '',
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        # change to absolute sinusoidal positional encoding
        self.sin_pe = sin_pe
        if sin_pe:
            print('using abs pe')
            # fixed (non-trainable) 2-D sin-cos embedding; a non-persistent buffer so it follows .to(device)
            self.register_buffer('pos_embed', build_sincos2d_pos_embed(
                [img_size[0] // patch_size, img_size[1] // patch_size], dim=embed_dim), persistent=False)
        else:
            print('using default pe')
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.attn_type = attn_type
        self.copy_ckpt_path = copy_ckpt_path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                attn_type=attn_type,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            if weight_init in ('jax', 'jax_nlhb', 'moco', ''):
                print('init weights...')
                self.init_weights(weight_init)
            else:
                self.weight_init = weight_init
                self.depth = depth
                self.num_heads = num_heads
                self.attn_size = [(img_size[0]//patch_size),(img_size[1]//patch_size)]
                self.embed_dim = embed_dim
                self.pseudo_input1 = pseudo_input1
                self.pseudo_input2 = pseudo_input2
                self.init_once = init_once
                self._init_weights(self)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # don't need trunc normal init when using sin pe
        if not self.sin_pe:
            print('using trunc pe initialization')
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        # init_weights_vit_timm(m)
        # modified this funcition as our initialization
        mode = self.weight_init
        if 'mimetic' in mode:
            alpha, beta = mode[7:].split('_')
            alpha = float(alpha)
            beta = float(beta)
            print(f'using mimetic init with alpha={alpha}, beta={beta}')
            head_dim = self.embed_dim // self.num_heads
            for i in range(self.depth):
                d = i / float(self.depth - 1)

                for h in range(self.num_heads):
                    Q, K = get_ortho_like(self.embed_dim, -float('inf'), alpha, beta, 1)
                    Q = Q[:,:head_dim]
                    K = K.T[:,:head_dim]

                    self.blocks[i].attn.qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = torch.tensor(Q.T).float()
                    self.blocks[i].attn.qkv.weight.data[self.embed_dim+(h*head_dim):self.embed_dim+((h+1)*head_dim)] = torch.tensor(K.T).float()

            for block in self.blocks:
                V, Proj = get_ortho_like(self.embed_dim, self.num_heads, 0.4, 0.4, -1)
                block.attn.qkv.weight.data[2*self.embed_dim:] = torch.tensor(V).float()
                block.attn.proj.weight.data = torch.tensor(Proj).float()
        elif 'impulse' in mode:
            a = mode[7:]
            # scale = float(a)
            scale = (self.embed_dim // self.num_heads) ** -0.5
            norm_epoch = 100 # int(b)
            ff = int(a)
            print(f'using impulse init with scale={scale}, norm_epoch={norm_epoch}, kernel size={ff}')
            head_dim = self.embed_dim // self.num_heads

            def get_pseudo_input(code,self):
                if code == 'P':
                    return self.pos_embed
                elif code == 'U':
                    return 2*torch.rand(self.pos_embed.shape,device=self.pos_embed.device)-1
                elif code == 'G':
                    return trunc_normal_(torch.zeros_like(self.pos_embed), std=torch.sqrt(torch.tensor(1/2)))
                elif code == 'A':
                    return self.pos_embed+2*torch.rand(self.pos_embed.shape,device=self.pos_embed.device)-1
                elif code == 'B':
                    return self.pos_embed+trunc_normal_(torch.zeros_like(self.pos_embed), std=torch.sqrt(torch.tensor(1/2)))
                else:
                    return None
            
            for i in range(self.depth):
                d = i / float(self.depth - 1)
                if i == 0:
                    Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input1,self),norm=norm_epoch)
                elif i == 1:
                    if not self.pseudo_input1==self.pseudo_input2:
                        Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
                    elif not self.init_once:
                        Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
                else:
                    if not self.init_once:
                        Q, K = impulse_init(self.num_heads,self.attn_size,head_dim,ff,scale=scale,spatial_pe=get_pseudo_input(self.pseudo_input2,self),norm=norm_epoch)
                for h in range(self.num_heads):
                    self.blocks[i].attn.qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = torch.tensor(Q[h].T).float()
                    self.blocks[i].attn.qkv.weight.data[self.embed_dim+(h*head_dim):self.embed_dim+((h+1)*head_dim)] = torch.tensor(K[h]).float()
        elif mode == 'copy':
            assert self.copy_ckpt_path, 'copy_ckpt_path must be set for copy init'
            print(f'using copy init from {self.copy_ckpt_path}')
            ckpt = torch.load(self.copy_ckpt_path, map_location='cpu', weights_only=False)
            teacher_state = ckpt['model'] if 'model' in ckpt else ckpt
            dim = self.embed_dim
            t_qkv0 = teacher_state.get('blocks.0.attn.qkv.weight')
            if t_qkv0 is None or t_qkv0.shape[1] != dim:
                raise RuntimeError(
                    f'copy init: checkpoint embed_dim {None if t_qkv0 is None else t_qkv0.shape[1]} does not match '
                    f'the model embed_dim {dim} (check --model / the checkpoint size)')
            for i in range(self.depth):
                qkv_w_key = f'blocks.{i}.attn.qkv.weight'
                qkv_b_key = f'blocks.{i}.attn.qkv.bias'
                proj_w_key = f'blocks.{i}.attn.proj.weight'
                proj_b_key = f'blocks.{i}.attn.proj.bias'
                student_qkv_w = self.blocks[i].attn.qkv.weight.data
                teacher_qkv_w = teacher_state.get(qkv_w_key)
                if teacher_qkv_w is not None:
                    # Teacher has dim*3 rows; student may have more (e.g., TTT: dim*3 + head_dim*3)
                    copy_rows = min(student_qkv_w.shape[0], teacher_qkv_w.shape[0])
                    copy_cols = min(student_qkv_w.shape[1], teacher_qkv_w.shape[1])
                    student_qkv_w[:copy_rows, :copy_cols].copy_(teacher_qkv_w[:copy_rows, :copy_cols])
                teacher_qkv_b = teacher_state.get(qkv_b_key)
                if teacher_qkv_b is not None and self.blocks[i].attn.qkv.bias is not None:
                    student_qkv_b = self.blocks[i].attn.qkv.bias.data
                    copy_len = min(student_qkv_b.shape[0], teacher_qkv_b.shape[0])
                    student_qkv_b[:copy_len].copy_(teacher_qkv_b[:copy_len])
                # Proj: only copy if shapes match (drop-in methods); skip if different (TTT, MHLA)
                teacher_proj_w = teacher_state.get(proj_w_key)
                if teacher_proj_w is not None:
                    student_proj_w = self.blocks[i].attn.proj.weight.data
                    if student_proj_w.shape == teacher_proj_w.shape:
                        student_proj_w.copy_(teacher_proj_w)
                teacher_proj_b = teacher_state.get(proj_b_key)
                if teacher_proj_b is not None and self.blocks[i].attn.proj.bias is not None:
                    student_proj_b = self.blocks[i].attn.proj.bias.data
                    if student_proj_b.shape == teacher_proj_b.shape:
                        student_proj_b.copy_(teacher_proj_b)
            print(f'copied attention weights from {self.depth} blocks (attn_type={self.attn_type})')
            del ckpt, teacher_state
        elif mode == 'fullcopy':
            assert self.copy_ckpt_path, 'copy_ckpt_path must be set for fullcopy init'
            print(f'using fullcopy init from {self.copy_ckpt_path}')
            ckpt = torch.load(self.copy_ckpt_path, map_location='cpu', weights_only=False)
            teacher_state = ckpt['model'] if 'model' in ckpt else ckpt
            # Always skip head (classifier) for fair comparison across all datasets
            skip_keys = {'head.weight', 'head.bias'}
            student_state = self.state_dict()
            copied, skipped = 0, 0
            for key in teacher_state:
                if key in skip_keys:
                    skipped += 1
                    continue
                if key in student_state and student_state[key].shape == teacher_state[key].shape:
                    student_state[key].copy_(teacher_state[key])
                    copied += 1
                else:
                    skipped += 1
            self.load_state_dict(student_state, strict=False)
            if copied == 0:
                raise RuntimeError(f'fullcopy init: no parameter of {self.copy_ckpt_path} matches the model (check --model / the checkpoint size)')
            print(f'fullcopy: copied {copied} params, skipped {skipped} (head/shape mismatch/missing)')
            del ckpt, teacher_state
        elif mode == 'ffncopy':
            assert self.copy_ckpt_path, 'copy_ckpt_path must be set for ffncopy init'
            print(f'using ffncopy init from {self.copy_ckpt_path}')
            ckpt = torch.load(self.copy_ckpt_path, map_location='cpu', weights_only=False)
            teacher_state = ckpt['model'] if 'model' in ckpt else ckpt
            skip_keys = {'head.weight', 'head.bias'}
            # Skip all attention weights: qkv, proj (and any extra attn submodules)
            attn_keywords = {'.attn.'}
            student_state = self.state_dict()
            copied, skipped_attn, skipped_other = 0, 0, 0
            for key in teacher_state:
                if key in skip_keys:
                    skipped_other += 1
                    continue
                if any(kw in key for kw in attn_keywords):
                    skipped_attn += 1
                    continue
                if key in student_state and student_state[key].shape == teacher_state[key].shape:
                    student_state[key].copy_(teacher_state[key])
                    copied += 1
                else:
                    skipped_other += 1
            self.load_state_dict(student_state, strict=False)
            if copied == 0:
                raise RuntimeError(f'ffncopy init: no MLP parameter of {self.copy_ckpt_path} matches the model (check --model / the checkpoint size)')
            print(f'ffncopy: copied {copied} params, skipped {skipped_attn} attn + {skipped_other} other (head/shape/missing)')
            del ckpt, teacher_state


    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)
        last_index_to_take = max(take_indices)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks[: last_index_to_take + 1]):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            patch_size = self.patch_embed.patch_size
            batch, _, height, width = x.size()
            outputs = [
                out.reshape(batch, int(math.ceil(height / patch_size[0])), int(math.ceil(width / patch_size[1])), -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_device() -> str:
    """Device used by the Mimetic/Impulse initialization solver (GPU when available)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm
    
def get_ortho_like(dim, heads, alpha, beta, sign=1, dist='uniform'):
    if dist == 'normal':
        A = alpha * np.random.normal(size=(dim,dim)) / (dim**0.5) + sign * beta * np.eye(dim)
    if dist == 'uniform':
        A = alpha * np.random.uniform(size=(dim,dim), low=-3**0.5 / (dim**0.5), high = 3**0.5 / (dim**0.5))\
    + sign * beta * np.eye(dim)

    U, S, V = np.linalg.svd(A)
    L = U @ np.diag(np.sqrt(S))
    R = np.diag(np.sqrt(S)) @ V
    return L, R

def impulse_init(heads,img_size,att_rank,ff,scale=1.0,spatial_pe=None,norm=1):
    weight = torch.zeros((heads,img_size[0]*img_size[1],img_size[0]*img_size[1]))
    k = torch.randint(0,ff**2,(heads,))
    for i in range(heads):
        m = (k[i]//ff)-(ff//2)
        n = (k[i]%ff)-(ff//2)
        tmp_weight = torch.zeros((img_size[1],img_size[1]))
        for j in range(0-min(0,n),img_size[1]-max(0,n)):
            tmp_weight[j,j+n] = 1
        for j in range(0-min(0,m),img_size[0]-max(0,m)):
            weight[i,j*img_size[1]:(j+1)*img_size[1],(j+m)*img_size[1]:(j+m+1)*img_size[1]] = tmp_weight
    # weight = np.sqrt(1/3)*weight
    class PermuteM(nn.Module):
        def __init__(self, heads, att_size, att_rank,scale=1.0,spatial_pe=None):
            super().__init__()
            self.scale = scale
            if spatial_pe is None:
                self.spatial_pe = False
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,att_size,att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,att_rank,att_size)-1)
            else:
                self.spatial_pe = True
                self.pe = torch.nn.functional.layer_norm(spatial_pe.to(_init_device()),[spatial_pe.shape[1]])
                weights_Q = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads,spatial_pe.shape[1],att_rank)-1)
                weights_K = np.sqrt(1/att_rank/heads)*(2*torch.rand(heads, att_rank, spatial_pe.shape[1])-1)

            self.weights_K = nn.Parameter(weights_K)
            self.weights_Q = nn.Parameter(weights_Q)
        def forward(self):
            if self.spatial_pe:
                M = self.pe@self.weights_Q@self.weights_K@(self.pe.T)
            else:
                M = torch.bmm(self.weights_Q,self.weights_K)
            return torch.softmax(M*self.scale,-1)
    
    net = PermuteM(heads,img_size[0]*img_size[1],att_rank,scale,spatial_pe)
    net.to(_init_device())

    nq = net.weights_Q.detach().cpu().norm(dim=(1)).mean()
    weight = weight.to(_init_device())
    num_epoch = 10000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#,weight_decay=1e-6)
    for i in range(num_epoch):
        if i%norm==0:
            with torch.no_grad():
                net.weights_Q.div_(net.weights_Q.detach().norm(dim=(1),keepdim=True)/nq)
                net.weights_K.div_(net.weights_K.detach().norm(dim=(1),keepdim=True)/nq)
        optimizer.zero_grad()
        outputs = net()
        loss = criterion(outputs, weight)
        loss.backward()
        optimizer.step()
    print(loss.data)

    return net.weights_Q.detach().cpu(),net.weights_K.detach().cpu()


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb, gs_new, gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: VisionTransformer,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        state_dict = _convert_openai_clip(state_dict, model, prefix='module.visual.')
    elif "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    elif "encoder" in state_dict:
        # IJEPA, vit in an 'encoder' submodule
        state_dict = state_dict['encoder']
        prefix = 'module.'
    elif 'visual.trunk.pos_embed' in state_dict:
        # OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'vit_tiny_patch16_224': _cfg(),
    'vit_small_patch16_224': _cfg(),
    'vit_base_patch16_224': _cfg(),
})


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> VisionTransformer:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )


@register_model
def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (ViT-Ti/16): embed_dim 192, depth 12, 3 heads. """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    return _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16): embed_dim 384, depth 12, 6 heads. """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    return _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16): embed_dim 768, depth 12, 12 heads. """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
