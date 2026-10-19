"""Dual-stream ViT for Attention Distillation.

Teacher (frozen softmax DeiT) + Student (linear attention ViT) run in parallel.
MSE loss on per-layer attention outputs guides the student to match teacher attention.

Usage:
    model = DualVisionTransformer(
        teacher_ckpt='path/to/deit_small.pth',
        distill_lambda=3.0,
        student_kwargs=dict(attn_type='elu1', weight_init='copy', ...),
    )
    # Forward returns (student_logits, distill_loss)
    logits, distill_loss = model(images)
    loss = ce_loss(logits, targets) + distill_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer, Attention


class AttentionOutputHook:
    """Hook to capture attention output from each block."""

    def __init__(self):
        self.outputs = []
        self.handles = []

    def register(self, model):
        """Register forward hooks on all attention modules."""
        self.outputs = []
        for block in model.blocks:
            handle = block.attn.register_forward_hook(self._hook_fn)
            self.handles.append(handle)

    def _hook_fn(self, module, input, output):
        self.outputs.append(output)

    def get_and_clear(self):
        outputs = self.outputs
        self.outputs = []
        return outputs

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class AttentionInputHook:
    """Hook to capture the INPUT tokens of each block's attention module.

    Used by the map-based distillation losses (ce_map / mse_map), which
    recompute the attention map A from the stored input with the module's
    own weights. Keeps the module forward untouched.
    """

    def __init__(self):
        self.inputs = []
        self.handles = []
        self.modules = []

    def register(self, model):
        self.inputs = []
        self.modules = []
        for block in model.blocks:
            self.modules.append(block.attn)
            handle = block.attn.register_forward_hook(self._hook_fn)
            self.handles.append(handle)

    def _hook_fn(self, module, input, output):
        self.inputs.append(input[0])

    def get_and_clear(self):
        inputs = self.inputs
        self.inputs = []
        return inputs

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _attention_map(module, x):
    """Recompute the row-stochastic attention map of `module` on input x.

    Supports the softmax `Attention` (teacher) and `LinearAttention` with
    elu1/relu kernels (student): A = softmax(qk^T) or phi(q)phi(k)^T
    row-normalized. Returns [B, H, N, N].
    """
    B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    if type(module).__name__ == 'InLineAttention':
        # Subtraction normalization (identity kernel): rows sum to 1 exactly,
        # entries may be negative. A = (s/N) q k^T + (1 - s q.kbar)/N.
        qk = q @ k.transpose(-2, -1)                          # [B, H, N, N]
        kbar = k.mean(dim=2, keepdim=True)                    # [B, H, 1, d]
        qkbar = (q * kbar).sum(dim=-1, keepdim=True)          # [B, H, N, 1]
        attn = module.scale / N * qk + (1.0 - module.scale * qkbar) / N
    elif hasattr(module, '_phi'):  # LinearAttention
        q = module._phi(q)
        k = module._phi(k)
        scores = q @ k.transpose(-2, -1)                      # [B, H, N, N], >= 0
        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
    else:  # softmax Attention
        attn = (q * module.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
    return attn


class DualVisionTransformer(nn.Module):
    """Dual-stream model: frozen softmax teacher + trainable linear student.

    Args:
        teacher_ckpt: Path to pretrained DeiT checkpoint (softmax attention)
        distill_lambda: Weight for distillation loss
        student_kwargs: kwargs passed to VisionTransformer for the student
        teacher_kwargs: kwargs for teacher (default: same as student but softmax)
        model_name: timm entrypoint (e.g. 'vit_small_patch16_224') that fixes embed_dim / depth / num_heads
            for both networks; without it the kwargs must carry these sizes themselves
    """

    def __init__(
        self,
        teacher_ckpt: str,
        distill_lambda: float = 3.0,
        distill_loss_type: str = 'mse_out',
        student_kwargs: dict = None,
        teacher_kwargs: dict = None,
        model_name: str = None,
    ):
        super().__init__()
        self.distill_lambda = distill_lambda
        if distill_loss_type not in ('mse_out', 'ce_map', 'mse_map'):
            raise ValueError(f'Unknown distill_loss_type: {distill_loss_type}')
        self.distill_loss_type = distill_loss_type

        def _build(kwargs):
            if model_name:
                from timm.models import create_model
                return create_model(model_name, pretrained=False, **kwargs)
            return VisionTransformer(**kwargs)

        # Build student (linear attention)
        self.student = _build(student_kwargs)

        # Build teacher (softmax attention, always CLS token + learned PE)
        if teacher_kwargs is None:
            teacher_kwargs = dict(student_kwargs)
            teacher_kwargs['attn_type'] = 'softmax'
            teacher_kwargs['weight_init'] = 'skip'
            teacher_kwargs.pop('copy_ckpt_path', None)
            # Teacher always uses DeiT-native config regardless of student config
            teacher_kwargs['class_token'] = True
            teacher_kwargs['no_embed_class'] = False
            teacher_kwargs['sin_pe'] = False
        self.teacher = _build(teacher_kwargs)

        # Load teacher weights (skip head and any shape-mismatched keys)
        ckpt = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        teacher_state = ckpt['model'] if 'model' in ckpt else ckpt
        model_state = self.teacher.state_dict()
        filtered = {k: v for k, v in teacher_state.items()
                    if k in model_state and model_state[k].shape == v.shape}
        if not filtered:
            raise RuntimeError(
                f'Teacher checkpoint {teacher_ckpt} matches no parameter of the teacher model: '
                f'the model size (--model / embed_dim, depth, num_heads) does not fit the checkpoint.')
        msg = self.teacher.load_state_dict(filtered, strict=False)
        print(f'Teacher loaded from {teacher_ckpt}: {len(filtered)} params loaded, {len(msg.missing_keys)} missing')
        del ckpt, teacher_state, filtered

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # Register hooks to capture attention outputs (mse_out)
        # or attention-module inputs (ce_map / mse_map).
        if self.distill_loss_type == 'mse_out':
            self.teacher_hook = AttentionOutputHook()
            self.student_hook = AttentionOutputHook()
        else:
            for block in self.student.blocks:
                attn = block.attn
                # Maps supported: softmax Attention, elu1/relu LinearAttention
                # (row-stochastic), and InLineAttention for mse_map only —
                # its subtraction-normalized map has negative entries, so CE
                # is ill-posed on it. MHLA (windowed) / TTT (no map) fail loudly.
                name = type(attn).__name__
                ok = (name == 'Attention') or (
                    hasattr(attn, '_phi') and getattr(attn, 'kernel', None) in ('elu1', 'relu')) or (
                    name == 'InLineAttention' and self.distill_loss_type == 'mse_map')
                if not ok:
                    raise ValueError(
                        f'{self.distill_loss_type} unsupported for student attention '
                        f'{name} (InLine allows mse_map only; MHLA/TTT have no map)')
            self.teacher_hook = AttentionInputHook()
            self.student_hook = AttentionInputHook()
        self.teacher_hook.register(self.teacher)
        self.student_hook.register(self.student)

    def forward(self, x):
        """Forward pass.

        Returns:
            logits: Student classification output [B, num_classes]
            distill_loss: MSE distillation loss (scalar), already scaled by lambda
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            self.teacher.eval()
            _ = self.teacher(x)
        teacher_attn_outs = self.teacher_hook.get_and_clear()

        # Student forward
        logits = self.student(x)
        student_attn_outs = self.student_hook.get_and_clear()

        distill_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.distill_lambda > 0:
            if self.distill_loss_type == 'mse_out':
                # MSE on per-layer attention outputs
                num_layers = min(len(teacher_attn_outs), len(student_attn_outs))
                for i in range(num_layers):
                    t_out = teacher_attn_outs[i].detach()
                    s_out = student_attn_outs[i]
                    # Handle sequence length mismatch: teacher has CLS (N+1), student may not (N)
                    if t_out.shape[1] > s_out.shape[1]:
                        t_out = t_out[:, 1:, :]  # strip teacher CLS token
                    elif s_out.shape[1] > t_out.shape[1]:
                        s_out = s_out[:, :t_out.shape[1], :]
                    # Handle embed dim mismatch (e.g. TTT has different output dim)
                    if s_out.shape[-1] != t_out.shape[-1]:
                        min_dim = min(s_out.shape[-1], t_out.shape[-1])
                        distill_loss = distill_loss + F.mse_loss(s_out[..., :min_dim], t_out[..., :min_dim])
                    else:
                        distill_loss = distill_loss + F.mse_loss(s_out, t_out)
            else:
                # ce_map / mse_map: per-layer loss on attention MAPS,
                # recomputed from the captured module inputs.
                num_layers = min(len(teacher_attn_outs), len(student_attn_outs))
                for i in range(num_layers):
                    with torch.no_grad():
                        t_map = _attention_map(self.teacher.blocks[i].attn,
                                               teacher_attn_outs[i].detach())
                    s_map = _attention_map(self.student.blocks[i].attn,
                                           student_attn_outs[i])
                    # Align token count: strip teacher CLS row+col if student has none
                    if t_map.shape[-1] > s_map.shape[-1]:
                        t_map = t_map[:, :, 1:, 1:]
                        t_map = t_map / (t_map.sum(dim=-1, keepdim=True) + 1e-6)
                    if self.distill_loss_type == 'mse_map':
                        distill_loss = distill_loss + F.mse_loss(s_map, t_map)
                    else:  # ce_map: row-wise cross-entropy, mean over B/H/rows
                        distill_loss = distill_loss + (-(t_map * torch.log(s_map + 1e-8)).sum(dim=-1)).mean()
            distill_loss = self.distill_lambda * distill_loss

        return logits, distill_loss

    def get_classifier(self):
        return self.student.get_classifier()

    def reset_classifier(self, num_classes, global_pool=''):
        self.student.reset_classifier(num_classes, global_pool)

    @property
    def num_classes(self):
        return self.student.num_classes
