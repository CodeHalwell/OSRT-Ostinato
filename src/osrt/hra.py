"""High Rank Adaptation (HRA) for expanding model capacity during post-training.

Injects high-rank adapter matrices (A @ B) alongside existing linear layers.
The pretrained weights are preserved; new capacity comes from the adapters.

Forward: y = Linear(x) + scale * (x @ A @ B)

Unlike LoRA (rank 4-64), HRA uses rank 128-512 to add substantial learning
capacity rather than parameter-efficient fine-tuning.

Usage:
    model = RecursiveOSRT(cfg).to(device)
    load_pretrained(model, checkpoint_path, device)
    hra_params = inject_hra(model, rank=256)

    # Differential LR: lower for pretrained, higher for HRA
    optimizer = AdamW([
        {"params": base_params, "lr": 2e-5},
        {"params": hra_params, "lr": 1e-4},
    ])
"""

import torch
import torch.nn as nn
from torch import Tensor


class HRALinear(nn.Module):
    """Linear layer with a parallel high-rank adapter.

    Wraps an existing nn.Linear and adds a trainable A @ B path.
    The original weight can optionally be frozen.

    Args:
        original: The existing nn.Linear to wrap.
        rank: Rank of the adapter (A is in_features x rank, B is rank x out_features).
        scale: Scaling factor for the adapter output.
        freeze_original: If True, freeze the original linear weight.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        scale: float = 1.0,
        freeze_original: bool = False,
    ) -> None:
        super().__init__()
        self.original = original
        self.scale = scale

        in_f = original.in_features
        out_f = original.out_features
        device = original.weight.device
        dtype = original.weight.dtype

        # Kaiming init for A, zero init for B (adapter starts as identity)
        self.adapter_a = nn.Parameter(
            torch.randn(in_f, rank, device=device, dtype=dtype) * (2.0 / in_f) ** 0.5
        )
        self.adapter_b = nn.Parameter(
            torch.zeros(rank, out_f, device=device, dtype=dtype)
        )

        if freeze_original:
            for p in self.original.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.original(x)
        hra_out = (x @ self.adapter_a) @ self.adapter_b
        return base_out + self.scale * hra_out

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    @property
    def weight(self) -> Tensor:
        return self.original.weight

    @property
    def bias(self) -> Tensor | None:
        return self.original.bias


def inject_hra(
    model: nn.Module,
    rank: int = 256,
    scale: float = 1.0,
    freeze_pretrained: bool = False,
    target_modules: tuple[str, ...] = (
        "q_proj", "kv_down", "v_from_k", "out_proj", "w_gate", "w_up", "w_down",
    ),
) -> list[nn.Parameter]:
    """Inject HRA adapters into all target linear layers in the model.

    Replaces target nn.Linear modules with HRALinear wrappers.
    Returns the list of new HRA parameters (for separate optimizer group).

    Args:
        model: The model to inject into.
        rank: Adapter rank. 256 adds ~11M params, 512 adds ~22M.
        scale: Scaling factor for adapter outputs.
        freeze_pretrained: If True, freeze original weights (only train adapters).
        target_modules: Names of Linear modules to wrap.

    Returns:
        List of new HRA parameters.
    """
    hra_params = []
    replacements = []

    # Collect replacements (can't modify dict during iteration)
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_modules:
                replacements.append((module, child_name, child))

    for parent, child_name, original_linear in replacements:
        hra_linear = HRALinear(
            original_linear,
            rank=rank,
            scale=scale,
            freeze_original=freeze_pretrained,
        )
        setattr(parent, child_name, hra_linear)
        hra_params.append(hra_linear.adapter_a)
        hra_params.append(hra_linear.adapter_b)

    n_hra = sum(p.numel() for p in hra_params)
    n_layers = len(replacements)
    print(f"  HRA injected: {n_layers} layers, rank {rank}, "
          f"+{n_hra:,} params ({n_hra / 1e6:.1f}M)")
    if freeze_pretrained:
        n_frozen = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        print(f"  Pretrained weights frozen: {n_frozen:,} params")

    return hra_params


def get_param_groups(
    model: nn.Module,
    hra_params: list[nn.Parameter],
    base_lr: float,
    hra_lr: float,
    weight_decay: float = 0.1,
) -> list[dict]:
    """Create differential learning rate param groups.

    Args:
        model: The full model.
        hra_params: HRA adapter parameters (from inject_hra).
        base_lr: Learning rate for pretrained parameters.
        hra_lr: Learning rate for HRA parameters (typically 5-10x base).
        weight_decay: Weight decay for both groups.

    Returns:
        List of param group dicts for the optimizer.
    """
    hra_ids = {id(p) for p in hra_params}
    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in hra_ids
    ]

    groups = []
    if base_params:
        groups.append({
            "params": base_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "group_name": "pretrained",
        })
    groups.append({
        "params": hra_params,
        "lr": hra_lr,
        "weight_decay": weight_decay,
        "group_name": "hra",
    })

    print(f"  Optimizer groups: pretrained ({len(base_params)} tensors, lr={base_lr}) "
          f"+ HRA ({len(hra_params)} tensors, lr={hra_lr})")

    return groups
