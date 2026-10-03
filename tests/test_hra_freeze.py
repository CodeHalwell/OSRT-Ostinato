"""`inject_hra(freeze_pretrained=True)` must freeze the WHOLE base.

Regression for the 2026-10-03 review finding: on the v7 preset the old code
left 80,585,091 base parameters trainable (tied embedding, MTP heads, routers,
norms) and `get_param_groups` put them in a "pretrained" optimiser group.
"""
import torch

from osrt.hra import get_param_groups, inject_hra
from osrt.model import OSRTForCausalLM
from osrt.presets import OSRT_V7, build_config


def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_frozen_base_trains_only_adapters_on_the_v7_preset():
    cfg = build_config(
        vocab_size=49280, real_vocab_size=OSRT_V7["real_vocab_size"],
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    with torch.device("meta"):
        model = OSRTForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    hra = inject_hra(model, rank=256, freeze_pretrained=True)
    n_hra = sum(p.numel() for p in hra)
    assert _trainable(model) == n_hra
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if "adapter" not in n)
    # Nothing but adapters is trainable, so the embedding, MTP heads, routers
    # and norms are all frozen.
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert frozen == total
    groups = get_param_groups(model, hra, base_lr=1e-5, hra_lr=1e-4)
    assert [g["group_name"] for g in groups] == ["hra"]
    assert sum(p.numel() for p in groups[0]["params"]) == n_hra


def test_unfrozen_injection_keeps_the_base_trainable():
    cfg = build_config(
        vocab_size=49280, real_vocab_size=OSRT_V7["real_vocab_size"],
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    with torch.device("meta"):
        model = OSRTForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    hra = inject_hra(model, rank=256, freeze_pretrained=False)
    assert _trainable(model) == total + sum(p.numel() for p in hra)
    groups = get_param_groups(model, hra, base_lr=1e-5, hra_lr=1e-4)
    assert [g["group_name"] for g in groups] == ["pretrained", "hra"]
