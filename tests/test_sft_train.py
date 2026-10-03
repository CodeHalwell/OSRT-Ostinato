"""SFT trainer plumbing on CPU: frozen base, frozen routing, LR schedule."""
import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.sft_train import _lr_at, build_sft_model
from osrt.train_config import SFTProbeConfig


def _tiny() -> OSRTConfig:
    return OSRTConfig(
        dim=64, heads=2, head_dim=32, num_kv_heads=1,
        vocab_size=128, real_vocab_size=100,
        num_blocks=1, recursive_loops=2,
        num_routed_experts=4, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=64,
        use_hra=False, max_position_embeddings=64,
        router_bias_in_gates=True, mtp_loss_weight=0.0,
    )


def test_build_sft_model_trains_adapters_only_and_freezes_routing(tmp_path):
    torch.manual_seed(0)
    base = OSRTForCausalLM(_tiny())
    path = tmp_path / "base.pt"
    torch.save({"model_state_dict": base.state_dict(), "step": 1}, path)
    cfg = SFTProbeConfig(hra_rank=4)
    model, hra = build_sft_model(_tiny(), cfg, str(path), torch.device("cpu"))
    n_hra = sum(p.numel() for p in hra)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == n_hra
    for m in model.modules():
        if hasattr(m, "balance_accum_enabled"):
            assert m.balance_accum_enabled is False
        if hasattr(m, "gumbel_tau"):
            assert float(m.gumbel_tau) == 0.0
    # Zero-initialised B: the adapted model computes the base's forward.
    ids = torch.randint(0, 100, (2, 16))
    base.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, base(ids).logits,
                                   atol=1e-5, rtol=1e-5)


def test_lr_schedule_warms_up_then_decays_to_lr_min():
    cfg = SFTProbeConfig(total_steps=100, warmup_steps=10, lr=1e-3, lr_min=1e-4)
    assert _lr_at(0, cfg) < _lr_at(9, cfg) <= cfg.lr
    assert abs(_lr_at(10, cfg) - cfg.lr) < 1e-9
    assert abs(_lr_at(99, cfg) - cfg.lr_min) < 2e-6
    mid = _lr_at(55, cfg)
    assert cfg.lr_min < mid < cfg.lr
