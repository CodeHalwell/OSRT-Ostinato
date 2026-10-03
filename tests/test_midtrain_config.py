"""MidtrainConfig: one phase, weights sum to one, schedule and init fields."""
from osrt.data import FORMAT_FN_PRETRAIN
from osrt.train_config import MidtrainConfig, PretrainConfig


def test_midtrain_config_is_one_phase_with_valid_entries():
    cfg = MidtrainConfig()
    assert list(cfg.phases) == ["midtrain"]
    ph = cfg.phases["midtrain"]
    assert ph["start"] == 0 and ph["end"] == cfg.total_steps
    assert abs(sum(d["weight"] for d in ph["datasets"]) - 1.0) < 1e-6
    for d in ph["datasets"]:
        fmt = d.get("format")
        assert fmt is None or fmt in FORMAT_FN_PRETRAIN, d["name"]
    assert cfg.total_tokens() == cfg.total_steps * 6 * 11 * 4096
    assert cfg.lr_schedule == "cosine" and cfg.peak_lr == 2e-4
    assert cfg.router_gumbel_tau_init == 0.0
    assert cfg.eval_seq_len == 4096 and cfg.eval_batch_size == 6
    assert cfg.init_weights_path == "" and PretrainConfig().init_weights_path == ""
    assert "init_weights_path" not in __import__("osrt.train", fromlist=["x"]) \
        ._STRICT_TRAIN_RECIPE_KEYS
