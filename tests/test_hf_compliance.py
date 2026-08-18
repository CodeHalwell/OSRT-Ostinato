"""HF transformers compliance: save_pretrained / from_pretrained round-trip.

The model is HF-loadable when the osrt package is importable:
AutoModelForCausalLM.from_pretrained(dir) returns an OSRTForCausalLM whose
forward is bit-identical to the source, and generate() works.

Regression guard for the rope bug: rope_cos/sin must be PERSISTENT buffers.
They're derived from config, but from_pretrained builds the skeleton on the
meta device — non-persistent buffers materialise as uninitialised garbage,
silently corrupting RoPE on every reloaded model (forward diverged by ~0.9
with bit-identical weights before the fix).
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import OSRTForCausalLM  # noqa: E402


def _roundtrip(model):
    d = tempfile.mkdtemp()
    model.save_pretrained(d)
    return d


def test_rope_buffers_are_persistent():
    """Non-persistent rope → garbage after from_pretrained meta init."""
    m = OSRTForCausalLM(tiny_config())
    sd = m.state_dict()
    assert "model.rope_cos" in sd, "rope_cos must be persistent (in state_dict)"
    assert "model.rope_sin" in sd, "rope_sin must be persistent (in state_dict)"


def test_from_pretrained_forward_parity():
    torch.manual_seed(0)
    m = OSRTForCausalLM(tiny_config()).eval()
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        out1 = m(input_ids=x).logits
    d = _roundtrip(m)
    m2 = OSRTForCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        out2 = m2(input_ids=x).logits
    assert torch.allclose(out1, out2, atol=1e-5), (out1 - out2).abs().max().item()


def test_automodel_roundtrip():
    from transformers import AutoModelForCausalLM
    torch.manual_seed(1)
    m = OSRTForCausalLM(tiny_config()).eval()
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        out1 = m(input_ids=x).logits
    d = _roundtrip(m)
    m2 = AutoModelForCausalLM.from_pretrained(d).eval()
    assert isinstance(m2, OSRTForCausalLM)
    with torch.no_grad():
        out2 = m2(input_ids=x).logits
    assert torch.allclose(out1, out2, atol=1e-5)


def test_generate_on_reloaded_model():
    m = OSRTForCausalLM(tiny_config()).eval()
    d = _roundtrip(m)
    m2 = OSRTForCausalLM.from_pretrained(d).eval()
    x = torch.randint(0, 512, (1, 6))
    g = m2.generate(x, max_new_tokens=4)
    assert g.shape[1] == 6 + 4


def test_auto_map_written():
    """save_pretrained writes auto_map + copies the modeling/config files
    (the trust_remote_code hook)."""
    import json
    m = OSRTForCausalLM(tiny_config())
    d = _roundtrip(m)
    cfg = json.load(open(os.path.join(d, "config.json")))
    assert "auto_map" in cfg
    assert cfg["model_type"] == "osrt"


def test_gradient_checkpointing_not_hf_advertised():
    """We manage checkpointing via the private _osrt_grad_ckpt gate, so HF's
    mechanism must NOT be advertised (it trips post_init and isn't what runs)."""
    assert OSRTForCausalLM.supports_gradient_checkpointing is False
