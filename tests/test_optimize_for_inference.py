"""optimize_for_inference() — the decode-speed inference prep (Idea #1).

Guards the property that makes it safe: turning MoE/loop telemetry OFF must not
change the model's outputs. The compile() half is a torch.compile fusion (no
math change, verified bit-exact on GPU: max|Δlogit|=0, 100% greedy token-match)
and isn't exercised here — CPU inductor is slow/flaky in CI and adds no signal
beyond the telemetry gate, which is the only thing that touches forward outputs.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import OSRTForCausalLM  # noqa: E402


def _fixed_input(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.real_vocab_size, (2, 16), generator=g)


def test_telemetry_off_is_output_identical():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg).eval()
    x = _fixed_input(cfg)

    with torch.no_grad():
        ref = model(x).logits.clone()  # telemetry ON (default)
        model.set_moe_telemetry(False)
        off = model(x).logits.clone()  # telemetry OFF

    assert torch.equal(ref, off), (
        f"telemetry gate changed logits: max|Δ|={(ref - off).abs().max()}"
    )


def test_optimize_for_inference_no_compile_matches_eager():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg)
    x = _fixed_input(cfg)

    with torch.no_grad():
        ref = model.eval()(x).logits.clone()
        ret = model.optimize_for_inference(compile_model=False)
        opt = model(x).logits.clone()

    assert ret is model                       # returns self for chaining
    assert model.training is False            # eval() applied
    assert torch.equal(ref, opt)              # outputs unchanged


def test_optimize_for_inference_disables_telemetry_on_all_blocks():
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg)
    model.optimize_for_inference(compile_model=False)
    for blk in model.model.blocks:
        assert blk.moe.telemetry_enabled is False


def test_generate_actually_compiles_via_fwd():
    """Regression guard for the 'compile bypassed in generate()' bug: a bare
    model.compile() wraps __call__, but generate() dispatches through
    self._fwd, which must hit self._compiled_forward. Prove a graph compiled
    by routing self._compiled_forward through a counting backend and asserting
    generate() triggered it (frame_count > 0). CompileCounter is a no-codegen
    backend, so this is fast and CPU-safe."""
    from torch._dynamo.testing import CompileCounter

    torch.manual_seed(0)
    model = OSRTForCausalLM(tiny_config()).eval()
    model.set_moe_telemetry(False)
    cnt = CompileCounter()
    model._compiled_forward = torch.compile(model.forward, backend=cnt)

    ids = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        model.generate(ids, max_new_tokens=3, temperature=0.0)

    assert cnt.frame_count > 0, (
        "generate() did not exercise the compiled forward — compile is bypassed"
    )


def test_prepack_matches_per_call_stack_and_is_nonpersistent():
    """prepack_expert_weights must build exactly the tensors _grouped_ffn
    builds per call (stack fp32 -> transpose -> cast bf16), and must NOT
    change the checkpoint layout (non-persistent buffers)."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(tiny_config())
    moe = model.model.blocks[0].moe
    keys_before = set(model.state_dict())

    moe.prepack_expert_weights()

    ref_gate = torch.stack(
        [e.w_gate.weight.t() for e in moe.experts]).to(torch.bfloat16)
    ref_down = torch.stack(
        [e.w_down.weight.t() for e in moe.experts]).to(torch.bfloat16)
    assert torch.equal(moe._packed_w_gate, ref_gate)
    assert torch.equal(moe._packed_w_down, ref_down)
    assert moe._packed_w_gate.dtype == torch.bfloat16
    # state_dict unchanged -> old checkpoints load strict into a prepacked model
    assert set(model.state_dict()) == keys_before


def test_optimize_for_inference_prepacks_every_block():
    model = OSRTForCausalLM(tiny_config())
    model.optimize_for_inference(compile_model=False)
    for blk in model.model.blocks:
        assert getattr(blk.moe, "_packed_w_up", None) is not None


def test_prepack_invalidated_by_train_and_load():
    """Stale-pack guards: .train(True) and load_state_dict must drop the packs
    (weights are about to change / just changed), and eval() must NOT."""
    model = OSRTForCausalLM(tiny_config())
    moe = model.model.blocks[0].moe

    moe.prepack_expert_weights()
    model.train()
    assert moe._packed_w_gate is None          # train() invalidates

    moe.prepack_expert_weights()
    model.eval()
    assert moe._packed_w_gate is not None      # eval() keeps the packs

    sd = model.state_dict()
    model.load_state_dict(sd, strict=True)
    assert moe._packed_w_gate is None          # new weights invalidate


def test_fwd_falls_back_to_eager_when_not_optimized():
    """Without optimize_for_inference, _fwd must be the plain eager forward
    (bit-identical) — the default training/eval path is untouched."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(tiny_config()).eval()
    x = _fixed_input(model.config)
    with torch.no_grad():
        assert model._compiled_forward is None
        a = model.forward(x).logits
        b = model._fwd(x).logits
    assert torch.equal(a, b)
