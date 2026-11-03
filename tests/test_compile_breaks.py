"""The training forward must compile as ONE graph on the grouped-GEMM path.

A graph break per MoE layer (18 per micro-batch) left half the step's GPU
time in eager glue (profiled on a B200, 2026-11-02).
"""
import torch
import torch._dynamo as dynamo

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM


def test_training_forward_has_no_graph_breaks_on_grouped_path():
    dynamo.reset()
    dynamo.config.capture_scalar_outputs = True
    dynamo.config.capture_dynamic_output_shape_ops = True
    cfg = OSRTConfig(
        dim=64, heads=2, head_dim=32, num_kv_heads=1, vocab_size=128,
        real_vocab_size=100, num_blocks=1, recursive_loops=2,
        num_routed_experts=4, top_k_experts=2, expert_hidden=64,
        shared_expert_hidden=64, use_hra=False, max_position_embeddings=64,
        moe_grouped_gemm=True,
    )
    model = OSRTForCausalLM(cfg)
    model.train()
    model.set_moe_telemetry(False)
    ids = torch.randint(0, 100, (2, 16))
    ex = dynamo.explain(lambda x: model(x, labels=x).loss)(ids)
    assert ex.graph_break_count == 0, [str(r.reason)[:120] for r in ex.break_reasons]
    assert ex.graph_count == 1
