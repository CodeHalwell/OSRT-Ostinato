"""Batched left-padded generation must match per-sequence generation.

Eval batches heterogeneous-length prompts; correctness requires that a prompt
generated inside a left-padded batch (with an attention_mask) yields the SAME
greedy tokens as that prompt generated alone. The padding + RoPE positions must
be invisible to the real tokens.
"""

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM


def _tiny_model() -> tuple[OSRTForCausalLM, OSRTConfig]:
    cfg = OSRTConfig(
        dim=128, heads=4, head_dim=32,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = OSRTForCausalLM(cfg).eval()
    return model, cfg


# eos out of real vocab → never argmax'd → no early stop, exactly N new tokens.
_NO_EOS = 10**9
_N = 6


@torch.no_grad()
def test_left_padded_batch_matches_per_sequence_greedy():
    model, cfg = _tiny_model()
    pad = 0

    # Two prompts of different lengths.
    prompt_a = [11, 22, 33, 44]          # len 4
    prompt_b = [5, 6, 7, 8, 9, 10]       # len 6

    def gen_single(ids: list[int]) -> list[int]:
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=_N, temperature=0.0,
                             eos_token_id=_NO_EOS)
        return out[0, len(ids):len(ids) + _N].tolist()

    ref_a = gen_single(prompt_a)
    ref_b = gen_single(prompt_b)

    # Left-pad A up to B's length; mask marks real tokens (1) vs pad (0).
    width = len(prompt_b)
    padded_a = [pad] * (width - len(prompt_a)) + prompt_a
    batch = torch.tensor([padded_a, prompt_b], dtype=torch.long)
    attn = torch.tensor(
        [[0] * (width - len(prompt_a)) + [1] * len(prompt_a),
         [1] * len(prompt_b)],
        dtype=torch.long,
    )

    out = model.generate(batch, attention_mask=attn, max_new_tokens=_N,
                         temperature=0.0, eos_token_id=_NO_EOS)

    batched_a = out[0, width:width + _N].tolist()
    batched_b = out[1, width:width + _N].tolist()

    assert batched_a == ref_a, f"row A: batched {batched_a} != single {ref_a}"
    assert batched_b == ref_b, f"row B: batched {batched_b} != single {ref_b}"


@torch.no_grad()
def test_three_row_batch_varied_padding_matches_per_sequence():
    """Generalises beyond two rows: three prompts with three different pad
    widths must each match their per-sequence greedy completion."""
    model, cfg = _tiny_model()
    pad = 0
    prompts = [[7, 8], [11, 22, 33, 44], [1, 2, 3, 4, 5]]   # lens 2, 4, 5

    def gen_single(ids):
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=_N, temperature=0.0,
                             eos_token_id=_NO_EOS)
        return out[0, len(ids):len(ids) + _N].tolist()

    refs = [gen_single(p) for p in prompts]

    width = max(len(p) for p in prompts)
    batch = torch.tensor(
        [[pad] * (width - len(p)) + p for p in prompts], dtype=torch.long,
    )
    attn = torch.tensor(
        [[0] * (width - len(p)) + [1] * len(p) for p in prompts],
        dtype=torch.long,
    )
    out = model.generate(batch, attention_mask=attn, max_new_tokens=_N,
                         temperature=0.0, eos_token_id=_NO_EOS)

    for i, ref in enumerate(refs):
        got = out[i, width:width + _N].tolist()
        assert got == ref, f"row {i} (len {len(prompts[i])}): {got} != {ref}"
