"""OSRT — Optimized Sparse Recursive Transformer.

A recursive sparse-MoE language model: 3 physical blocks applied over 6 loops
(18 effective layers), 1 shared + 8 routed (top-2) experts per block, HRA
adapters, Muon-optimized. Target build ~608M physical / ~279M active.

Top-level imports expose the current architecture. Earlier versions (v1/v2/v3
dense recursive, v4 MoE with dense FFN crutch, v5 ~363M) are preserved under
`archive/` for reference but not importable here.
"""

from osrt.config import OSRTConfig
from osrt.model import (
    ExpertFFN,
    MoELayer,
    OSRTForCausalLM,
    OSRTModel,
    OSRTPreTrainedModel,
    RecursiveBlock,
    orthogonal_expert_init,
)
from osrt.quant import (
    QuantizedKV,
    dequantize_kv_latent,
    quantize_kv_latent,
)

__all__ = [
    "ExpertFFN",
    "MoELayer",
    "OSRTConfig",
    "OSRTForCausalLM",
    "OSRTModel",
    "OSRTPreTrainedModel",
    "QuantizedKV",
    "RecursiveBlock",
    "dequantize_kv_latent",
    "orthogonal_expert_init",
    "quantize_kv_latent",
]
__version__ = "0.6.0"
