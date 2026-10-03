"""Decode-speed ladder for a v7 checkpoint on one H100 (Modal).

Measures greedy generation, batch 1 unless stated, over the same prompts in
four modes, each warmed up first, and reports tok/s plus agreement with the
eager run (continuations compared up to and including the first stop):

  eager      plain Python decode loop, latent KV cache (what lm-eval used)
  compiled   optimize_for_inference(): telemetry off, prepacked experts,
             torch.compile(fullgraph, dynamic) of forward; latent cache
  graphs     + reduce_overhead=True and generate(cache_impl="static"):
             static KV cache, CUDA-graph decode step
  mtp-spec   compiled latent + speculative decoding from the two MTP heads
  eager-b8   eager, the prompt repeated 8x (throughput, not latency)

The inductor cache persists on the volume, so a second run skips the cold
compile. Pulls the checkpoint from the private HF mirror when it is not on
the volume (same rules as scripts/lm_eval_trunk.py).

    MODAL_PROFILE=inference-syn uv run modal run --detach scripts/bench_decode.py \\
        --ckpt osrt-v7-sft-probe_merged_step_1000.pt --hf-subdir sft --chat
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "TORCHINDUCTOR_CACHE_DIR": "/vol/inductor_cache"})
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.3.0", "triton==3.6.0", "tokenizers==0.22.2",
        "safetensors==0.7.0", "huggingface_hub>=0.35", "numpy",
    )
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "src"), "/root/src")
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "tokenizer"), "/root/tokenizer")
)
vol = modal.Volume.from_name("osrt-v7-ladder-ckpt", create_if_missing=True)
app = modal.App("osrt-bench-decode", image=image)

PLAIN_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Question: A bakery sells 12 muffins per tray and bakes 7 trays. "
    "How many muffins is that?\nAnswer:",
    "In 1905, Albert Einstein published",
]
CHAT_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "A bakery sells 12 muffins per tray and bakes 7 trays. How many muffins "
    "is that? Think step by step and give the final number.",
    "Explain in three sentences what Albert Einstein published in 1905.",
]


@app.function(gpu="H100", timeout=3600, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("hf-secret")])
def bench(ckpt: str, hf_subdir: str, hf_repo: str, chat: bool,
          max_new_tokens: int, legacy_gates: bool, skip_graphs: bool) -> str:
    import os
    import time

    import torch

    sys.path.insert(0, "/root/src")
    from transformers import AutoTokenizer

    from osrt.chat_format import END_TURN, render_chat
    from osrt.model import OSRTForCausalLM
    from osrt.presets import OSRT_V7, build_config

    vol.reload()
    path = f"/vol/{hf_subdir}/{ckpt}"
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        print(f"[bench] pulling {hf_subdir}/{ckpt} from {hf_repo}", flush=True)
        path = hf_hub_download(hf_repo, f"{hf_subdir}/{ckpt}", local_dir="/root/ckpt")

    tok = AutoTokenizer.from_pretrained("/root/tokenizer")
    real = len(tok)
    padded = ((real + 127) // 128) * 128
    assert real == OSRT_V7["real_vocab_size"], real
    cfg = build_config(
        vocab_size=padded, real_vocab_size=real,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, router_bias_in_gates=legacy_gates,
    )
    model = OSRTForCausalLM(cfg)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del ck, sd
    model = model.to("cuda", dtype=torch.bfloat16).eval()
    n_params = sum(p.numel() for p in model.parameters())
    lines = [f"[bench] {ckpt} | params {n_params:,} | missing={len(missing)} "
             f"unexpected={len(unexpected)} | H100 bf16 | max_new={max_new_tokens} "
             f"| torch {torch.__version__}"]
    print(lines[-1], flush=True)

    eos = tok.eos_token_id
    end_turn = tok.convert_tokens_to_ids(END_TURN)
    stop_ids = [end_turn] if chat else None
    prompts = CHAT_PROMPTS if chat else PLAIN_PROMPTS
    texts = [render_chat([{"role": "user", "content": p}], add_generation_prompt=True)
             if chat else p for p in prompts]
    from osrt.chat_format import encode_with_markers
    prompt_ids = [torch.tensor([encode_with_markers(tok, t)], device="cuda")
                  for t in texts]

    def to_stop(seq: list[int]) -> list[int]:
        stops = {eos} | set(stop_ids or [])
        for i, t in enumerate(seq):
            if t in stops:
                return seq[: i + 1]
        return seq

    def run_mode(name: str, **kw) -> dict:
        """Warm up on every prompt (compile / graph capture), then time."""
        t_warm = time.perf_counter()
        for ids in prompt_ids:
            model.generate(ids, max_new_tokens=8, temperature=0.0,
                           stop_token_ids=stop_ids, **kw)
        for ids in prompt_ids:
            model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0,
                           stop_token_ids=stop_ids, **kw)
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - t_warm
        outs, tot_tok, tot_s = [], 0, 0.0
        for ids in prompt_ids:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0,
                                 stop_token_ids=stop_ids, **kw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            new = out[0, ids.shape[1]:].tolist()
            outs.append(to_stop(new))
            tot_tok += len(new)
            tot_s += dt
        stats = getattr(model, "last_spec_stats", None) if "speculative" in kw else None
        r = {"name": name, "tps": tot_tok / tot_s, "tokens": tot_tok, "s": tot_s,
             "warm_s": warm_s, "outs": outs, "spec": stats}
        extra = (f" | accept {stats['acceptance_rate']:.2f} tok/fwd "
                 f"{stats['tokens_per_forward']:.2f}") if stats else ""
        line = (f"  {name:10} {r['tps']:7.1f} tok/s  ({tot_tok} tok in {tot_s:.1f}s; "
                f"warmup {warm_s:.0f}s){extra}")
        lines.append(line)
        print(line, flush=True)
        return r

    results = {}
    results["eager"] = run_mode("eager")
    base_outs = results["eager"]["outs"]
    for name, o in zip(prompts, base_outs):
        s = f"    eager> {name[:40]!r}: {tok.decode(o)[:120]!r}"
        lines.append(s)
        print(s, flush=True)

    # batch-8 throughput, eager (identical rows -> no padding needed)
    ids8 = prompt_ids[1].repeat(8, 1)
    model.generate(ids8, max_new_tokens=8, temperature=0.0, stop_token_ids=stop_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out8 = model.generate(ids8, max_new_tokens=max_new_tokens, temperature=0.0,
                          stop_token_ids=stop_ids)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n8 = (out8.shape[1] - ids8.shape[1]) * 8
    line = f"  {'eager-b8':10} {n8 / dt:7.1f} tok/s  ({n8} tok in {dt:.1f}s; 8 rows)"
    lines.append(line)
    print(line, flush=True)

    model.optimize_for_inference(compile_model=True, reduce_overhead=not skip_graphs)
    results["compiled"] = run_mode("compiled")
    if not skip_graphs:
        try:
            results["graphs"] = run_mode("graphs", cache_impl="static")
        except Exception as exc:  # noqa: BLE001
            line = f"  graphs     FAILED: {type(exc).__name__}: {str(exc)[:300]}"
            lines.append(line)
            print(line, flush=True)
    try:
        results["mtp-spec"] = run_mode("mtp-spec", speculative=True,
                                       spec_drafter="mtp")
    except Exception as exc:  # noqa: BLE001
        line = f"  mtp-spec   FAILED: {type(exc).__name__}: {str(exc)[:300]}"
        lines.append(line)
        print(line, flush=True)

    for name, r in results.items():
        if name == "eager":
            continue
        same = sum(int(a == b) for a, b in zip(r["outs"], base_outs))
        first_div = []
        for a, b in zip(r["outs"], base_outs):
            d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), -1)
            first_div.append(d)
        line = (f"  {name:10} identical_to_stop vs eager {same}/{len(base_outs)} "
                f"| first divergence {first_div}")
        lines.append(line)
        print(line, flush=True)
    summary = " | ".join(f"{k} {v['tps']:.1f}" for k, v in results.items())
    lines.append(f"TOTAL tok/s: {summary}")
    print(lines[-1], flush=True)
    out_dir = "/vol/bench"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{Path(ckpt).stem}_decode.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def main(ckpt: str = "osrt_soup_17000_17500_18000.pt", hf_subdir: str = "trunk",
         hf_repo: str = "HallD/OSRT-Ostinato-trunk", chat: bool = False,
         max_new_tokens: int = 128, legacy_gates: bool = True,
         skip_graphs: bool = False):
    print(bench.remote(ckpt, hf_subdir, hf_repo, chat, max_new_tokens,
                       legacy_gates, skip_graphs))
