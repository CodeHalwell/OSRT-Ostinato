# Inference, KV Cache & Speculative Decoding

> Part of the OSRT-605M `docs/` architecture series. This chapter explains how
> the trained model actually turns a prompt into text: the two phases of
> generation (prefill and decode), the unusual *latent-only* KV cache (KDV,
> Key-Derived Value), the CPU-GPU-sync-aware standard decode loop, sampling,
> and a **greedy-only** speculative-decode accelerator. It cross-references
> `docs/02-attention.md` (the attention sub-block / MLA latent),
> `docs/06-recursion.md` (the 3-blocks-×-6-loops depth recurrence),
> `ARCHITECTURE.md` §12 (inference) and §13 (KV cache), and the shipping
> implementation in `src/osrt/model.py`.

---

## 1. Purpose — how the trained model generates text

OSRT-605M is an autoregressive decoder: given a sequence of token ids it
produces a probability distribution over the next token, and text is built one
token at a time by repeatedly feeding the model its own output. Everything in
this document is about doing that **fast** without changing what the model would
have predicted token-for-token.

The naïve way to generate is to re-run the full forward over the entire
sequence-so-far at every step. That is O(N²) total work for an N-token
generation, because step *t* re-attends over all *t* prior tokens from scratch.
The standard fix is a **KV cache**: keep the per-layer key/value tensors of
every token you have already processed, so each new step only computes attention
for the single new token against the cached past. That turns each decode step
from O(t) into O(1) amortised model work (plus an O(t) attention read).

The entry point is `OSRTForCausalLM.generate()` at
`src/osrt/model.py:1841`. Its docstring (`src/osrt/model.py:1856-1864`) states
the cached decode is roughly 3× faster than the non-cached path for a 256-token
generation on this architecture. Defaults are *IFEval-safe*: greedy
(`temperature=0.0`), no repetition penalty (`src/osrt/model.py:1844-1850`).

Two architectural quirks of OSRT make its inference path different from a vanilla
transformer, and both get their own section below:

1. **The cache stores only a compressed latent, not K and V.** OSRT uses an
   MLA-inspired attention where K and V are both *linear functions of one cached
   latent* `c_kv` (see `docs/02-attention.md` and `ARCHITECTURE.md` §6.2-6.3).
   We cache `c_kv` and rebuild K (with RoPE) and V on the fly. V in particular
   is the **Key-Derived Value (KDV)**: a single learned
   `Linear(512→512)+bias` (`v_from_k`) reading the cached latent and producing
   V. That is ~half a normal GQA cache.
2. **Depth comes from recursion, not distinct layers.** The model runs 3
   physical decoder blocks 6 times (`docs/06-recursion.md`). Each
   *(loop, block)* pair is an **effective layer** with its own cache slot, so the
   cache has `num_blocks × recursive_loops = 3 × 6 = 18` slots — and the variable
   loop-count knob (§8) changes how many of those slots exist.

---

## 2. Prefill vs decode — the two phases

Every `generate()` call has exactly two phases.

**Prefill** — one full forward over the whole prompt, with `use_cache=True`:

```python
context = input_ids[:, -self.config.max_position_embeddings:]
out = self.forward(context, use_cache=True, num_loops=num_loops)
past_key_values = cast("PastKV | None", out.past_key_values)
```
(`src/osrt/model.py:1917-1919`)

Note the prompt is left-truncated to `max_position_embeddings` (4096 in the
shipped `configs/osrt-605m-a279m/config.json`) before prefill. Prefill is the
expensive, parallel phase: all prompt positions are processed at once, and it
seeds the cache for every effective layer. The logits at the *last* prompt
position are the model's prediction for the first generated token.

**Decode** — one forward per new token, feeding only the newest token plus the
cache:

```python
new_tok = generated[:, cursor - 1:cursor]
out = self.forward(
    new_tok,
    past_key_values=past_key_values,
    use_cache=True,
    num_loops=num_loops,
)
```
(`src/osrt/model.py:1957`, `1972-1977`)

Each decode forward processes a single-token sequence (`S == 1`), reads the
cached past, and returns an updated cache one position longer. This is the O(1)
amortised step that the cache buys us.

The same `num_loops` is threaded through **both** phases. This is not optional
bookkeeping: the cache layout is loop-count-specific (it has
`num_blocks × n_loops` slots), so prefill and every decode step must agree on the
loop count or the per-effective-layer slots would misalign. The docstring at
`src/osrt/model.py:1879-1883` makes this explicit, and `forward()` enforces it by
validating `len(past_key_values) == num_blocks * n_loops_to_run`
(`src/osrt/model.py:1361-1370`).

---

## 3. The KV cache — latent-only design

### 3.1 What's actually stored

A normal GQA transformer caches two tensors per layer per token: the rotated
keys K and the values V. OSRT caches **one** tensor per effective layer per
token: the *un-rotated* compressed latent `c_kv`.

In the attention sub-block the hidden state is projected down to a single latent:

```python
c_kv_new = self.kv_down(h)            # (B, S, kv_dim) — un-rotated latent
```
(`src/osrt/model.py:1000`)

with `kv_dim = num_kv_heads × head_dim = 8 × 64 = 512` (config:
`num_kv_heads=8`, `head_dim=64`). This 512-wide latent is the **only** thing the
cache holds. The cache returned to `generate()` is a plain Python list of
per-effective-layer tensors, each shaped `(B, seq_len, 512)`.

### 3.2 The 18 slots and the index formula

The cache is built in `OSRTModel.forward()`. Slots are appended in
*(loop, block)* order:

```python
for loop in range(n_loops_to_run):
    for block_idx, block in enumerate(self.blocks):
        idx = loop * self.config.num_blocks + block_idx
        ...
        if presents is not None:
            presents.append(present_kv)
```
(`src/osrt/model.py:1461-1463`, `1503-1504`)

So the cache slot for the *block_idx*-th block on the *loop*-th iteration is

```
idx = loop * num_blocks + block_idx
```

With `num_blocks = 3` and `recursive_loops = 6` that is 18 slots: `(loop 0,
block 0) → 0`, `(loop 0, block 1) → 1`, …, `(loop 5, block 2) → 17`. Each loop
genuinely recomputes a fresh latent — the input to loop *r* is loop *r-1*'s
output, so the 18 latents are 18 different representations, not 6 copies (see
`ARCHITECTURE.md` §13.4 and `docs/06-recursion.md`). If `num_loops=K` is set
(§8), only the first `K × num_blocks` slots exist, and that is exactly what
`forward()` validates the incoming cache against
(`src/osrt/model.py:1361`, `expected_past_layers = num_blocks * n_loops_to_run`).

### 3.3 Why un-rotated, and how K and V are recovered

The cache deliberately stores the latent **before** RoPE is applied. The code
comment spells out why both K and V must be rebuilt from the un-rotated latent
every step (`src/osrt/model.py:1002-1004`):

```python
# The cache holds ONLY the un-rotated latent. K and V are recomputed
# from the full latent every step: RoPE is positional and KDV
# (Key-Derived Value) must operate on un-rotated K, so neither may
# be cached rotated.
```

There are two reasons, and they compound:

- **RoPE is positional.** A key rotated for absolute position *p* is only valid
  at position *p*. If we cached rotated K we could not re-derive anything; storing
  the un-rotated latent lets us apply RoPE freshly over the whole span at
  attention time.
- **V is a linear function of the un-rotated latent — the KDV (Key-Derived Value) contract.** OSRT does not store V at
  all. It derives V from the latent with a learned affine map
  `v_from_k` (`src/osrt/model.py:921`, `1015`); this is what we call
  **Key-Derived Value (KDV)**: V at each token is *derived from* its
  cached key latent by a single `Linear(512→512)+bias`:

```python
k = c_kv.view(B, total_len, self.kv_heads, self.head_dim)
v = self.v_from_k(c_kv).view(B, total_len, self.kv_heads, self.head_dim)
```
(`src/osrt/model.py:1013-1015`)

K is just the latent reshaped (then QK-normed and RoPE'd at attention time); V is
`W·c_kv + b` over the same latent (the **KDV / Key-Derived Value** affine map).
Because both K and V are linear in `c_kv`, the single cached latent loses no
expressivity relative to caching K and V separately — it's the same trick as
DeepSeek MLA's shared `c_KV` (`src/osrt/model.py:909-917`; see
`docs/02-attention.md` §"V derived from K / KDV").

The cache update itself is a concatenation along the sequence axis
(`src/osrt/model.py:1005-1009`):

```python
if past_key_value is not None:
    c_kv = torch.cat([past_key_value, c_kv_new], dim=1)  # (B, L+S, kv_dim)
else:
    c_kv = c_kv_new
present_kv = c_kv if use_cache else None
```

### 3.4 The memory win

Per token per effective layer the cache holds a single 512-d latent. Against a
hypothetical GQA baseline that caches K **and** V (2 × 512 floats), this is
exactly **half**. `ARCHITECTURE.md` §13.2-13.3 works the numbers: at BF16 the
K-only baseline is 18 layers × 512 × 2 bytes = **18 KB/token**, vs ~36 KB/token
for K+V. (`ARCHITECTURE.md` §13.2's worked examples quote an 8K context and an
`eos_token_id` of 1 from an earlier draft; the shipped 605M config uses
`max_position_embeddings=4096` and `eos_token_id=2`, so treat those §13.2/§12.1
constants as illustrative, not as the deployed values.)

---

## 4. Why the cache isn't trimmed

A tempting optimisation when a generation runs long is to left-truncate the cache
(drop the oldest positions) to bound memory. OSRT **deliberately does not** do
this, and the reasoning is the most important RoPE subtlety in the inference
path. From the decode loop (`src/osrt/model.py:1958-1971`):

```python
# Don't trim past_key_values when the cache exceeds
# max_position_embeddings — left-truncating the cache
# shifts the absolute RoPE indices that the forward
# derives from past_key_values[idx].shape[1] (the
# past_length read in OSRTModel.forward), so cached K
# (rotated at original absolute positions) and the new K
# (rotated at the post-trim shifted index) end up in
# different positional bases and attention breaks.
```

The mechanism: `forward()` infers each token's absolute position **from the cache
length itself** — `past_length` is read off `past_key_values[idx].shape[1]`
(`src/osrt/model.py:1379-1387`), and the new query rotates at
`[past_len:total_len]` while K rebuilds over `[0:total_len]`
(`src/osrt/model.py:1020-1028`). If you trimmed *L* positions off the front, the
surviving cached latents were produced as un-rotated K destined for positions
*L, L+1, …*, but the next forward would now believe the cache starts at position
0 and rotate the new token relative to a shifted base. Cached K (effectively
anchored to its original positions when re-rotated) and new K would live in
incompatible positional frames, and attention would silently corrupt.

Letting the cache grow unbounded is safe because `forward()` recomputes RoPE
on demand when the required length exceeds the precomputed table
(`src/osrt/model.py:1392-1408`, the `else` branch of the `rope_cos` slice). The
code notes that if growth ever becomes a real constraint, the correct fix is a
sliding window **with re-rotation**, not a naïve trim (`src/osrt/model.py:1969-1971`).

> Note the asymmetry with the speculative path: that path *does* slice its cache
> (§7.3), but only along the sequence axis to drop a **stale speculative tail
> that was never committed** — it never drops committed prefix positions, so the
> absolute-position argument is not violated.

---

## 5. Standard autoregressive decode

The standard decode loop (`src/osrt/model.py:1954-2072`) is greedy/sampling
agnostic and has been deliberately rewritten to minimise both memory bandwidth
and CPU-GPU synchronisation. On a GPU, anything that forces the host to read a
value back from the device (a `.item()`, a `bool(tensor)`, a Python-side
branch on tensor contents) stalls the pipeline: the CPU must wait for the GPU to
finish so it can read the result. A decode loop runs hundreds of times, so each
per-step sync is paid hundreds of times. The whole loop is engineered to avoid
them.

### 5.1 Preallocated output buffer + cursor

The old pattern grew the output with `torch.cat` per token, paying
O(prompt + step) memory bandwidth on **every** step. The new code preallocates
the full output once and writes in place (`src/osrt/model.py:1946-1952`):

```python
total_len = input_ids.shape[1] + max_new_tokens
generated = torch.zeros(
    batch_size, total_len,
    dtype=input_ids.dtype, device=input_ids.device,
)
generated[:, :input_ids.shape[1]] = input_ids
cursor = input_ids.shape[1]
```

Each step writes the new token in place and bumps the cursor
(`src/osrt/model.py:2048-2049`):

```python
generated[:, cursor:cursor + 1].copy_(next_token)
cursor += 1
```

`cursor` tracks the next-write position. Because the tail of `generated` is
zero-filled until written, anything that reads the sequence-so-far must slice
`generated[:, :cursor]` — both the repetition penalty and the final return do
this so the zero padding never leaks into observable output
(`src/osrt/model.py:1996`, `2070-2072`).

### 5.2 The finished mask — `torch.where`, no per-step sync

Each row of the batch must stop independently once it emits EOS or a stop token,
but the loop must not synchronise to check that on every step. The solution is a
per-row boolean `finished` mask kept entirely on-device
(`src/osrt/model.py:1931-1933`). Finished rows are *forced* to keep emitting EOS
via a vectorised `torch.where` (no host branch) so the output tensor stays
rectangular (`src/osrt/model.py:2038-2043`):

```python
if eos_token_id is not None:
    next_token = torch.where(
        finished.unsqueeze(-1),
        torch.full_like(next_token, eos_token_id),
        next_token,
    )
```

The mask is updated with pure tensor ops (`src/osrt/model.py:2062-2066`):

```python
nt = next_token.squeeze(-1)
if eos_token_id is not None:
    finished = finished | (nt == eos_token_id)
if stop_tensor is not None:
    finished = finished | torch.isin(nt, stop_tensor)
```

The **only** per-step host sync is the all-finished early break
(`src/osrt/model.py:2067`), `bool(finished.all())` — and that is the one we
actually want, since it lets us stop early when every row is done. (It is a real
sync, but a single cheap scalar read, not a per-row `.any()` scan.)

### 5.3 Stop tokens — `torch.isin` against a precomputed tensor

Callers can pass `stop_token_ids` (e.g. chat-template markers like `<|/answer|>`
or `<|user|>`) to terminate cleanly on those (`src/osrt/model.py:2057-2061`).
The list is turned into a device tensor **once** before the loop
(`src/osrt/model.py:1922-1924`):

```python
stop_tensor = None
if stop_token_ids:
    stop_tensor = torch.tensor(list(stop_token_ids), device=input_ids.device)
```

and membership is tested vectorially with `torch.isin(nt, stop_tensor)` — no
Python `in` over a list, no host round-trip per token.

### 5.4 Vectorised repetition penalty

Repetition penalty is off by default (`repetition_penalty=1.0`). When enabled it
is fully vectorised with gather/scatter — there is **no** Python loop over
previously-seen tokens (`src/osrt/model.py:1992-2010`):

```python
already = generated[:, :cursor]
gen_clamped = already.clamp(max=vocab - 1)
in_vocab = (already < vocab)
score = torch.gather(logits_last, 1, gen_clamped)
penalised = torch.where(
    score > 0,
    score / repetition_penalty,
    score * repetition_penalty,
)
penalised = torch.where(in_vocab, penalised, score)
logits_last = logits_last.clone()
logits_last.scatter_(1, gen_clamped, penalised)
```

`gather` pulls each previously-seen token's current logit, scales it (divide if
positive, multiply if negative — the standard CTRL-style penalty), and scatters
it back. Duplicate ids write the same scaled value, so last-write-wins on
`scatter_` is semantically identical to "apply once per unique id". Two subtleties
the code handles: the slice to `[:, :cursor]` avoids penalising token 0 from the
zero-filled tail (`src/osrt/model.py:1994-1996`), and the `in_vocab` mask writes
the original score back for any clamped out-of-vocab index so scatter cannot
corrupt a real logit (`src/osrt/model.py:2005-2008`).

---

## 6. Sampling — temperature / top-p / top-k

With `temperature == 0` the loop is pure greedy: `logits_last.argmax(...)`
(`src/osrt/model.py:2032-2033`). With `temperature > 0` it runs the standard
top-k → top-p (nucleus) → multinomial pipeline (`src/osrt/model.py:2012-2031`):

```python
next_logits = logits_last / temperature
if top_k > 0:
    topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
    next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")
if top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumprobs - sorted_probs >= top_p
    sorted_logits[sorted_mask] = float("-inf")
    next_logits.scatter_(1, sorted_indices, sorted_logits)
probs = F.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

Temperature scales the logits; top-k restricts to the *k* highest; top-p keeps
the smallest set of tokens whose cumulative probability first reaches `top_p`
(the `cumprobs - sorted_probs` form keeps the token that crosses the threshold).
The order is top-k then top-p, both applied before the final softmax + sample.
Logits are sliced to `:real_vocab_size` (`src/osrt/model.py:1936`, `1981`) so any
padding ids in the embedding table can never be emitted.

---

## 7. Speculative decoding — greedy accelerator

> **Honest caveat up front.** OSRT's speculative path is a **greedy throughput
> trick, not distribution-preserving speculative sampling.** It is provably
> identical to plain greedy (temperature-0) decoding from the full model, and it
> is *not* a valid sampler for `temperature > 0`. The code says so itself in a
> boxed docstring banner (`src/osrt/model.py:2087-2100`). There is no
> accept/reject correction step. Use it for greedy / very-low-temperature
> generation only.

`generate(speculative=True)` routes to `_generate_speculative()`
(`src/osrt/model.py:1901-1910`, `2075`). The idea exploits OSRT's recursion: a
*cheap draft* is produced by running only the first `spec_draft_loops` loops
(3 by default, config `spec_draft_loops=3`), and the *expensive verifier* runs
the full loop count. The per-loop aux LM-head training (`docs/06-recursion.md`,
`ARCHITECTURE.md` §9.2) is what makes the low-loop draft predictive of the
full-loop output, so the draft and verifier agree often.

### 7.1 The draft/verify/commit round

Each round (`src/osrt/model.py:2206-2335`) does three things:

1. **DRAFT** — autoregressively greedy-decode `D = spec_draft_tokens` tokens, one
   at a time, at the cheap draft loop count, advancing a draft-side cache
   (`src/osrt/model.py:2209-2227`).
2. **VERIFY** — run **one** full-loop forward over `[pending, draft_0, …,
   draft_{D-1}]` (D+1 tokens). The verifier's logits at position *i* predict the
   token following the *i*-th input, so position *i<D* re-predicts the slot
   `drafts[i]` fills, and position *D* is a free **bonus** token for after the
   block (`src/osrt/model.py:2229-2245`).
3. **COMMIT** — accept the longest prefix where the draft matches the verifier's
   greedy token, emit the verifier's correction at the first mismatch, and on a
   full accept additionally emit the bonus token.

`draft_loops` is clamped so the drafter never runs more compute than the verifier
(`src/osrt/model.py:2146-2147`): `draft_loops = min(spec_draft_loops, full_loops)`.

### 7.2 Greedy argmax verify and commit matching

When `repetition_penalty == 1.0` the verifier's predictions for all D+1 positions
are a **single batched argmax** — no per-position Python loop
(`src/osrt/model.py:2248-2249`):

```python
verify_preds = v_logits[:, :D + 1, :self.config.real_vocab_size].float().argmax(dim=-1)
```

(When a repetition penalty is active it falls back to a per-position greedy that
threads the penalty exactly as the standard loop, so the two paths agree
token-for-token — `src/osrt/model.py:2250-2257`.)

The accept count comes from a vectorised prefix match across the whole batch
(`src/osrt/model.py:2259-2270`):

```python
all_match = (drafts == verify_preds[:, :D]).all(dim=0)  # (D,)
mismatches = (~all_match).nonzero(as_tuple=True)[0]
accept = int(mismatches[0].item()) if mismatches.numel() > 0 else D
new_cols = [drafts[:, i:i + 1] for i in range(accept)]
new_cols.append(verify_preds[:, accept:accept + 1])  # correction OR bonus
```

`all_match` is the per-position AND across rows (conservative but correct for
batch > 1 — a position is accepted only if *every* row agrees), and
`(~all_match).nonzero()` finds the first mismatch. `new_cols` is the accepted
drafts plus one always-correct token: on a mismatch it's the verifier's
correction at slot `accept`; on a full accept it's the bonus at slot `D`. Either
way at least one token is committed per round, and a fully-accepted round commits
`D + 1` tokens for the cost of one verifier forward — that is the speed-up. The
final return trims a possible one-token overshoot from a full-accept round's bonus
(`src/osrt/model.py:2337-2340`).

### 7.3 Two caches and why this one *can* be sliced

The cache is loop-count-specific, so the speculative path keeps **two**: a draft
cache at `draft_loops` and a verify cache at `full_loops`
(`src/osrt/model.py:2118-2129`, `2191-2203`). After committing, the stale
speculative tail is dropped by slicing each per-layer latent along the sequence
axis (`src/osrt/model.py:2133-2141`, `2308-2310`):

```python
keep = cache_len + accept + 1
verify_past = _trunc(verify_past_full, keep)
draft_past = _trunc(draft_past, min(keep, cache_len + D))
```

This is allowed precisely because it only drops positions **past the acceptance
point** — speculative latents that were never committed. An accepted draft equals
the verifier's prediction (the actually-committed token), so the kept latents are
exactly the committed-token latents. No committed prefix position is ever removed,
so the §4 absolute-position argument is not violated. The verifier feeds D+1
inputs while the drafter feeds only D, so on a full accept the draft cache is one
short and a single extra forward re-adds the missing latent
(`src/osrt/model.py:2316-2327`). The whole re-establishment of the cache invariant
is done by truncation (plus that rare one-token extend), with no extra full
forward.

> Correctness relies on MoE capacity drops being disabled in eval
> (`MoELayer.forward` keys off `self.training`), so the one-shot parallel verify
> reproduces exactly what a token-by-token decode would have produced — making the
> acceptance check exact (`src/osrt/model.py:2126-2129`). Callers must
> `model.eval()`.

### 7.4 Reconciling with `ARCHITECTURE.md` §12.3

`ARCHITECTURE.md` §12.3 sketches the same loop-as-draft idea and quotes an
expected accept rate of 60-75% and ~1.8-2.4× net speed-up. Two honest
clarifications against the shipped code: (a) the doc's pseudocode reuses a single
`kv_cache`, whereas the implementation keeps **separate** draft/verify caches
because the cache is loop-count-specific; and (b) the doc frames "accept matching
prefix" without stressing that there is **no probabilistic accept/reject** — the
shipped routine is explicitly greedy-only, as its banner states.

---

## 8. `num_loops` — variable test-time compute

`num_loops` is an inference knob (`ARCHITECTURE.md` §12.2;
`docs/06-recursion.md`) that runs fewer than the trained 6 loops at every step
for a speed/quality trade-off. It is validated by `_resolve_num_loops()`
(`src/osrt/model.py:1293-1310`): `None` → `config.recursive_loops` (full
quality, the default, bit-identical to before); otherwise `K` must lie in
`[1, recursive_loops]` and only the first `K` loops run.

The crucial constraint for inference is that the **same** `K` is threaded through
prefill and every decode step (§2), because the cache has `num_blocks × K` slots
and `forward()` validates the incoming cache against exactly that count
(`src/osrt/model.py:1361`). Mixing loop counts mid-generation would misalign the
per-effective-layer cache and is therefore disallowed. In the speculative path
`K` also caps the draft loops (`src/osrt/model.py:2146-2147`). `ARCHITECTURE.md`
§12.2 gives the indicative trade-off: loops=3 ≈ 2× faster at ~85% quality, up to
loops=6 = baseline full quality.

---

## 9. Performance notes — what's optimised, what's deferred

**Already optimised in the standard decode loop:**

- **Preallocated buffer + cursor** replaces per-token `torch.cat`, eliminating
  O(prompt+step) per-step copies (`src/osrt/model.py:1946-1952`, §5.1).
- **No per-step `.any()` sync.** The finished mask is updated and applied with
  `torch.where` / boolean ops entirely on-device; the only host read is the
  single all-finished early break (`src/osrt/model.py:2038-2067`, §5.2).
- **Precomputed stop tensor + `torch.isin`** instead of a Python membership test
  per token (`src/osrt/model.py:1922-1924`, `2065-2066`, §5.3).
- **Vectorised repetition penalty** (gather/scatter, no Python loop over tokens)
  (`src/osrt/model.py:1992-2010`, §5.4).
- **RoPE recomputed on demand** beyond the precomputed table so the cache can
  grow without a trim (`src/osrt/model.py:1392-1408`, §4).
- **Speculative verify uses a single batched argmax** when no repetition penalty
  is active, and re-establishes its cache by truncation rather than re-forwarding
  (`src/osrt/model.py:2248-2249`, `2294-2327`, §7).

**Deferred / left to a GPU-specific pass (cross-ref `docs/02-attention.md`):**

- **Attention is flash SDPA — the sink (and its fused-kernel follow-up) is
  retired by default.** The shipping preset sets `attention_sink=False`
  (`src/osrt/presets.py:54`), so every attention call goes through fused
  `F.scaled_dot_product_attention` and the score matrix is never materialised
  (`src/osrt/model.py:1177-1183`). The learnable per-head attention sink
  (`ARCHITECTURE.md` §6.6) needed an extra term in the softmax denominator that
  SDPA cannot express, so when it was on it routed to a manual log-sum-exp
  rescale path (`_attention_with_sink`, `src/osrt/model.py:1187-1251`) that
  materialised the full `(B, H, S, total_len)` score matrix per head. At the
  seq-8192 instruction phase that matrix (~12 GB at batch 2, *recomputed* in the
  gradient-checkpointed backward) OOMed the run (>85 GB); flash never builds the
  score matrix, so the same seq-8192 / batch-2 config fits at 35.9 GB — which is
  why the sink was dropped (see `docs/02-attention.md` §6.3). The
  `flex_attention(return_lse=True)` "flash + lse" route (the natural way to keep
  a fused kernel *and* recover the log-sum-exp the sink rescale needs) was
  investigated but rejected for the current target (torch 2.12, CPU): without
  `torch.compile` it materialises the full score matrix anyway, emits a
  `return_lse` deprecation warning, and needs a custom `mask_mod` for GQA
  (`_attention_with_sink` docstring, `src/osrt/model.py:1207-1215`). With the
  sink off none of this is on the critical path; the manual path and the
  fused-kernel follow-up matter only if `attention_sink` is ever re-enabled.
- **Fused cross-entropy.** The training-time losses use stock
  `F.cross_entropy` (`src/osrt/model.py:1678`, `1727`, `1780`); a fused CE is a
  training-throughput item, not an inference one, and is out of scope for the
  decode loop.

The non-speculative `temperature=0` decode is bit-identical to the pre-optimisation
implementation; the optimisations changed *how* the loop runs (memory traffic and
sync points), not *what* it predicts.

---

## 10. Putting it together — a worked decode

For a greedy generation of `max_new_tokens` from a prompt of length `P`:

1. **Prefill:** one forward over the (≤ `max_position_embeddings`) prompt with
   `use_cache=True`; seeds 18 latent cache slots (or `3×K` under `num_loops=K`).
   Logits at position `P-1` give token 0.
2. **Decode step *t*:** feed the single newest token + cache → updated cache
   (one position longer) + logits; apply repetition penalty (if any) over
   `generated[:, :cursor]`; `argmax` (or sample); force EOS on finished rows;
   write in place at `cursor`; update the on-device `finished` mask.
3. **Stop** when every row has emitted EOS or a stop token (the one host sync), or
   at `max_new_tokens`.
4. **Return** `generated[:, :cursor]` so the zero-filled tail never escapes.

With `speculative=True`, steps 2-3 are replaced by draft/verify/commit rounds
that commit 1…D+1 greedy-identical tokens per verifier forward — same emitted
text as greedy decode, fewer expensive forwards.

---

### Cross-references

- `docs/02-attention.md` — GQA + MLA latent, KDV (Key-Derived Value), RoPE, attention sink.
- `docs/06-recursion.md` — 3 blocks × 6 loops, loop embeddings, per-loop aux head.
- `ARCHITECTURE.md` §6 (attention), §12 (inference), §13 (KV cache).
- `src/osrt/model.py` — `generate()` (1841), `_generate_speculative()` (2075),
  `_attention()` (979), `OSRTModel.forward()` cache loop (1461).
