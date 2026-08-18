# Output Heads & Training Losses

> Part of the OSRT-605M `docs/` architecture series. This chapter explains how
> the model turns its final hidden state into token predictions, and the full
> stack of training losses that shape it: the tied LM head and its
> cross-entropy task loss, the per-loop auxiliary heads, the multi-token
> prediction (MTP) heads, and the three MoE router losses — plus exactly how
> they are normalized and summed into one scalar.

A note on sourcing. Where this document states a *mechanic* it cites
`src/osrt/model.py`, `src/osrt/config.py`, and `src/osrt/presets.py` by line —
that is the source of truth. `ARCHITECTURE.md` is cited only for *intent* and
*history*. Its §9/§11 pseudocode is explicitly "illustrative ... the
implementation in `model.py` is the source of truth," and in a few places the
prose has drifted from the code. **When they disagree, the code wins, and the
discrepancy is flagged.** Two such drifts are called out below (MTP offsets, and
"MTP heads add no params").

Param counts come from `scripts/compute_budget.py`, which builds the canonical
preset `OSRT_605M_A288M` on a meta device and counts real parameters.

---

## 1. Purpose — heads convert hidden states into predictions and gradient

The body of the model (recursion + MoE + attention; see chapters 02–06)
produces, for each position, a single `dim=1536` hidden vector. By itself that
vector predicts nothing — it has to be projected into the 65,536-token
vocabulary to become logits, and those logits have to be scored against the
labels to produce a gradient. That is the job of the **heads** and **losses**.

OSRT does not stop at one head and one loss. At training time it stacks several
prediction heads on top of the body, each producing its own cross-entropy, plus
three router-health losses that come out of the MoE layers. The single
deployed inference path is tiny — *one* tied LM head — but during training the
model is supervised much more densely:

- **The tied LM head** — the real output. Predicts the next token (+1). The
  only head used at inference.
- **Per-loop aux LM heads** — the same tied head, re-applied to each
  intermediate recursive loop. Anti-loop-collapse. Adds **0 parameters**.
- **MTP heads** — predict tokens at +2 and +3. Densify the training signal.
  Training-time only; **dropped at deploy**.
- **Router losses** — `balance_loss`, `z_loss`, `seq_balance_loss`. Keep the
  MoE router healthy. Computed inside the MoE, summed across loops, normalized,
  and added with small coefficients.

A useful mental model: the inference graph is *narrow* (one head), but the
training graph is *wide* — eight full-vocab logit tensors and three scalar
router penalties all flowing gradient into the same shared body. Everything
beyond the +1 LM head exists to make that shared body learn faster and more
robustly, then is discarded.

Everything in this chapter except the main task CE is **train-only**: the entire
auxiliary sum is gated behind `if self.training` and the eval loss collapses to
pure task cross-entropy (`src/osrt/model.py:1793-1804`). More on that in §7.

---

## 2. The tied LM head and the main task loss

### 2.1 Weight tying

OSRT has **no dedicated output projection matrix**. The LM head reuses the input
embedding's weight — classic weight tying:

```python
# src/osrt/model.py:1657-1658
# Weight-tied LM head
logits = F.linear(hidden, self.model.embedding.weight)
```

`F.linear(x, W)` computes `x @ W.T`. With `W = embedding.weight` of shape
`(vocab=65536, dim=1536)`, this maps the `(B, S, 1536)` hidden state to
`(B, S, 65536)` logits. The *same* `65536 × 1536` matrix that turns token IDs
into vectors at the input turns vectors back into token scores at the output.

Why tie? The embedding matrix is the single largest tensor in the model —
`compute_budget.py` reports the `embedding` category (input embedding + the tied
LM head, counted once) at **100,690,944 params** (~100.7M), ~17% of the 601M
physical budget. An untied LM head would *double* that to ~200M for no quality
gain at this scale. Tying also couples "what a token means as input" to "what
predicting that token requires," which is a mild but real inductive prior.

The hidden state fed in (`hidden`) is the final loop's residual stream after the
dedicated mHC collapse and `norm_out` (`src/osrt/model.py:1533-1535`); see
chapter 04 for the mHC collapse and chapter 06 for the recursion that produces
it.

### 2.2 The main task loss (next-token cross-entropy)

```python
# src/osrt/model.py:1675-1682
shift_logits = logits[..., :-1, :self.config.real_vocab_size]
shift_logits = shift_logits.contiguous().float()
shift_labels = labels[..., 1:].contiguous()
task_loss = F.cross_entropy(
    shift_logits.view(-1, self.config.real_vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

Four details worth understanding:

1. **The shift.** Position `i`'s logits must predict token `i+1`, so we drop the
   last logit (`[:-1]`) and the first label (`[1:]`). Standard causal-LM
   teacher forcing.
2. **`real_vocab_size`.** Logits are sliced to `:real_vocab_size` before the
   loss. In the canonical preset `vocab_size == real_vocab_size == 65536`
   (`src/osrt/presets.py:27-28`), so the slice is a no-op there; it exists so a
   padded-vocab config (allocated vocab > real tokens) never trains the model to
   emit pad-only ids.
3. **`.float()`.** The CE is computed in fp32 even under bf16 autocast. The
   `(B,S,65536)` softmax is numerically delicate; fp32 keeps it stable. This is
   also *the* dominant activation-memory cost — see §8.
4. **`ignore_index=-100`.** Positions labelled `-100` (prompt prefixes during
   SFT, padding) contribute no loss and no gradient. The body still *sees* those
   positions as context; it just isn't asked to predict them.

`task_loss` is the only term present in the eval loss. Everything below is added
on top, at training time only.

---

## 3. Per-loop auxiliary LM-head losses — the loop-collapse fix

### 3.1 The problem

OSRT runs its 3 physical blocks **6 times** (`recursive_loops=6`). Without
intervention, gradient flows cleanly only to the *last* loop — the one feeding
the LM head — and the model learns to do all real prediction work in loop 6,
treating loops 1–5 as trivial pass-throughs. The recursion stops buying depth.
This is **loop collapse** (diagnosed in `probe_recursion`, 2026-06-05; full
treatment in chapter 06).

### 3.2 The fix: re-apply the tied head to every intermediate loop

During the forward pass, `OSRTModel` captures the residual stream at the end of
each non-final loop:

```python
# src/osrt/model.py:1524-1527
if capture_aux and loop < n_loops_to_run - 1:
    # Collapse the mHC stream to a single vector for the aux head
    intermediate_hiddens.append(self._collapse(x) if self.use_mhc else x)
```

The LM wrapper then runs each captured hidden through `norm_out` + the **same
tied embedding** and computes CE against the **same shifted labels**:

```python
# src/osrt/model.py:1721-1732
for i, h_loop in enumerate(intermediate_hiddens):
    h_norm = self.model.norm_out(h_loop)
    h_logits = F.linear(h_norm, self.model.embedding.weight)
    h_shift = h_logits[..., :-1, :self.config.real_vocab_size].contiguous().float()
    aux_l = F.cross_entropy(
        h_shift.view(-1, self.config.real_vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    per_loop_aux.append(aux_l)
```

**Key insight — zero new parameters.** The aux heads are *not* new modules. They
reuse `norm_out` and the tied `embedding.weight`. The entire anti-collapse
mechanism costs 0 params; the cost is purely the extra forward CE and its
activation memory (it adds 5 more full-vocab logit tensors — §8). Each
intermediate loop is now *directly* graded on whether its hidden state already
predicts the next token, so the recursion is forced to make monotone progress
loop-over-loop rather than dumping all work into loop 6.

This same property is what makes **reduced-loop inference** and **speculative
decoding** viable: a loop-3 readout is a usable draft because loop 3 was trained
to predict. See chapter 06 §7.

### 3.3 Weighting

With `recursive_loops=6` there are 5 intermediate loops (0…4), so 5 aux CEs. The
canonical preset uses uniform weighting via `aux_loop_loss_weight=0.05`
(`src/osrt/presets.py:49`):

```python
# src/osrt/model.py:1737-1744
if per_loop_weights is not None and i < len(per_loop_weights):
    w = per_loop_weights[i]
else:
    w = 1.0
aux_loop_total = aux_loop_total + w * aux_l
```

`aux_loop_total` then gets multiplied by `aux_weight` (= `aux_loop_loss_weight`)
in the final sum (§7). So with 5 loops at uniform 0.05 the aux block contributes
roughly `5 × 0.05 = 0.25×` the main loss in magnitude.

**A footgun worth flagging.** `config.per_loop_aux_weights` is described in the
config as *overriding* the uniform weight (`src/osrt/config.py:139-145`), but the
code does **not** override — it sets the per-loop factor `w` and *still*
multiplies the whole sum by `aux_loop_loss_weight` (`src/osrt/model.py:1800`).
The two **compound**. And the entire aux block only runs at all when
`aux_loop_loss_weight > 0` (`src/osrt/model.py:1716-1718`), so setting
per-loop weights with `aux_loop_loss_weight=0` silently disables them. The
canonical preset leaves `per_loop_aux_weights=None` (uniform 0.05 across all 5
intermediate loops); the per-loop override exists for front-loading gradient onto
early loops but must be reasoned about as a multiplier, not a replacement.

---

## 4. MTP heads — multi-token prediction

### 4.1 What MTP is and why it densifies the signal

Standard next-token training gives one supervision target per position: token
`+1`. **Multi-token prediction** (DeepSeek-V3/V4) adds heads that, from the
*same* final hidden state, also predict tokens further ahead. OSRT adds two:
**+2 and +3** (`mtp_heads=2`, `src/osrt/presets.py:53`).

The point is *signal density*, not faster decoding. Forcing one hidden vector to
simultaneously be a good predictor of the next *three* tokens pressures the
representation to carry longer-range structure — it can't just memorize the
immediate bigram continuation. Three CE gradients per position instead of one
means the body learns more per token of data, which matters on a fixed-budget
($350) training run.

### 4.2 The head module

Unlike the aux heads, MTP heads **do** own parameters:

```python
# src/osrt/model.py:1197-1203  (class MTPHead)
def __init__(self, dim: int) -> None:
    super().__init__()
    self.norm = nn.RMSNorm(dim)
    self.proj = nn.Linear(dim, dim, bias=False)

def forward(self, x_final: Tensor) -> Tensor:
    return self.proj(self.norm(x_final))
```

Each head is a small `RMSNorm(1536)` + `Linear(1536, 1536, bias=False)`
projection. The *vocab* projection still reuses the tied embedding — only this
small per-head transform is new. So an MTP head is "partly tied": new
`dim×dim` projection, tied final readout.

> **Discrepancy flag.** `ARCHITECTURE.md §9.3` says MTP heads "are tied with
> embedding too — no separate params." That is wrong for the projection layer:
> the `RMSNorm + Linear(dim,dim)` per head is genuinely new weight. The
> `compute_budget.py` `mtp_heads` category confirms **4,721,664 params** for the
> two heads. (Contrast the §3 aux heads, which really do add 0 params.)

### 4.3 The MTP loss

```python
# src/osrt/model.py:1759-1786
if self.training and len(self.mtp_heads) > 0:
    seq_len = labels.shape[-1]
    for k, head in enumerate(self.mtp_heads):
        offset = k + 2  # head k (0-indexed) → future offset +(2+k)
        if offset >= seq_len:
            per_mtp.append(torch.tensor(0.0, device=task_loss.device))
            continue
        head_hidden = head(hidden)
        head_logits = F.linear(head_hidden, self.model.embedding.weight)
        m_shift_logits = head_logits[..., :-offset, :self.config.real_vocab_size].contiguous().float()
        m_shift_labels = labels[..., offset:].contiguous()
        mtp_l = F.cross_entropy(
            m_shift_logits.view(-1, self.config.real_vocab_size),
            m_shift_labels.view(-1),
            ignore_index=-100,
        )
        per_mtp.append(mtp_l)
        mtp_total = mtp_total + mtp_l
```

Mechanics:

- **Offsets are +2 and +3.** `offset = k + 2`, so head 0 → +2, head 1 → +3. The
  main LM head already covers +1; MTP extends past it. (The brief's "+2/+3" is
  correct and matches code.)

  > **Discrepancy flag.** `ARCHITECTURE.md §9.3/§11.4` describes MTP at offsets
  > **+1/+2**. The code uses **+2/+3** (`offset = k + 2`,
  > `src/osrt/model.py:1762`). The ARCHITECTURE doc is stale here; the code and
  > this chapter use +2/+3.

- **Per-offset slicing.** Logits at position `i` predict `i+offset`, so the last
  `offset` positions have no in-range target and are dropped (`[:-offset]`),
  with labels shifted by `offset`.
- **Short-sequence guard.** If `offset >= seq_len` there are no valid positions;
  the head contributes a literal `0.0` that step.
- **Same CE recipe.** fp32, `ignore_index=-100`, sliced to `real_vocab_size`.

`mtp_total` is the *unweighted* sum of the per-head CEs; the
`mtp_loss_weight=0.3` scaling is applied in the final sum (§7).

### 4.4 Training-only, droppable at deploy

The MTP heads are created only when `mtp_heads > 0` and the loss is gated by
`self.training` (`src/osrt/model.py:1759`). `generate()` never touches them.
They contribute **0 active inference params** — `active_per_token` in
`compute_budget.py:71-72` explicitly skips the `mtp_heads` category. At
deployment the heads can be deleted from the checkpoint with no behavioral
change.

---

## 5. Router auxiliary losses — keeping the MoE healthy

The three router losses are produced *inside* each MoE layer as side effects of
routing, then collected by `OSRTModel.forward`. They are computed on the **raw,
pre-bias, pre-Gumbel router logits** — the *learned* router itself — not on the
bias-corrected "clean" path or the noisy dispatch path. The reasoning
(`src/osrt/model.py:631-635`): Gumbel noise is exploration and the balance bias
is an external controller; the *gradient* must push the learned router away from
collapse, so the aux losses must act on the raw router output, not on a path that
some other mechanism has already corrected.

### 5.1 `balance_loss` (Switch load balance)

```python
# src/osrt/model.py:647-653
raw_balance_f = raw_balance_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)
raw_balance_p = raw_router_probs.float().mean(dim=0)
self.balance_loss = self.num_routed * (raw_balance_f * raw_balance_p).sum()
```

The Switch loss: `E · Σ_i f_i · p_i`, where `f_i` is the fraction of
token-expert pairs routed to expert `i` (hard assignment) and `p_i` is the mean
softmax prob for expert `i`. It is minimized (= 1.0) at uniform load. It
penalizes *imbalance* without forcing uniform *probabilities* — the router may
develop sharp preferences as long as tokens spread roughly evenly. **What it
prevents:** expert collapse, where a few experts hog all tokens and the rest go
dead. (The canonical `router_aux_loss_coeff=0.10` was tuned upward from the
Switch default 0.01 after sanity runs showed an 8-expert router collapsing to
~2–3 active experts — see `src/osrt/config.py:80-103`.)

### 5.2 `z_loss` (log-partition magnitude bound)

```python
# src/osrt/model.py:662-663
z = torch.logsumexp(router_logits.float(), dim=-1)  # (N,)
self.z_loss = (z ** 2).mean()
```

The ST-MoE router z-loss: `mean_token (logsumexp(logits))²`. Two jobs: (1) keeps
raw logit magnitudes O(1) so bf16/fp8 softmax exponentials don't overflow, and
(2) keeps early softmax distributions flatter so cold experts retain a non-zero
gradient through LR warmup. **What it prevents:** numerical blow-up and
early-training dead experts. Coefficient `router_z_loss_coeff=1e-3` — small
enough not to compete with the task loss.

### 5.3 `seq_balance_loss` (per-sequence balance)

```python
# src/osrt/model.py:674-683
seq_one_hot = raw_balance_one_hot.float().view(B, S, self.top_k, self.num_routed)
f_seq = seq_one_hot.sum(dim=(1, 2)) / (S * self.top_k)         # (B, E)
p_seq = raw_router_probs.float().view(B, S, self.num_routed).mean(dim=1)  # (B, E)
self.seq_balance_loss = self.num_routed * (f_seq * p_seq).sum(dim=-1).mean()
```

DeepSeek-V3 §5.2 sequence-wise balance: the same Switch formula but computed
*within each sequence* and averaged over the batch. **What it prevents:** a
single long document dominating one expert even when the *global* batch is
balanced — a long-context (phase 3, `seq_len=8192`) failure mode.

> **Status: implemented but OFF in the canonical config.**
> `router_seq_balance_loss_coeff` defaults to `0.0` (`src/osrt/config.py:122`)
> and the preset does not set it. So `seq_balance_loss` is *computed* and
> *logged* every step but contributes **exactly 0** to the gradient until you
> opt in (intended for the long-context phase). It appears in the total-loss
> formula below for completeness; in the standard run its coefficient zeroes it
> out. (Ignore `ARCHITECTURE.md §11.3`'s `α=0.0001` — that section is stale and
> conflates the global and sequence balance losses.)

The aux-loss-**free** balance-bias controller is a *separate* heuristic: a
per-expert additive bias updated outside the gradient. It is **not** one of these
three differentiable losses and is covered in **chapter 03 (MoE & routing)**, not
here.

---

## 6. Normalization by *actual* loops run (the fixed bug)

Each router loss returned by `OSRTModel.forward` is the **sum** across every MoE
application — `num_blocks × loops_run` of them — because the loop in
`src/osrt/model.py:1510-1517` accumulates `block.moe.balance_loss` etc. on every
block of every loop. To turn a sum-over-layers into a per-layer-comparable
quantity, the wrapper divides by the number of MoE applications:

```python
# src/osrt/model.py:1694-1697
n_moe_layers = self.config.num_blocks * max(1, len(loop_rms))
balance_norm = balance_loss / n_moe_layers
z_norm = z_loss / n_moe_layers
seq_balance_norm = seq_balance_loss / n_moe_layers
```

The load-bearing detail is **`len(loop_rms)`, not `config.recursive_loops`.**
`loop_rms` has exactly one entry per loop that *actually ran* this batch. Under
**loop dropout** (chapter 06 §6) the loop chain is randomly truncated to fewer
than `recursive_loops` loops on some batches. If the divisor were the *configured*
depth (6) but only 3 loops ran, the regularizer would be silently **halved on
exactly the stochastic-depth batches that need it most** — the truncated batches
have fewer MoE applications and their per-layer pressure would be diluted by a
divisor counting layers that never executed. Dividing by the actual count keeps
the per-layer coefficient constant regardless of how many loops ran.

The `max(1, …)` guards a degenerate zero-loop edge case. This normalization fix
is cross-referenced from chapter 06; it lives in the loss code, so it is detailed
here.

---

## 7. Assembling the total loss

Putting it together, the training loss is a weighted sum of the task CE, the
three normalized router losses, the weighted aux-loop sum, and the weighted MTP
sum:

```python
# src/osrt/model.py:1793-1804
if self.training:
    loss = (
        task_loss
        + self.config.router_aux_loss_coeff * balance_norm
        + self.config.router_z_loss_coeff * z_norm
        + self.config.router_seq_balance_loss_coeff * seq_balance_norm
        + aux_weight * aux_loop_total
        + self.config.mtp_loss_weight * mtp_total
    )
else:
    loss = task_loss
```

With the canonical `OSRT_605M_A288M` coefficients:

```
L_train = task_loss
        + 0.10  · (Σ_layers balance_loss   / n_moe_layers)
        + 1e-3  · (Σ_layers z_loss          / n_moe_layers)
        + 0.0   · (Σ_layers seq_balance_loss / n_moe_layers)   # off in canonical
        + 0.05  · Σ_{r=1}^{5} aux_loop_CE_r                     # uniform per-loop
        + 0.30  · (mtp_CE_{+2} + mtp_CE_{+3})

n_moe_layers = num_blocks(3) × loops_actually_run
```

Two things to internalize:

1. **Eval loss is pure `task_loss`.** The entire aux stack is added only inside
   `if self.training`. Held-out perplexity and checkpoint comparisons therefore
   reflect *next-token quality alone*, never the auxiliary hyperparameter
   choices. (Training loops that want aux signals at eval time read the
   `last_*_normalised` telemetry attributes set at
   `src/osrt/model.py:1822-1829`, which are populated regardless of mode.)
2. **The task loss is never down-weighted.** It always carries coefficient 1.0;
   every auxiliary term is a *small* additive nudge. The router losses are
   normalized so 0.10/1e-3 are per-layer weights; the aux-loop sum is ~0.25× the
   main loss in practice; MTP is ~0.3× the summed future-token CE. The primary
   objective dominates by construction.

For telemetry, the per-component detached values (`last_task_loss`,
`last_balance_loss_normalised`, `last_aux_loop_total`, `last_mtp_loss`, etc.) are
stashed for the training loop's logging (`src/osrt/model.py:1806-1829`).

---

## 8. The (optional) fused linear-CE — activation-memory optimization

### 8.1 The problem it targets

Look back at how many `(B, S, 65536)` **fp32** logit tensors a single training
forward materializes:

- **1** main LM head (§2.2)
- **5** per-loop aux heads (§3, one per intermediate loop at `recursive_loops=6`)
- **2** MTP heads (§4)

That is **8 full-vocab fp32 logit tensors** live at once. At, say, `B·S = 8192`
tokens × 65536 vocab × 4 bytes ≈ **2.1 GB per tensor**, ~17 GB just for logits —
by far the dominant activation cost of the whole model, dwarfing the body's
hidden states. (This is real regardless of any optimization: it is a direct
consequence of stacking eight full-vocab heads, §1.)

### 8.2 The idea: fused, chunked, checkpointed linear-CE

A *fused linear cross-entropy* computes the CE for each head **without ever
materializing the full `(B,S,65536)` fp32 logit tensor**: it streams the
hidden→vocab projection and the cross-entropy together in chunks over the token
dimension, gradient-checkpointing the projection so only one chunk of logits is
resident at a time. The math is identical to `F.linear` + `F.cross_entropy`; only
the memory schedule changes. The intended knob was `fused_cross_entropy_chunks`
(0 = off, the materialized path; >0 = chunk count), and it was parity-tested
against the dense path.

### 8.3 Status: designed, not landed on `main`

> **Important discrepancy with the task brief.** The brief describes
> `src/osrt/fused_ce.py` and a `fused_cross_entropy_chunks` config field as
> present-but-default-off. **Neither exists on the current `main` tree** (HEAD
> `eff034b`). Verified:
>
> - `find` / `grep` over the whole repo: no `src/osrt/fused_ce.py`, no
>   `fused_cross_entropy_chunks` anywhere in `src/` or `scripts/`.
> - `grep -rn cross_entropy src/osrt` returns **only** the three plain
>   `F.cross_entropy` calls in `model.forward` (`src/osrt/model.py:1678, 1727,
>   1780`) — the materialized path. There is no chunked implementation hiding
>   under another name.
> - `git log --all -- src/osrt/fused_ce.py` shows the file was added in commit
>   `49dc802` ("perf: optional fused linear-CE (B2) + flex attention-sink (B1)"),
>   but `git merge-base --is-ancestor 49dc802 HEAD` is **false**: that commit
>   lives only on the unmerged branch `b1b2-attn-sink-fused-ce`. It was never
>   merged into `main`.
>
> So: the fused linear-CE was **prototyped on a feature branch and not landed**.
> The current trained model uses the dense, materialized 8-logit-tensor path
> described in §§2–4. The optimization remains available to merge when the GPU
> phase needs the activation memory back; until then this section documents the
> *design and rationale*, not live code.
>
> The **attention-sink (B1)** half of that same `b1b2-attn-sink-fused-ce` branch
> likewise never landed, and the canonical model does **not** use an attention
> sink. The preset sets `attention_sink=False` (`src/osrt/presets.py:54`), so
> attention routes through flash `F.scaled_dot_product_attention`
> (`src/osrt/model.py:1177-1183`) rather than the manual sink-rescale path. The
> `sink_logits` / `_attention_with_sink` machinery still exists in the model
> guarded behind that flag, but it is off in the trained model. (Attention itself
> is covered in chapter 02; mentioned here only because the same branch name
> couples it to the fused-CE history above.)

---

## 9. Parameter cost of the heads

From `scripts/compute_budget.py` on the canonical `OSRT_605M_A288M` preset:

| Head / component        | Params       | Inference-active? | Notes |
|-------------------------|--------------|-------------------|-------|
| Tied LM head            | (in embedding, 100,690,944) | yes | shares `embedding.weight`; counted once in the `embedding` category |
| Per-loop aux heads (×5) | **0**        | n/a (train-only)  | reuse `norm_out` + tied embedding |
| MTP heads (×2)          | **4,721,664** | **no** (dropped at deploy) | each is `RMSNorm(1536) + Linear(1536,1536)` |

Takeaways:

- **The LM head and the aux heads add no new parameters.** The LM head *is* the
  tied embedding; the aux heads *re-apply* it. The whole anti-loop-collapse
  mechanism (§3) is free in parameter terms.
- **The MTP heads cost ~4.72M params, all training-only.** They are excluded
  from the active-per-token count (`compute_budget.py:71-72`) and droppable at
  deployment, so they buy denser training signal at **zero** inference cost.
- For reference, `compute_budget.py` reports the full preset at **~601M physical
  / ~278M active per token** (46.3% of physical). (The preset is *branded*
  "605M / 288M" in `presets.py:21`; that label is slightly stale — the live
  budget script is authoritative at ~601M / ~278M.)

---

## 10. Cross-references

- **Tied embedding / mHC collapse producing `hidden`** — chapter 04 (mHC).
- **Recursion, loop dropout, why aux heads exist** — chapter 06 (recursion); the
  normalization-by-actual-loops fix (§6 here) is referenced from there.
- **The aux-loss-free balance-bias controller** (the non-gradient heuristic that
  shares load at deploy time) — chapter 03 (MoE & routing). It is *not* one of
  the three differentiable router losses in §5.
- **MTP / DeepSeek-V3-V4 intent** — `ARCHITECTURE.md §9.3, §11.4` (note the
  +1/+2 vs code's +2/+3 drift flagged in §4.3).
