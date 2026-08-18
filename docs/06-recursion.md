# Recursion: Depth Recurrence, Loop Embeddings & Loop Aux

> Part of the OSRT-605M `docs/` architecture series. This chapter explains how
> the model gets deep without getting big: it runs **3 physical decoder blocks
> 6 times** (recursive depth recurrence), how it keeps the six iterations from
> collapsing into one, and the training machinery (loop embeddings, per-loop aux
> heads, loop dropout) that makes the recursion actually carry its weight.

A note on sourcing. Where this document states a *mechanic* it cites
`src/osrt/model.py` (and `src/osrt/config.py`) by line — that is the source of
truth. `ARCHITECTURE.md` and `RESEARCH.md` are cited only for *intent*, project
*history*, and *research grounding*. `ARCHITECTURE.md §10` itself says its
pseudocode is "illustrative ... the implementation in `model.py` is the source
of truth," and in two places below the code and the prose disagree. When they
do, the code wins and the discrepancy is flagged.

---

## 1. Purpose — depth recurrence as parameter-efficient depth

A standard 18-layer transformer stores 18 distinct sets of attention + FFN
weights. OSRT stores **3** and applies them **6 times in sequence**:

```
3 physical blocks  ×  6 loops  =  18 effective layers
```

The 3 blocks' *weights are reused* on every loop. That reuse is the entire
parameter-efficiency argument: you pay for 3 layers of weights but, at run time,
the residual stream is transformed 18 times. The bet — validated by the looped-LM
literature (Ouro, Huginn; see §9) — is that a block's job ("refine the residual
a little") is roughly the same on every iteration, so the same weights can do it
repeatedly. Effective *depth* (sequential compute / refinement steps) is what
unlocks reasoning; *parameters* are what you pay for. Recurrence decouples them.

The configured depth lives in two knobs (`src/osrt/config.py:50-51`):

```python
num_blocks: int = 3,
recursive_loops: int = 6,
```

`ARCHITECTURE.md §16.1` makes the contract explicit: the recursive forward
"MUST apply the SAME 3 physical blocks 6 times (not 18 different block
instances)." If you accidentally instantiated 18 blocks you'd have a vanilla
deep transformer and would have thrown away the whole design.

---

## 2. The loop structure — nested loop × block, weight reuse

The recursion is a plain nested loop in `OSRTModel.forward`
(`src/osrt/model.py:1461-1531`):

```python
for loop in range(n_loops_to_run):
    for block_idx, block in enumerate(self.blocks):
        idx = loop * self.config.num_blocks + block_idx     # 1463
        adapter_a = self.adapters_a[idx]
        adapter_b = self.adapters_b[idx]
        layer_past = past_key_values[idx] if past_key_values is not None else None
        x, present_kv = block(
            x, adapter_a, adapter_b, self.adapter_scale, cos, sin,
            loop_idx=loop, past_key_value=layer_past, ...,
        )
        ...
    # end-of-loop bookkeeping: aux capture, loop_rms, norm_loop
```

`self.blocks` is a `ModuleList` of exactly `num_blocks` (=3) `RecursiveBlock`s
(`src/osrt/model.py:1250-1251`). The *same three module objects* are pulled out
of that list on every one of the 6 iterations — that is the weight reuse, made
literal: `for block in enumerate(self.blocks)` runs over the same 3 objects each
loop.

So the reused blocks are not perfectly identical across loops. Three things make
the 18 effective layers differentiate even though the *block weights* repeat:

1. **Loop embeddings** — a per-loop additive vector injected into the MoE router
   (§3). Iteration conditioning.
2. **Per-(loop,block) low-rank adapters** — `adapters_a[idx]`/`adapters_b[idx]`
   selected by `idx = loop*num_blocks + block_idx` (line 1463). There are
   `num_blocks * recursive_loops = 18` distinct adapter pairs
   (`src/osrt/model.py:1255`), so each of the 18 effective layers gets its own
   small low-rank delta on top of the shared block. (Mechanism detailed in the
   HRA chapter; mentioned here only so "weight reuse" isn't overstated.)
3. **Per-loop router balance bias** — a buffer of shape
   `(num_loops, num_routed)` (`src/osrt/model.py:236-240`), so even routing
   load-balancing is corrected per loop.

> **Correction to a common simplification.** It is *not* true that the loop
> embedding is "the only per-loop-unique parameter." The 18 adapter pairs and the
> per-loop balance bias are also loop-specific. The loop embedding is the main
> piece of *iteration conditioning that drives expert selection*, but the reused
> blocks get loop-specific differentiation from all three sources above.

The other end-of-loop bookkeeping (lines 1519-1531) — aux capture, `loop_rms`,
and the inter-loop `norm_loop` — is covered in §4 and below.

---

## 3. Loop embeddings — symmetry breaking, why each loop must differ

If you run the same weights six times with no per-iteration signal, every loop
sees a similar residual and does a similar transform — the iterations have no way
to *specialize* ("loop 0, you do early syntactic work; loop 5, you do final
output shaping"). The recurrence degenerates toward a single fixed-point step
repeated. The looped-LM literature is blunt about this: "pure weight-tying
without loop conditioning degenerates" and "loop embeddings ... are critical"
(`RESEARCH.md:276-277`). They are the Universal-Transformer "depth embedding"
idea: tell the shared block *which iteration it is on*.

### Where the loop embedding actually enters (read this carefully)

The loop embedding lives **inside the MoE layer**, not at the top of the loop.
Each `MoELayer` owns its own table (`src/osrt/model.py:213-214`):

```python
self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
self.loop_embeddings._osrt_init_std = config.loop_embedding_init_std
```

Because there are 3 physical blocks and the table is a per-block module, there
are **3 separate loop-embedding tables** (one per `MoELayer`), each of shape
`(recursive_loops=6, dim=1536)`. It is consumed on the **learned-router path**
of `MoELayer.forward` (`src/osrt/model.py:514-516`):

```python
# Router: add loop embedding, project to expert scores
loop_emb = self.loop_embeddings.weight[loop_idx].view(1, 1, D)
router_input = x + loop_emb            # 516
router_logits = self.router(router_input.reshape(N, D))
```

So the loop embedding is added to the **router's input only**, biasing **which
experts get selected at this iteration**. It is *not* added to the residual
stream that flows through attention and the expert FFNs. The conditioning is
"loop 5 should route to a different mix of experts than loop 0," which is exactly
the lever you want for per-iteration specialization in a sparse MoE.

> **Discrepancy flagged (code vs. docs).** `ARCHITECTURE.md §5.2` and the
> common project shorthand describe the loop embedding as a residual *bias* —
> "Added BEFORE the first physical block at each loop ... the only parameter
> that differs across loop iterations." The shipped code does **not** do that.
> An exhaustive grep finds the loop embedding added in exactly one place:
> `router_input = x + loop_emb` (line 516), inside `MoELayer`, on the learned-
> router path. There is no `x = x + loop_emb` at loop start anywhere in
> `OSRTModel.forward`. Trust the code: **loop conditioning enters through the MoE
> router, producing loop-dependent expert selection** — not as a residual bias on
> the hidden state.

Init scale is small and deliberate (`config.py:248`,
`loop_embedding_init_std = 0.1`): start the iterations nearly indistinguishable
and let training pull them apart only as much as it needs.

### How hash-routed early blocks break symmetry *without* the loop embedding

There is a subtlety: early blocks can use deterministic hash routing instead of
the learned router (`hash_routing_blocks`, `ARCHITECTURE.md §7.5`). On that path
the code `return`s *before* the `loop_emb` addition (`src/osrt/model.py:506-512`),
so a hash-routed block never reads `loop_embeddings`. Those loops still
differentiate because the hash itself is **loop-indexed**
(`src/osrt/model.py:503`):

```
expert_id = (token_id + loop_idx) % E      # loop-indexed top-1 hash
```

Adding `loop_idx` into the modular hash means the same token routes to a
different expert on each iteration — symmetry is broken by the routing rule, not
by a learned vector. So: learned-router blocks break loop symmetry via
`loop_embeddings`; hash-routed blocks break it via the loop-indexed hash. Either
way no two loops are forced to do identical work.

---

## 4. Per-loop aux LM-head losses — the loop-collapse fix

This is the model's **signature failure mode**, and the fix is the most
important idea in this chapter.

### The failure: loop collapse

With weight-tied recurrence, the gradient has a lazy shortcut available. Only the
**final** loop's hidden state feeds the LM head and therefore the task loss. So
gradient naturally flows into "whatever the last loop does," and the optimizer is
free to learn a solution where loops 0..N-2 do almost nothing useful — the early
iterations become near-no-ops and **loop N-1 absorbs all the real work**. You
"have" 18 effective layers but you're effectively running ~3. The model.py
comment names the mechanism exactly (`src/osrt/model.py:1415-1421`):

```python
# We capture the hidden state at the END of each non-final loop ... so the
# aux LM head can ... predict the next token from the intermediate
# representation. Forces gradient signal into loops 0..N-2 instead of letting
# loop N-1 absorb everything.
```

This is not hypothetical. Loop collapse was hit empirically — the project's v5
history lists "loop collapse" among the late-discovered failures
(`README.md:45,135`), and a dedicated training stage (`loop_fix` / `loop_fix_v2`)
was built specifically to repair it. (Note: this is distinct from the *router*/
representation collapse in `RESEARCH.md:645-651` Cells B/D — that is a routing
load-balance failure. Loop collapse is about *iterations going idle*, a different
problem with a different fix.)

### The fix: apply the tied LM head to every intermediate loop

The anti-collapse fix gives each non-final loop its own next-token prediction
objective. Two pieces cooperate.

**(a) Capture, in `OSRTModel.forward`** (`src/osrt/model.py:1524-1531`):

```python
if capture_aux and loop < n_loops_to_run - 1:
    # Collapse the mHC stream to a single vector for the aux head
    intermediate_hiddens.append(self._collapse(x) if self.use_mhc else x)

loop_rms.append(x.float().pow(2).mean().sqrt())
if loop < n_loops_to_run - 1:
    x = self.norm_loop(x)
```

The intermediate hidden is captured at the **end of each non-final loop**, after
the 3 blocks but **before `norm_loop`** (the `norm_loop` is applied on the *next*
line, only for non-final loops). It uses the dedicated learnable collapse head
`mhc_collapse` via `_collapse(...)` (`src/osrt/model.py:1288-1291`) — never a
stale dynamic mixing matrix — to mix the multi-channel residual stream down to a
single `dim`-vector. `capture_aux` is on only when
`aux_loop_loss_weight > 0 and self.training` (`src/osrt/model.py:1423-1426`), so
this entire path is **train-only**.

**(b) Loss, in `OSRTForCausalLM.forward`** (`src/osrt/model.py:1716-1744`):

```python
for i, h_loop in enumerate(intermediate_hiddens):
    h_norm = self.model.norm_out(h_loop)                       # 1722
    h_logits = F.linear(h_norm, self.model.embedding.weight)   # tied LM head
    h_shift = h_logits[..., :-1, :real_vocab_size].contiguous().float()
    aux_l = F.cross_entropy(
        h_shift.view(-1, real_vocab_size),
        shift_labels.view(-1), ignore_index=-100,
    )
    per_loop_aux.append(aux_l)
    aux_loop_total = aux_loop_total + w * aux_l
```

Three things to notice:

- **The LM head is the tied embedding** — `F.linear(h_norm, embedding.weight)`.
  No new parameters: the same matrix that maps tokens→embeddings (and final
  hidden→logits) is reused on every intermediate loop. `ARCHITECTURE.md §9.2`:
  "the LM head is SHARED across all loop outputs (it IS the embedding). No
  additional parameters."
- **The normalization applied to the capture is `norm_out`** (line 1722), the
  *output* norm — the same one the final hidden gets (`src/osrt/model.py:1535`) —
  **not** `norm_loop`. The capture was deliberately taken *before* `norm_loop` so
  the aux head can put the intermediate hidden into the same "about to hit the LM
  head" frame the final loop uses.
- **Targets are the same shifted labels** as the main loss — each intermediate
  loop is asked to predict the *same next tokens*. That is what forces it to
  produce a coherent, decode-ready representation rather than arbitrary scratch
  work.

The aux term enters the total loss only in training (`src/osrt/model.py:1793-1800`),
weighted by `aux_loop_loss_weight`; eval loss stays pure task CE so perplexity
isn't polluted. Per-loop weights can override the uniform weight via
`per_loop_aux_weights` (`config.py:139-145`, applied at `model.py:1737-1744`).

> **Weight value.** The *code default* is `aux_loop_loss_weight = 0.0`
> (`config.py:136`) — i.e. off unless a training stage turns it on. The training
> stages set it to **0.05** (pretrain/MOPD/SFT) and 0.03 during GRPO
> (`ARCHITECTURE.md §11.2`). `ARCHITECTURE.md §16.1` warns: if it is 0, "training
> MUST monitor for loop collapse." So 0.05 is the *operating* value; 0.0 is just
> the inert default.

The payoff is twofold (`ARCHITECTURE.md §9.2`): (1) loops 0..N-2 actually
contribute, defeating collapse; and (2) because each intermediate loop now emits
a usable next-token distribution, you can read off a *draft* prediction at an
early loop and verify it at the full loop count — the speculative-decode hook of
§7.

### 4.4 Detecting collapse at runtime — the per-loop residual-update telemetry

The aux losses and loop dropout (§5) *prevent* collapse; a separate telemetry
hook *detects* it, so a regressing run is caught early. The signal is the
**relative residual update** each effective layer makes to the hidden stream,
`||Δx|| / ||x||`. A block whose update has decayed to ~0 has become a no-op — a
collapsed loop — so a monotone decay through the deep loops is the at-a-glance
collapse fingerprint.

It is computed inside the recursion loop in `OSRTModel.forward`
(`src/osrt/model.py:1636-1685`). Before each block it snapshots the residual,
and after the block records the ratio per effective layer
`idx = loop*num_blocks + block_idx`:

```python
# src/osrt/model.py:1681-1685
if collect_loop:
    base = x_prev.norm().clamp_min(1e-6)
    upd = (x.detach() - x_prev).norm()
    self.last_loop_update_norm[idx] = (upd / base).item()
    self.last_loop_hidden_norm[idx] = base.item()
```

Two pieces are stored, one per effective layer: the relative update
`last_loop_update_norm` and the raw hidden norm `last_loop_hidden_norm`
(`src/osrt/model.py:1417-1419`).

**Gated exactly like the MoE telemetry.** `collect_loop = self.telemetry_enabled`
(`src/osrt/model.py:1641`), and the *same* `set_moe_telemetry` call that toggles
the MoE diagnostics also flips this hook (`src/osrt/model.py:1743`,
`"# gates the loop-collapse hook"`). On normal compiled steps it is off, so the
`.item()`/`.detach()` syncs never run and the B4 fullgraph (chapter 03 §7) stays
clean; on logging steps it adds the same kind of sync the MoE telemetry already
pays for.

The trainer's `_collect_moe_metrics` surfaces it (`src/osrt/train.py:452-464`):
each layer as `loop/update_norm_l{idx}` in W&B, plus aggregate
`loop/update_norm_min` / `_last` / `_mean`. Stdout prints a per-layer
`loop |dx|/|x|: L0=.. L1=..` line and a `collapse:` line whose `loop_upd
min/last/mean` come straight from these values (`src/osrt/train.py:1218-1228`),
sitting next to the MoE-side `dead_experts` / `bias_abs_max` (chapter 03 §10).

> **Scope.** This detects *recursive-loop* collapse (a deep loop gone idle) — the
> §4 failure mode — and is distinct from the MoE *expert* collapse the
> dead-expert count watches (chapter 03 §13). Both ride the one
> `telemetry_enabled` gate and print on the same `collapse:` line, but they
> measure different things.

---

## 5. Loop dropout (stochastic depth) — making early loops stand alone

Aux heads push intermediate loops to predict *in parallel* with the final loop.
Loop dropout goes further: some fraction of the time it makes an intermediate
loop's output **the actual model output**, so the truncation-point loop has to be
genuinely standalone-useful, not just a parallel rehearsal.

The mechanism (`src/osrt/model.py:1444-1452`):

```python
if (
    self.training
    and getattr(self.config, "loop_dropout_prob", 0.0) > 0.0
    and random.random() < self.config.loop_dropout_prob
):
    min_loops = max(2, getattr(self.config, "loop_dropout_min_loops", 3))
    max_loops = n_loops_to_run
    if max_loops > min_loops:
        n_loops_to_run = random.randint(min_loops, max_loops)
```

With probability `loop_dropout_prob` during training, the loop chain is truncated
to a random length in `[loop_dropout_min_loops, n_loops_to_run]` (defaults:
`loop_dropout_prob = 0.0`, `loop_dropout_min_loops = 3` — `config.py:171-172`).
The whole forward then runs only that many loops; the final (truncated) loop's
hidden feeds the LM head and the main task loss flows from a *shorter* chain. The
in-code comment frames it as the complement to the aux loss
(`src/osrt/model.py:1428-1435`): "aux pushes intermediate loops to predict in
parallel, dropout makes their predictions become the actual model output some
fraction of the time."

Two guards:

- **Floor of 2** — `max(2, ...)` (line 1449) ensures the chain is never
  truncated below 2 loops; collapsing all the way to 1 would defeat the point of
  recurrence.
- **Train-only.** The whole block is gated on `self.training`. At inference
  (`model.eval()`) loop dropout never fires, so an eval/serving forward always
  runs the full resolved loop count. This is standard stochastic-depth
  discipline: regularize during training, run the full network at test time.

Loop dropout composes cleanly with the §7 inference knob: the ceiling
`max_loops` is the **already-resolved** `n_loops_to_run`, so an explicit
`num_loops=K` caps the chain at K and dropout may only shorten it further
(`src/osrt/model.py:1437-1443`).

---

## 6. Loss normalization by *actual* loops run (the fixed bug)

Subtle but real: the router regularizers — Switch balance loss, router z-loss,
sequence-balance loss — are **summed** across every MoE application inside
`OSRTModel.forward` (one add per block per loop). To make the configured
coefficient mean "per-layer weight" rather than "per-whole-model sum," the
wrapper divides that sum by the number of MoE applications
(`src/osrt/model.py:1694-1697`):

```python
n_moe_layers = self.config.num_blocks * max(1, len(loop_rms))
balance_norm = balance_loss / n_moe_layers
z_norm       = z_loss / n_moe_layers
seq_balance_norm = seq_balance_loss / n_moe_layers
```

The load-bearing detail is `len(loop_rms)`, **not** `self.config.recursive_loops`.
`loop_rms` gets one entry appended per loop that actually executed
(`src/osrt/model.py:1529`), so its length always equals the number of loops the
forward really ran.

Why dividing by the configured depth was wrong: under loop dropout the chain is
often shorter than 6. Say dropout truncates to 3 loops. Then the *numerator*
(the summed regularizers) only accumulated 3 loops' worth of penalties — but
dividing by the configured `num_blocks * 6` would normalize as if 6 loops ran.
That **halves the regularizer on exactly the short batches**, the stochastic-depth
batches where balance is most fragile (fewer routing steps, less averaging). The
comment at `src/osrt/model.py:1686-1693` spells this out: dividing by full depth
"halves the regularizer exactly on the stochastic-depth batches that need it
most." Dividing by the *actual* loop count keeps the per-MoE-layer coefficient
constant regardless of how dropout truncated the chain.

The `max(1, ...)` is a defensive floor against a degenerate `len(loop_rms) == 0`
(no divide-by-zero). In the default `num_loops=None`, no-dropout path
`len(loop_rms) == recursive_loops`, so this is bit-identical to the old behavior
there — the change only bites when dropout (or the §7 knob) shortens the chain.

---

## 7. The `num_loops` inference knob — variable test-time compute

Recurrence gives a knob the dense baseline can't: **spend less compute by running
fewer loops**. `num_loops` (default `None`) is threaded from
`OSRTForCausalLM.forward` → `OSRTModel.forward` and resolved by
`_resolve_num_loops` (`src/osrt/model.py:1293-1310`):

```python
def _resolve_num_loops(self, num_loops: int | None) -> int:
    if num_loops is None:
        return self.config.recursive_loops
    if not (1 <= num_loops <= self.config.recursive_loops):
        raise ValueError(
            f"num_loops must be in [1, recursive_loops="
            f"{self.config.recursive_loops}], got {num_loops}"
        )
    return num_loops
```

- `None` → run the full trained `recursive_loops` (=6). Bit-identical to the
  historical path: every downstream count and index is unchanged.
- `K` in `[1, recursive_loops]` → run only the **first K** loops before
  collapse/`norm_out` + the LM head. Fewer loops = faster, slightly lower
  quality. `ARCHITECTURE.md §12.2` sketches a rough trade-off (≈3 loops at ~85%
  quality, 5 at ~98%); treat those numbers as targets, not measurements.

What makes a reduced K *usable* rather than garbage is the §4 aux training: each
intermediate loop was trained to emit a coherent next-token distribution, so
"stop at loop K and read the LM head" produces a real prediction. Without the aux
heads, an early loop's hidden would be uncalibrated scratch state and
short-circuiting would degrade badly.

**Speculative-draft connection.** The same property powers greedy speculative
decoding (`src/osrt/model.py:2082-2147`, `ARCHITECTURE.md §12.3`): the **drafter**
runs at the cheap `spec_draft_loops` count and the **verifier** runs the full
loop count, accepting the longest greedy-matching prefix. The drafter is capped
at the verifier's loop count — `draft_loops = min(spec_draft_loops, full_loops)`
(`src/osrt/model.py:2146-2147`) — so the draft never costs more compute than the
verify. (Caveat documented in code: this routine is greedy-only / not a
distribution-preserving sampler — see the box at `src/osrt/model.py:2087-2100`.)

---

## 8. How recursion interacts with the KV cache

Each *effective* layer keeps its own cache slot. With `use_cache=True` the cache
is a flat list of per-effective-layer latents indexed exactly as the blocks are
walked (`src/osrt/model.py:1463`):

```
idx = loop * num_blocks + block_idx
```

So for `recursive_loops=6, num_blocks=3` there are **18** cache entries per
token — one per effective layer — and the entry for (loop, block) lives at
`loop*3 + block_idx`. Each loop computes its own fresh K from that loop's input
(`ARCHITECTURE.md §16.5`: "Each loop's K is computed FRESH from that loop's
input"), so loop 0's block 0 and loop 5's block 0 occupy *different* cache slots
even though they share weights.

This is also why `num_loops` must be held constant across a cached decode: the
cache length is keyed off the loops actually run
(`expected_past_layers = num_blocks * n_loops_to_run`,
`src/osrt/model.py:1361`), so prefill and every decode step must use the same K
or the layer indices stop lining up (`src/osrt/model.py:1335-1340`). `generate()`
enforces this by threading a single `num_loops` through the whole call. The
speculative path keeps **two** independent caches — a draft cache at
`draft_loops` and a verify cache at `full_loops` — precisely because cache length
is loop-count-specific (`src/osrt/model.py:2118-2122`). Full cache mechanics
(latent/MLA layout, quantization) are in the inference chapter; this section only
makes the recursion↔cache indexing explicit.

---

## 9. Research grounding (Ouro / Huginn)

The recursive design is not invented here; it tracks a validated line of
looped-LM research (`RESEARCH.md §4`, lines 245-296):

- **Huginn-3.5B** (Geiping et al., arXiv 2502.05171): prelude → recurrent core
  (a few blocks applied N times) → coda, trained with variable iteration counts
  and truncated backprop. At many unrolls, a handful of physical blocks act like
  a very deep stack.
- **Ouro / LoopLM** (Zhu et al., arXiv 2510.25741): a looped LM trained on 7.7T
  tokens that **beats dense Qwen3-4B on GSM8K and MATH500** at far fewer
  parameters, reporting 2–3× parameter-efficiency gains. Ouro also reports
  **instability at high recursion** — R4 is its sweet spot, R8+ needs careful
  stabilization. OSRT's choice of R=6 and its hard refusal to run past
  `recursive_loops` (§10) are downstream of that finding.
- **Loop embeddings** as the Universal-Transformer depth-conditioning idea
  (`RESEARCH.md:276-277`): weight-tying without per-iteration conditioning
  degenerates.

What OSRT adopted (`RESEARCH.md:282-287`): the recursive 3×6 design, loop
embeddings, a cap at R=6, and the aux per-loop LM-head losses. Sandwich RMSNorm
in the recurrent block (`ARCHITECTURE.md §5.3`) follows Huginn's recipe for
surviving many iterations without blowing up.

---

## 10. Caveats

- **No safe unrolling beyond the trained loop count.** The loop-embedding tables
  are sized to `recursive_loops` (`nn.Embedding(recursive_loops, dim)`,
  `src/osrt/model.py:213`) and indexed directly with no clamp
  (`weight[loop_idx]`, line 515). `_resolve_num_loops` *raises* `ValueError` for
  any `K > recursive_loops` (`src/osrt/model.py:1305-1309`) — there is **no**
  in-code mechanism to run more loops than were trained. (Older notes describe a
  `min(r, 7)` "hard wall at R=8"; that clamp is **not** in the current code. The
  real boundary is the table size + the validation `ValueError`.) Running *fewer*
  loops is supported and graceful (§7); running *more* is simply disallowed.

- **Fewer loops trades quality for speed.** Reduced-K inference is usable only
  because of the aux per-loop training; quality still degrades as K drops
  (`ARCHITECTURE.md §12.2`). It is a throughput dial, not a free lunch.

- **The aux heads are a *training* fix, not an inference component.** Their job
  is to keep gradient flowing into intermediate loops during training. At
  inference you read the tied LM head off whatever loop you stop at; there is no
  separate "loop-3 head" module — it is the same tied embedding.

- **Loop dropout and the aux path are train-only.** `model.eval()` disables both
  (`self.training` gates at `src/osrt/model.py:1425,1445`). Serving always runs
  the full resolved loop count with no stochastic truncation.

- **Loop collapse can recur if you turn the aux weight off.** With
  `aux_loop_loss_weight = 0`, the architecture has nothing forcing intermediate
  loops to work, and `ARCHITECTURE.md §16.1` explicitly mandates monitoring for
  collapse in that regime. The 0.05 operating value is load-bearing.
