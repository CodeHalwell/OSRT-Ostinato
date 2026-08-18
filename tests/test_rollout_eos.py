"""Regression test: the SFT rollout builder must terminate targets with EOS.

Without a terminal EOS the target ended at <|/answer|> + masked padding, so
EOS was never a training target and the SFT model never learned to stop
(observed in SFT-v2: generations ran to the length cap in both modes). This
locks the fix in RolloutDataset._build_sequence.
"""
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from osrt.data import RolloutDataset

# Repo-root relative, never CWD relative: a bare relative path makes
# AutoTokenizer.from_pretrained fall back to a Hugging Face repo-id lookup
# and hit the network. v7's tokenizer is open (roadmap gate G2); skip
# cleanly when the artefact is absent.
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizer"

pytestmark = pytest.mark.skipif(
    not (TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason="tokenizer/ artefact absent — pending roadmap gate G2",
)

TOK = str(TOKENIZER_DIR)
def _build(rec, seq_len=256):
    ds = RolloutDataset(jsonl_path="/dev/null", seq_len=seq_len,
                        tok_name=TOK, seed=0)
    tok = AutoTokenizer.from_pretrained(TOK)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = ds._build_sequence(rec, tok, pad_id)
    return out, tok


def test_rollout_target_ends_with_eos():
    rec = {"system": "You are helpful.", "prompt": "What is 2+2?",
           "thinking": "2 plus 2 is 4.", "response": "4"}
    out, tok = _build(rec)
    assert out is not None
    ids, labels = out
    ids, labels = ids.tolist(), labels.tolist()

    # last REAL label (before -100 padding) must be EOS — i.e. the model is
    # trained to emit EOS right after <|/answer|>.
    real = [(i, tid) for i, (tid, lab) in enumerate(zip(ids, labels)) if lab != -100]
    assert real, "no supervised target tokens"
    last_idx, last_tok = real[-1]
    assert last_tok == tok.eos_token_id, (
        f"last supervised token {last_tok} != eos {tok.eos_token_id}")
    # and the label at that position equals the token (EOS is a real label)
    assert labels[last_idx] == tok.eos_token_id


def test_rollout_answer_only_also_ends_with_eos():
    # no `thinking` → direct <|answer|> path must still terminate with EOS
    rec = {"prompt": "Name a color.", "response": "Blue is a color."}
    out, tok = _build(rec)
    assert out is not None
    ids, labels = out
    real = [tid for tid, lab in zip(ids.tolist(), labels.tolist()) if lab != -100]
    assert real[-1] == tok.eos_token_id
