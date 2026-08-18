"""Regression tests for the verifiable-reward stack.

Focused on the bugs surfaced in review/deep-dive-code-review-2026-06-08.md:
P2 — mbpp_test_reward used to collapse to all-or-nothing. A 3-of-4-pass
completion scored the same as 0-of-4, killing the partial-reward
gradient signal that GRPO needs to escape uniform reward groups.
"""

import sys

import pytest

from osrt.rewards import mbpp_test_reward


def _completion(code: str) -> str:
    """Wrap a python code block in the expected <|answer|>...</|answer|>
    format. mbpp_test_reward calls extract_answer_text under the hood and
    prefers fenced ```python blocks inside the answer.
    """
    return f"<|answer|>\n```python\n{code}\n```\n<|/answer|>"


def test_mbpp_all_pass():
    """4 of 4 assertions pass -> reward_pass, verdict all_pass."""
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        "assert add(1, 2) == 3",
        "assert add(0, 0) == 0",
        "assert add(-1, 1) == 0",
        "assert add(10, 5) == 15",
    ]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "all_pass", bd
    assert bd["passed"] == 4
    assert bd["total"] == 4
    assert r == 2.0  # reward_pass default


def test_mbpp_partial_credit_not_collapsed_to_zero():
    """The P2 regression: when SOME assertions pass and others fail, the
    reward must reflect the pass rate, NOT collapse to all_fail (which is
    what the old `passed = len(test_list) if rc == 0 else 0` logic did).
    """
    # Implementation is correct for the first two tests, wrong for the
    # other two. The buggy implementation scored this as 0 of 4.
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        "assert add(1, 2) == 3",      # passes
        "assert add(0, 0) == 0",      # passes
        "assert add(1, 1) == 99",     # fails — wrong expected value
        "assert add(2, 2) == 99",     # fails — wrong expected value
    ]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        reward_partial=1.0,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "partial", (
        f"expected 'partial' verdict, got {bd!r}. The old all-or-"
        f"nothing implementation would return 'all_fail' here — if "
        f"that's what you see, P2 has silently regressed."
    )
    assert bd["passed"] == 2
    assert bd["total"] == 4
    # 1.0 * (2 / 4) = 0.5
    assert r == pytest.approx(0.5)


def test_mbpp_all_fail():
    """0 of N assertions pass -> penalty_fail."""
    code = "def add(a, b):\n    return 999\n"   # always wrong
    tests = [
        "assert add(1, 2) == 3",
        "assert add(0, 0) == 0",
    ]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "all_fail", bd
    assert bd["passed"] == 0
    assert bd["total"] == 2
    assert r == -1.5  # penalty_fail default


def test_mbpp_one_of_one():
    """Single test that passes -> all_pass (not partial)."""
    code = "def f(x):\n    return x * 2\n"
    tests = ["assert f(3) == 6"]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "all_pass"
    assert bd["passed"] == 1
    assert bd["total"] == 1


def test_mbpp_one_of_three_partial():
    """1 of 3 passing — confirm rate-proportional reward."""
    code = "def f(x):\n    return x if x == 1 else None\n"
    tests = [
        "assert f(1) == 1",       # passes
        "assert f(2) == 2",       # fails (returns None)
        "assert f(3) == 3",       # fails (returns None)
    ]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        reward_partial=2.0,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "partial"
    assert bd["passed"] == 1
    # 2.0 * (1 / 3)
    assert r == pytest.approx(2.0 / 3.0)


def test_mbpp_model_code_crashes():
    """If the model code itself raises at import time, all_fail (no
    marker reaches stdout)."""
    code = "raise ValueError('model code is broken')\n"
    tests = ["assert True"]
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=tests,
        allow_unsafe_exec=True,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "all_fail"
    assert bd["passed"] == 0
    assert r == -1.5


def test_mbpp_no_tests():
    code = "def f(): return 1"
    r, bd = mbpp_test_reward(
        _completion(code), test_list=[], allow_unsafe_exec=True,
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "no_tests"
    assert r == 0.0


def test_mbpp_sandbox_off_by_default():
    """The unsafe-exec opt-in must be explicit — confirms the default-off
    behaviour didn't drift while we were touching the same function.
    """
    code = "def f(): return 1"
    r, bd = mbpp_test_reward(
        _completion(code),
        test_list=["assert f() == 1"],
        # allow_unsafe_exec=False (default)
        python_executable=sys.executable,
    )
    assert bd["verdict"] == "exec_disabled_set_allow_unsafe_exec"
    assert r == 0.0
