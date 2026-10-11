"""SIGTERM/SIGINT handler emits a durable `study_aborted` line.

Kubernetes sends SIGTERM ~30 s before SIGKILL on Job deletion,
preemption, or Pod eviction. Without a handler the harness dies with
no trace and study-progress.sh / the uber aggregator have to
reconstruct the outcome from Job conditions. The handler emits one
last `study_aborted` progress line and flips state.json phase before
letting the process terminate normally.

The tests drive the handler synchronously (calling it as a plain
function with a signal number) rather than sending real signals —
that keeps the tests hermetic and portable to non-POSIX CI.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pricing_reversal_harness.__main__ import _install_abort_handler
from pricing_reversal_harness.progress import Progress


@pytest.fixture
def progress_tmp(tmp_path: Path) -> Progress:
    return Progress(tmp_path)


def _invoke_handler_directly(signum: int) -> None:
    """Fetch the registered handler for `signum` and call it, then
    restore the default so the test never actually kills its own
    process even if the handler forgets to swap it back."""
    h = signal.getsignal(signum)
    try:
        # os.kill inside the handler would kill this test process; block it.
        with patch("os.kill"):
            h(signum, None)  # type: ignore[misc]
    finally:
        signal.signal(signum, signal.SIG_DFL)


def test_handler_emits_study_aborted_on_sigterm(tmp_path: Path,
                                                progress_tmp: Progress) -> None:
    _install_abort_handler(progress_tmp)
    _invoke_handler_directly(signal.SIGTERM)

    text = (tmp_path / "progress.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.strip().split("\n") if line]
    aborted = [r for r in lines if r["phase"] == "study_aborted"]
    assert len(aborted) == 1
    assert aborted[0]["signal"] == "SIGTERM"


def test_handler_updates_state_phase(tmp_path: Path,
                                     progress_tmp: Progress) -> None:
    _install_abort_handler(progress_tmp)
    _invoke_handler_directly(signal.SIGTERM)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "aborted"
    assert state["signal"] == "SIGTERM"


def test_handler_reraises_default_action_after_emit(tmp_path: Path,
                                                    progress_tmp: Progress) -> None:
    """After emit, the handler swaps itself out for SIG_DFL and
    forwards the signal. A second SIGTERM would then kill the process
    normally — this test verifies the swap happened."""
    _install_abort_handler(progress_tmp)
    _invoke_handler_directly(signal.SIGTERM)
    # After invocation, the SIGTERM disposition should be back to
    # default so the follow-up SIGKILL doesn't get intercepted.
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_handler_emit_failure_does_not_raise(tmp_path: Path,
                                             progress_tmp: Progress) -> None:
    """If emit() itself fails (disk full, PVC unmounted, etc.), the
    handler must still swap to SIG_DFL + forward — swallowing the
    exception is correct because we're already shutting down."""
    _install_abort_handler(progress_tmp)
    with patch.object(Progress, "emit", side_effect=OSError("disk full")), \
         patch("os.kill"):
        # Should not raise
        h = signal.getsignal(signal.SIGTERM)
        h(signal.SIGTERM, None)  # type: ignore[misc]
    # And the disposition should still be swapped.
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
