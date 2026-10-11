"""Structured progress log + heartbeat.

Two overlapping signals so nothing gets lost, written under the
`results_dir` handed to Progress.__init__ (uber: /results/<study-id>/uber/;
worker: /results/<study-id>/workers/<model>/):
  * append-only <results_dir>/progress.jsonl (durable, machine-readable)
  * stdout JSON lines (visible via `oc logs -f`)
  * <results_dir>/heartbeat.txt updated every N seconds
  * <results_dir>/state.json atomic snapshot of the last known phase

The public entry point is Progress.emit(event, **kwargs). It fsyncs the
progress log so a Pod eviction never loses the last line.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Progress:
    """Progress emitter shared by orchestrator and runners."""

    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.results_dir / "progress.jsonl"
        self.state_path = self.results_dir / "state.json"
        self.heartbeat_path = self.results_dir / "heartbeat.txt"
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._load_state()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_detail: dict[str, Any] = {}

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            except json.JSONDecodeError:
                return {}
        return {}

    # ---- progress events -------------------------------------------------
    def emit(self, phase: str, **kwargs: Any) -> None:
        """Emit one progress event; durable + also to stdout."""

        record = {
            "event": "progress",
            "ts": _utcnow_iso(),
            "phase": phase,
            **kwargs,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.progress_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
                fp.flush()
                os.fsync(fp.fileno())
            sys.stdout.write(line)
            sys.stdout.flush()

    # ---- state snapshot --------------------------------------------------
    def update_state(self, **kwargs: Any) -> None:
        """Atomically replace the state snapshot."""

        with self._lock:
            self._state.update(kwargs)
            tmp = self.state_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fp:
                json.dump(self._state, fp, ensure_ascii=False, indent=2)
                fp.flush()
                os.fsync(fp.fileno())
            tmp.replace(self.state_path)

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    # ---- heartbeat -------------------------------------------------------
    def set_heartbeat_detail(self, **kwargs: Any) -> None:
        with self._lock:
            self._heartbeat_detail = dict(kwargs)

    def start_heartbeat(self, interval_seconds: float) -> None:
        if self._heartbeat_thread is not None:
            return

        def _loop() -> None:
            while not self._heartbeat_stop.wait(interval_seconds):
                try:
                    payload = {
                        "ts": _utcnow_iso(),
                        "state": self._state,
                        "detail": self._heartbeat_detail,
                    }
                    tmp = self.heartbeat_path.with_suffix(".txt.tmp")
                    with tmp.open("w", encoding="utf-8") as fp:
                        json.dump(payload, fp, ensure_ascii=False)
                        fp.flush()
                        os.fsync(fp.fileno())
                    tmp.replace(self.heartbeat_path)
                except Exception:
                    # Heartbeat failures should never kill the run.
                    pass

        t = threading.Thread(target=_loop, name="progress-heartbeat", daemon=True)
        self._heartbeat_thread = t
        t.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
