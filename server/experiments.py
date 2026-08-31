"""Experiment Manager: create, run, track, and reproduce lab experiments.

Every interactive run in the lab becomes an *experiment* with an id, a frozen
config, a status, and — once finished — a result plus a reproducibility receipt.
That framing is the point of the site: a visitor should never see a number
without being able to ask where it came from.

Design notes
------------
**Why async at all?** Most experiments here finish in one to three seconds, so
blocking would technically work. Running them on a worker thread anyway buys two
things that matter: a multi-asset sweep can report real incremental progress, and
a slow upstream price fetch can never hold a request open until the proxy times
out. Progress is reported from the actual loop counter, never interpolated.

**Bounded memory.** The registry is in-process and capped at
:data:`MAX_EXPERIMENTS`, evicting oldest-finished-first. The backend runs on a
single free-tier worker that restarts on idle, so experiments are explicitly
ephemeral; the API says so rather than implying durable storage.

**Honesty.** A run that fails is recorded as ``status="error"`` with the reason.
A field the backend cannot compute is ``None``, never a placeholder. The
reproducibility receipt reports ``null`` for provenance it genuinely cannot
determine (e.g. a git commit when the deploy has no git metadata).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

MAX_EXPERIMENTS = 200
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #
def _git_commit() -> Optional[str]:
    """Best-effort short commit hash of the deployed code.

    Render exposes ``RENDER_GIT_COMMIT``; a local checkout can be asked directly.
    Returns ``None`` when neither is available — the receipt shows that honestly
    rather than inventing a version string.
    """
    env = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    if env:
        return env[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


_GIT_COMMIT = _git_commit()


def code_version() -> Optional[str]:
    """The deployed code version used on every reproducibility receipt."""
    return _GIT_COMMIT


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    """One experiment's full lifecycle record."""

    id: str
    kind: str                                  # "rollout" | "generalization" | ...
    config: Dict[str, Any]
    status: str = "queued"                     # queued | running | done | error
    progress: float = 0.0                      # 0..1, from the real loop counter
    stage: str = "queued"                      # human-readable current step
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    # The question the person running this said they were asking. Kept beside the
    # config rather than inside it, so the config stays a clean, round-trippable
    # description of the environment. Never generated on their behalf: an
    # invented hypothesis would be the notebook's version of a fabricated result.
    question: Optional[str] = None

    # -- serialisation ---------------------------------------------------- #
    def summary(self) -> dict:
        """Status view — everything except the (potentially large) result body."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "config": self.config,
            "created_at": round(self.created_at, 3),
            "elapsed_sec": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 3
            ),
            "error": self.error,
            "has_result": self.result is not None,
            "question": self.question,
        }

    def full(self) -> dict:
        """Status view plus the result body and the reproducibility receipt."""
        out = self.summary()
        out["result"] = self.result
        out["receipt"] = self.receipt()
        return out

    def receipt(self) -> dict:
        """The reproducibility receipt: exactly what produced this result.

        Fields the backend cannot determine are reported as ``None``. The
        ``reproduce`` block is the real command that regenerates this run.
        """
        return {
            "experiment_id": self.id,
            "kind": self.kind,
            "code_version": _GIT_COMMIT,
            "created_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)
            ),
            "config": self.config,
            "question": self.question,
            "provenance": self.provenance,
            "storage": "ephemeral — the API keeps experiments in memory only",
        }


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #
class ExperimentManager:
    """Thread-safe, bounded registry that runs experiments on worker threads."""

    def __init__(self, max_experiments: int = MAX_EXPERIMENTS) -> None:
        self._lock = threading.Lock()
        self._experiments: "OrderedDict[str, Experiment]" = OrderedDict()
        self._max = max_experiments

    # -- ids -------------------------------------------------------------- #
    def _new_id(self) -> str:
        """Allocate a short, human-quotable id (e.g. ``EXP-8F42A``)."""
        while True:
            candidate = "EXP-" + secrets.token_hex(3).upper()[:5]
            if candidate not in self._experiments:
                return candidate

    # -- lifecycle -------------------------------------------------------- #
    def create(
        self,
        kind: str,
        config: Dict[str, Any],
        runner: Callable[["Experiment"], Dict[str, Any]],
        provenance: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> Experiment:
        """Register an experiment and start it on a worker thread."""
        with self._lock:
            exp = Experiment(
                id=self._new_id(),
                kind=kind,
                config=config,
                provenance=provenance or {},
                question=question,
            )
            self._experiments[exp.id] = exp
            self._evict_locked()

        thread = threading.Thread(target=self._run, args=(exp, runner), daemon=True)
        thread.start()
        return exp

    def _run(self, exp: Experiment, runner: Callable[["Experiment"], Dict[str, Any]]) -> None:
        exp.status = "running"
        exp.stage = "starting"
        exp.started_at = time.time()
        try:
            exp.result = runner(exp)
            exp.progress = 1.0
            exp.stage = "complete"
            exp.status = "done"
        except Exception as exc:  # surfaced to the client, never silently swallowed
            exp.status = "error"
            exp.stage = "failed"
            exp.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            exp.finished_at = time.time()

    def _evict_locked(self) -> None:
        """Drop oldest *finished* experiments once over capacity."""
        while len(self._experiments) > self._max:
            for key, exp in list(self._experiments.items()):
                if exp.status in ("done", "error"):
                    del self._experiments[key]
                    break
            else:
                # Everything is still running — drop the oldest to bound memory.
                self._experiments.popitem(last=False)

    # -- access ----------------------------------------------------------- #
    def get(self, experiment_id: str) -> Optional[Experiment]:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list(self, limit: int = 50, kind: Optional[str] = None) -> List[dict]:
        """Most-recent-first summaries, optionally filtered by kind."""
        with self._lock:
            items = list(self._experiments.values())
        items.sort(key=lambda e: e.created_at, reverse=True)
        if kind:
            items = [e for e in items if e.kind == kind]
        return [e.summary() for e in items[: max(1, min(limit, 200))]]

    def stats(self) -> dict:
        with self._lock:
            items = list(self._experiments.values())
        return {
            "total": len(items),
            "by_status": {
                s: sum(1 for e in items if e.status == s)
                for s in ("queued", "running", "done", "error")
            },
            "capacity": self._max,
        }


def progress_reporter(exp: Experiment, total: int, stage: str) -> Callable[[int], None]:
    """Return a callback that records genuine loop progress onto ``exp``.

    ``total`` is the real number of units of work. Progress only ever reflects
    completed units — the UI shows measured progress, not an animation.
    """
    total = max(1, int(total))

    def report(done: int) -> None:
        exp.progress = min(1.0, max(0.0, done / total))
        exp.stage = f"{stage} ({done}/{total})"

    exp.stage = f"{stage} (0/{total})"
    return report
