"""Result files: run directories, JSON writing, the command log, and timing summaries.

Contract: specs/001-cpu-path-audit/contracts/results.md.

* Every file carries ``schema_version`` and ``run_id`` at the top level (rule 1).
* A condition that produced no measurement keeps its place with a ``status`` other than
  ``measured`` and a ``reason`` (rule 2). Items are validated before anything is written.
* Times are milliseconds, memory is bytes (rule 3).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

SCHEMA_VERSION = 1

#: Default results root. Git-ignored (constitution V). Tests pass their own root.
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

#: p95 from fewer samples than this is flagged (research.md R6).
LOW_SAMPLE_P95 = 20

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Status(str, Enum):
    """The only statuses a result item may carry."""

    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    PARTIAL = "partial"


class ResultValidationError(ValueError):
    """A result item or file breaks the results contract."""


def new_run_id() -> str:
    """Default run id: a UTC timestamp, sortable and safe as a directory name."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_dir(run_id: Optional[str] = None, root: Union[str, Path, None] = None) -> Path:
    """Create (or reuse) ``<root>/<run_id>/`` and return it."""
    if run_id is None:
        run_id = new_run_id()
    if not _RUN_ID.match(run_id):
        raise ValueError(
            "run id %r must be 1-128 characters of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit" % run_id)
    path = Path(root) if root is not None else RESULTS_ROOT
    path = path / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_item(item: Dict[str, Any], where: str = "item") -> None:
    """Check one result item: a known status, and a reason whenever it is not ``measured``."""
    if not isinstance(item, dict):
        raise ResultValidationError("%s: a result item must be a dict, got %s"
                                    % (where, type(item).__name__))
    if "status" not in item:
        raise ResultValidationError("%s: missing 'status'" % where)
    status = item["status"]
    if isinstance(status, Status):
        status = status.value
    allowed = [s.value for s in Status]
    if status not in allowed:
        raise ResultValidationError("%s: status %r is not one of %s" % (where, status, allowed))
    if status != Status.MEASURED.value:
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ResultValidationError(
                "%s: status %r needs a non-empty 'reason' (contracts/results.md rule 2)"
                % (where, status))


def _validate_payload(payload: Dict[str, Any], name: str) -> None:
    if "status" in payload:
        validate_item(payload, name)
    items = payload.get("items")
    if items is not None:
        if not isinstance(items, list):
            raise ResultValidationError("%s: 'items' must be a list" % name)
        for i, it in enumerate(items):
            validate_item(it, "%s#%d" % (name, i))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None  # JSON has no NaN/inf; a missing value is stated as null
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    item = getattr(obj, "item", None)  # numpy / torch scalars
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return obj


def write_json(run_path: Union[str, Path], name: str, payload: Union[Dict[str, Any], List[Any]]) -> Path:
    """Write ``<run_path>/<name>`` with ``schema_version`` and ``run_id`` at the top level.

    A list payload is stored as ``{"items": [...]}``. Items are validated first, so an invalid
    result never reaches disk. The write is atomic: a crash leaves the old file or the new one.
    """
    run_path = Path(run_path)
    if isinstance(payload, list):
        payload = {"items": payload}
    if not isinstance(payload, dict):
        raise ResultValidationError("%s: payload must be a dict or a list" % name)
    _validate_payload(payload, name)
    body = {"schema_version": SCHEMA_VERSION, "run_id": run_path.name}
    body.update({k: v for k, v in payload.items() if k not in ("schema_version", "run_id")})
    text = json.dumps(_jsonable(body), indent=2, ensure_ascii=False, allow_nan=False)
    target = run_path / name
    target.parent.mkdir(parents=True, exist_ok=True)      # `name` may carry a subdirectory (tier_e/probs.L512.json)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix="." + target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def read_json(run_path: Union[str, Path], name: str) -> Dict[str, Any]:
    """Read a result file and check its top-level contract fields."""
    path = Path(run_path) / name
    with open(path, encoding="utf-8") as f:
        body = json.load(f)
    if not isinstance(body, dict) or "schema_version" not in body or "run_id" not in body:
        raise ResultValidationError("%s: missing schema_version or run_id" % path)
    return body


def format_command(argv: Sequence[str], prefix: Sequence[str] = ("python", "-m", "experiments")) -> str:
    """The command line as the user would type it on this platform."""
    parts = list(prefix) + [str(a) for a in argv]
    if sys.platform == "win32":
        return subprocess.list2cmdline(parts)
    import shlex
    return " ".join(shlex.quote(p) for p in parts)


def append_command(run_path: Union[str, Path], argv: Sequence[str]) -> Path:
    """Append the exact command line to ``commands.txt`` (never overwritten)."""
    run_path = Path(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    path = run_path / "commands.txt"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(format_command(argv) + "\n")
    return path


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default method), q in [0, 100]."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def summarize(samples_ms: Iterable[float]) -> Dict[str, Any]:
    """min, p50, p95, mean, sample std and n of timings in milliseconds.

    ``low_sample_p95`` is true when there are fewer than 20 samples (research.md R6).
    """
    vals = [float(v) for v in samples_ms]
    if not vals:
        raise ValueError("summarize() needs at least one sample")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("summarize() got a non-finite timing")
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in s) / (n - 1)) if n > 1 else 0.0
    return {
        "min": s[0],
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "mean": mean,
        "std": std,
        "n": n,
        "low_sample_p95": n < LOW_SAMPLE_P95,
    }


# --------------------------------------------------------------------------- Phase 2 helpers (T007)
#
# Contract: specs/002-decision-benchmark-baselines/contracts/results.md. Predictions are
# append-only JSON lines so an interrupted run resumes without rewriting finished items.

#: Refusal reasons of contracts/cli.md ("Exit and refusal reasons").
REFUSAL_REASONS = ("fingerprint_mismatch", "split_locked", "audit_required", "audit_stale",
                   "reference_missing", "device_mismatch")


class Refusal(ValueError):
    """A Phase 2 command refuses to proceed; `reason` is one of `REFUSAL_REASONS`."""

    def __init__(self, reason: str, message: str):
        if reason not in REFUSAL_REASONS:
            raise ValueError("unknown refusal reason %r" % reason)
        super().__init__("%s: %s" % (reason, message))
        self.reason = reason


class SplitLocked(Refusal):
    """A final-split item was about to be scored (FR-013, SC-008)."""

    def __init__(self, message: str = "the final test split is not scored in this phase (FR-013)"):
        super().__init__("split_locked", message)


def _drop_partial_tail(path: Path) -> None:
    """Cut a last line that an interrupted append left without its newline.

    `read_jsonl` skips such a fragment, but an append glued onto it would corrupt the new record too.
    """
    try:
        with open(path, "rb+") as f:
            if f.seek(0, 2) == 0:
                return
            f.seek(-1, 2)
            if f.read(1) == b"\n":
                return
            f.seek(0)
            data = f.read()
            f.truncate(data.rfind(b"\n") + 1)
    except FileNotFoundError:
        pass


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Append one JSON object as one line; earlier lines are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
    _drop_partial_tail(path)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSON-lines file. A last line cut short by an interrupted write is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            if i >= len(lines) - 2:      # truncated tail from an interrupted append
                continue
            raise
    return out


def _param_text(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    return "-".join("%s%s" % (k, params[k]) for k in sorted(params))


def condition_id(name: str, params: Optional[Dict[str, Any]], variant: str, device: str, length: Any) -> str:
    """Stable id ``<name>[.<params>].<variant>.<device>.L<length>``."""
    parts = [name]
    text = _param_text(params)
    if text:
        parts.append(text)
    parts += [variant or "none", device, "L%s" % length]
    return ".".join(parts)


def assert_no_final(items: Iterable[Dict[str, Any]]) -> None:
    """Raise `SplitLocked` when any item belongs to the final split."""
    for it in items:
        if it.get("split") == "final":
            raise SplitLocked("item %r belongs to the final test split (FR-013)" % it.get("item_id"))
