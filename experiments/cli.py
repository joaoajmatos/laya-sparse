"""Command-line entry point: ``python -m experiments <command> [options]``.

Contract: specs/001-cpu-path-audit/contracts/cli.md.

Commands register themselves with `command(...)`; later tasks add manifest, audit, sweep,
profile, kernels, report and all. Exit codes: 0 when the command completed, including when some
conditions were recorded as ``unsupported`` or ``failed`` (those are results). Non-zero only when
the tool itself could not run (bad arguments, model cannot load, device is not CPU).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import results

DEFAULT_MODEL = "convaiinnovations/laya"
EXIT_OK = 0
EXIT_TOOL_ERROR = 1
EXIT_USAGE = 2


class ToolError(Exception):
    """The tool cannot run: reported on stderr and turned into a non-zero exit."""


@dataclass
class Command:
    name: str
    help: str
    run: Callable[[argparse.Namespace], Optional[int]]
    add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None
    #: run directory used when `--run-id` is not given (Phase 2 data commands log to one stable place)
    default_run_id: Optional[str] = None


#: name -> Command, in registration order.
COMMANDS: Dict[str, Command] = {}


def command(name: str, help: str, add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None,
            default_run_id: Optional[str] = None):
    """Decorator registering `run(args)` as a subcommand."""
    def register(run):
        if name in COMMANDS:
            raise ValueError("command %r registered twice" % name)
        COMMANDS[name] = Command(name, help, run, add_arguments, default_run_id)
        return run
    return register


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def int_list(text: str) -> List[int]:
    """Parse ``"128,512,4096"``."""
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers, got %r" % text)
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("expected positive integers, got %r" % text)
    return values


def common_parser() -> argparse.ArgumentParser:
    """Options every command accepts (contracts/cli.md, "Common options")."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("common options")
    g.add_argument("--run-id", default=None,
                   help="results directory name under experiments/results/ (default: UTC timestamp)")
    g.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint id or local path (default: %(default)s)")
    g.add_argument("--revision", default=None,
                   help="commit to pin; 'reviewed' uses laya.revisions.PINNED_REVISIONS")
    g.add_argument("--threads", type=_positive_int, default=None,
                   help="torch intra-op threads, fixed for the whole run (default: physical core count)")
    g.add_argument("--seed", type=int, default=0, help="base seed (default: %(default)s)")
    g.add_argument("--mha-fastpath", choices=["on", "off"], default=None,
                   help="PyTorch's TransformerEncoderLayer inference fast path, used by Laya's decision head. "
                        "'on' (default) is native Laya; 'off' is a labelled variant (research.md R17). One setting per run.")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Phase 1 CPU path audit and measurement for Laya (specs/001-cpu-path-audit).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for cmd in COMMANDS.values():
        # A fresh copy of the common options per command: `parents=` shares the Action objects, so a
        # command that changes a default (Phase 2 commands default to the fine-tuned checkpoint) would
        # otherwise change it for every other command too.
        sp = sub.add_parser(cmd.name, help=cmd.help, description=cmd.help, parents=[common_parser()])
        if cmd.add_arguments:
            cmd.add_arguments(sp)
    if not COMMANDS:
        parser.epilog = "No commands are registered yet."
    return parser


def resolve_common(args: argparse.Namespace) -> argparse.Namespace:
    """Fill defaults that need torch or the filesystem, and create the run directory.

    The thread count is resolved once here and passed to every child, so every condition in a
    run uses the same value (research.md R12). Its source is recorded.
    """
    if args.threads is None:
        from .runner import default_threads
        args.threads = default_threads()
        args.threads_source = "default"
    else:
        args.threads_source = "user"
    args.run_path = results.run_dir(args.run_id)
    args.run_id = args.run_path.name
    _match_run_manifest(args)
    if getattr(args, "mha_fastpath", "on") is None:
        args.mha_fastpath = "on"
    return args


def _match_run_manifest(args: argparse.Namespace) -> None:
    """Keep one model and revision per run directory.

    When the run already has a manifest, a command that names no revision uses the manifest's
    resolved commit (so `sweep` after `audit --revision reviewed` measures the same weights, and
    works offline). A different model or revision in the same run is refused.
    """
    try:
        man = results.read_json(args.run_path, "manifest.json")
    except (OSError, ValueError):
        return
    model = man.get("model") or {}
    recorded_fp = (man.get("runtime") or {}).get("mha_fastpath", True)
    if getattr(args, "mha_fastpath", "on") is None:
        args.mha_fastpath = "on" if recorded_fp else "off"   # inherit, like the revision
    elif (args.mha_fastpath == "on") != bool(recorded_fp):
        raise ToolError("run %r was recorded with --mha-fastpath %s; use another --run-id for the other setting"
                        % (args.run_id, "on" if recorded_fp else "off"))
    if model.get("id") and model["id"] != args.model:
        raise ToolError("run %r was recorded with model %r; this command asks for %r. Use another --run-id."
                        % (args.run_id, model["id"], args.model))
    recorded = model.get("revision")
    if not recorded or recorded == "local" or model.get("revision_source") == "local":
        return
    if args.revision in (None, ""):
        args.revision = recorded
        print("using revision %s from this run's manifest.json" % recorded)
        return
    from .runner import resolve_model_revision
    try:
        wanted, _ = resolve_model_revision(args.model, args.revision)
    except ValueError as exc:
        raise ToolError(str(exc))
    if wanted != recorded:
        raise ToolError("run %r was recorded at revision %s; this command asks for %s. Use another --run-id."
                        % (args.run_id, recorded, wanted))


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_commands()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    cmd = COMMANDS[args.command]
    if args.run_id is None and cmd.default_run_id:
        args.run_id = cmd.default_run_id
    try:
        resolve_common(args)
        results.append_command(args.run_path, argv)
        code = cmd.run(args)
    except ToolError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_TOOL_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return EXIT_OK if code is None else int(code)


def _load_commands() -> None:
    """Commands are registered below in this module; nothing else to import yet."""
    return None


# --------------------------------------------------------------------------- shared helpers

def load_model(args: argparse.Namespace):
    """Load the agent on CPU for a command; tool-level failures become `ToolError`."""
    from .runner import load_agent
    try:
        return load_agent(args.model, revision=args.revision, threads=args.threads,
                          mha_fastpath=args.mha_fastpath == "on")
    except Exception as exc:  # model cannot load, or device is not CPU
        raise ToolError("could not load %r on CPU: %s: %s" % (args.model, type(exc).__name__, exc))


def write_manifest(args: argparse.Namespace, agent, info) -> str:
    from .manifest import ManifestError, build_manifest
    try:
        body = build_manifest(agent, info, run_id=args.run_id, seed=args.seed,
                              threads_source=args.threads_source)
    except ManifestError as exc:
        raise ToolError(str(exc))
    if body["device"]["effective"] != "cpu":
        raise ToolError("effective device is %s, not cpu" % body["device"]["effective"])
    return str(results.write_json(args.run_path, "manifest.json", body))


# --------------------------------------------------------------------------- manifest / audit (T019)

@command("manifest", "Write manifest.json: the pinned environment of this run.")
def _cmd_manifest(args: argparse.Namespace) -> int:
    agent, info = load_model(args)
    path = write_manifest(args, agent, info)
    print(path)
    return EXIT_OK


def _audit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=None,
                   help="comma-separated total lengths to check against positional capacity")


@command("audit", "Write audit.json (and manifest.json): what the loaded model actually is.", _audit_args)
def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import build_audit, format_schedule
    agent, info = load_model(args)
    print(write_manifest(args, agent, info))
    body = build_audit(agent, info, lengths=args.lengths)
    print(format_schedule(body))
    print(results.write_json(args.run_path, "audit.json", body))
    return EXIT_OK


# --------------------------------------------------------------------------- sweep / profile (T027, T028)

DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_TIME_CAP = 900.0


def _repeats_arg(text: str):
    if text == "auto":
        return "auto"
    try:
        return _positive_int(text)
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")


def _known_position_limit(run_path) -> Optional[int]:
    try:
        return int(results.read_json(run_path, "audit.json")["positional"]["max_position_embeddings"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def session_info() -> Dict[str, Any]:
    """Machine state when a measuring command starts: the power plan may differ from the manifest's."""
    import datetime as _dt
    from .manifest import hardware_info
    hw = hardware_info()
    return {"started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "power_scheme": hw.get("power_scheme", "not recorded"),
            "on_ac_power": hw.get("on_ac_power", "not recorded")}


def sweep_conditions(lengths, questions, batch_sizes, grid: str = "axes"):
    """(length, questions, batch) tuples. ``axes`` varies one factor at a time around the primary
    curve (questions=1, batch=1); ``full`` is the complete product."""
    out = []
    for L in lengths:
        if grid == "full":
            out.extend((L, q, b) for q in questions for b in batch_sizes)
            continue
        out.append((L, 1, 1))
        out.extend((L, q, 1) for q in questions if q != 1)
        out.extend((L, 1, b) for b in batch_sizes if b != 1)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _abort_on_load_failure(item: Dict[str, Any], args: argparse.Namespace, name: str, body) -> None:
    """A model that cannot load is a tool error (contracts/cli.md), not a failed measurement."""
    if item.get("exception_type") != "ModelLoadError":
        return
    results.write_json(args.run_path, name, body)
    raise ToolError("the model could not be loaded in the measurement process, so nothing was measured "
                    "(%s). Partial file: %s" % (item.get("reason"), args.run_path / name))


def _unsupported(L, q, opts, b, limit):
    from .timing import condition_kind
    kind = condition_kind(q, b)
    return {"status": "unsupported",
            "reason": "%d tokens exceeds positional capacity %d (from audit.json); no model call made" % (L, limit),
            "condition": {"total_tokens": L, "questions": q, "options_per_question": opts, "batch_size": b},
            "kind": kind, "call": "predict_batch" if kind == "batch" else "predict"}


def _sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--questions", type=int_list, default=[1, 2, 5, 10])
    p.add_argument("--options", type=_positive_int, default=2, help="options per question (default 2)")
    p.add_argument("--batch-sizes", type=int_list, default=[1, 4, 8])
    p.add_argument("--grid", choices=["axes", "full"], default="axes",
                   help="axes: vary questions and batch size one at a time around questions=1, batch=1 "
                        "(default); full: every combination")
    p.add_argument("--repeats", type=_repeats_arg, default="auto")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--time-cap", type=float, default=DEFAULT_TIME_CAP,
                   help="seconds per condition, model load included (default %(default)s)")
    _canary_args(p)


def _canary_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--canary-length", type=int, default=512,
                   help="drift canary: single-question length re-measured before each length and at "
                        "the end (0 disables; default %(default)s)")
    p.add_argument("--canary-repeats", type=_positive_int, default=5)
    p.add_argument("--drift-threshold", type=float, default=0.05,
                   help="flag conditions whose surrounding canaries differ from the first by more than "
                        "this fraction (default %(default)s)")


class Canary:
    """Re-measures one fixed short condition to detect machine drift (heat, turbo budget, other
    programs) during a long run. Canary results never enter the measurement curves."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.enabled = bool(args.canary_length)
        self.runs: List[Dict[str, Any]] = []

    def measure(self, before_item: Optional[int]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        from .runner import run_condition
        a = self.args
        spec = {"model": a.model, "revision": a.revision, "threads": a.threads, "mha_fastpath": a.mha_fastpath == "on", "seed": a.seed + 7_000_000,
                "total_tokens": a.canary_length, "questions": 1, "options": 2, "batch_size": 1,
                "repeats": a.canary_repeats, "warmup": 2}
        item = run_condition("experiments.timing:measure_condition", spec, time_cap=a.time_cap)
        p50 = (item.get("timings_ms") or {}).get("p50")
        base = self.baseline
        run = {"index": len(self.runs), "before_item": before_item, "status": item["status"],
               "p50_ms": p50, "samples_ms": item.get("samples_ms"),
               "ratio_to_first": (p50 / base) if (p50 and base) else (1.0 if p50 and not self.runs else None)}
        if item.get("reason"):
            run["reason"] = item["reason"]
        if item.get("exception_type"):
            run["exception_type"] = item["exception_type"]
        self.runs.append(run)
        print("  canary L=%d: %s" % (a.canary_length, "p50 %.1f ms (x%.3f of first)" % (p50, run["ratio_to_first"])
                                     if p50 else item["status"]), flush=True)
        return item

    @property
    def baseline(self) -> Optional[float]:
        for r in self.runs:
            if r["p50_ms"]:
                return r["p50_ms"]
        return None

    def annotate(self, items: List[Dict[str, Any]]) -> None:
        """Give every item the canaries measured just before and just after it."""
        if not self.enabled:
            return
        thr = self.args.drift_threshold
        for i, it in enumerate(items):
            before = [r for r in self.runs if r["before_item"] is not None and r["before_item"] <= i]
            after = [r for r in self.runs if r["before_item"] is None or r["before_item"] > i]
            b = before[-1] if before else None
            a = after[0] if after else None
            ratios = [r["ratio_to_first"] for r in (b, a) if r and r.get("ratio_to_first")]
            worst = max((abs(x - 1.0) for x in ratios), default=None)
            it["drift"] = {
                "canary_before": b["index"] if b else None,
                "canary_after": a["index"] if a else None,
                "ratios_to_first": ratios,
                "max_deviation": worst,
                "flagged": bool(worst is not None and worst > thr),
                "derived_from": ["%s#canary.runs.%d" % (self.args._results_file, r["index"]) for r in (b, a) if r],
            }

    def block(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        dev = [abs(r["ratio_to_first"] - 1.0) for r in self.runs if r.get("ratio_to_first")]
        return {"enabled": True, "length": self.args.canary_length, "repeats": self.args.canary_repeats,
                "threshold": self.args.drift_threshold, "baseline_p50_ms": self.baseline,
                "max_deviation": max(dev) if dev else None, "runs": self.runs}


@command("sweep", "Clean latency sweep, one subprocess per condition; writes sweep.json.", _sweep_args)
def _cmd_sweep(args: argparse.Namespace) -> int:
    from .runner import run_condition
    limit = _known_position_limit(args.run_path)
    items = []
    conds = sweep_conditions(args.lengths, args.questions, args.batch_sizes, args.grid)
    args._results_file = "sweep.json"
    canary = Canary(args)
    last_len = None

    session = session_info()

    def body():
        canary.annotate(items)
        return {"grid": args.grid, "session": session, "items": items, "canary": canary.block()}

    for n, (L, q, b) in enumerate(conds, 1):
        if L != last_len and not (limit is not None and L > limit):
            _abort_on_load_failure(canary.measure(len(items)) or {}, args, "sweep.json", body())
            last_len = L
        if limit is not None and L > limit:
            item = _unsupported(L, q, args.options, b, limit)
        else:
            spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                    "seed": args.seed, "total_tokens": L, "questions": q, "options": args.options,
                    "batch_size": b, "repeats": args.repeats, "warmup": args.warmup}
            item = run_condition("experiments.timing:measure_condition", spec, time_cap=args.time_cap)
            item.setdefault("condition", {"total_tokens": L, "questions": q,
                                          "options_per_question": args.options, "batch_size": b})
        items.append(item)
        _abort_on_load_failure(item, args, "sweep.json", body())
        p50 = (item.get("timings_ms") or {}).get("p50")
        print("[%d/%d] L=%d q=%d b=%d: %s%s" % (n, len(conds), L, q, b, item["status"],
                                              " p50 %.1f ms" % p50 if p50 is not None else
                                              " (%s)" % item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "sweep.json", body())
    canary.measure(None)
    results.write_json(args.run_path, "sweep.json", body())
    flagged = [i for i, it in enumerate(items) if (it.get("drift") or {}).get("flagged")]
    if flagged:
        print("drift above %.0f%% around %d condition(s): %s" % (100 * args.drift_threshold, len(flagged), flagged))
    print(args.run_path / "sweep.json")
    return EXIT_OK


def _profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--repeats", type=_positive_int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--time-cap", type=float, default=DEFAULT_TIME_CAP)
    p.add_argument("--overhead-length", type=int, default=512,
                   help="length for the wrappers on/off overhead check (0 to skip)")
    _canary_args(p)


def _clean_p50(run_path) -> Dict[int, Dict[str, Any]]:
    try:
        sweep = results.read_json(run_path, "sweep.json")
    except OSError:
        return {}
    out = {}
    for i, it in enumerate(sweep.get("items", [])):
        c = it.get("condition", {})
        if it.get("kind") == "single" and it.get("status") == "measured" and it.get("timings_ms"):
            out[c["total_tokens"]] = {"p50": it["timings_ms"]["p50"], "ref": "sweep.json#%d" % i}
    return out


@command("profile", "Profiled component breakdown per length (not headline latency); writes profile.json.",
         _profile_args)
def _cmd_profile(args: argparse.Namespace) -> int:
    from . import profile as P
    from .runner import run_condition
    limit = _known_position_limit(args.run_path)
    items = []
    args._results_file = "profile.json"
    canary = Canary(args)
    session = session_info()
    for n, L in enumerate(args.lengths, 1):
        if limit is not None and L > limit:
            item = _unsupported(L, 1, 2, 1, limit)
            item["kind"] = "profile"
        else:
            _abort_on_load_failure(canary.measure(len(items)) or {}, args, "profile.json", {"items": items})
            spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                    "seed": args.seed, "total_tokens": L, "repeats": args.repeats, "warmup": args.warmup}
            item = run_condition("experiments.profile:profile_condition", spec, time_cap=args.time_cap)
            item.setdefault("condition", {"total_tokens": L, "questions": 1,
                                          "options_per_question": 2, "batch_size": 1})
            item["profile_run"] = True
        items.append(item)
        _abort_on_load_failure(item, args, "profile.json", {"items": items})
        print("[%d/%d] profile L=%d: %s" % (n, len(args.lengths), L, item["status"]), flush=True)
    canary.measure(None)
    canary.annotate(items)
    clean = _clean_p50(args.run_path)
    for it in items:
        c = clean.get(it["condition"]["total_tokens"])
        if c and it.get("total_profiled_ms"):
            it["total_clean_ms"] = c["p50"]
            it["profiled_to_clean_ratio"] = it["total_profiled_ms"] / c["p50"]
            it["derived_from"] = [c["ref"]]
    scal = P.scaling(items)
    window = None
    try:
        audit = results.read_json(args.run_path, "audit.json")
        window = audit["encoder"].get("local_attention")
    except OSError:
        audit = None
    verdict = P.executed_work_verdict(scal, window)
    body = {"session": session, "items": items, "scaling": scal, "executed_work": verdict, "canary": canary.block()}
    if args.overhead_length:
        spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                "seed": args.seed, "total_tokens": args.overhead_length, "repeats": 10, "warmup": 2}
        body["overhead_check"] = run_condition("experiments.timing:compare_paths", spec,
                                               time_cap=args.time_cap)
    print(results.write_json(args.run_path, "profile.json", body))
    if audit is not None:
        audit["executed_work_note"] = dict(verdict, derived_from=["profile.json#scaling"])
        print(results.write_json(args.run_path, "audit.json", audit))
    else:
        print("audit.json not found in this run; the executed-work verdict is only in profile.json")
    print("executed work: %s - %s" % (verdict["verdict"], verdict["note"]))
    return EXIT_OK


# --------------------------------------------------------------------------- kernels (T037)

def _kernels_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--impls", default="dense,dense_masked,local,gas",
                   help="comma-separated from dense, dense_masked, local, gas (default: all)")
    p.add_argument("--block-sizes", type=int_list, default=[128, 256, 512])
    p.add_argument("--selection-sizes", type=int_list, default=[512, 1024],
                   help="tokens gathered per query block by gas; must be a multiple of the block "
                        "size and at least 3 blocks (default 512,1024)")
    p.add_argument("--repeats", type=_repeats_arg, default="auto")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--time-cap", type=float, default=300.0)


def kernel_conditions(impls, lengths, blocks, selections):
    out = []
    for L in lengths:
        for impl in impls:
            if impl == "dense":
                out.append({"impl": impl, "length": L, "block": None, "selection_blocks": None})
            elif impl in ("dense_masked", "local"):
                out.extend({"impl": impl, "length": L, "block": b, "selection_blocks": None} for b in blocks)
            else:
                for b in blocks:
                    for s in selections:
                        out.append({"impl": impl, "length": L, "block": b, "selection_tokens": s,
                                    "selection_blocks": s // b if s % b == 0 else None})
    return out


@command("kernels", "Attention microbenchmarks with correctness checks; writes kernels.json and floor.json.",
         _kernels_args)
def _cmd_kernels(args: argparse.Namespace) -> int:
    from .floor import build_floor
    from .kernels.bench import IMPLS, executed_work
    from .kernels.reference import score_matrix_bytes
    from .runner import run_condition
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    bad = [i for i in impls if i not in IMPLS]
    if bad:
        raise ToolError("unknown implementation(s) %s; choose from %s" % (bad, sorted(IMPLS)))
    try:
        audit = results.read_json(args.run_path, "audit.json")
    except OSError:
        raise ToolError("kernels takes its shapes from audit.json; run `audit` for run %r first" % args.run_id)
    heads, head_dim = audit["encoder"]["num_heads"], audit["encoder"]["head_dim"]
    limit = audit["positional"]["max_position_embeddings"]
    items = []
    session = session_info()
    conds = kernel_conditions(impls, args.lengths, args.block_sizes, args.selection_sizes)
    for n, c in enumerate(conds, 1):
        shape = {"batch": 1, "heads": heads, "length": c["length"], "head_dim": head_dim,
                 "block": c["block"], "selection_blocks": c.get("selection_blocks"),
                 "selection_tokens": c.get("selection_tokens")}
        if c["impl"] == "gas" and (c["selection_blocks"] is None or c["selection_blocks"] < 3):
            item = {"impl": c["impl"], "shape": shape, "status": "unsupported",
                    "reason": "selection of %s tokens is not a whole number of at least 3 blocks of %d"
                              % (c.get("selection_tokens"), c["block"])}
        elif c["length"] > limit:
            item = {"impl": c["impl"], "shape": shape, "status": "unsupported",
                    "reason": "%d exceeds the model's positional capacity %d" % (c["length"], limit)}
        else:
            spec = {"impl": c["impl"], "length": c["length"], "heads": heads, "head_dim": head_dim,
                    "block": c["block"], "selection_blocks": c.get("selection_blocks"),
                    "repeats": args.repeats, "warmup": args.warmup, "threads": args.threads,
                    "seed": args.seed}
            item = run_condition("experiments.kernels.bench:bench_condition", spec, time_cap=args.time_cap)
            item.setdefault("impl", c["impl"])
            item.setdefault("shape", shape)
        item.setdefault("score_matrix_bytes_analytical", score_matrix_bytes(1, heads, c["length"]))
        items.append(item)
        p50 = (item.get("timings_ms") or {}).get("p50")
        print("[%d/%d] %s L=%d block=%s sel=%s: %s%s" % (
            n, len(conds), c["impl"], c["length"], c["block"], c.get("selection_blocks"), item["status"],
            " p50 %.1f ms" % p50 if p50 is not None else " (%s)" % item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "kernels.json", {"shapes_from": "audit.json", "items": items})
    body = {"shapes_from": "audit.json", "session": session, "heads": heads, "head_dim": head_dim, "items": items,
            "executed_work": executed_work(items), "not_run": []}
    try:
        floor = build_floor(args.run_path)
        print(results.write_json(args.run_path, "floor.json", floor))
    except FileNotFoundError as exc:
        body["not_run"].append({"check": "cost floor (floor.json)", "reason": str(exc)})
        print("floor.json not written: %s" % exc)
    print(results.write_json(args.run_path, "kernels.json", body))
    for g in body["executed_work"]["groups"]:
        print("  %-12s block=%-4s sel=%-4s exponent=%s -> %s" % (
            g["impl"], g["block"], g["selection_blocks"],
            "%.2f" % g["time_exponent"] if g["time_exponent"] is not None else "n/a", g["verdict"]))
    return EXIT_OK


# --------------------------------------------------------------------------- GPU reference (R15)

def _gpu_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=[512, 2048])
    p.add_argument("--repeats", type=_positive_int, default=30)
    p.add_argument("--warmup", type=int, default=5)


@command("gpu-reference",
         "GPU-only check of the historical ~33 ms figure; writes gpu_reference.json (never a CPU result).",
         _gpu_args)
def _cmd_gpu_reference(args: argparse.Namespace) -> int:
    from .gpu import GpuUnavailable, measure_gpu
    try:
        body = measure_gpu(args.model, args.revision, args.lengths, args.repeats, args.warmup, args.seed)
    except GpuUnavailable as exc:
        raise ToolError(str(exc))
    for it in body["items"]:
        p50 = (it.get("timings_ms") or {}).get("p50")
        print("GPU L=%d: %s" % (it["condition"]["total_tokens"],
                                "p50 %.1f ms" % p50 if p50 is not None else it.get("reason")))
    print(results.write_json(args.run_path, "gpu_reference.json", body))
    return EXIT_OK


# --------------------------------------------------------------------------- report / all (T040)

@command("report", "Assemble report.json and report.md from this run's result files.")
def _cmd_report(args: argparse.Namespace) -> int:
    from .report import ReportError, write_report
    try:
        j, m = write_report(args.run_path)
    except ReportError as exc:
        raise ToolError(str(exc))
    print(j)
    print(m)
    return EXIT_OK


#: step -> options of `all` it receives (only when given)
_ALL_STEPS = [
    ("audit", ["--lengths"]),
    ("sweep", ["--lengths", "--questions", "--batch-sizes", "--repeats", "--time-cap", "--canary-length"]),
    ("profile", ["--lengths", "--repeats", "--time-cap", "--canary-length", "--overhead-length"]),
    ("kernels", ["--lengths", "--impls", "--block-sizes", "--selection-sizes", "--repeats"]),
    ("report", []),
]


def _all_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", default=None)
    p.add_argument("--questions", default=None)
    p.add_argument("--batch-sizes", default=None)
    p.add_argument("--repeats", default=None)
    p.add_argument("--time-cap", default=None)
    p.add_argument("--canary-length", default=None)
    p.add_argument("--overhead-length", default=None)
    p.add_argument("--impls", default=None)
    p.add_argument("--block-sizes", default=None)
    p.add_argument("--selection-sizes", default=None)


@command("all", "Run audit (with manifest), sweep, profile, kernels and report in order.", _all_args)
def _cmd_all(args: argparse.Namespace) -> int:
    parser = build_parser()
    common = ["--run-id", args.run_id, "--model", args.model, "--threads", str(args.threads), "--seed", str(args.seed)]
    common += ["--mha-fastpath", args.mha_fastpath]
    if args.revision:
        common += ["--revision", args.revision]
    for step, passed in _ALL_STEPS:
        extra = []
        for opt in passed:
            val = getattr(args, opt.lstrip("-").replace("-", "_"))
            if val is not None:
                extra += [opt, str(val)]
        ns = parser.parse_args([step] + common + extra)
        ns.threads_source = args.threads_source
        ns.run_path = args.run_path
        _match_run_manifest(ns)
        print("== %s ==" % step, flush=True)
        code = COMMANDS[step].run(ns)
        if code:
            return code
    return EXIT_OK


# --------------------------------------------------------------------------- instrumentation A/B (R16)

def _abtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=[2048, 8192])
    p.add_argument("--repeats", type=_positive_int, default=4)
    p.add_argument("--modes", default="clean,wrappers,labels,profiler,all")
    p.add_argument("--time-cap", type=float, default=2400.0)


@command("abtest", "Same documents in one process with and without each kind of instrumentation; "
                   "writes abtest.json.", _abtest_args)
def _cmd_abtest(args: argparse.Namespace) -> int:
    from .abtest import MODES
    from .runner import run_condition
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or "clean" not in modes:
        raise ToolError("modes must include clean and come from %s" % (MODES,))
    session = session_info()
    items = []
    for L in args.lengths:
        spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on", "seed": args.seed,
                "total_tokens": L, "repeats": args.repeats, "modes": modes}
        item = run_condition("experiments.abtest:abtest_condition", spec, time_cap=args.time_cap)
        item.setdefault("condition", {"total_tokens": L, "questions": 1, "options_per_question": 2, "batch_size": 1})
        items.append(item)
        _abort_on_load_failure(item, args, "abtest.json", {"session": session, "items": items})
        ratios = item.get("ratio_to_clean") or {}
        print("L=%d %s: %s" % (L, item["status"], ", ".join("%s x%.3f" % (m, r) for m, r in ratios.items() if r)
                               or item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "abtest.json", {"session": session, "items": items})
    print(args.run_path / "abtest.json")
    return EXIT_OK


# --------------------------------------------------------------------------- Phase 2 data commands (specs/002)

def _phase2_defaults(p: argparse.ArgumentParser) -> None:
    """Phase 2 commands default to the pinned fine-tuned checkpoint at its reviewed revision."""
    from .evalrun import CHECKPOINTS
    p.set_defaults(model=CHECKPOINTS["fine_tuned"], revision="reviewed")


def _data_root_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default=None, help="data directory (default experiments/data)")


def _data_import_args(p: argparse.ArgumentParser) -> None:
    from . import data
    _data_root_arg(p)
    p.add_argument("--data-revision", default=data.DATASET_REVISION,
                   help="dataset revision to import (default: the pinned sha)")
    p.add_argument("--refresh", action="store_true", help="re-download the files and verify them again")


@command("data-import", "Import the pinned upstream typed-decisions dataset; write experiments/data/manifest.json.",
         _data_import_args, default_run_id="data")
def _cmd_data_import(args: argparse.Namespace) -> int:
    from . import data
    from .results import Refusal
    try:
        man = data.import_dataset(revision=args.data_revision, root=args.data_root, refresh=args.refresh)
    except (Refusal, data.DataError) as exc:
        raise ToolError(str(exc))
    root = data.data_root(args.data_root)
    print("dataset %s @ %s" % (man["source"], man["revision"]))
    for s in data.SPLITS:
        sp = man["splits"][s]
        print("  %s: %d cases, %d questions %s" % (s, sp["cases"], sp["questions"], sp["questions_by_type"]))
    print("fingerprint %s" % man["fingerprint"]["combined"])
    print("low-confidence cutoff %.4f (bottom quartile of train)" % man["low_confidence"]["cutoff"])
    print(root / "manifest.json")
    return EXIT_OK


def _length_profile_args(p: argparse.ArgumentParser) -> None:
    _data_root_arg(p)
    _phase2_defaults(p)


@command("length-profile", "Token-length profile of the upstream cases under Laya's tokenizer.",
         _length_profile_args, default_run_id="data")
def _cmd_length_profile(args: argparse.Namespace) -> int:
    import json
    from . import lengths
    agent, _ = load_model(args)
    try:
        path = lengths.write_length_profile(agent, args.model, args.data_root)
    except FileNotFoundError as exc:
        raise ToolError(str(exc))
    prof = json.loads(path.read_text(encoding="utf-8"))
    t = prof["by_split"]["test"]
    print("test rows: %s" % {k: t["overall"]["rows"][k] for k in ("min", "median", "p90", "p99", "max")})
    print("share of test cases within each length: %s" % t["overall"]["cases"]["fit_share"])
    print(path)
    return EXIT_OK


def _splits_args(p: argparse.ArgumentParser) -> None:
    from . import data
    _data_root_arg(p)
    p.add_argument("--split-seed", type=int, default=data.DEFAULT_SPLIT_SEED, help="split seed (recorded)")
    p.add_argument("--check", action="store_true", help="verify the existing split manifest without writing")


@command("splits", "Create (or --check) the frozen dev/calibration/final split manifest.", _splits_args,
         default_run_id="data")
def _cmd_splits(args: argparse.Namespace) -> int:
    from . import data
    from .results import Refusal
    try:
        if args.check:
            print(data.check_splits(args.data_root))
            return EXIT_OK
        sp = data.make_splits(seed=args.split_seed, root=args.data_root)
    except (Refusal, FileNotFoundError) as exc:
        raise ToolError(str(exc))
    for s in data.EVAL_SPLITS:
        e = sp["splits"][s]
        print("%s: %d cases %s fingerprint %s" % (s, e["n_cases"], e["n_per_workflow"], e["fingerprint"][:16]))
    print(data.data_root(args.data_root) / "splits.json")
    return EXIT_OK


def _solvability_args(p: argparse.ArgumentParser) -> None:
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("--models", default="fine_tuned,base", help="comma-separated checkpoint names (fine_tuned, base)")
    p.add_argument("--split", default="dev", choices=["dev", "calibration"])
    p.add_argument("--time-cap", type=float, default=None, help="seconds per checkpoint (resume with the same command)")


@command("solvability", "Run both native checkpoints on original-length dev cases and pick the quality reference.",
         _solvability_args)
def _cmd_solvability(args: argparse.Namespace) -> int:
    from . import evalrun
    from .results import Refusal
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in evalrun.CHECKPOINTS]
    if bad:
        raise ToolError("unknown checkpoint names %s; choose from %s" % (bad, list(evalrun.CHECKPOINTS)))
    try:
        rep = evalrun.run_solvability(args.run_path, models=models, split=args.split, threads=args.threads,
                                      threads_source=args.threads_source, seed=args.seed,
                                      data_root=args.data_root, time_cap=args.time_cap, revision=args.revision)
    except (Refusal, FileNotFoundError) as exc:
        raise ToolError(str(exc))
    for name, r in rep["models"].items():
        if r.get("status") == "failed":
            print("%s: FAILED %s" % (name, r.get("reason")))
            continue
        acc = r["summary"]["accuracy"]["overall"]
        print("%s: accuracy %.3f (majority %.3f, paired lo %.3f) solves=%s"
              % (name, acc, r["majority_accuracy"], r["paired_vs_majority"]["lo"], r["solves"]))
    print("reference checkpoint: %s" % rep["reference_checkpoint"])
    print(args.run_path / "solvability.json")
    return EXIT_OK


# --------------------------------------------------------------------------- families and audit (T031)

def _families_args(p: argparse.ArgumentParser) -> None:
    from . import families
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("--split", required=True, choices=["dev", "calibration", "final"],
                   help="split to build items for (final is built and fingerprinted, never scored)")
    p.add_argument("--lengths", type=int_list, default=list(families.DEFAULT_LENGTHS))
    p.add_argument("--variants", default=",".join(families.DEFAULT_VARIANTS),
                   help="comma-separated: neutral@mid, distractor@mid, distractor@begin, distractor@end, oracle")
    p.add_argument("--rule-version", default=families.RULE_VERSION)


@command("families", "Build controlled long-context evaluation items for a split.", _families_args,
         default_run_id="data")
def _cmd_families(args: argparse.Namespace) -> int:
    from . import families
    agent, _ = load_model(args)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in families.DEFAULT_VARIANTS]
    if bad:
        raise ToolError("unknown variants %s; choose from %s" % (bad, list(families.DEFAULT_VARIANTS)))
    try:
        meta = families.build_split(agent, args.split, lengths=args.lengths, variants=variants, seed=args.seed,
                                    rule_version=args.rule_version, root=args.data_root,
                                    tokenizer_name=args.model,
                                    progress=lambda n: print("  %d items" % n, flush=True))
    except (FileNotFoundError, families.FamilyError) as exc:
        raise ToolError(str(exc))
    e = meta["splits"][args.split]
    print("families %s, split %s: %d items from %d cases, fingerprint %s"
          % (meta["families_id"], args.split, e["n_items"], e["n_cases"], e["fingerprint"][:16]))
    print(families.items_dir(meta["families_id"], args.data_root) / ("%s.jsonl.gz" % args.split))
    return EXIT_OK


def _audit_sample_args(p: argparse.ArgumentParser) -> None:
    from . import audit_items
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("--n", type=int, default=audit_items.DEFAULT_AUDIT_ITEMS, help="items to audit (at least 50)")
    p.add_argument("--families-id", default=None, help="families to audit (default: the most recent)")


@command("audit-sample", "Write the hand-audit sheet for the dev items of the current families.", _audit_sample_args,
         default_run_id="data")
def _cmd_audit_sample(args: argparse.Namespace) -> int:
    from . import audit_items, data, families
    agent, _ = load_model(args)
    try:
        fid = families.current_families_id(args.data_root, args.families_id)
        items = list(families.read_items(families.items_dir(fid, args.data_root) / "dev.jsonl.gz"))
        cases = {c["id"]: c for c in data.load_cases("test", args.data_root)}
        sheet = audit_items.audit_sample(agent, items, cases, fid, n=args.n, seed=args.seed, root=args.data_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(str(exc))
    root = data.data_root(args.data_root)
    print("audit sheet for families %s: %d items" % (fid, sheet["n"]))
    print(root / "audit_sheet.md")
    print("Review it, write audit_result_draft.json, then run: python -m experiments audit-record --from <draft>")
    return EXIT_OK


def _audit_record_args(p: argparse.ArgumentParser) -> None:
    _data_root_arg(p)
    p.add_argument("--from", dest="draft", required=True, help="the researcher's draft verdicts (JSON)")


@command("audit-record", "Ingest the audit verdicts and compute the shares (audit_result.json).", _audit_record_args,
         default_run_id="data")
def _cmd_audit_record(args: argparse.Namespace) -> int:
    from . import audit_items, data
    try:
        res = audit_items.record_audit(args.draft, root=args.data_root)
    except (OSError, ValueError, KeyError) as exc:
        raise ToolError(str(exc))
    print("audited %d items: answer changed %.1f%%, ambiguous %.1f%%, evidence intact %s -> %s"
          % (res["n_audited"], 100 * res["share_answer_changed"], 100 * res["share_ambiguous"],
             res["all_evidence_intact"], "PASSED" if res["passed"] else "NOT PASSED (fix the rule and rebuild)"))
    print(data.data_root(args.data_root) / "audit_result.json")
    return EXIT_OK


# --------------------------------------------------------------------------- evaluation commands (T047)

def _condition_args(p: argparse.ArgumentParser) -> None:
    from . import baselines, families, variants
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("--conditions", default=",".join(baselines.CONDITION_NAMES),
                   help="comma-separated: %s" % ", ".join(baselines.CONDITION_NAMES))
    p.add_argument("--variants", default="none",
                   help="optimized-system variants of native, comma-separated: %s (default none = the architectural "
                        "baselines)" % ", ".join(variants.VARIANTS))
    p.add_argument("--lengths", type=int_list, default=None,
                   help="default: 512,1024,2048,4096,8192 (variants: 512,2048,8192)")
    p.add_argument("--families-id", default=None, help="families to use (default: the most recent)")


def _variant_subset(case_ids, n):
    """The first `n` cases of the variant sample, spread evenly over the workflows (all of them when `n` is unset)."""
    ids = sorted(case_ids)
    if not n or n >= len(ids):
        return ids
    by_wf = {}
    for c in ids:
        by_wf.setdefault(c.rsplit("_", 1)[0], []).append(c)
    if n < len(by_wf):
        raise ToolError("--variant-cases %d is below the number of workflows (%d)" % (n, len(by_wf)))
    take = {w: n // len(by_wf) for w in by_wf}
    for w in sorted(by_wf)[:n % len(by_wf)]:
        take[w] += 1
    return sorted(c for w, cs in by_wf.items() for c in cs[:take[w]])


def _plan_conditions(args: argparse.ArgumentParser, run_path, device: str = "cpu", sample: str = "all"):
    """(conditions, restricted, case_ids) for the requested conditions, variants and lengths.

    `restricted` means the fixed 20-case variant sample and `distractor@mid` items only: used for the optimized
    variants and for the CPU parity subset (``--sample variant``)."""
    from . import baselines, data, evalrun, families, variants
    names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in names if c not in baselines.CONDITION_NAMES]
    if bad:
        raise ToolError("unknown conditions %s; choose from %s" % (bad, list(baselines.CONDITION_NAMES)))
    vs = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in vs if v not in variants.VARIANTS]
    if bad:
        raise ToolError("unknown variants %s; choose from %s" % (bad, list(variants.VARIANTS)))
    variant_run = vs != ["none"]
    small = variant_run or sample == "variant"
    lengths = args.lengths or (list(evalrun.VARIANT_LENGTHS) if small else list(families.DEFAULT_LENGTHS))
    tuned = evalrun.load_tuned_params(run_path)
    conds = []
    if variant_run:
        for v in vs:
            conds += [evalrun.make_condition("native", L, variant=v, device=device, model=args.model) for L in lengths]
        splits = data.read_data_json("splits.json", args.data_root)
        return conds, True, _variant_subset(splits["variant_sample"], getattr(args, "variant_cases", None))
    for name in names:
        for L in lengths:
            params = {"size": tuned["size"]} if name == "window" else None
            conds.append(evalrun.make_condition(name, L, device=device, params=params, model=args.model))
    if sample == "variant":
        splits = data.read_data_json("splits.json", args.data_root)
        return conds, True, list(splits["variant_sample"])
    return conds, False, None


def _eval_args(p: argparse.ArgumentParser) -> None:
    _condition_args(p)
    p.add_argument("--split", default="dev", choices=["dev", "calibration", "final"],
                   help="split to score (final is refused: split_locked)")
    p.add_argument("--tune", action="store_true",
                   help="choose the window size on dev only, write conditions.json and stop")
    p.add_argument("--time-cap", type=float, default=1800.0,
                   help="seconds one measuring process may run before it is relaunched to continue (default 1800)")
    p.add_argument("--max-cases", type=int, default=None, help="pilot: keep only the first N cases (recorded in the log)")
    p.add_argument("--variant-cases", type=int, default=None,
                   help="optimized variants only: run on N cases of the 20-case variant sample, spread evenly over the "
                        "workflows (default all 20). The parity subset always uses all 20")
    p.add_argument("--resume", action="store_true", default=True, help="continue from existing predictions (always on)")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                   help="device that scores quality (default cpu). gpu is fast (about an hour for the whole dev grid) and needs "
                        ".venv-gpu; every result is labeled with its device (FR-024)")
    p.add_argument("--sample", default="all", choices=["all", "variant"],
                   help="variant = the fixed 20-case dev sample at 512, 2,048 and 8,192 tokens: the CPU parity subset, "
                        "run with --device cpu next to the GPU results")


@command("eval", "Quality run of the baseline conditions on a split (resumable, one subprocess per condition).", _eval_args)
def _cmd_eval(args: argparse.Namespace) -> int:
    from . import audit_items, evalrun, families
    from .results import Refusal, SplitLocked
    if args.split == "final":
        raise ToolError(str(SplitLocked("eval does not accept --split final: the final test split is not scored in this phase (FR-013)")))
    try:
        fid = families.current_families_id(args.data_root, args.families_id)
        say = lambda m: print(m, flush=True)   # noqa: E731
        if args.tune:
            body = evalrun.tune_window(args.run_path, args.model, fid, revision=args.revision, threads=args.threads,
                                       data_root=args.data_root, chunk_seconds=args.time_cap, log=say, device=args.device)
            print("window tuned on dev (scored on %s): size %s; grid %s" % (args.device, body["window"]["size"], body["grid"]))
            print(args.run_path / "conditions.json")
            return EXIT_OK
        conds, restricted, case_ids = _plan_conditions(args, args.run_path, device=args.device, sample=args.sample)
        item_variants = ["distractor@mid"] if restricted else list(families.CONTEXT_VARIANTS)
        out = evalrun.run_eval(args.run_path, args.split, conds, args.model, fid, item_variants,
                               revision=args.revision, threads=args.threads, data_root=args.data_root,
                               chunk_seconds=args.time_cap, case_ids=case_ids, max_cases=args.max_cases, log=say)
    except (Refusal, FileNotFoundError, ValueError) as exc:
        raise ToolError(str(exc))
    bad = [o for o in out if o.get("status") not in ("measured",)]
    print("%d conditions run, %d not complete: %s" % (len(out), len(bad), [(o.get("condition_id"), o.get("status")) for o in bad]))
    return EXIT_OK


def _calibrate_args(p: argparse.ArgumentParser) -> None:
    _phase2_defaults(p)


@command("calibrate", "Fit per-question-type temperatures on the calibration-split results (calibration.json).",
         _calibrate_args)
def _cmd_calibrate(args: argparse.Namespace) -> int:
    from . import summary
    body = summary.calibrate(args.run_path)
    for cid, by_type in body["conditions"].items():
        print("%s: %s" % (cid, {t: round(v["temperature"], 3) for t, v in by_type.items()}))
    if not body["conditions"]:
        print("no calibration-split results yet; run: eval --split calibration")
    print(args.run_path / "calibration.json")
    return EXIT_OK


def _latency_args(p: argparse.ArgumentParser) -> None:
    _condition_args(p)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="gpu needs the .venv-gpu environment")
    p.add_argument("--repeats", default=None, help="auto (default) or passes over the fixed sample")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--time-cap", type=float, default=None, help="seconds per condition, model load included")


@command("latency", "Latency of the conditions on the fixed dev sample: CPU (primary) or GPU (labeled).", _latency_args)
def _cmd_latency(args: argparse.Namespace) -> int:
    from . import families, latency
    from .results import Refusal
    try:
        fid = families.current_families_id(args.data_root, args.families_id)
        conds, _, _ = _plan_conditions(args, args.run_path, device=args.device)
        out = latency.run_latency(args.run_path, conds, args.model, fid, revision=args.revision, threads=args.threads,
                                  data_root=args.data_root, repeats=args.repeats, warmup=args.warmup,
                                  time_cap=args.time_cap, log=lambda m: print(m, flush=True))
    except (Refusal, FileNotFoundError, ValueError) as exc:
        raise ToolError(str(exc))
    print("%d latency conditions written under %s" % (len(out), latency.latency_dir(args.run_path, args.device)))
    return EXIT_OK


def _margin_float(text: str) -> float:
    """A probability margin: finite and not negative (a negative margin would make every disagreement a failure)."""
    import math
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number, got %r" % text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0, got %r" % text)
    return value


def _summary_args(p: argparse.ArgumentParser) -> None:
    _phase2_defaults(p)
    p.add_argument("--n-boot", type=int, default=5000, help="bootstrap resamples (default 5000)")
    p.add_argument("--parity-margin", type=_margin_float, default=None,
                   help="CPU/GPU parity: a label disagreement is a near-tie flip, not a failure, when the CPU top-two "
                        "probability margin is at most this (default: summary.PARITY_NEAR_TIE_MARGIN, 0.01)")


@command("eval-summary", "Summaries, paired comparisons and the cell table of the dev results (summary.json).",
         _summary_args)
def _cmd_eval_summary(args: argparse.Namespace) -> int:
    from . import summary
    margin = summary.PARITY_NEAR_TIE_MARGIN if args.parity_margin is None else args.parity_margin
    body = summary.build_summary(args.run_path, split="dev", n_boot=args.n_boot, seed=args.seed, parity_margin=margin)
    for c in body["cells"]:
        acc = "-" if c["accuracy"] is None else "%.3f" % c["accuracy"]
        print("%-46s %-11s n=%d/%d acc=%s" % (c["condition_id"], c["status"], c["n_measured"], c["n_items"], acc))
    print("selected retrieval budget on dev: %s" % body["retrieval_budget"]["selected"])
    print(args.run_path / "summary.json")
    return EXIT_OK


# --------------------------------------------------------------------------- plan, report, all (T057)

def _freeze_args(p: argparse.ArgumentParser) -> None:
    _phase2_defaults(p)
    p.add_argument("--available-cases", type=int, default=200, help="final-split cases available (default 200)")
    p.add_argument("--new-version", action="store_true", help="create a new plan version (needs --reason)")
    p.add_argument("--reason", default="", help="why a new plan version is needed")


@command("freeze-plan", "Freeze the final-test protocol and sample size from the dev pilot (evaluation_plan.json).",
         _freeze_args)
def _cmd_freeze_plan(args: argparse.Namespace) -> int:
    from . import evalplan
    from .results import Refusal
    try:
        plan = evalplan.freeze_plan(args.run_path, available=args.available_cases, seed=args.seed,
                                    new_version=args.new_version, reason=args.reason)
    except (Refusal, ValueError) as exc:
        raise ToolError(str(exc))
    print("plan version %s frozen; fingerprint %s" % (plan["version"], plan["fingerprint"][:16]))
    print(plan["statement"])
    print("final-split items scored: %d" % plan["final_scored_items"])
    print(args.run_path / "evaluation_plan.md")
    return EXIT_OK


def _report2_args(p: argparse.ArgumentParser) -> None:
    _phase2_defaults(p)
    _data_root_arg(p)


@command("phase2-report", "Assemble report2.json and report2.md: curves, verdicts, Phase 3 evidence, limits.", _report2_args)
def _cmd_phase2_report(args: argparse.Namespace) -> int:
    from . import report2
    try:
        j, m = report2.write_report(args.run_path, args.data_root)
    except report2.ReportError as exc:
        raise ToolError(str(exc))
    print(j)
    print(m)
    return EXIT_OK


#: Rough seconds per native question row on the target CPU, from Phase 1 (fast path on, one document). Estimated.
_ESTIMATE_SECONDS = {512: 2.0, 1024: 4.0, 2048: 9.0, 4096: 25.0, 8192: 80.0}


def estimate_hours(cases: int = 120, questions: int = 5, variants: int = 4) -> Dict[str, float]:
    """A rough wall-time estimate (labeled estimated in every use) of the architectural-baseline quality runs on dev."""
    def rows(length: int) -> int:
        n = cases // 2 if length >= 4096 else cases
        return n * questions * variants
    native = {L: rows(L) * s / 3600.0 for L, s in _ESTIMATE_SECONDS.items()}
    # trunc512/truncCap ~ short rows, window ~ native, retrieve ~ a 1-2K row, oracle ~ one short row per case-question
    total = 0.0
    for L, h in native.items():
        total += h                                        # native
        total += rows(L) * _ESTIMATE_SECONDS[512] / 3600.0 * 2      # two truncations
        total += h * (0.8 if L >= 2048 else 0.4)          # window
        total += rows(L) * _ESTIMATE_SECONDS[2048] / 3600.0 * 0.6 * 3   # three retrieval budgets
    total += cases * questions * 2.0 / 3600.0             # oracle, once per case-question
    gpu = sum(rows(L) * s for L, s in _ESTIMATE_GPU_SECONDS.items()) / 3600.0 * (total / max(1e-9, sum(native.values())))
    # CPU parity subset: the 20-case variant sample (100 questions) at 512, 2,048 and 8,192 tokens, distractor@mid only
    parity = 100 * (sum(_ESTIMATE_SECONDS[L] for L in (512, 2048, 8192))              # native
                    + 3 * 4.0 + 3 * 6.0) / 3600.0                                       # truncCap and retrieve1024
    return {"native_hours": sum(native.values()), "all_baselines_hours": total,
            "gpu_all_baselines_hours": gpu, "cpu_parity_hours": parity}


#: Rough seconds per native question row on the RTX 4060 (estimated: Phase 1 measured 45 ms at 512 tokens).
_ESTIMATE_GPU_SECONDS = {512: 0.05, 1024: 0.08, 2048: 0.15, 4096: 0.5, 8192: 1.5}


def _gpu_python():
    """The CUDA environment's interpreter (`.venv-gpu`), or None when it does not exist."""
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent
    for rel in ("Scripts/python.exe", "bin/python"):
        p = REPO_ROOT / ".venv-gpu" / rel
        if p.exists():
            return p
    return None


def _all2_args(p: argparse.ArgumentParser) -> None:
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("--dry-run", action="store_true", help="print the ordered plan and the estimated wall time, run nothing")


@command("phase2-all", "Run the Phase 2 pipeline in order on dev and calibration; stops at the audit gate.", _all2_args)
def _cmd_phase2_all(args: argparse.Namespace) -> int:
    from . import audit_items, families
    from .results import Refusal
    rid = ["--run-id", args.run_id]
    steps = [
        ("data-import", []), ("splits", []), ("length-profile", []),
        ("solvability", rid), ("families --split dev", []), ("families --split calibration", []),
        ("families --split final", []), ("audit-sample", []),
        ("AUDIT GATE", []),
        # Quality is scored on the GPU (fast); GPU steps run in the .venv-gpu environment (FR-024, research.md R20).
        ("eval --tune --device gpu", rid), ("eval --split dev --device gpu", rid),
        ("eval --split calibration --device gpu", rid), ("calibrate", rid),
        # The CPU parity subset: the same items on CPU, so GPU-scored quality can be called CPU-equivalent.
        ("eval --split dev --device cpu --sample variant --conditions native,truncCap,retrieve1024", rid),
        ("latency --device cpu", rid), ("eval --variants fastpath_off,int8_encoder,int8_all_nofast --variant-cases 10", rid),
        ("latency --variants fastpath_off,int8_encoder,int8_all_nofast", rid), ("latency --device gpu", rid),
        ("eval-summary", rid), ("freeze-plan", rid), ("phase2-report", rid),
    ]
    if args.dry_run:
        est = estimate_hours()
        print("Ordered plan for run %s:" % args.run_id)
        for i, (s, extra) in enumerate(steps, 1):
            print("  %2d. python -m experiments %s %s" % (i, s, " ".join(extra)) if s != "AUDIT GATE"
                  else "  %2d. -- audit gate: review the sheet, then: python -m experiments audit-record --from <draft> --" % i)
        print("Estimated wall time (estimated; correct it from the first measured runs): quality scored on the GPU "
              "about %.1f hours for all architectural baselines on dev, plus the calibration split; the CPU parity subset "
              "about %.1f hours; CPU latency and the variants on top. Scoring the same grid on CPU instead would take "
              "about %.0f hours (native alone %.0f)." % (est["gpu_all_baselines_hours"], est["cpu_parity_hours"],
                                                        est["all_baselines_hours"], est["native_hours"]))
        print("Steps marked --device gpu need the .venv-gpu environment; phase2-all runs them there when it exists.")
        return EXIT_OK
    parser = build_parser()
    common = ["--data-root", str(args.data_root)] if args.data_root else []
    # Every step gets the pipeline's own settings, not the step's defaults (a seed or thread count given to
    # phase2-all would otherwise be silently ignored, and the steps would disagree with each other).
    shared = ["--seed", str(args.seed), "--model", args.model, "--mha-fastpath", args.mha_fastpath]
    if args.revision:
        shared += ["--revision", args.revision]
    if args.threads_source == "user":
        shared += ["--threads", str(args.threads)]
    for s, extra in steps:
        if s == "AUDIT GATE":
            try:
                audit_items.require_audit(families.current_families_id(args.data_root), args.data_root)
            except (Refusal, FileNotFoundError) as exc:
                print("stopped at the audit gate: %s" % exc)
                print("Review experiments/data/audit_sheet.md, write your verdicts, run audit-record, then run phase2-all again.")
                return EXIT_OK
            continue
        argv = s.split() + extra + shared
        if s.split()[0] not in ("calibrate", "eval-summary", "freeze-plan", "phase2-report"):
            argv += common                                   # these four take no --data-root
        if "--device gpu" in s:                              # GPU steps run in the CUDA environment
            gpu_py = _gpu_python()
            if gpu_py is None:
                raise ToolError("step %r needs the .venv-gpu environment (experiments/setup_gpu.ps1)" % s)
            print("== %s (in %s) ==" % (s, gpu_py), flush=True)
            import subprocess
            code = subprocess.call([str(gpu_py), "-m", "experiments"] + argv, cwd=str(_gpu_python().parent.parent.parent))
            if code:
                return code
            continue
        try:
            ns = parser.parse_args(argv)
        except SystemExit:
            raise ToolError("internal: could not parse step %r" % s)
        ns.threads_source = args.threads_source
        ns.run_path = args.run_path
        print("== %s ==" % s, flush=True)
        if ns.run_id is None and COMMANDS[ns.command].default_run_id:     # as `main` does for a direct call
            ns.run_id = COMMANDS[ns.command].default_run_id
        resolve_common(ns)
        code = COMMANDS[ns.command].run(ns)
        if code:
            return code
    return EXIT_OK


# --------------------------------------------------------------------------- golden fixtures (laya:004, laya:005)

_GOLDEN_ACTIONS = ("weights-inventory", "export", "kernels", "tokenizer")


def _golden_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=_GOLDEN_ACTIONS,
                   help="weights-inventory | export (forward-pass goldens) | kernels | tokenizer")
    p.add_argument("--out", default=None,
                   help="output directory (default: artifacts/raya-golden/, tokenizer: artifacts/raya-golden/tokenizer/)")
    p.add_argument("--checkpoint", default=None,
                   help="local real checkpoint directory (weights-inventory, tokenizer); default is the offline fixture")
    p.add_argument("--lengths", type=int_list, default=None, help="kernels: sequence lengths (default 256,1000,2048)")


@command("golden", "Golden fixtures for Raya from the offline fixture model (laya:004, laya:005); no timings.",
         _golden_args, default_run_id="golden")
def _cmd_golden(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .golden import common
    out = Path(args.out) if args.out else (common.DEFAULT_OUT / "tokenizer" if args.action == "tokenizer"
                                           else common.DEFAULT_OUT)
    if args.action == "weights-inventory":
        from .golden.weights import write_inventory
        print(write_inventory(out, checkpoint=args.checkpoint))
    elif args.action == "export":
        from .golden.forward import export_all
        export_all(out)
        print(out)
    elif args.action == "kernels":
        from .golden.kernels import LENGTHS, export_kernels
        export_kernels(out, lengths=tuple(args.lengths or LENGTHS))
        print(out / "kernels")
    else:
        from .golden.tokenizer import export_tokenizer
        export_tokenizer(out, checkpoint=args.checkpoint)
        print(out)
    return EXIT_OK


def _tier_e_args(p: argparse.ArgumentParser) -> None:
    from . import tier_e
    _data_root_arg(p)
    _phase2_defaults(p)
    p.add_argument("action", choices=["probs", "compare", "report"],
                   help="probs: E1 (probabilities and per-local-layer differences, CPU fp32); compare: E2 (predictions vs "
                        "a reference run); report: E1/E2/E3 verdicts")
    p.add_argument("--families-id", default=None, help="item families (default: the most recently built)")
    p.add_argument("--lengths", type=int_list, default=None,
                   help="probs: lengths to measure (default %s); one batch per call keeps each run short" % (tier_e.E1_LENGTHS,))
    p.add_argument("--items-per-length", type=_positive_int, default=tier_e.E1_ITEMS_PER_LENGTH)
    p.add_argument("--candidate-condition", default="native.local_exact_fastpath_off.cpu",
                   help="compare: condition id prefix of the candidate's quality predictions (length appended as .L<n>)")
    p.add_argument("--candidate-run", default=None, help="compare/report: run id holding the candidate's eval and latency results")
    p.add_argument("--reference-run", default="p2-dev", help="compare: run id holding the reference native CPU predictions")
    p.add_argument("--reference-root", default=None,
                   help="compare: results directory that holds the reference run (default: this checkout's experiments/results; "
                        "a gate run from a clean worktree points it at the main checkout's, read only)")
    p.add_argument("--latency-run", default=None, help="report: run id of the latency measurements (default: --candidate-run)")
    p.add_argument("--allow-dirty", action="store_true", help="tests only: a gate run needs a clean git tree")
    p.add_argument("--dry-run", action="store_true", help="probs: print the item counts and the estimated CPU hours, run nothing")


def _tier_e_estimate_hours(lengths, n_items: int) -> float:
    # Estimated (labeled): native fastpath_off p50 from p2-dev (512/2K/8K measured; 1K and 4K interpolated from the
    # fast-path-on measurements), exact kernel about 0.6x of native at 8K and 1.0x at 512; two passes (hooks add a little).
    native = {512: 1.8, 1024: 4.0, 2048: 8.1, 4096: 24.0, 8192: 60.0}
    ratio = {512: 1.0, 1024: 0.9, 2048: 0.8, 4096: 0.7, 8192: 0.6}
    return sum(n_items * native[L] * (1.0 + ratio[L]) for L in lengths) / 3600.0


@command("tier-e", "Tier E check of the compression gate (laya:007): exact local-attention kernel E1/E2/E3.", _tier_e_args)
def _cmd_tier_e(args: argparse.Namespace) -> int:
    from pathlib import Path
    from . import data, families, results, tier_e
    from .results import Refusal
    run = args.run_path
    try:
        if args.action == "probs":
            lengths = list(args.lengths or tier_e.E1_LENGTHS)
            bad = [L for L in lengths if L not in tier_e.E1_LENGTHS]
            if bad:
                raise ToolError("lengths must be among %s, got %s" % (tier_e.E1_LENGTHS, bad))
            est = _tier_e_estimate_hours(lengths, args.items_per_length)
            print("E1 probs: %d lengths x %d items, estimated %.2f CPU hours (estimate, i7-8700 fp32)"
                  % (len(lengths), args.items_per_length, est))
            if args.dry_run:
                return EXIT_OK
            code = tier_e.require_clean_tree(args.allow_dirty)
            fid = families.current_families_id(args.data_root, args.families_id)
            splits = data.read_data_json("splits.json", args.data_root)
            items = list(families.read_items(families.items_dir(fid, args.data_root) / "dev.jsonl.gz"))
            by_len = {L: tier_e.e1_items(items, splits["variant_sample"], L, args.items_per_length) for L in lengths}
            short = {L: len(v) for L, v in by_len.items() if len(v) < args.items_per_length}
            if short:
                raise ToolError("fewer items than requested at %s" % short)
            paths = tier_e.run_probs(run, args.model, args.revision, args.threads, by_len, code, log=print)
            for p in paths:
                print(p)
            return EXIT_OK
        code = tier_e.require_clean_tree(args.allow_dirty)
        if args.action == "compare":
            cand_run = results.run_dir(args.candidate_run or args.run_id)
            ref_run = results.run_dir(args.reference_run, args.reference_root)
            cand, ref = {}, {}
            for L in tier_e.PARITY_LENGTHS:
                cand.update(tier_e._load_predictions(cand_run, "quality", "%s.L%d" % (args.candidate_condition, L)))
                ref.update(tier_e._load_predictions(ref_run, "quality", "native.none.cpu.L%d" % L))
            out = tier_e.compare_predictions(cand, ref, {512: 95, 2048: 100, 8192: 100})
            print(results.write_json(run, "%s/e2.json" % tier_e.SUBDIR, out))
            print("E2: %d items, %d changed, %d missing -> %s" % (out["n"], out["changed"], out["missing"],
                                                              "passed" if out["passed"] else "FAILED"))
            return EXIT_OK
        # report
        base = Path(run) / tier_e.SUBDIR
        e1 = None
        per = {}
        for L in tier_e.E1_LENGTHS:
            f = base / ("probs.L%d.json" % L)
            if f.exists():
                per[L] = results.read_json(run, "%s/%s" % (tier_e.SUBDIR, f.name))
        if per:
            e1 = tier_e.summarize_e1(per)
        e2 = results.read_json(run, "%s/e2.json" % tier_e.SUBDIR) if (base / "e2.json").exists() else None
        lat_run = results.run_dir(args.latency_run or args.candidate_run or args.run_id)
        lat = tier_e.read_latency(lat_run)
        e3 = tier_e.summarize_e3(lat) if lat else None
        rep = tier_e.build_report(run, e1, e2, e3, code)
        print(results.write_json(run, "%s/report.json" % tier_e.SUBDIR, rep))
        print("verdicts:", rep["verdicts"])
        return EXIT_OK
    except (Refusal, FileNotFoundError, ValueError, tier_e.TierEError) as exc:
        raise ToolError(str(exc))

def _parity_draw_args(p: argparse.ArgumentParser) -> None:
    from . import parity_draw
    _data_root_arg(p)
    p.add_argument("--parity-seed", type=int, default=parity_draw.PARITY_SEED, help="default: the seed fixed in docs/gate-plan-v2.md")
    p.add_argument("--out", default=None, help="write the drawn ids (JSON) here; default: print only")


@command("parity-draw", "Draw the two fresh 20-case parity sets from splits.json (deterministic, no model, no run).",
         _parity_draw_args)
def _cmd_parity_draw(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path
    from . import data, parity_draw
    try:
        splits = data.read_data_json("splits.json", args.data_root)
        body = parity_draw.draw_fresh_sets(splits, seed=args.parity_seed)
    except (FileNotFoundError, parity_draw.ParityDrawError) as exc:
        raise ToolError(str(exc))
    problems = parity_draw.check_sets(body, splits)
    if problems:
        raise ToolError("the drawn sets break the plan's rule: %s" % problems)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(args.out)
    print("seed %d, ids sha256 %s" % (body["seed"], body["ids_sha256"]))
    return EXIT_OK
