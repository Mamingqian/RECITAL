"""Command line of the control layer: preflight, run one cell, run a campaign, read status.

    python -m recital.control.cli preflight --config C --output DIR
    python -m recital.control.cli run --config C --output DIR [--arm A --seed N --scheduler slurm]
    python -m recital.control.cli campaign --config C --output-root ROOT --arms A B C --seeds 0 1 2
                                             [--transport MODULE:CLASS]
    python -m recital.control.cli fixed-recipe --config C --output-root ROOT --steps 4000 8192 --seeds 0 1 2
    python -m recital.control.cli arm-search --config C --arm random|rule|module:Class --output-root ROOT --seeds 0 1 2
                                                [--match-root ROOT_OF_SEARCHES]
    python -m recital.control.cli status DIR [DIR ...]
    python -m recital.control.cli human-search --config C --output DIR    (control.modules.modes.mode: human_only)
    python -m recital.control.cli show --run DIR                          (what a copilot or human-only cell waits for)
    python -m recital.control.cli note --run DIR --minutes M --text T [--kind direction|note]
    python -m recital.control.cli approve --run DIR --minutes M [--decision ID] [--veto INDEX ...] [--text T]
    python -m recital.control.cli propose --run DIR --minutes M --file RESPONSE.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from .campaign import Cell, JobGate, run_campaign
from .contract import ControlContract, endpoint_host
from .errors import ControlError
from .export import export
from .launch import ARM, Prepared, prepare, require_arm, reward_role_policy
from .runtime_build import build_runtime, preflight
from .scheduler import LocalScheduler, Scheduler
from .state import fold
from .store import read_events
from .events import from_record
from .transport import ResponsesTransport, read_key_file

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = (ARM,)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ControlError("the run config must be a mapping")
    for key, target in (("seed", "seed"), ("total_updates", "total_updates"), ("arm", "protocol_condition")):
        value = getattr(args, key, None)
        if value is not None:
            config[target] = value
    if getattr(args, "max_concurrent_jobs", None) is not None:
        config["scheduler"]["max_concurrent_jobs"] = args.max_concurrent_jobs
    if getattr(args, "partition", None) is not None:
        config["scheduler"]["partition"] = args.partition
    return config


def make_scheduler(kind: str, run_dir: Path) -> Scheduler:
    if kind == "local":
        return LocalScheduler([run_dir / "jobs"])
    if kind == "slurm":
        from .slurm import SlurmScheduler
        return SlurmScheduler()
    raise ControlError(f"unknown scheduler {kind!r}")


def make_transport(config: dict[str, Any]) -> ResponsesTransport:
    llm = ControlContract.from_config(config).llm
    return ResponsesTransport(model=llm.model, max_output_tokens=llm.max_output_tokens, base_url=llm.base_url,
                              api_key_file=llm.api_key_file, schema_dialect=llm.schema_dialect)


def make_campaign_transport(config: dict[str, Any], spec: str | None, run_dir: Path):
    """The LLM transport of one campaign cell: the config's own, or the class `MODULE:CLASS` of `--transport` over it.

    The class is called as `Class(inner=<the transport the config builds>, config=<the cell's config>, run_dir=<the cell's
    directory>)`; it stands in for the transport the engine talks to and may forward any request to `inner`. When it has a
    `bind` method, `cmd_campaign` calls `bind(engine)` once the cell is prepared, as `cmd_arm_search` does for its transports.
    """
    inner = make_transport(config)
    if spec is None:
        return inner
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ControlError(f"--transport takes MODULE:CLASS, got {spec!r}")
    try:
        transport_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as error:
        raise ControlError(f"--transport {spec!r} names no importable class: {error}") from error
    return transport_class(inner=inner, config=config, run_dir=run_dir)


def make_components(parts: dict[str, Any]):
    from .coder import build_runner
    return build_runner(parts["settings"].control, backend=parts["backend"], synthesis=parts["synthesis"],
                        transport=parts["transport"])


def cmd_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config, args)
    control = ControlContract.from_config(config)
    if control.llm.api_key_file is not None:
        read_key_file(control.llm.api_key_file)  # a cell that names a key file cannot start without it
    runtime = build_runtime(config, repo_root=args.repo_root, output=args.output, device="cpu",
                            audit_oracle_factories={})
    declaration = preflight(runtime, device=args.device)
    task = runtime["contract"]
    require_arm(task)
    control.validate_against(task.visible_sources, list(task.conditions), int(config["delivery"]["count"]))
    reward_role_policy(runtime)  # raises when this arm cannot see a source the task's policy requires enabled
    print(json.dumps({"declaration": declaration, "control_contract_hash": control.contract_hash(),
                      "llm": {"model": control.llm.model, "endpoint": endpoint_host(control.llm.base_url)}},
                     indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


def _install_stop_flag() -> list[bool]:
    flag = [False]

    def handler(signum, frame):  # noqa: ARG001 - signature fixed by `signal`
        flag[0] = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return flag


def audit_device() -> str:
    """The controller's own device for the terminal blind evaluation, which may run on a node with no GPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _finish(prepared: Prepared, args: argparse.Namespace) -> dict[str, Any]:
    state = prepared.engine.state()
    result: dict[str, Any] = {"status": state.finished_status, "output": str(prepared.settings.run_dir)}
    if state.finished_status == "frozen" and not args.no_audit:
        from .terminal_audit import run_terminal_audit
        result["terminal_audit"] = run_terminal_audit(
            yaml.safe_load(prepared.settings.config_path.read_text()), prepared.settings.run_dir,
            repo_root=prepared.settings.repo_root, device=audit_device())["status"]
    return result


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args)
    return run_cell(args, config, make_transport(config), make_components)


def run_cell(args: argparse.Namespace, config: dict[str, Any], transport, components) -> int:
    run_dir = args.output.expanduser().resolve()
    stop = _install_stop_flag()
    prepared = prepare(config, run_dir=run_dir, repo_root=args.repo_root, scheduler=make_scheduler(args.scheduler, run_dir),
                       transport=transport, device=args.device, components=components)
    try:
        cell = Cell(run_dir.name, prepared.engine)
        run_campaign([cell], poll_seconds=prepared.settings.control.scheduler.poll_seconds,
                     should_stop=lambda: stop[0])
        print(json.dumps(_finish(prepared, args) if cell.done else {"status": "interrupted"}, sort_keys=True))
    finally:
        prepared.close()
    return 0


def cmd_human_search(args: argparse.Namespace) -> int:
    """A cell whose Strategist is a person: the contract, validators, selection and delivery are the searches'."""
    from .arms import arm_config
    from .ext.modes import HumanTransport

    base = load_config(args.config, args)
    declared = ((base.get("control") or {}).get("modules") or {}).get("modes") or {}
    if declared.get("mode") != "human_only":
        raise ControlError("human-search runs a cell that declares control.modules.modes.mode: human_only")
    config = {**arm_config(base, "human"), "protocol_condition": "no_self_validation"}
    return run_cell(args, config, HumanTransport(), None)


def _fold_run(run: Path):
    events = read_events(Path(run) / "events" / "log.jsonl")
    if not events:
        raise ControlError(f"{run} holds no event log")
    return fold(from_record(event) for event in events)


def cmd_note(args: argparse.Namespace) -> int:
    from .ext.modes import write_inbox

    text = args.text if args.text is not None else args.file.read_text()
    print(json.dumps({"posted": write_inbox(args.run, args.kind, minutes=args.minutes, text=text).name}))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    from .ext.modes import awaiting_approval, write_inbox

    waiting = awaiting_approval(_fold_run(args.run))
    decision = args.decision or (waiting[0] if waiting else None)
    if decision is None:
        raise ControlError("no proposal is waiting for approval")
    posted = write_inbox(args.run, "approve", minutes=args.minutes, decision_id=decision, vetoed=args.veto or [],
                         text=args.text or "")
    print(json.dumps({"posted": posted.name, "decision": decision, "vetoed": args.veto or []}))
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    from .ext.modes import write_inbox

    response = json.loads(args.file.read_text())
    print(json.dumps({"posted": write_inbox(args.run, "propose", minutes=args.minutes, response=response).name}))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from .ext.modes import effort, pending_holds

    state = _fold_run(args.run)
    waiting = pending_holds(args.run, state)
    for path in waiting:
        print(f"==== {path} ====\n{path.read_text()}")
    print(json.dumps({"run": Path(args.run).name, "status": state.finished_status or "running",
                      "waiting_for_you": [path.stem for path in waiting], "effort": effort(state),
                      "refused": (state.ext.get("modes") or {}).get("refused", [])}, indent=1, sort_keys=True))
    return 0


def cell_dir(root: Path, config: dict[str, Any], arm: str, seed: int) -> Path:
    return root / f"{config.get('task_family')}_{arm}_seed{seed}"


def cmd_campaign(args: argparse.Namespace) -> int:
    base = load_config(args.config, args)
    cap = base["scheduler"]["max_concurrent_jobs"]
    gate, prepared_cells, cells = JobGate(cap), [], []
    stop = _install_stop_flag()
    try:
        # The job gate serves cells in this order, so each seed's arms are adjacent and advance together.
        for seed in args.seeds:
            for arm in args.arms:
                config = {**base, "protocol_condition": arm, "seed": seed}
                run_dir = cell_dir(args.output_root.expanduser().resolve(), config, arm, seed)
                spec = getattr(args, "transport", None)
                transport = make_campaign_transport(config, spec, run_dir)
                prepared = prepare(config, run_dir=run_dir, repo_root=args.repo_root,
                                   scheduler=make_scheduler(args.scheduler, run_dir), transport=transport,
                                   device=args.device, components=make_components, may_submit=gate.allow)
                if spec is not None and hasattr(transport, "bind"):
                    transport.bind(prepared.engine)
                prepared_cells.append(prepared)
                cells.append(Cell(run_dir.name, prepared.engine))
        gate.bind(cells)
        states = run_campaign(cells, poll_seconds=base["scheduler"]["poll_seconds"], should_stop=lambda: stop[0])
        summary = {name: state.finished_status for name, state in states.items()}
        if not stop[0] and not args.no_audit:
            for cell, prepared in zip(cells, prepared_cells):
                if states[cell.name].finished_status == "frozen":
                    summary[cell.name] = {"frozen": _finish(prepared, args).get("terminal_audit")}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for prepared in prepared_cells:
            prepared.close()
    return 0


def cmd_drill_coder(args: argparse.Namespace) -> int:
    from .drill import run_coder_drill

    config = load_config(args.config, args)
    run_dir = args.output.expanduser().resolve()
    result = run_coder_drill(
        config, run_dir=run_dir, repo_root=args.repo_root, scheduler=make_scheduler(args.scheduler, run_dir),
        coder_transport=make_transport(config), device=args.device,
        poll_seconds=config["scheduler"]["poll_seconds"], oracles=tuple(args.drill_oracles))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "frozen" and all(
        item["status"] == "published" for item in result["components"].values()) else 1


def parse_variant(text: str) -> tuple[str, dict[str, Any]]:
    """`LABEL=JSON`: a fixed-recipe variant, the JSON object holding candidate fields (see `fixed_recipe.EDITABLE`)."""
    label, sep, body = text.partition("=")
    if not sep or not label.isidentifier():
        raise ControlError(f"a variant is LABEL=JSON with an identifier label, got {text[:40]!r}")
    edits = json.loads(body)
    if not isinstance(edits, dict):
        raise ControlError("a variant's JSON must be an object")
    return label, edits


def cmd_fixed_recipe(args: argparse.Namespace) -> int:
    """No-search control cells: the declared initial recipe trained once for each of `--steps`, per seed."""
    from .fixed_recipe import prepare_cell

    base = load_config(args.config, args)
    gate, prepared_cells, cells = JobGate(base["scheduler"]["max_concurrent_jobs"]), [], []
    stop = _install_stop_flag()
    try:
        variants = [parse_variant(text) for text in args.variant] if args.variant else [(None, None)]
        for seed in args.seeds:
            for label, edits in variants:
                for steps in args.steps:
                    config = {**base, "protocol_condition": "no_self_validation", "seed": seed}
                    name = f"{config.get('task_family')}_fixedrecipe{'' if label is None else '_' + label}_steps{steps}_seed{seed}"
                    run_dir = args.output_root.expanduser().resolve() / name
                    prepared = prepare_cell(config, steps=steps, run_dir=run_dir, repo_root=args.repo_root,
                                            scheduler=make_scheduler(args.scheduler, run_dir), device=args.device,
                                            may_submit=gate.allow, edits=edits, label=label)
                    prepared_cells.append(prepared)
                    cells.append(Cell(run_dir.name, prepared.engine))
        gate.bind(cells)
        states = run_campaign(cells, poll_seconds=base["scheduler"]["poll_seconds"], should_stop=lambda: stop[0])
        summary = {name: state.finished_status for name, state in states.items()}
        if not stop[0] and not args.no_audit:
            for cell, prepared in zip(cells, prepared_cells):
                if states[cell.name].finished_status == "frozen":
                    summary[cell.name] = {"frozen": _finish(prepared, args).get("terminal_audit")}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for prepared in prepared_cells:
            prepared.close()
    return 0


def status_line(run_dir: Path) -> dict[str, Any]:
    events = read_events(run_dir / "events" / "log.jsonl")
    if not events:
        return {"run": run_dir.name, "status": "not_started"}
    state = fold(from_record(event) for event in events)
    counts: dict[str, int] = {}
    for candidate in state.candidates.values():
        counts[candidate.status] = counts.get(candidate.status, 0) + 1
    usage = state.usage
    attention = run_dir / "needs_attention.json"
    line = {
        "run": run_dir.name, "status": state.finished_status or "running", "rounds_done": state.rounds_done,
        "open_round": state.open_round.index if state.open_round else None, "candidates": counts,
        "baseline": state.baseline.status, "committed_updates": state.committed_updates,
        "decisions": len(state.decisions),
        "invalid_decisions": sum(1 for d in state.decisions.values() if d.status == "invalid"),
        "tokens": {"prompt": usage.prompt, "cached": usage.cached, "completion": usage.completion,
                   "reasoning": usage.reasoning},
        "endpoint": endpoint_host(state.run.llm.get("base_url")), "resolved_models": state.resolved_models,
        "resumes": state.resumes, "first_event": events[0].ts, "last_event": events[-1].ts,
        "needs_attention": json.loads(attention.read_text()) if attention.exists() and state.finished_status
        == "needs_attention" else None,
    }
    if state.ext.get("modes"):
        from .ext.modes import effort
        line["human"] = effort(state)
    return line


def cmd_status(args: argparse.Namespace) -> int:
    for directory in args.directories:
        print(json.dumps(status_line(Path(directory)), sort_keys=True))
    return 0


def matched_updates(match_root: Path, task_family: str, seed: int) -> int:
    """The updates the comparison search of this seed committed, which an arm's budget is set to."""
    cell = match_root.expanduser().resolve() / f"{task_family}_no_self_validation_seed{seed}"
    line = status_line(cell)
    if line.get("status") in (None, "not_started"):
        raise ControlError(f"no comparison search at {cell}")
    return int(line["committed_updates"])


def cmd_arm_search(args: argparse.Namespace) -> int:
    """Cells whose Strategist is a non-LLM arm (`arms.py`): the contract, validators, selection and delivery are the searches'."""
    from .arms import ArmTransport, arm_class, arm_config

    base = load_config(args.config, args)
    arm_type = arm_class(args.arm)
    gate, prepared_cells, cells = JobGate(base["scheduler"]["max_concurrent_jobs"]), [], []
    stop = _install_stop_flag()
    try:
        for seed in args.seeds:
            updates = (matched_updates(args.match_root, base.get("task_family"), seed) if args.match_root
                       else args.total_updates)
            config = {**arm_config(base, arm_type.name, total_updates=updates),
                      "protocol_condition": "no_self_validation", "seed": seed}
            run_dir = args.output_root.expanduser().resolve() / f"{config.get('task_family')}_{arm_type.name}_seed{seed}"
            transport = ArmTransport(arm_type(seed))
            prepared = prepare(config, run_dir=run_dir, repo_root=args.repo_root,
                               scheduler=make_scheduler(args.scheduler, run_dir), transport=transport,
                               device=args.device, components=None, may_submit=gate.allow)
            transport.bind(prepared.engine)
            prepared_cells.append(prepared)
            cells.append(Cell(run_dir.name, prepared.engine))
        gate.bind(cells)
        states = run_campaign(cells, poll_seconds=base["scheduler"]["poll_seconds"], should_stop=lambda: stop[0])
        summary = {name: state.finished_status for name, state in states.items()}
        if not stop[0] and not args.no_audit:
            for cell, prepared in zip(cells, prepared_cells):
                if states[cell.name].finished_status == "frozen":
                    summary[cell.name] = {"frozen": _finish(prepared, args).get("terminal_audit")}
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for prepared in prepared_cells:
            prepared.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    for directory in args.directories:
        summary = export(Path(directory))
        print(json.dumps({"run": Path(directory).name, "status": summary["status"],
                          "candidates": summary["candidates"]["allocated"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recital.control.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int)
        p.add_argument("--total-updates", dest="total_updates", type=int)
        p.add_argument("--max-concurrent-jobs", dest="max_concurrent_jobs", type=int)
        p.add_argument("--partition", help="replaces scheduler.partition of the config (kept in the cell's config.yaml)")
        p.add_argument("--scheduler", choices=("local", "slurm"), default="slurm")
        p.add_argument("--no-audit", action="store_true", help="skip the terminal blind audit")

    pre = sub.add_parser("preflight")
    common(pre)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--arm", choices=ARMS)
    pre.set_defaults(func=cmd_preflight)
    run = sub.add_parser("run")
    common(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS)
    run.set_defaults(func=cmd_run)
    camp = sub.add_parser("campaign")
    common(camp)
    camp.add_argument("--output-root", type=Path, required=True)
    camp.add_argument("--arms", nargs="+", choices=ARMS, default=["no_self_validation"])
    camp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    camp.add_argument("--transport", metavar="MODULE:CLASS",
                      help="a class built over the config's LLM transport as Class(inner=, config=, run_dir=) and bound to "
                           "the cell's engine (bind(engine)) when it has that method; default: the config's transport")
    camp.set_defaults(func=cmd_campaign)
    drill = sub.add_parser("drill-coder", help="scripted Strategist, real Coder and real training")
    common(drill)
    drill.add_argument("--output", type=Path, required=True)
    drill.add_argument("--arm", choices=ARMS)
    drill.add_argument("--drill-oracles", nargs="+", default=["malinois", "grammar_prior"],
                       help="enabled oracles of the task: the first gets the requested utility")
    drill.set_defaults(func=cmd_drill_coder)
    fixed = sub.add_parser("fixed-recipe", help="no-search control: the declared initial recipe trained once per --steps")
    common(fixed)
    fixed.add_argument("--output-root", type=Path, required=True)
    fixed.add_argument("--steps", nargs="+", type=int, required=True)
    fixed.add_argument("--seeds", nargs="+", type=int, required=True)
    fixed.add_argument("--variant", action="append", metavar="LABEL=JSON",
                       help="train this edit of the declared recipe instead of the declared recipe itself; repeat for "
                            "several variants (candidate fields: lr kl group_size enabled_oracles aggregator "
                            "oracle_utilities penalties gates)")
    fixed.set_defaults(func=cmd_fixed_recipe)
    arm = sub.add_parser("arm-search", help="cells whose Strategist is a non-LLM arm (random, rule, or module:Class)")
    common(arm)
    arm.add_argument("--arm", required=True)
    arm.add_argument("--output-root", type=Path, required=True)
    arm.add_argument("--seeds", nargs="+", type=int, required=True)
    arm.add_argument("--match-root", type=Path,
                     help="a campaign root of Strategist searches; each cell's total updates is set to the updates "
                          "its seed's search committed")
    arm.set_defaults(func=cmd_arm_search)
    human = sub.add_parser("human-search", help="a cell whose Strategist is a person (modes.mode human_only)")
    common(human)
    human.add_argument("--output", type=Path, required=True)
    human.set_defaults(func=cmd_human_search)

    def posting(p: argparse.ArgumentParser, *, minutes: bool = True) -> None:
        p.add_argument("--run", type=Path, required=True)
        if minutes:
            p.add_argument("--minutes", type=float, required=True,
                           help="the active minutes this took you, reading included: the effort the search records")

    show = sub.add_parser("show", help="what a copilot or human-only cell waits for you to do, and your effort so far")
    posting(show, minutes=False)
    show.set_defaults(func=cmd_show)
    note = sub.add_parser("note", help="post the direction that opens a copilot search, or a note between rounds")
    posting(note)
    note.add_argument("--kind", choices=("direction", "note"), default="note")
    source = note.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    note.set_defaults(func=cmd_note)
    approve = sub.add_parser("approve", help="release a proposal to training, vetoing the candidates named by --veto")
    posting(approve)
    approve.add_argument("--decision", help="the proposal (default: the oldest one waiting)")
    approve.add_argument("--veto", type=int, nargs="*", metavar="INDEX", help="candidates[INDEX] of the proposal")
    approve.add_argument("--text", help="why; the Strategist reads it beside a veto")
    approve.set_defaults(func=cmd_approve)
    propose = sub.add_parser("propose", help="post your response for the round a human_only cell waits on")
    posting(propose)
    propose.add_argument("--file", type=Path, required=True, help="JSON in the Strategist's response schema")
    propose.set_defaults(func=cmd_propose)
    stat = sub.add_parser("status")
    stat.add_argument("directories", nargs="+")
    stat.set_defaults(func=cmd_status)
    exp = sub.add_parser("export", help="write each cell's candidates, self-validation and decisions tables")
    exp.add_argument("directories", nargs="+")
    exp.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
