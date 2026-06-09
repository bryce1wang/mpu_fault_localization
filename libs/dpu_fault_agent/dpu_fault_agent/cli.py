from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

from dpu_fault_agent.checkpoint import DEFAULT_CHECKPOINT_PATH, sqlite_checkpointer
from dpu_fault_agent.graph import build_graph, make_initial_state
from dpu_fault_agent.model import load_model_config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpu-fault-agent")
    parser.add_argument(
        "--checkpoint-db",
        default=DEFAULT_CHECKPOINT_PATH,
        help="SQLite checkpoint database path.",
    )
    parser.add_argument("--config", help="YAML config file with `llm` settings.")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-max-tool-steps", type=int)
    parser.add_argument("--no-llm", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    run = subparsers.add_parser("run", help="Start a fault-localization run.")
    run.add_argument("--thread-id", required=True)
    run.add_argument("--case-id")
    run.add_argument("--problem", required=True)
    run.add_argument("--log", action="append", default=[], dest="logs")
    run.add_argument("--source")
    run.add_argument("--skills", action="append", default=[], dest="skills")
    run.set_defaults(func=cmd_run)

    status = subparsers.add_parser("status", help="Show checkpointed run status.")
    status.add_argument("--thread-id", required=True)
    status.set_defaults(func=cmd_status)

    resume = subparsers.add_parser("resume", help="Resume from the human gate.")
    resume.add_argument("--thread-id", required=True)
    group = resume.add_mutually_exclusive_group()
    group.add_argument("--approve", help="Comma-separated hypothesis IDs to approve.")
    group.add_argument("--reject", action="store_true", help="Reject all hypotheses.")
    resume.add_argument("--log", action="append", default=[], dest="logs")
    resume.add_argument("--source")
    resume.add_argument("--note", default="")
    resume.set_defaults(func=cmd_resume)

    report = subparsers.add_parser("report", help="Export the final report.")
    report.add_argument("--thread-id", required=True)
    report.add_argument("--output")
    report.set_defaults(func=cmd_report)
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args.thread_id)
    initial = make_initial_state(
        thread_id=args.thread_id,
        case_id=args.case_id,
        problem=args.problem,
        log_paths=args.logs,
        source_root=args.source,
        skill_dirs=args.skills,
        llm_config=_model_config_from_args(args),
    )
    with sqlite_checkpointer(args.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        _stream_updates(graph, initial, config)
        snapshot = graph.get_state(config)
        _print_snapshot(snapshot.values, snapshot.next)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _config(args.thread_id)
    with sqlite_checkpointer(args.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = graph.get_state(config)
        _print_snapshot(snapshot.values, snapshot.next)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config = _config(args.thread_id)
    if not any([args.approve, args.reject, args.logs, args.source, args.note]):
        print(
            "error: resume requires --approve, --reject, --log, --source, or --note",
            file=sys.stderr,
        )
        return 1
    if args.reject:
        decision: dict[str, Any] = {"status": "rejected", "note": args.note}
    else:
        decision = {"note": args.note}
        if args.approve:
            approved = [
                item.strip() for item in args.approve.split(",") if item.strip()
            ]
            decision.update({"status": "approved", "approved_ids": approved})
    if args.logs:
        decision["log_paths"] = args.logs
    if args.source:
        decision["source_root"] = args.source
    with sqlite_checkpointer(args.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        _stream_updates(graph, Command(resume=decision), config)
        snapshot = graph.get_state(config)
        _print_snapshot(snapshot.values, snapshot.next)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = _config(args.thread_id)
    with sqlite_checkpointer(args.checkpoint_db) as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = graph.get_state(config)
        values = snapshot.values
    report = values.get("final_report")
    if not report:
        print("No final report is available yet.", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(report)
    return 0


def _stream_updates(graph: Any, input_value: Any, config: dict[str, Any]) -> None:
    for event in graph.stream(input_value, config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupts = event["__interrupt__"]
            print("== human_gate: interrupted ==")
            for item in interrupts:
                print(
                    json.dumps(
                        getattr(item, "value", item), indent=2, ensure_ascii=False
                    )
                )
            continue
        for node, update in event.items():
            print(f"== {node} ==")
            if isinstance(update, dict):
                print(_summarize_update(update))
            else:
                print(update)


def _print_snapshot(values: dict[str, Any], next_nodes: tuple[str, ...]) -> None:
    print("== status ==")
    print(f"thread_id: {values.get('thread_id', '')}")
    print(f"case_id: {values.get('case_id', '')}")
    print(f"next: {', '.join(next_nodes) if next_nodes else 'END'}")
    print(f"observations: {len(values.get('observations', []))}")
    print(f"hypotheses: {len(values.get('hypotheses', []))}")
    print(f"matched_skills: {len(values.get('matched_skills', []))}")
    print(f"tool_calls: {len(values.get('tool_calls', []))}")
    print(f"approval: {values.get('approval', {}).get('status', 'unknown')}")
    if values.get("pending_action"):
        action = values["pending_action"]
        print(f"pending_action: {action.get('tool')} {action.get('command', '')}")
    plan = values.get("diagnosis_plan", {})
    if plan:
        print(f"diagnosis_plan: {plan.get('summary', '')}")
    if values.get("final_report"):
        print("final_report: available")


def _summarize_update(update: dict[str, Any]) -> str:
    parts: list[str] = []
    if "observations" in update:
        parts.append(f"observations={len(update['observations'])}")
    if "hypotheses" in update:
        parts.append(f"hypotheses={len(update['hypotheses'])}")
    if "matched_skills" in update:
        parts.append(f"matched_skills={len(update['matched_skills'])}")
    if "tool_calls" in update:
        parts.append(f"tool_calls={len(update['tool_calls'])}")
    if "pending_action" in update and update["pending_action"]:
        parts.append("pending_action=available")
    if "diagnosis_plan" in update:
        parts.append("diagnosis_plan=available")
    if "approval" in update:
        parts.append(f"approval={update['approval'].get('status')}")
    if "final_report" in update:
        parts.append("final_report=available")
    if "messages" in update:
        last = update["messages"][-1]
        parts.append(f"message={getattr(last, 'content', last)}")
    return ", ".join(parts) if parts else json.dumps(update, default=str)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "model": args.llm_model,
        "base_url": args.llm_base_url,
        "api_key_env": args.llm_api_key_env,
        "temperature": args.llm_temperature,
        "max_tool_steps": args.llm_max_tool_steps,
    }
    if args.no_llm:
        overrides["enabled"] = False
    elif any(
        value is not None
        for value in (
            args.llm_model,
            args.llm_base_url,
            args.llm_api_key_env,
            args.llm_temperature,
            args.llm_max_tool_steps,
        )
    ):
        overrides["enabled"] = True
    config = load_model_config(args.config, overrides)
    return config.to_dict()
