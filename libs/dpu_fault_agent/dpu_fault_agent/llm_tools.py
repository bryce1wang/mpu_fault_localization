from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dpu_fault_agent.script_runner import run_skill_scripts
from dpu_fault_agent.tools import (
    derive_search_terms,
    normalize_paths,
    search_source,
    triage_logs,
)

MAX_TOOL_OUTPUT_CHARS = 6000
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
AUTO_READ_PREFIXES = (
    "get-childitem",
    "ls",
    "dir",
    "select-string",
    "grep",
    "findstr",
    "get-content",
    "type",
    "cat",
    "test-path",
    "resolve-path",
    "git status",
    "git show",
    "git diff",
)
RISK_TOKENS = (
    ">",
    ">>",
    "remove-item",
    "rm ",
    "del ",
    "erase ",
    "rmdir",
    "move-item",
    "mv ",
    "set-content",
    "add-content",
    "out-file",
    "new-item",
    "copy-item",
    "cp ",
    "pip install",
    "uv add",
    "npm install",
    "pnpm install",
    "yarn add",
    "curl ",
    "wget ",
    "invoke-webrequest",
    "git clone",
    "git pull",
    "flash",
    "burn",
    "jtag",
    "serial",
    "uart",
    "dfu",
)


def registered_tools(matched_skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skill_scripts = []
    for skill in matched_skills:
        for script in skill.get("scripts", []):
            skill_scripts.append(f"{skill.get('id')}/{script.get('name')}")
    return [
        {
            "name": "log_triage",
            "description": "Read configured DPU logs and extract error-like lines, modules, and codes.",
            "risk_level": "low",
            "requires_approval": False,
        },
        {
            "name": "source_search",
            "description": "Search the configured source root for terms.",
            "risk_level": "low",
            "requires_approval": False,
        },
        {
            "name": "read_file",
            "description": "Read a small text file snippet.",
            "risk_level": "low",
            "requires_approval": False,
        },
        {
            "name": "skill_script",
            "description": "Run a matched skill script by skill id and optional script name.",
            "risk_level": "low",
            "requires_approval": False,
            "available": skill_scripts,
        },
        {
            "name": "shell",
            "description": "Run a shell command. Read-only commands may run automatically; risky commands require approval.",
            "risk_level": "dynamic",
            "requires_approval": "dynamic",
        },
    ]


def parse_tool_plan(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        msg = "LLM tool plan must be a JSON object."
        raise ValueError(msg)
    calls = data.get("tool_calls", [])
    if not isinstance(calls, list):
        msg = "LLM tool plan `tool_calls` must be a list."
        raise ValueError(msg)
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(calls, start=1):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        tool = str(item.get("tool") or item.get("name") or "")
        if not tool:
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            args = {}
        normalized.append(
            {
                "id": str(item.get("id") or f"T{idx}"),
                "tool": tool,
                "args": args,
                "reason": str(item.get("reason", "")),
                "status": "proposed",
            }
        )
    return {
        "tool_calls": normalized,
        "reasoning_summary": str(data.get("reasoning_summary", "")),
        "reflection": str(data.get("reflection", "")),
    }


def execute_tool_calls(
    state: dict[str, Any], tool_calls: list[dict[str, Any]], *, max_steps: int
) -> dict[str, Any]:
    observations = list(state.get("observations", []))
    updated_calls: list[dict[str, Any]] = []
    pending_action: dict[str, Any] = {}
    executed = 0
    for idx, call in enumerate(tool_calls):
        if call.get("status") != "proposed":
            updated_calls.append(call)
            continue
        if executed >= max_steps:
            updated_calls.append({**call, "status": "skipped_budget"})
            continue
        decision = assess_tool_risk(state, call)
        call = {
            **call,
            "risk_level": decision["risk_level"],
            "requires_approval": decision["requires_approval"],
        }
        if decision["requires_approval"]:
            call = {**call, "status": "pending_approval"}
            updated_calls.append(call)
            pending_action = _pending_action(call, decision)
            updated_calls.extend(tool_calls[idx + 1 :])
            break
        result = _execute_approved_tool(state, call)
        observations.extend(result["observations"])
        updated_calls.append({**call, **result["tool_call_update"]})
        executed += 1
    return {
        "tool_calls": updated_calls,
        "observations": observations,
        "pending_action": pending_action,
        "executed_count": executed,
    }


def execute_pending_action(
    state: dict[str, Any], pending_action: dict[str, Any]
) -> dict[str, Any]:
    tool_call_id = pending_action.get("tool_call_id", "")
    tool_calls = []
    observations = list(state.get("observations", []))
    for call in state.get("tool_calls", []):
        if call.get("id") != tool_call_id:
            tool_calls.append(call)
            continue
        result = _execute_approved_tool(state, {**call, "requires_approval": False})
        observations.extend(result["observations"])
        tool_calls.append({**call, **result["tool_call_update"]})
    return {
        "tool_calls": tool_calls,
        "observations": observations,
        "pending_action": {},
    }


def reject_pending_action(
    state: dict[str, Any], pending_action: dict[str, Any], note: str = ""
) -> dict[str, Any]:
    tool_call_id = pending_action.get("tool_call_id", "")
    tool_calls = []
    for call in state.get("tool_calls", []):
        if call.get("id") == tool_call_id:
            tool_calls.append(
                {
                    **call,
                    "status": "rejected",
                    "result_summary": note or "Rejected by reviewer.",
                }
            )
        else:
            tool_calls.append(call)
    return {"tool_calls": tool_calls, "pending_action": {}}


def assess_tool_risk(state: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    tool = call.get("tool", "")
    if tool != "shell":
        return {
            "risk_level": "low",
            "requires_approval": False,
            "risk_reason": "Registered non-shell diagnostic tool.",
        }
    args = call.get("args", {})
    command = str(args.get("command", ""))
    timeout = int(args.get("timeout_seconds", DEFAULT_SHELL_TIMEOUT_SECONDS))
    lowered = command.strip().lower()
    if timeout > DEFAULT_SHELL_TIMEOUT_SECONDS:
        return _risky("Shell timeout exceeds the automatic execution limit.")
    if any(token in lowered for token in RISK_TOKENS):
        return _risky("Shell command matches a protected operation pattern.")
    if not any(lowered.startswith(prefix) for prefix in AUTO_READ_PREFIXES):
        return _risky("Shell command is not in the read-only allowlist.")
    cwd = _resolve_cwd(state, args.get("cwd"))
    if not cwd.exists():
        return _risky("Shell working directory does not exist.")
    return {
        "risk_level": "low",
        "requires_approval": False,
        "risk_reason": "Read-only shell command matched the allowlist.",
    }


def _execute_approved_tool(
    state: dict[str, Any], call: dict[str, Any]
) -> dict[str, Any]:
    tool = call.get("tool", "")
    try:
        if tool == "log_triage":
            observations = triage_logs(_log_paths(state, call))
            return _tool_result(call, observations, "completed")
        if tool == "source_search":
            terms = _terms(state, call)
            source_root = _source_root(state, call)
            observations = search_source(source_root, terms) if source_root else []
            return _tool_result(call, observations, "completed")
        if tool == "read_file":
            observation = _read_file_observation(state, call)
            return _tool_result(call, [observation], "completed")
        if tool == "skill_script":
            observations = _run_skill_tool(state, call)
            return _tool_result(call, observations, "completed")
        if tool == "shell":
            observation, update = _run_shell(state, call)
            return {"observations": [observation], "tool_call_update": update}
    except Exception as exc:
        observation = {
            "kind": "tool_error",
            "summary": f"Tool `{tool}` failed: {exc}",
            "severity": "error",
            "evidence": str(exc),
        }
        return _tool_result(call, [observation], "failed")
    observation = {
        "kind": "tool_error",
        "summary": f"Unknown tool `{tool}` requested by LLM.",
        "severity": "error",
    }
    return _tool_result(call, [observation], "failed")


def _run_shell(
    state: dict[str, Any], call: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = call.get("args", {})
    command = str(args.get("command", ""))
    cwd = _resolve_cwd(state, args.get("cwd"))
    timeout = int(args.get("timeout_seconds", DEFAULT_SHELL_TIMEOUT_SECONDS))
    if os.name == "nt":
        shell_command = ["powershell", "-NoProfile", "-Command", command]
    else:
        shell_command = ["/bin/sh", "-lc", command]
    completed = subprocess.run(
        shell_command,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout = _trim(_decode_output(completed.stdout))
    stderr = _trim(_decode_output(completed.stderr))
    evidence = "\n".join(part for part in [stdout, stderr] if part)
    observation = {
        "kind": "shell",
        "summary": f"Shell command exited with code {completed.returncode}: {command}",
        "path": str(cwd),
        "severity": "info" if completed.returncode == 0 else "error",
        "evidence": evidence,
    }
    update = {
        "status": "completed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "result_summary": observation["summary"],
    }
    return observation, update


def _read_file_observation(
    state: dict[str, Any], call: dict[str, Any]
) -> dict[str, Any]:
    args = call.get("args", {})
    path = _resolve_path(state, str(args.get("path", "")))
    start_line = max(1, int(args.get("start_line", 1)))
    max_lines = max(1, min(200, int(args.get("max_lines", 40))))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    snippet = "\n".join(lines[start_line - 1 : start_line - 1 + max_lines])
    return {
        "kind": "file",
        "summary": f"Read {path.name}:{start_line}",
        "path": str(path),
        "line": start_line,
        "severity": "info",
        "evidence": snippet,
    }


def _run_skill_tool(
    state: dict[str, Any], call: dict[str, Any]
) -> list[dict[str, Any]]:
    args = call.get("args", {})
    skill_id = str(args.get("skill_id", ""))
    script_name = str(args.get("script_name", ""))
    matched = []
    for skill in state.get("matched_skills", []):
        if skill_id and skill.get("id") != skill_id:
            continue
        scripts = skill.get("scripts", [])
        if script_name:
            scripts = [
                script for script in scripts if script.get("name") == script_name
            ]
        if scripts:
            matched.append({**skill, "scripts": scripts})
    return run_skill_scripts(state, matched)


def _tool_result(
    call: dict[str, Any], observations: list[dict[str, Any]], status: str
) -> dict[str, Any]:
    summary = "; ".join(obs.get("summary", "") for obs in observations[:3])
    return {
        "observations": observations,
        "tool_call_update": {
            "status": status,
            "result_summary": summary,
        },
    }


def _pending_action(call: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    args = call.get("args", {})
    return {
        "tool_call_id": str(call.get("id", "")),
        "tool": str(call.get("tool", "")),
        "args": args,
        "command": str(args.get("command", "")),
        "cwd": str(args.get("cwd", "")),
        "risk_level": str(decision.get("risk_level", "high")),
        "risk_reason": str(decision.get("risk_reason", "")),
        "expected_benefit": str(call.get("reason", "")),
        "possible_impact": "May modify files, environment, devices, network state, or run longer than the automatic limit.",
    }


def _risky(reason: str) -> dict[str, Any]:
    return {"risk_level": "high", "requires_approval": True, "risk_reason": reason}


def _log_paths(state: dict[str, Any], call: dict[str, Any]) -> list[str]:
    args = call.get("args", {})
    if args.get("log_paths"):
        return normalize_paths([str(path) for path in args["log_paths"]])
    return [str(path) for path in state.get("artifacts", {}).get("log_paths", [])]


def _source_root(state: dict[str, Any], call: dict[str, Any]) -> str:
    args = call.get("args", {})
    if args.get("source_root"):
        return normalize_paths([str(args["source_root"])])[0]
    return str(state.get("artifacts", {}).get("source_root", ""))


def _terms(state: dict[str, Any], call: dict[str, Any]) -> list[str]:
    args = call.get("args", {})
    terms = args.get("terms")
    if isinstance(terms, str):
        return [terms]
    if isinstance(terms, list):
        return [str(term) for term in terms]
    return derive_search_terms(
        str(state.get("problem_statement", "")), state.get("observations", [])
    )


def _resolve_path(state: dict[str, Any], raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_resolve_cwd(state, None) / path).resolve()


def _resolve_cwd(state: dict[str, Any], raw_cwd: Any) -> Path:
    if raw_cwd:
        path = Path(str(raw_cwd))
        return path.resolve() if not path.is_absolute() else path
    source_root = state.get("artifacts", {}).get("source_root", "")
    if source_root:
        return Path(str(source_root)).resolve()
    return Path.cwd().resolve()


def _trim(value: str) -> str:
    value = value.strip()
    if len(value) <= MAX_TOOL_OUTPUT_CHARS:
        return value
    return value[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"


def _decode_output(value: bytes | str) -> str:
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", "gbk"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")
