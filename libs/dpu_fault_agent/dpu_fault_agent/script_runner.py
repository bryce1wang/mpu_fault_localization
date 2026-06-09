from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 8000


def run_skill_scripts(
    state: dict[str, Any], matched_skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for skill in matched_skills:
        for script in skill.get("scripts", []):
            observations.append(_run_script(state, skill, script))
    return observations


def _run_script(
    state: dict[str, Any], skill: dict[str, Any], script: dict[str, Any]
) -> dict[str, Any]:
    script_path = Path(str(script["path"]))
    command = [sys.executable, str(script_path), *_script_args(script)]
    timeout_seconds = int(script.get("timeout_seconds", 30))
    env = _script_env(state, skill)
    try:
        completed = subprocess.run(
            command,
            cwd=str(script_path.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = _trim_output(_to_text(exc.stdout) + "\n" + _to_text(exc.stderr))
        return {
            "kind": "skill_script",
            "summary": (
                f"Skill script `{skill.get('id')}/{script.get('name')}` timed out "
                f"after {timeout_seconds}s"
            ),
            "path": str(script_path),
            "symbol": str(skill.get("id", "")),
            "severity": "error",
            "evidence": output,
        }

    combined = _trim_output(completed.stdout + "\n" + completed.stderr)
    severity = "info" if completed.returncode == 0 else "error"
    return {
        "kind": "skill_script",
        "summary": (
            f"Skill script `{skill.get('id')}/{script.get('name')}` exited "
            f"with code {completed.returncode}"
        ),
        "path": str(script_path),
        "symbol": str(skill.get("id", "")),
        "severity": severity,
        "evidence": combined,
    }


def _script_args(script: dict[str, Any]) -> list[str]:
    return [str(item) for item in script.get("args", [])]


def _script_env(state: dict[str, Any], skill: dict[str, Any]) -> dict[str, str]:
    artifacts = state.get("artifacts", {})
    env = dict(os.environ)
    env.update(
        {
            "DPU_FAULT_AGENT_THREAD_ID": str(state.get("thread_id", "")),
            "DPU_FAULT_AGENT_PROBLEM": str(state.get("problem_statement", "")),
            "DPU_FAULT_AGENT_SKILL_ID": str(skill.get("id", "")),
            "DPU_FAULT_AGENT_SKILL_NAME": str(skill.get("name", "")),
            "DPU_FAULT_AGENT_FEATURE": str(skill.get("feature", "")),
            "DPU_FAULT_AGENT_MODULE": str(skill.get("module", "")),
            "DPU_FAULT_AGENT_PROBLEM_TYPE": str(skill.get("problem_type", "")),
            "DPU_FAULT_AGENT_LOG_PATHS": os.pathsep.join(
                str(path) for path in artifacts.get("log_paths", [])
            ),
            "DPU_FAULT_AGENT_SOURCE_ROOT": str(artifacts.get("source_root", "")),
        }
    )
    return env


def _trim_output(output: str) -> str:
    output = output.strip()
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    return output[:MAX_OUTPUT_CHARS] + "\n...[truncated]"


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
