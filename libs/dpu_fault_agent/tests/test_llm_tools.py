from __future__ import annotations

import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from dpu_fault_agent.graph import build_graph, make_initial_state
from dpu_fault_agent.llm_tools import assess_tool_risk


def _llm_config(tool_calls: list[dict], *, max_tool_steps: int = 5) -> dict:
    return {
        "enabled": True,
        "provider": "mock",
        "model": "mock",
        "max_tool_steps": max_tool_steps,
        "mock_response": json.dumps(
            {"reasoning_summary": "Use targeted tools.", "tool_calls": tool_calls}
        ),
    }


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    source.mkdir()
    (source / "driver.c").write_text(
        "int dpu_init(void) {\n    return DPU_ERR_TIMEOUT;\n}\n",
        encoding="utf-8",
    )
    return source


def test_llm_builtin_read_file_tool_adds_observation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "llm-read-file"}}
    initial = make_initial_state(
        thread_id="llm-read-file",
        problem="Inspect DPU timeout source",
        source_root=str(source),
        llm_config=_llm_config(
            [
                {
                    "id": "T1",
                    "tool": "read_file",
                    "args": {"path": "driver.c", "start_line": 1, "max_lines": 2},
                    "reason": "Inspect the matched source file.",
                }
            ]
        ),
    )

    list(graph.stream(initial, config, stream_mode="updates"))
    snapshot = graph.get_state(config)

    assert any(obs.get("kind") == "file" for obs in snapshot.values["observations"])
    assert snapshot.values["tool_calls"][0]["status"] == "completed"


def test_llm_low_risk_shell_runs_without_approval(tmp_path: Path) -> None:
    source = _source(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "llm-shell-read"}}
    initial = make_initial_state(
        thread_id="llm-shell-read",
        problem="List source files",
        source_root=str(source),
        llm_config=_llm_config(
            [
                {
                    "id": "T1",
                    "tool": "shell",
                    "args": {"command": "Get-ChildItem"},
                    "reason": "List available source files.",
                }
            ]
        ),
    )

    list(graph.stream(initial, config, stream_mode="updates"))
    snapshot = graph.get_state(config)

    assert snapshot.next == ("human_gate",)
    assert snapshot.values["tool_calls"][0]["status"] == "completed"
    assert any(obs.get("kind") == "shell" for obs in snapshot.values["observations"])


def test_llm_risky_shell_requires_approval_and_reject_does_not_execute(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    target = source / "driver.c"
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "llm-shell-risk"}}
    initial = make_initial_state(
        thread_id="llm-shell-risk",
        problem="Remove bad file",
        source_root=str(source),
        llm_config=_llm_config(
            [
                {
                    "id": "T1",
                    "tool": "shell",
                    "args": {"command": "Remove-Item driver.c"},
                    "reason": "Risky cleanup should require approval.",
                }
            ]
        ),
    )

    events = list(graph.stream(initial, config, stream_mode="updates"))
    snapshot = graph.get_state(config)

    assert "__interrupt__" in events[-1]
    assert snapshot.values["approval"]["status"] == "pending_tool_approval"
    assert snapshot.values["pending_action"]["command"] == "Remove-Item driver.c"
    assert target.exists()

    result = graph.invoke(Command(resume={"status": "rejected"}), config)

    assert target.exists()
    assert result["tool_calls"][0]["status"] == "rejected"


def test_llm_risky_shell_executes_after_approval(tmp_path: Path) -> None:
    source = _source(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "llm-shell-approve"}}
    initial = make_initial_state(
        thread_id="llm-shell-approve",
        problem="Run explicit diagnostic echo",
        source_root=str(source),
        llm_config=_llm_config(
            [
                {
                    "id": "T1",
                    "tool": "shell",
                    "args": {"command": "Write-Output approved"},
                    "reason": "Explicitly requested diagnostic shell output.",
                }
            ]
        ),
    )
    graph.invoke(initial, config)

    result = graph.invoke(Command(resume={"status": "approved"}), config)

    assert result["tool_calls"][0]["status"] == "completed"
    assert "approved" in result["tool_calls"][0]["stdout"]


def test_llm_tool_budget_skips_extra_calls(tmp_path: Path) -> None:
    source = _source(tmp_path)
    calls = [
        {
            "id": f"T{idx}",
            "tool": "shell",
            "args": {"command": "Get-ChildItem"},
            "reason": "Read directory.",
        }
        for idx in range(1, 7)
    ]
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "llm-budget"}}
    initial = make_initial_state(
        thread_id="llm-budget",
        problem="List source repeatedly",
        source_root=str(source),
        llm_config=_llm_config(calls, max_tool_steps=5),
    )

    list(graph.stream(initial, config, stream_mode="updates"))
    snapshot = graph.get_state(config)

    statuses = [call["status"] for call in snapshot.values["tool_calls"]]
    assert statuses.count("completed") == 5
    assert statuses.count("skipped_budget") == 1


def test_shell_risk_classifier_marks_network_and_writes_for_approval() -> None:
    state = {"artifacts": {"source_root": ""}}

    assert (
        assess_tool_risk(
            state, {"tool": "shell", "args": {"command": "Get-ChildItem"}}
        )["requires_approval"]
        is False
    )
    assert (
        assess_tool_risk(
            state, {"tool": "shell", "args": {"command": "curl https://example.com"}}
        )["requires_approval"]
        is True
    )
    assert (
        assess_tool_risk(
            state, {"tool": "shell", "args": {"command": "Set-Content a.txt x"}}
        )["requires_approval"]
        is True
    )
