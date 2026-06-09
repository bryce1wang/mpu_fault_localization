from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from dpu_fault_agent.graph import build_graph, make_initial_state


def _case(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "boot.log"
    log.write_text(
        "[dpu_drv] init failed with DPU_ERR_TIMEOUT status=0xdead\n",
        encoding="utf-8",
    )
    source = tmp_path / "src"
    source.mkdir()
    (source / "driver.c").write_text(
        "int dpu_init(void) {\n    return DPU_ERR_TIMEOUT;\n}\n",
        encoding="utf-8",
    )
    return log, source


def _script_skill(root: Path) -> Path:
    skill_dir = root / "virtualization" / "vf" / "vf-init-timeout"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "collect.py").write_text(
        "import os\n"
        "print('script evidence for ' + os.environ['DPU_FAULT_AGENT_PROBLEM'])\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text(
        """---
name: vf-init-timeout
description: Diagnose VF init timeout and DPU_ERR_TIMEOUT during VF bring-up.
feature: virtualization
module: vf
problem_type: init-timeout
modules: [vf, dpu]
keywords: [vf, init, DPU_ERR_TIMEOUT]
symptoms: [timeout, failed]
required_evidence: [failing log window]
triage_steps: [Run skill collector, Check VF init stage]
common_causes: [queue setup timeout]
validation_steps: [Compare known-good VF init log]
scripts:
  - name: collect
    path: scripts/collect.py
    timeout_seconds: 5
---

VF initialization troubleshooting notes.
""",
        encoding="utf-8",
    )
    return root


def test_graph_interrupts_at_human_gate(tmp_path: Path) -> None:
    log, source = _case(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-1"}}
    initial = make_initial_state(
        thread_id="case-1",
        problem="DPU init timeout",
        log_paths=[str(log)],
        source_root=str(source),
    )

    events = list(graph.stream(initial, config, stream_mode="updates"))

    assert "__interrupt__" in events[-1]
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_gate",)
    assert snapshot.values["approval"]["status"] == "pending"
    assert snapshot.values["diagnosis_plan"]["steps"]


def test_graph_resume_generates_report(tmp_path: Path) -> None:
    log, source = _case(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-2"}}
    initial = make_initial_state(
        thread_id="case-2",
        problem="DPU init timeout",
        log_paths=[str(log)],
        source_root=str(source),
    )
    graph.invoke(initial, config)

    result = graph.invoke(
        Command(resume={"status": "approved", "approved_ids": ["H1"]}),
        config,
    )

    assert result["approval"]["status"] == "approved"
    assert result["final_report"].startswith("# DPU Fault Localization Report")
    assert "DPU_ERR_TIMEOUT" in result["final_report"]


def test_rejected_hypothesis_is_not_primary(tmp_path: Path) -> None:
    log, source = _case(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-3"}}
    initial = make_initial_state(
        thread_id="case-3",
        problem="DPU init timeout",
        log_paths=[str(log)],
        source_root=str(source),
    )
    graph.invoke(initial, config)

    result = graph.invoke(Command(resume={"status": "rejected"}), config)

    assert "No hypothesis was approved" in result["final_report"]
    assert result["hypotheses"][0]["status"] == "rejected"


def test_problem_only_run_generates_plan_and_requests_evidence() -> None:
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-4"}}
    initial = make_initial_state(
        thread_id="case-4",
        problem="VF init timeout after DPU reset",
    )

    events = list(graph.stream(initial, config, stream_mode="updates"))

    assert "__interrupt__" in events[-1]
    snapshot = graph.get_state(config)
    assert snapshot.values["diagnosis_plan"]["evidence_gaps"]
    assert snapshot.values["approval"]["status"] == "needs_more_evidence"
    assert snapshot.values["matched_skills"][0]["id"] == "generic-dpu"


def test_resume_with_supplemental_material_updates_plan(tmp_path: Path) -> None:
    log, source = _case(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-5"}}
    initial = make_initial_state(
        thread_id="case-5",
        problem="VF init timeout after DPU reset",
    )
    graph.invoke(initial, config)

    result = graph.invoke(
        Command(
            resume={
                "log_paths": [str(log)],
                "source_root": str(source),
                "note": "failure happens after queue setup",
            }
        ),
        config,
    )

    assert result["approval"]["status"] == "approved"
    assert result["observations"]
    assert result["artifacts"]["notes"] == ["failure happens after queue setup"]
    assert "DPU_ERR_TIMEOUT" in result["final_report"]


def test_matched_skill_scripts_add_observations(tmp_path: Path) -> None:
    skill_root = _script_skill(tmp_path / "skills")
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "case-6"}}
    initial = make_initial_state(
        thread_id="case-6",
        problem="VF init timeout after DPU reset",
        skill_dirs=[str(skill_root)],
    )

    events = list(graph.stream(initial, config, stream_mode="updates"))

    assert "__interrupt__" in events[-1]
    snapshot = graph.get_state(config)
    script_observations = [
        obs
        for obs in snapshot.values["observations"]
        if obs.get("kind") == "skill_script"
    ]
    assert script_observations
    assert "script evidence for VF init timeout" in script_observations[0]["evidence"]
