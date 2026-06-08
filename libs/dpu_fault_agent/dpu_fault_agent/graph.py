from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dpu_fault_agent.report import render_report
from dpu_fault_agent.state import Approval, DpuFaultState, Hypothesis, Observation
from dpu_fault_agent.tools import (
    derive_search_terms,
    format_ref,
    normalize_paths,
    search_source,
    triage_logs,
)


def build_graph(*, checkpointer: Any | None = None):
    builder = StateGraph(DpuFaultState)
    builder.add_node("intake", intake)
    builder.add_node("log_triage", log_triage)
    builder.add_node("source_search", source_search)
    builder.add_node("hypothesis_builder", hypothesis_builder)
    builder.add_node("human_gate", human_gate)
    builder.add_node("validation_planner", validation_planner)
    builder.add_node("report_writer", report_writer)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "log_triage")
    builder.add_edge("log_triage", "source_search")
    builder.add_edge("source_search", "hypothesis_builder")
    builder.add_edge("hypothesis_builder", "human_gate")
    builder.add_edge("human_gate", "validation_planner")
    builder.add_edge("validation_planner", "report_writer")
    builder.add_edge("report_writer", END)
    return builder.compile(checkpointer=checkpointer)


def make_initial_state(
    *,
    thread_id: str,
    problem: str,
    log_paths: list[str],
    source_root: str,
    case_id: str | None = None,
) -> DpuFaultState:
    return {
        "messages": [HumanMessage(content=problem)],
        "case_id": case_id or thread_id,
        "thread_id": thread_id,
        "problem_statement": problem,
        "artifacts": {
            "log_paths": log_paths,
            "source_root": source_root,
            "config_paths": [],
        },
        "observations": [],
        "hypotheses": [],
        "approval": {"status": "not_reviewed", "approved_ids": [], "rejected_ids": []},
        "metadata": {},
    }


def intake(state: DpuFaultState) -> dict[str, Any]:
    artifacts = state["artifacts"]
    log_paths = normalize_paths(artifacts.get("log_paths", []))
    source_root = normalize_paths([artifacts.get("source_root", "")])[0]
    missing = [path for path in log_paths if not Path(path).is_file()]
    if missing:
        msg = f"Missing log file(s): {', '.join(missing)}"
        raise FileNotFoundError(msg)
    if not Path(source_root).is_dir():
        msg = f"Missing source root: {source_root}"
        raise FileNotFoundError(msg)
    return {
        "artifacts": {
            **artifacts,
            "log_paths": log_paths,
            "source_root": source_root,
        },
        "messages": [AIMessage(content="Intake complete: inputs were validated.")],
    }


def log_triage(state: DpuFaultState) -> dict[str, Any]:
    observations = triage_logs(state["artifacts"].get("log_paths", []))
    return {
        "observations": observations,
        "messages": [
            AIMessage(content=f"Log triage collected {len(observations)} observation(s).")
        ],
    }


def source_search(state: DpuFaultState) -> dict[str, Any]:
    observations = list(state.get("observations", []))
    terms = derive_search_terms(state["problem_statement"], observations)
    source_hits = search_source(state["artifacts"]["source_root"], terms)
    return {
        "observations": observations + source_hits,
        "metadata": {**state.get("metadata", {}), "search_terms": terms},
        "messages": [
            AIMessage(content=f"Source search found {len(source_hits)} source hit(s).")
        ],
    }


def hypothesis_builder(state: DpuFaultState) -> dict[str, Any]:
    observations = state.get("observations", [])
    source_observations = [obs for obs in observations if obs.get("kind") == "source"]
    log_observations = [obs for obs in observations if obs.get("kind", "").startswith("log")]
    hypotheses: list[Hypothesis] = []

    if source_observations and log_observations:
        for idx, source_obs in enumerate(source_observations[:3], start=1):
            related_logs = log_observations[:3]
            hypotheses.append(
                {
                    "id": f"H{idx}",
                    "title": f"Log signal may originate near `{Path(source_obs.get('path', '')).name}`",
                    "confidence": max(0.35, 0.8 - (idx - 1) * 0.15),
                    "evidence": [obs.get("summary", "") for obs in related_logs]
                    + [source_obs.get("summary", "")],
                    "source_refs": [format_ref(source_obs)],
                    "validation_steps": [
                        f"Inspect `{format_ref(source_obs)}` and confirm the matched condition is reachable.",
                        "Compare failing logs with a known-good boot or initialization log.",
                        "Check DPU firmware, driver, and configuration versions for this module.",
                    ],
                    "status": "candidate",
                }
            )
    elif log_observations:
        hypotheses.append(
            {
                "id": "H1",
                "title": "The failure is visible in logs but source evidence is weak",
                "confidence": 0.35,
                "evidence": [obs.get("summary", "") for obs in log_observations[:5]],
                "source_refs": [],
                "validation_steps": [
                    "Broaden source search terms using the module and error-code tokens from the log.",
                    "Collect a higher-verbosity DPU driver log around the failure window.",
                ],
                "status": "candidate",
            }
        )
    else:
        hypotheses.append(
            {
                "id": "H1",
                "title": "No strong evidence was found in the provided logs",
                "confidence": 0.2,
                "evidence": ["No error-like log lines or source-code matches were found."],
                "source_refs": [],
                "validation_steps": [
                    "Confirm the provided log captures the failure window.",
                    "Re-run with debug logging enabled for the DPU driver and firmware path.",
                ],
                "status": "candidate",
            }
        )

    return {
        "hypotheses": hypotheses,
        "approval": {"status": "pending", "approved_ids": [], "rejected_ids": []},
        "messages": [
            AIMessage(content=f"Built {len(hypotheses)} candidate hypothesis/hypotheses.")
        ],
    }


def human_gate(state: DpuFaultState) -> dict[str, Any]:
    approval = state.get("approval", {})
    if approval.get("status") in {"approved", "rejected"}:
        return {}
    decision = interrupt(
        {
            "stage": "human_gate",
            "message": "Review candidate hypotheses and approve the primary path.",
            "hypotheses": state.get("hypotheses", []),
        }
    )
    normalized = normalize_approval(decision, state.get("hypotheses", []))
    return {
        "approval": normalized,
        "messages": [
            AIMessage(content=f"Human gate completed with `{normalized['status']}`.")
        ],
    }


def validation_planner(state: DpuFaultState) -> dict[str, Any]:
    approval = state.get("approval", {})
    approved = set(approval.get("approved_ids", []))
    rejected = set(approval.get("rejected_ids", []))
    updated: list[Hypothesis] = []
    for hypothesis in state.get("hypotheses", []):
        status = "candidate"
        if hypothesis.get("id") in approved:
            status = "approved"
        elif hypothesis.get("id") in rejected:
            status = "rejected"
        updated.append({**hypothesis, "status": status})
    return {
        "hypotheses": updated,
        "messages": [AIMessage(content="Validation plan finalized.")],
    }


def report_writer(state: DpuFaultState) -> dict[str, Any]:
    report = render_report(state)
    return {
        "final_report": report,
        "messages": [AIMessage(content="Final report generated.")],
    }


def normalize_approval(decision: Any, hypotheses: list[Hypothesis]) -> Approval:
    ids = [item.get("id", "") for item in hypotheses]
    if isinstance(decision, str):
        decision = decision.strip()
        if decision.lower() in {"reject", "rejected", "none"}:
            return {
                "status": "rejected",
                "approved_ids": [],
                "rejected_ids": ids,
                "note": "All hypotheses rejected by reviewer.",
            }
        approved = [part.strip() for part in decision.split(",") if part.strip()]
        return {
            "status": "approved",
            "approved_ids": approved,
            "rejected_ids": [item for item in ids if item not in approved],
            "note": "Approved from CLI shorthand.",
        }
    if isinstance(decision, dict):
        status = str(decision.get("status", "approved"))
        approved_ids = [str(item) for item in decision.get("approved_ids", [])]
        rejected_ids = [str(item) for item in decision.get("rejected_ids", [])]
        if status == "rejected" and not rejected_ids:
            rejected_ids = ids
        if status == "approved" and not approved_ids and ids:
            approved_ids = [ids[0]]
        if not rejected_ids:
            rejected_ids = [item for item in ids if item not in approved_ids]
        return {
            "status": status,
            "approved_ids": approved_ids,
            "rejected_ids": rejected_ids,
            "note": str(decision.get("note", "")),
        }
    first = ids[:1]
    return {
        "status": "approved",
        "approved_ids": first,
        "rejected_ids": [item for item in ids if item not in first],
        "note": "Defaulted to the first hypothesis.",
    }
