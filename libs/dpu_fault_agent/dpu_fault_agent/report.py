from __future__ import annotations

from dpu_fault_agent.state import DpuFaultState, Hypothesis, Observation
from dpu_fault_agent.tools import format_ref


def render_report(state: DpuFaultState) -> str:
    observations = state.get("observations", [])
    hypotheses = state.get("hypotheses", [])
    approval = state.get("approval", {})
    approved_ids = set(approval.get("approved_ids", []))
    primary = [item for item in hypotheses if item.get("id") in approved_ids]
    if not primary and approval.get("status") != "rejected":
        primary = hypotheses[:1]

    lines = [
        "# DPU Fault Localization Report",
        "",
        "## Problem",
        "",
        state.get("problem_statement", ""),
        "",
        "## Inputs",
        "",
        f"- Case ID: `{state.get('case_id', '')}`",
        f"- Thread ID: `{state.get('thread_id', '')}`",
        f"- Source root: `{state.get('artifacts', {}).get('source_root', '')}`",
    ]
    for log_path in state.get("artifacts", {}).get("log_paths", []):
        lines.append(f"- Log: `{log_path}`")

    lines.extend(["", "## Key Observations", ""])
    if observations:
        for obs in observations[:15]:
            ref = format_ref(obs)
            lines.append(f"- [{obs.get('kind', 'signal')}] {obs.get('summary', '')} ({ref})")
    else:
        lines.append("- No strong log or source-code observations were found.")

    lines.extend(["", "## Primary Hypotheses", ""])
    if primary:
        for hypothesis in primary:
            lines.extend(_render_hypothesis(hypothesis))
    else:
        lines.append("- No hypothesis was approved at the human gate.")

    rejected = [
        item
        for item in hypotheses
        if item.get("id") in set(approval.get("rejected_ids", []))
    ]
    if rejected:
        lines.extend(["", "## Rejected Hypotheses", ""])
        for hypothesis in rejected:
            lines.append(f"- {hypothesis.get('id')}: {hypothesis.get('title')}")

    lines.extend(["", "## Validation Plan", ""])
    validation_steps = _collect_validation_steps(primary or hypotheses)
    if validation_steps:
        for step in validation_steps:
            lines.append(f"- {step}")
    else:
        lines.append("- Collect additional logs and broaden source search terms.")

    lines.extend(
        [
            "",
            "## Human Gate",
            "",
            f"- Status: `{approval.get('status', 'not_reviewed')}`",
            f"- Note: {approval.get('note', '') or 'N/A'}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_hypothesis(hypothesis: Hypothesis) -> list[str]:
    lines = [
        f"### {hypothesis.get('id', '')}: {hypothesis.get('title', '')}",
        "",
        f"- Confidence: {hypothesis.get('confidence', 0):.2f}",
    ]
    for evidence in hypothesis.get("evidence", [])[:5]:
        lines.append(f"- Evidence: {evidence}")
    for ref in hypothesis.get("source_refs", [])[:5]:
        lines.append(f"- Source: `{ref}`")
    return lines + [""]


def _collect_validation_steps(hypotheses: list[Hypothesis]) -> list[str]:
    steps: list[str] = []
    for hypothesis in hypotheses:
        for step in hypothesis.get("validation_steps", []):
            if step not in steps:
                steps.append(step)
    return steps[:10]
