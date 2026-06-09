from __future__ import annotations

from typing import TYPE_CHECKING

from dpu_fault_agent.tools import format_ref

if TYPE_CHECKING:
    from dpu_fault_agent.state import DpuFaultState, Hypothesis
else:
    DpuFaultState = dict
    Hypothesis = dict


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
            lines.append(
                f"- [{obs.get('kind', 'signal')}] {obs.get('summary', '')} ({ref})"
            )
    else:
        lines.append("- No strong log or source-code observations were found.")

    lines.extend(["", "## Problem Analysis", ""])
    analysis = state.get("problem_analysis", {})
    lines.append(f"- Keywords: {', '.join(analysis.get('keywords', [])) or 'N/A'}")
    lines.append(f"- Symptoms: {', '.join(analysis.get('symptoms', [])) or 'N/A'}")
    lines.append(
        f"- Missing information: {', '.join(analysis.get('missing_info', [])) or 'N/A'}"
    )

    lines.extend(["", "## Matched Skills", ""])
    matches = state.get("matched_skills", [])
    if matches:
        for match in matches:
            reasons = ", ".join(match.get("reasons", []))
            lines.append(
                f"- `{match.get('id')}` {match.get('name')} "
                f"(score={match.get('score')}, reasons={reasons})"
            )
    else:
        lines.append("- No feature-specific skill matched; generic triage was used.")

    lines.extend(["", "## Diagnosis Plan", ""])
    plan = state.get("diagnosis_plan", {})
    lines.append(plan.get("summary", "No diagnosis plan was generated."))
    for step in plan.get("steps", []):
        lines.append(f"- Step: {step}")
    for gap in plan.get("evidence_gaps", []):
        lines.append(f"- Evidence gap: {gap}")

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
    for step in plan.get("next_actions", []):
        if step not in validation_steps:
            validation_steps.append(step)
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
