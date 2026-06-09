from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dpu_fault_agent.llm_tools import (
    execute_pending_action,
    execute_tool_calls,
    parse_tool_plan,
    registered_tools,
    reject_pending_action,
)
from dpu_fault_agent.model import ChatModelFactory, ModelConfig
from dpu_fault_agent.report import render_report
from dpu_fault_agent.script_runner import run_skill_scripts
from dpu_fault_agent.skills import default_skill_dirs, load_skills, match_skills
from dpu_fault_agent.state import Approval, DiagnosisPlan, DpuFaultState, Hypothesis
from dpu_fault_agent.tools import (
    TOKEN_RE,
    derive_search_terms,
    format_ref,
    normalize_paths,
    search_source,
    triage_logs,
)


def build_graph(*, checkpointer: Any | None = None):
    builder = StateGraph(DpuFaultState)
    builder.add_node("intake", intake)
    builder.add_node("problem_analyzer", problem_analyzer)
    builder.add_node("skill_loader", skill_loader)
    builder.add_node("evidence_collector", evidence_collector)
    builder.add_node("skill_router", skill_router)
    builder.add_node("skill_script_runner", skill_script_runner)
    builder.add_node("llm_tool_planner", llm_tool_planner)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("tool_reflector", tool_reflector)
    builder.add_node("diagnosis_planner", diagnosis_planner)
    builder.add_node("human_gate", human_gate)
    builder.add_node("hypothesis_builder", hypothesis_builder)
    builder.add_node("validation_planner", validation_planner)
    builder.add_node("report_writer", report_writer)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "problem_analyzer")
    builder.add_edge("problem_analyzer", "skill_loader")
    builder.add_edge("skill_loader", "evidence_collector")
    builder.add_edge("evidence_collector", "skill_router")
    builder.add_edge("skill_router", "skill_script_runner")
    builder.add_edge("skill_script_runner", "llm_tool_planner")
    builder.add_edge("llm_tool_planner", "tool_executor")
    builder.add_edge("tool_executor", "tool_reflector")
    builder.add_edge("tool_reflector", "diagnosis_planner")
    builder.add_edge("diagnosis_planner", "human_gate")
    builder.add_edge("human_gate", "hypothesis_builder")
    builder.add_edge("hypothesis_builder", "validation_planner")
    builder.add_edge("validation_planner", "report_writer")
    builder.add_edge("report_writer", END)
    return builder.compile(checkpointer=checkpointer)


def make_initial_state(
    *,
    thread_id: str,
    problem: str,
    log_paths: list[str] | None = None,
    source_root: str | None = None,
    skill_dirs: list[str] | None = None,
    case_id: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> DpuFaultState:
    return {
        "messages": [HumanMessage(content=problem)],
        "case_id": case_id or thread_id,
        "thread_id": thread_id,
        "problem_statement": problem,
        "artifacts": {
            "log_paths": log_paths or [],
            "source_root": source_root or "",
            "skill_dirs": default_skill_dirs(skill_dirs),
            "config_paths": [],
            "notes": [],
        },
        "problem_analysis": {},
        "matched_skills": [],
        "diagnosis_plan": {},
        "observations": [],
        "hypotheses": [],
        "tool_calls": [],
        "pending_action": {},
        "llm_analysis": {},
        "approval": {"status": "not_reviewed", "approved_ids": [], "rejected_ids": []},
        "metadata": {"llm_config": llm_config or ModelConfig().to_dict()},
    }


def intake(state: DpuFaultState) -> dict[str, Any]:
    artifacts = state["artifacts"]
    log_paths = normalize_paths(artifacts.get("log_paths", []))
    source_root = artifacts.get("source_root", "")
    normalized_source = normalize_paths([source_root])[0] if source_root else ""
    skill_dirs = default_skill_dirs(artifacts.get("skill_dirs", []))
    missing = [path for path in log_paths if not Path(path).is_file()]
    if missing:
        msg = f"Missing log file(s): {', '.join(missing)}"
        raise FileNotFoundError(msg)
    if normalized_source and not Path(normalized_source).is_dir():
        msg = f"Missing source root: {normalized_source}"
        raise FileNotFoundError(msg)
    return {
        "artifacts": {
            **artifacts,
            "log_paths": log_paths,
            "source_root": normalized_source,
            "skill_dirs": skill_dirs,
        },
        "messages": [AIMessage(content="Intake complete: inputs were validated.")],
    }


def problem_analyzer(state: DpuFaultState) -> dict[str, Any]:
    keywords = _keywords(state["problem_statement"])
    symptoms = [
        item
        for item in keywords
        if item in {"fail", "failed", "timeout", "drop", "drops", "panic", "reset"}
    ]
    missing_info = []
    artifacts = state["artifacts"]
    if not artifacts.get("log_paths"):
        missing_info.append("failing log window")
    if not artifacts.get("source_root"):
        missing_info.append("source root")
    return {
        "problem_analysis": {
            "keywords": keywords,
            "suspected_modules": keywords[:5],
            "symptoms": symptoms,
            "missing_info": missing_info,
        },
        "messages": [
            AIMessage(content=f"Problem analysis extracted {len(keywords)} keyword(s).")
        ],
    }


def skill_loader(state: DpuFaultState) -> dict[str, Any]:
    skill_dirs = state["artifacts"].get("skill_dirs", [])
    skills = load_skills(skill_dirs)
    return {
        "metadata": {
            **state.get("metadata", {}),
            "skills": [skill.to_dict() for skill in skills],
        },
        "messages": [AIMessage(content=f"Loaded {len(skills)} diagnostic skill(s).")],
    }


def evidence_collector(state: DpuFaultState) -> dict[str, Any]:
    observations = _collect_evidence(state)
    return {
        "observations": observations,
        "messages": [
            AIMessage(
                content=f"Evidence collection produced {len(observations)} item(s)."
            )
        ],
    }


def skill_router(state: DpuFaultState) -> dict[str, Any]:
    skills = _skills_from_metadata(state)
    keywords = state.get("problem_analysis", {}).get("keywords", [])
    matches = match_skills(
        skills, keywords=keywords, observations=state.get("observations", [])
    )
    if not matches:
        generic = [skill for skill in skills if skill.id == "generic-dpu"]
        if generic:
            matches = [generic[0].to_match(score=1, reasons=["fallback:generic"])]
    return {
        "matched_skills": matches[:3],
        "messages": [
            AIMessage(content=f"Matched {len(matches[:3])} diagnostic skill(s).")
        ],
    }


def skill_script_runner(state: DpuFaultState) -> dict[str, Any]:
    matched_skills = state.get("matched_skills", [])
    script_observations = run_skill_scripts(state, matched_skills)
    if not script_observations:
        return {
            "messages": [AIMessage(content="No skill scripts were declared to run.")]
        }
    observations = state.get("observations", []) + script_observations
    return {
        "observations": observations,
        "messages": [
            AIMessage(
                content=f"Skill scripts produced {len(script_observations)} observation(s)."
            )
        ],
    }


def llm_tool_planner(state: DpuFaultState) -> dict[str, Any]:
    config = _model_config(state)
    if not config.enabled:
        return {"messages": [AIMessage(content="LLM tool planning is disabled.")]}
    if state.get("tool_calls"):
        return {
            "messages": [
                AIMessage(
                    content="LLM tool planning skipped; tool calls already exist."
                )
            ]
        }
    try:
        prompt = _tool_planning_prompt(state)
        raw = ChatModelFactory(config).create().complete(prompt)
        plan = parse_tool_plan(raw)
    except Exception as exc:
        observations = state.get("observations", []) + [
            {
                "kind": "llm_error",
                "summary": f"LLM tool planning failed: {exc}",
                "severity": "error",
                "evidence": str(exc),
            }
        ]
        return {
            "observations": observations,
            "llm_analysis": {
                **state.get("llm_analysis", {}),
                "errors": state.get("llm_analysis", {}).get("errors", [])
                + [f"tool_planner: {exc}"],
            },
            "messages": [
                AIMessage(content="LLM tool planning failed; falling back to rules.")
            ],
        }
    return {
        "tool_calls": plan["tool_calls"],
        "llm_analysis": {
            **state.get("llm_analysis", {}),
            "tool_choice_reason": plan.get("reasoning_summary", ""),
        },
        "messages": [
            AIMessage(content=f"LLM proposed {len(plan['tool_calls'])} tool call(s).")
        ],
    }


def tool_executor(state: DpuFaultState) -> dict[str, Any]:
    config = _model_config(state)
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {"messages": [AIMessage(content="No LLM tool calls to execute.")]}
    result = execute_tool_calls(
        state, tool_calls, max_steps=max(0, config.max_tool_steps)
    )
    updates: dict[str, Any] = {
        "tool_calls": result["tool_calls"],
        "observations": result["observations"],
        "messages": [
            AIMessage(
                content=f"Executed {result['executed_count']} low-risk LLM tool call(s)."
            )
        ],
    }
    if result["pending_action"]:
        updates["pending_action"] = result["pending_action"]
    return updates


def tool_reflector(state: DpuFaultState) -> dict[str, Any]:
    calls = state.get("tool_calls", [])
    completed = len([call for call in calls if call.get("status") == "completed"])
    pending = len([call for call in calls if call.get("status") == "pending_approval"])
    skipped = len([call for call in calls if call.get("status") == "skipped_budget"])
    reflection = (
        f"Tool execution summary: completed={completed}, "
        f"pending_approval={pending}, skipped_budget={skipped}."
    )
    return {
        "llm_analysis": {**state.get("llm_analysis", {}), "reflection": reflection},
        "messages": [AIMessage(content=reflection)],
    }


def diagnosis_planner(state: DpuFaultState) -> dict[str, Any]:
    plan = _build_diagnosis_plan(state)
    status = "pending"
    if state.get("pending_action"):
        status = "pending_tool_approval"
    elif plan.get("evidence_gaps"):
        status = "needs_more_evidence"
    return {
        "diagnosis_plan": plan,
        "approval": {"status": status, "approved_ids": [], "rejected_ids": []},
        "messages": [AIMessage(content="Diagnosis plan generated.")],
    }


def hypothesis_builder(state: DpuFaultState) -> dict[str, Any]:
    observations = state.get("observations", [])
    source_observations = [obs for obs in observations if obs.get("kind") == "source"]
    log_observations = [
        obs for obs in observations if obs.get("kind", "").startswith("log")
    ]
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
        plan = state.get("diagnosis_plan", {})
        hypotheses.append(
            {
                "id": "H1",
                "title": "Evidence is insufficient; follow the diagnosis plan first",
                "confidence": 0.2,
                "evidence": plan.get("evidence_gaps", [])
                or ["No concrete log or source evidence is available yet."],
                "source_refs": [],
                "validation_steps": plan.get("next_actions", []),
                "status": "candidate",
            }
        )

    return {
        "hypotheses": hypotheses,
        "messages": [
            AIMessage(
                content=f"Built {len(hypotheses)} candidate hypothesis/hypotheses."
            )
        ],
    }


def human_gate(state: DpuFaultState) -> dict[str, Any]:
    approval = state.get("approval", {})
    if approval.get("status") in {"approved", "rejected"}:
        return {}
    pending_action = state.get("pending_action", {})
    if pending_action:
        decision = interrupt(
            {
                "stage": "human_gate",
                "message": "Review and approve or reject the pending tool action.",
                "pending_action": pending_action,
                "diagnosis_plan": state.get("diagnosis_plan", {}),
                "matched_skills": state.get("matched_skills", []),
            }
        )
        note = _decision_note(decision)
        if _decision_rejected(decision):
            updates = reject_pending_action(state, pending_action, note)
            return {
                **updates,
                "approval": {
                    "status": "approved",
                    "approved_ids": [],
                    "rejected_ids": [],
                    "note": note or "Pending tool action rejected by reviewer.",
                },
                "messages": [AIMessage(content="Pending tool action was rejected.")],
            }
        updates = execute_pending_action(state, pending_action)
        return {
            **updates,
            "approval": {
                "status": "approved",
                "approved_ids": [],
                "rejected_ids": [],
                "note": note or "Pending tool action approved and executed.",
            },
            "messages": [AIMessage(content="Pending tool action was executed.")],
        }
    decision = interrupt(
        {
            "stage": "human_gate",
            "message": "Review the diagnosis plan and provide missing evidence or approve continuing.",
            "diagnosis_plan": state.get("diagnosis_plan", {}),
            "matched_skills": state.get("matched_skills", []),
            "missing_info": state.get("problem_analysis", {}).get("missing_info", []),
        }
    )
    supplement = _extract_supplement(decision)
    updates: dict[str, Any] = {}
    if supplement:
        artifacts = _merge_artifacts(state["artifacts"], supplement)
        updated_state = {**state, "artifacts": artifacts}
        observations = _collect_evidence(updated_state)
        updated_state = {**updated_state, "observations": observations}
        skills = _skills_from_metadata(updated_state)
        matches = match_skills(
            skills,
            keywords=updated_state.get("problem_analysis", {}).get("keywords", []),
            observations=observations,
        )
        if not matches:
            generic = [skill for skill in skills if skill.id == "generic-dpu"]
            if generic:
                matches = [generic[0].to_match(score=1, reasons=["fallback:generic"])]
        script_observations = run_skill_scripts(
            {**updated_state, "matched_skills": matches[:3]}, matches[:3]
        )
        observations = observations + script_observations
        updated_state = {
            **updated_state,
            "observations": observations,
            "matched_skills": matches[:3],
        }
        updates["artifacts"] = artifacts
        updates["observations"] = observations
        updates["matched_skills"] = matches[:3]
        updates["diagnosis_plan"] = _build_diagnosis_plan(updated_state)
    normalized = normalize_approval(decision, state.get("hypotheses", []))
    if normalized["status"] == "pending":
        normalized["status"] = "approved"
        normalized["note"] = (
            normalized.get("note", "") or "Supplemental evidence accepted."
        )
    return {
        **updates,
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
        elif approval.get("status") == "rejected" and not rejected:
            status = "rejected"
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


def _collect_evidence(state: DpuFaultState) -> list[dict[str, Any]]:
    artifacts = state["artifacts"]
    observations: list[dict[str, Any]] = []
    log_paths = artifacts.get("log_paths", [])
    if log_paths:
        observations.extend(triage_logs(log_paths))
    source_root = artifacts.get("source_root", "")
    if source_root:
        terms = derive_search_terms(state["problem_statement"], observations)
        observations.extend(search_source(source_root, terms))
    return observations


def _build_diagnosis_plan(state: DpuFaultState) -> DiagnosisPlan:
    matches = state.get("matched_skills", [])
    observations = state.get("observations", [])
    if matches:
        primary = matches[0]
        steps = primary.get("triage_steps", [])
        required = primary.get("required_evidence", [])
        next_actions = primary.get("validation_steps", [])
        summary = f"Use `{primary.get('name')}` to localize the issue."
    else:
        steps = [
            "Identify the affected DPU feature or module.",
            "Collect the failing log window and reproduction context.",
            "Search source code for concrete error codes or module markers.",
        ]
        required = ["affected module", "failing log window", "source root"]
        next_actions = [
            "Ask the user for module name, log path, source path, or a matching skill.",
            "Keep conclusions low-confidence until evidence is available.",
        ]
        summary = "No module skill matched; use the generic DPU triage path."
    gaps = _evidence_gaps(required, observations, state["artifacts"])
    return {
        "summary": summary,
        "steps": steps,
        "required_evidence": required,
        "next_actions": next_actions,
        "evidence_gaps": gaps,
    }


def _evidence_gaps(
    required_evidence: list[str],
    observations: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> list[str]:
    gaps: list[str] = []
    has_logs = bool(artifacts.get("log_paths"))
    has_source = bool(artifacts.get("source_root"))
    for item in required_evidence:
        lowered = item.lower()
        if "log" in lowered and not has_logs:
            gaps.append(item)
        elif ("source" in lowered or "code" in lowered) and not has_source:
            gaps.append(item)
        elif ("module" in lowered or "feature" in lowered) and not observations:
            gaps.append(item)
    return gaps


def _keywords(text: str) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 2 or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _skills_from_metadata(state: DpuFaultState):
    from dpu_fault_agent.skills import Skill

    skills = []
    for item in state.get("metadata", {}).get("skills", []):
        skills.append(Skill.from_dict(item))
    return skills


def _model_config(state: DpuFaultState) -> ModelConfig:
    return ModelConfig.from_dict(state.get("metadata", {}).get("llm_config", {}))


def _tool_planning_prompt(state: DpuFaultState) -> str:
    payload = {
        "task": "Choose low-risk diagnostic tool calls for DPU fault localization. Return JSON only.",
        "output_schema": {
            "reasoning_summary": "short reason for tool choices",
            "tool_calls": [
                {
                    "id": "T1",
                    "tool": "log_triage|source_search|read_file|skill_script|shell",
                    "args": {},
                    "reason": "short expected benefit",
                }
            ],
        },
        "limits": {
            "max_tool_calls": _model_config(state).max_tool_steps,
            "avoid": "Do not propose source modifications unless necessary; risky shell commands require approval.",
        },
        "problem": state.get("problem_statement", ""),
        "artifacts": state.get("artifacts", {}),
        "problem_analysis": state.get("problem_analysis", {}),
        "matched_skills": state.get("matched_skills", []),
        "observations": state.get("observations", [])[:20],
        "available_tools": registered_tools(state.get("matched_skills", [])),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _decision_rejected(decision: Any) -> bool:
    if isinstance(decision, str):
        return decision.strip().lower() in {"reject", "rejected", "none"}
    if isinstance(decision, dict):
        return str(decision.get("status", "")).lower() == "rejected"
    return False


def _decision_note(decision: Any) -> str:
    if isinstance(decision, dict):
        return str(decision.get("note", ""))
    if isinstance(decision, str):
        return decision
    return ""


def _extract_supplement(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    return {
        key: decision[key]
        for key in ("log_paths", "source_root", "note")
        if decision.get(key)
    }


def _merge_artifacts(
    artifacts: dict[str, Any], supplement: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(artifacts)
    if supplement.get("log_paths"):
        extra_logs = normalize_paths(list(supplement["log_paths"]))
        merged["log_paths"] = list(
            dict.fromkeys(merged.get("log_paths", []) + extra_logs)
        )
    if supplement.get("source_root"):
        merged["source_root"] = normalize_paths([str(supplement["source_root"])])[0]
    if supplement.get("note"):
        merged["notes"] = merged.get("notes", []) + [str(supplement["note"])]
    return merged
