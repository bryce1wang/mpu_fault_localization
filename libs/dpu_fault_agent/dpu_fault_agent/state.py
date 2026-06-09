from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import NotRequired, TypedDict


class ArtifactSpec(TypedDict, total=False):
    log_paths: list[str]
    source_root: str
    config_paths: list[str]
    skill_dirs: list[str]
    notes: list[str]


class Observation(TypedDict, total=False):
    kind: str
    summary: str
    path: str
    line: int
    symbol: str
    severity: str
    evidence: str


class Hypothesis(TypedDict, total=False):
    id: str
    title: str
    confidence: float
    evidence: list[str]
    source_refs: list[str]
    validation_steps: list[str]
    status: str


class Approval(TypedDict, total=False):
    status: str
    approved_ids: list[str]
    rejected_ids: list[str]
    note: str


class ProblemAnalysis(TypedDict, total=False):
    keywords: list[str]
    suspected_modules: list[str]
    symptoms: list[str]
    missing_info: list[str]


class MatchedSkill(TypedDict, total=False):
    id: str
    name: str
    description: str
    feature: str
    module: str
    problem_type: str
    score: int
    reasons: list[str]
    modules: list[str]
    required_evidence: list[str]
    triage_steps: list[str]
    common_causes: list[str]
    validation_steps: list[str]
    body: str
    scripts: list[dict[str, Any]]


class DiagnosisPlan(TypedDict, total=False):
    summary: str
    steps: list[str]
    required_evidence: list[str]
    next_actions: list[str]
    evidence_gaps: list[str]


class DpuFaultState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    case_id: str
    thread_id: str
    problem_statement: str
    artifacts: ArtifactSpec
    problem_analysis: NotRequired[ProblemAnalysis]
    matched_skills: NotRequired[list[MatchedSkill]]
    diagnosis_plan: NotRequired[DiagnosisPlan]
    observations: NotRequired[list[Observation]]
    hypotheses: NotRequired[list[Hypothesis]]
    approval: NotRequired[Approval]
    final_report: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
