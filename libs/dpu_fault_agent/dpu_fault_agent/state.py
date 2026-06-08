from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, NotRequired, TypedDict


class ArtifactSpec(TypedDict, total=False):
    log_paths: list[str]
    source_root: str
    config_paths: list[str]


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


class DpuFaultState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    case_id: str
    thread_id: str
    problem_statement: str
    artifacts: ArtifactSpec
    observations: NotRequired[list[Observation]]
    hypotheses: NotRequired[list[Hypothesis]]
    approval: NotRequired[Approval]
    final_report: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
