"""DPU embedded software fault localization agent."""

from typing import Any

__all__ = ["DpuFaultState", "build_graph"]


def __getattr__(name: str) -> Any:
    if name == "build_graph":
        from dpu_fault_agent.graph import build_graph

        return build_graph
    if name == "DpuFaultState":
        from dpu_fault_agent.state import DpuFaultState

        return DpuFaultState
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
