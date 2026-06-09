from __future__ import annotations

from pathlib import Path

import yaml


def test_opencode_agent_is_primary_and_selectable() -> None:
    agent = (
        Path(__file__).resolve().parents[3]
        / ".opencode"
        / "agents"
        / "dpu_fault_agent.md"
    )
    text = agent.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, front_matter, body = text.split("---", 2)
    metadata = yaml.safe_load(front_matter)

    assert metadata["mode"] == "primary"
    assert "DPU embedded software fault localization" in metadata["description"]
    assert metadata["permission"]["edit"] == "deny"
    assert "D:\\anaconda\\python.exe -m dpu_fault_agent" in body
