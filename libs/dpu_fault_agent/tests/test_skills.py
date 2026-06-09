from __future__ import annotations

from pathlib import Path

import pytest

from dpu_fault_agent.skills import (
    SkillParseError,
    load_skills,
    match_skills,
    parse_skill_file,
)


def _skill(path: Path, *, skill_id: str = "vf_init") -> Path:
    path.write_text(
        f"""---
id: {skill_id}
name: VF init triage
modules: [vf, dpu]
keywords: [vf, init, DPU_ERR_TIMEOUT]
symptoms: [timeout, failed]
required_evidence: [failing log window, source root]
triage_steps: [Check VF init stage, Inspect timeout path]
common_causes: [queue setup timeout]
validation_steps: [Compare known-good VF init log]
---

VF initialization troubleshooting notes.
""",
        encoding="utf-8",
    )
    return path


def test_parse_markdown_yaml_skill(tmp_path: Path) -> None:
    skill = parse_skill_file(_skill(tmp_path / "vf.md"))

    assert skill.id == "vf_init"
    assert "vf" in skill.modules
    assert "VF initialization" in skill.body


def test_parse_skill_reports_missing_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("---\nid: bad\n---\nbody\n", encoding="utf-8")

    with pytest.raises(SkillParseError, match="missing required field"):
        parse_skill_file(bad)


def test_load_and_match_skills(tmp_path: Path) -> None:
    _skill(tmp_path / "vf.md")

    skills = load_skills([str(tmp_path)])
    matches = match_skills(
        skills,
        keywords=["vf", "timeout"],
        observations=[{"symbol": "DPU_ERR_TIMEOUT"}],
    )

    assert len(skills) == 1
    assert matches[0]["id"] == "vf_init"
    assert matches[0]["score"] > 0
