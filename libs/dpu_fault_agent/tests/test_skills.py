from __future__ import annotations

from pathlib import Path

import pytest

from dpu_fault_agent.skills import (
    SkillParseError,
    load_skills,
    match_skills,
    parse_skill_file,
)


def _skill(path: Path, *, name: str = "vf-init-timeout", scripts: str = "") -> Path:
    path.write_text(
        f"""---
name: {name}
description: Diagnose VF init timeout and DPU_ERR_TIMEOUT during VF bring-up.
feature: virtualization
module: vf
problem_type: init-timeout
modules: [vf, dpu]
keywords: [vf, init, DPU_ERR_TIMEOUT]
symptoms: [timeout, failed]
required_evidence: [failing log window, source root]
triage_steps: [Check VF init stage, Inspect timeout path]
common_causes: [queue setup timeout]
validation_steps: [Compare known-good VF init log]
{scripts}
---

VF initialization troubleshooting notes.
""",
        encoding="utf-8",
    )
    return path


def test_parse_markdown_yaml_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "vf-init-timeout"
    skill_dir.mkdir()
    skill = parse_skill_file(_skill(skill_dir / "SKILL.md"))

    assert skill.id == "vf-init-timeout"
    assert skill.feature == "virtualization"
    assert skill.module == "vf"
    assert skill.problem_type == "init-timeout"
    assert "vf" in skill.modules
    assert "VF initialization" in skill.body


def test_parse_skill_script_metadata(tmp_path: Path) -> None:
    skill_dir = tmp_path / "vf-init-timeout"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "collect.py").write_text("print('collect evidence')\n")
    skill = parse_skill_file(
        _skill(
            skill_dir / "SKILL.md",
            scripts="""scripts:
  - name: collect
    path: scripts/collect.py
    args: ["--fast"]
    timeout_seconds: 3""",
        )
    )

    assert skill.scripts[0].name == "collect"
    assert skill.scripts[0].args == ["--fast"]
    assert skill.scripts[0].timeout_seconds == 3
    assert skill.scripts[0].path.endswith("collect.py")


def test_skill_script_path_cannot_escape_skill_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / "vf-init-timeout"
    skill_dir.mkdir()
    (tmp_path / "escape.py").write_text("print('bad')\n")

    with pytest.raises(SkillParseError, match="must stay inside"):
        parse_skill_file(
            _skill(
                skill_dir / "SKILL.md",
                scripts="""scripts:
  - ../escape.py""",
            )
        )


def test_parse_skill_reports_missing_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("---\nid: bad\n---\nbody\n", encoding="utf-8")

    with pytest.raises(SkillParseError, match="missing required field"):
        parse_skill_file(bad)


def test_load_skills_recursively_and_match_skills(tmp_path: Path) -> None:
    skill_dir = tmp_path / "virtualization" / "vf" / "vf-init-timeout"
    skill_dir.mkdir(parents=True)
    _skill(skill_dir / "SKILL.md")

    skills = load_skills([str(tmp_path)])
    matches = match_skills(
        skills,
        keywords=["vf", "timeout"],
        observations=[{"symbol": "DPU_ERR_TIMEOUT"}],
    )

    assert len(skills) == 1
    assert matches[0]["id"] == "vf-init-timeout"
    assert matches[0]["feature"] == "virtualization"
    assert matches[0]["score"] > 0


def test_load_skills_supports_legacy_flat_md(tmp_path: Path) -> None:
    legacy = _skill(tmp_path / "legacy.md", name="legacy")

    skills = load_skills([str(tmp_path)])

    assert skills[0].path == str(legacy)


def test_directory_name_must_match_skill_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "wrong-name"
    skill_dir.mkdir()
    _skill(skill_dir / "SKILL.md", name="vf-init-timeout")

    with pytest.raises(SkillParseError, match="directory name"):
        parse_skill_file(skill_dir / "SKILL.md")


def test_duplicate_skill_names_fail_fast(tmp_path: Path) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    one = root_a / "one"
    other = root_b / "one"
    one.mkdir(parents=True)
    other.mkdir(parents=True)
    _skill(one / "SKILL.md", name="one")
    _skill(other / "SKILL.md", name="one")

    with pytest.raises(SkillParseError, match="Duplicate skill name"):
        load_skills([str(root_a), str(root_b)])


def test_feature_module_problem_type_influence_matching(tmp_path: Path) -> None:
    rx = tmp_path / "networking" / "rx" / "rx-drop"
    rx.mkdir(parents=True)
    (rx / "SKILL.md").write_text(
        """---
name: rx-drop
description: Diagnose RX packet drops.
feature: networking
module: rx
problem_type: drop
keywords: [rx, drop]
symptoms: [drop]
required_evidence: [failing log window]
triage_steps: [Check RX counters]
common_causes: [descriptor starvation]
validation_steps: [Compare RX counters]
---

RX drop notes.
""",
        encoding="utf-8",
    )

    matches = match_skills(
        load_skills([str(tmp_path)]), keywords=["rx", "drop"], observations=[]
    )

    assert matches[0]["id"] == "rx-drop"
