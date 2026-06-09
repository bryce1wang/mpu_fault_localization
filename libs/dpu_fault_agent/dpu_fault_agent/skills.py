from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SKILL_FIELDS = {
    "id",
    "name",
    "modules",
    "keywords",
    "symptoms",
    "required_evidence",
    "triage_steps",
    "common_causes",
    "validation_steps",
}
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillParseError(ValueError):
    """Raised when a Markdown skill file cannot be parsed."""


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    modules: list[str]
    keywords: list[str]
    symptoms: list[str]
    required_evidence: list[str]
    triage_steps: list[str]
    common_causes: list[str]
    validation_steps: list[str]
    body: str
    path: str

    def to_match(self, *, score: int, reasons: list[str]) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "score": score,
            "reasons": reasons,
            "modules": self.modules,
            "required_evidence": self.required_evidence,
            "triage_steps": self.triage_steps,
            "common_causes": self.common_causes,
            "validation_steps": self.validation_steps,
            "body": self.body,
        }


def load_skills(skill_dirs: list[str]) -> list[Skill]:
    skills: list[Skill] = []
    seen: set[str] = set()
    for directory in skill_dirs:
        path = Path(directory)
        if not path.exists():
            continue
        for skill_file in sorted(path.glob("*.md")):
            skill = parse_skill_file(skill_file)
            if skill.id in seen:
                continue
            seen.add(skill.id)
            skills.append(skill)
    return skills


def parse_skill_file(path: str | Path) -> Skill:
    skill_path = Path(path)
    text = skill_path.read_text(encoding="utf-8")
    front_matter, body = split_front_matter(text, str(skill_path))
    data = yaml.safe_load(front_matter) or {}
    if not isinstance(data, dict):
        msg = f"Skill front matter must be a mapping: {skill_path}"
        raise SkillParseError(msg)
    missing = sorted(REQUIRED_SKILL_FIELDS - set(data))
    if missing:
        msg = f"Skill {skill_path} missing required field(s): {', '.join(missing)}"
        raise SkillParseError(msg)
    return Skill(
        id=str(data["id"]),
        name=str(data["name"]),
        modules=_string_list(data["modules"]),
        keywords=_string_list(data["keywords"]),
        symptoms=_string_list(data["symptoms"]),
        required_evidence=_string_list(data["required_evidence"]),
        triage_steps=_string_list(data["triage_steps"]),
        common_causes=_string_list(data["common_causes"]),
        validation_steps=_string_list(data["validation_steps"]),
        body=body.strip(),
        path=str(skill_path),
    )


def split_front_matter(text: str, path: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        msg = f"Skill file must start with YAML front matter: {path}"
        raise SkillParseError(msg)
    end = text.find("\n---", 4)
    if end == -1:
        msg = f"Skill file missing closing YAML front matter marker: {path}"
        raise SkillParseError(msg)
    front_matter = text[4:end]
    body = text[end + 4 :]
    return front_matter, body


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def default_skill_dirs(extra_dirs: list[str] | None = None) -> list[str]:
    dirs = [str(DEFAULT_SKILLS_DIR)]
    if extra_dirs:
        dirs.extend(extra_dirs)
    return dirs


def match_skills(
    skills: list[Skill],
    *,
    keywords: list[str],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    haystack_terms = {term.lower() for term in keywords}
    for obs in observations:
        for value in (obs.get("symbol"), obs.get("summary")):
            if value:
                haystack_terms.update(_tokenize(str(value)))

    matches: list[dict[str, Any]] = []
    for skill in skills:
        score = 0
        reasons: list[str] = []
        for module in skill.modules:
            if module.lower() in haystack_terms:
                score += 5
                reasons.append(f"module:{module}")
        for keyword in skill.keywords:
            if keyword.lower() in haystack_terms:
                score += 3
                reasons.append(f"keyword:{keyword}")
        for symptom in skill.symptoms:
            if symptom.lower() in haystack_terms:
                score += 2
                reasons.append(f"symptom:{symptom}")
        if score > 0:
            matches.append(skill.to_match(score=score, reasons=reasons))
    return sorted(matches, key=lambda item: item.get("score", 0), reverse=True)


def _tokenize(text: str) -> set[str]:
    normalized = text.replace("_", " ").replace("-", " ").replace("/", " ")
    return {part.lower() for part in normalized.split() if len(part) > 2}
