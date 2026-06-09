from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SKILL_FIELDS = {
    "name",
    "description",
    "feature",
    "module",
    "problem_type",
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
class SkillScript:
    name: str
    path: str
    args: list[str]
    timeout_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "args": self.args,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> SkillScript:
        return cls(
            name=str(item["name"]),
            path=str(item["path"]),
            args=_string_list(item.get("args", [])),
            timeout_seconds=int(item.get("timeout_seconds", 30)),
        )


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    feature: str
    module: str
    problem_type: str
    modules: list[str]
    keywords: list[str]
    symptoms: list[str]
    required_evidence: list[str]
    triage_steps: list[str]
    common_causes: list[str]
    validation_steps: list[str]
    body: str
    path: str
    scripts: list[SkillScript]

    @property
    def id(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "feature": self.feature,
            "module": self.module,
            "problem_type": self.problem_type,
            "modules": self.modules,
            "keywords": self.keywords,
            "symptoms": self.symptoms,
            "required_evidence": self.required_evidence,
            "triage_steps": self.triage_steps,
            "common_causes": self.common_causes,
            "validation_steps": self.validation_steps,
            "body": self.body,
            "path": self.path,
            "scripts": [script.to_dict() for script in self.scripts],
        }

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> Skill:
        scripts = [
            script if isinstance(script, SkillScript) else SkillScript.from_dict(script)
            for script in item.get("scripts", [])
        ]
        return cls(
            name=str(item["name"]),
            description=str(item["description"]),
            feature=str(item["feature"]),
            module=str(item["module"]),
            problem_type=str(item["problem_type"]),
            modules=_string_list(item.get("modules", [])),
            keywords=_string_list(item.get("keywords", [])),
            symptoms=_string_list(item.get("symptoms", [])),
            required_evidence=_string_list(item.get("required_evidence", [])),
            triage_steps=_string_list(item.get("triage_steps", [])),
            common_causes=_string_list(item.get("common_causes", [])),
            validation_steps=_string_list(item.get("validation_steps", [])),
            body=str(item.get("body", "")),
            path=str(item["path"]),
            scripts=scripts,
        )

    def to_match(self, *, score: int, reasons: list[str]) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "feature": self.feature,
            "module": self.module,
            "problem_type": self.problem_type,
            "score": score,
            "reasons": reasons,
            "modules": self.modules,
            "required_evidence": self.required_evidence,
            "triage_steps": self.triage_steps,
            "common_causes": self.common_causes,
            "validation_steps": self.validation_steps,
            "body": self.body,
            "scripts": [script.to_dict() for script in self.scripts],
        }


def load_skills(skill_dirs: list[str]) -> list[Skill]:
    skills: list[Skill] = []
    seen: set[str] = set()
    for directory in skill_dirs:
        path = Path(directory)
        if not path.exists():
            continue
        for skill_file in _skill_files(path):
            skill = parse_skill_file(skill_file)
            if skill.name in seen:
                msg = f"Duplicate skill name `{skill.name}` found at {skill_file}"
                raise SkillParseError(msg)
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
    _validate_non_empty(skill_path, data)
    name = str(data["name"])
    if skill_path.name == "SKILL.md" and skill_path.parent.name != name:
        msg = (
            f"Skill directory name `{skill_path.parent.name}` must match "
            f"front matter name `{name}`"
        )
        raise SkillParseError(msg)
    module = str(data["module"])
    modules = _string_list(data.get("modules", []))
    if module not in modules:
        modules.insert(0, module)
    return Skill(
        name=name,
        description=str(data["description"]),
        feature=str(data["feature"]),
        module=module,
        problem_type=str(data["problem_type"]),
        modules=modules,
        keywords=_string_list(data["keywords"]),
        symptoms=_string_list(data["symptoms"]),
        required_evidence=_string_list(data["required_evidence"]),
        triage_steps=_string_list(data["triage_steps"]),
        common_causes=_string_list(data["common_causes"]),
        validation_steps=_string_list(data["validation_steps"]),
        body=body.strip(),
        path=str(skill_path),
        scripts=_parse_scripts(data.get("scripts", []), skill_path.parent),
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


def _parse_scripts(value: Any, skill_dir: Path) -> list[SkillScript]:
    scripts: list[SkillScript] = []
    if value in (None, "", []):
        return scripts
    raw_items = value if isinstance(value, list) else [value]
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            name = Path(item).stem
            script_path = item
            args: list[str] = []
            timeout_seconds = 30
        elif isinstance(item, dict):
            script_path = item.get("path")
            if not script_path:
                msg = f"Skill script entry #{index} in {skill_dir} missing `path`"
                raise SkillParseError(msg)
            name = str(item.get("name") or Path(str(script_path)).stem)
            args = _string_list(item.get("args", []))
            timeout_seconds = int(item.get("timeout_seconds", item.get("timeout", 30)))
        else:
            msg = (
                f"Skill script entry #{index} in {skill_dir} must be string or mapping"
            )
            raise SkillParseError(msg)
        resolved = _resolve_skill_script(skill_dir, str(script_path))
        scripts.append(
            SkillScript(
                name=name,
                path=str(resolved),
                args=args,
                timeout_seconds=timeout_seconds,
            )
        )
    return scripts


def _resolve_skill_script(skill_dir: Path, script_path: str) -> Path:
    root = skill_dir.resolve()
    resolved = (root / script_path).resolve()
    if not resolved.is_relative_to(root):
        msg = f"Skill script `{script_path}` must stay inside `{skill_dir}`"
        raise SkillParseError(msg)
    if not resolved.is_file():
        msg = f"Skill script `{script_path}` does not exist under `{skill_dir}`"
        raise SkillParseError(msg)
    if resolved.suffix.lower() != ".py":
        msg = f"Skill script `{script_path}` must be a Python `.py` file"
        raise SkillParseError(msg)
    return resolved


def default_skill_dirs(extra_dirs: list[str] | None = None) -> list[str]:
    dirs = [str(DEFAULT_SKILLS_DIR)]
    if extra_dirs:
        dirs.extend(extra_dirs)
    return list(dict.fromkeys(dirs))


def _skill_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.md"))
    files.extend(sorted(path.glob("**/SKILL.md")))
    return list(dict.fromkeys(files))


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
        if skill.feature.lower() in haystack_terms:
            score += 1
            reasons.append(f"feature:{skill.feature}")
        for module in skill.modules:
            if module.lower() in haystack_terms:
                score += 5
                reasons.append(f"module:{module}")
        for keyword in skill.keywords:
            if keyword.lower() in haystack_terms:
                score += 8 if _looks_like_error_code(keyword) else 3
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
    terms = {part.lower() for part in normalized.split() if len(part) > 2}
    terms.add(text.lower())
    return terms


def _validate_non_empty(path: Path, data: dict[str, Any]) -> None:
    for field in REQUIRED_SKILL_FIELDS:
        value = data.get(field)
        if value in (None, "", []):
            msg = f"Skill {path} field `{field}` must be non-empty"
            raise SkillParseError(msg)


def _looks_like_error_code(value: str) -> bool:
    return value.startswith("0x") or value.upper() == value and "_" in value
