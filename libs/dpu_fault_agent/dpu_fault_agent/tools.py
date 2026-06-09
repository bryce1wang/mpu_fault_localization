from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dpu_fault_agent.state import Observation
else:
    Observation = dict

ERROR_RE = re.compile(
    r"(?i)(error|err|fail(?:ed|ure)?|fault|panic|assert|timeout|reset|exception|abort)"
)
ERROR_CODE_RE = re.compile(r"\b(?:0x[0-9a-fA-F]{2,}|[A-Z][A-Z0-9_]{3,})\b")
MODULE_RE = re.compile(r"\[([A-Za-z0-9_.:/-]+)\]")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".py",
    ".rs",
    ".go",
    ".S",
    ".s",
    ".mk",
    ".cmake",
    ".cfg",
    ".conf",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
}


def normalize_paths(paths: list[str], *, base: Path | None = None) -> list[str]:
    root = base or Path.cwd()
    return [
        str((root / path).resolve())
        if not Path(path).is_absolute()
        else str(Path(path))
        for path in paths
    ]


def read_text_sample(path: str, *, max_bytes: int = 200_000) -> str:
    data = Path(path).read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def triage_logs(log_paths: list[str], *, max_findings: int = 40) -> list[Observation]:
    observations: list[Observation] = []
    module_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()

    for raw_path in log_paths:
        path = Path(raw_path)
        text = read_text_sample(str(path))
        for line_no, line in enumerate(text.splitlines(), start=1):
            modules = MODULE_RE.findall(line)
            module_counts.update(modules)
            codes = ERROR_CODE_RE.findall(line)
            code_counts.update(codes)
            if ERROR_RE.search(line) or codes:
                observations.append(
                    {
                        "kind": "log",
                        "summary": line.strip()[:300],
                        "path": str(path),
                        "line": line_no,
                        "severity": "error" if ERROR_RE.search(line) else "signal",
                        "evidence": line.strip(),
                    }
                )
            if len(observations) >= max_findings:
                break

    for module, count in module_counts.most_common(5):
        observations.append(
            {
                "kind": "log_module",
                "summary": f"Module marker `{module}` appears {count} time(s) in logs.",
                "symbol": module,
                "severity": "info",
            }
        )
    for code, count in code_counts.most_common(10):
        observations.append(
            {
                "kind": "log_code",
                "summary": f"Signal `{code}` appears {count} time(s) in logs.",
                "symbol": code,
                "severity": "signal",
            }
        )
    return observations


def derive_search_terms(
    problem_statement: str, observations: list[Observation]
) -> list[str]:
    terms: Counter[str] = Counter()
    for token in TOKEN_RE.findall(problem_statement):
        if len(token) > 3:
            terms[token] += 1
    for obs in observations:
        symbol = obs.get("symbol")
        if symbol and len(symbol) > 2:
            terms[symbol] += 5
        for token in TOKEN_RE.findall(obs.get("summary", "")):
            if len(token) > 3 and token.lower() not in {
                "error",
                "failed",
                "failure",
                "timeout",
            }:
                terms[token] += 1
    return [term for term, _ in terms.most_common(20)]


def iter_source_files(source_root: str) -> list[Path]:
    root = Path(source_root)
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SOURCE_EXTENSIONS or path.name in {
            "Makefile",
            "CMakeLists.txt",
        }:
            files.append(path)
    return files


def search_source(
    source_root: str,
    terms: list[str],
    *,
    max_hits: int = 60,
    context_lines: int = 2,
) -> list[Observation]:
    hits: list[Observation] = []
    if not terms:
        return hits
    lowered_terms = [(term, term.lower()) for term in terms]
    for path in iter_source_files(source_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            matched = [term for term, lowered in lowered_terms if lowered in line_lower]
            if not matched:
                continue
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            snippet = "\n".join(lines[start:end])
            hits.append(
                {
                    "kind": "source",
                    "summary": f"Matched {', '.join(matched[:3])} in {path.name}:{idx + 1}",
                    "path": str(path),
                    "line": idx + 1,
                    "symbol": matched[0],
                    "severity": "signal",
                    "evidence": snippet,
                }
            )
            if len(hits) >= max_hits:
                return hits
    return hits


def format_ref(obs: Observation) -> str:
    path = obs.get("path", "")
    line = obs.get("line")
    if path and line:
        return f"{path}:{line}"
    return path or obs.get("summary", "")
