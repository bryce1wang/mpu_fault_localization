from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool = False
    provider: str = "mock"
    model: str = "rule-based"
    base_url: str = ""
    temperature: float = 0.0
    api_key_env: str | None = None
    timeout_seconds: int = 30
    max_tokens: int = 2000
    max_tool_steps: int = 5
    mock_response: str = ""
    mock_responses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "max_tool_steps": self.max_tool_steps,
            "mock_response": self.mock_response,
            "mock_responses": self.mock_responses,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelConfig:
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=str(data.get("provider", "mock")),
            model=str(data.get("model", "rule-based")),
            base_url=str(data.get("base_url", "")),
            temperature=float(data.get("temperature", 0.0)),
            api_key_env=data.get("api_key_env"),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            max_tokens=int(data.get("max_tokens", 2000)),
            max_tool_steps=int(data.get("max_tool_steps", 5)),
            mock_response=str(data.get("mock_response", "")),
            mock_responses=[str(item) for item in data.get("mock_responses", [])],
        )


class DiagnosticModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockDiagnosticModel:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def complete(self, prompt: str) -> str:
        if self.config.mock_responses:
            return self.config.mock_responses[0]
        if self.config.mock_response:
            return self.config.mock_response
        return prompt


class OpenAICompatibleDiagnosticModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(self, prompt: str) -> str:
        if not self.config.base_url:
            msg = "OpenAI-compatible provider requires `base_url`."
            raise ValueError(msg)
        api_key = ""
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env, "")
        if self.config.api_key_env and not api_key:
            msg = f"Missing API key environment variable `{self.config.api_key_env}`."
            raise ValueError(msg)
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON for the requested diagnostic action.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self.config.timeout_seconds
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])


class ChatModelFactory:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def create(self) -> DiagnosticModel:
        if self.config.provider == "mock":
            return MockDiagnosticModel(self.config)
        if self.config.provider == "openai-compatible":
            return OpenAICompatibleDiagnosticModel(self.config)
        msg = (
            "Unsupported provider. Use `mock` or `openai-compatible`: "
            f"`{self.config.provider}`."
        )
        raise ValueError(msg)


def load_model_config(
    config_path: str | None = None, overrides: dict[str, Any] | None = None
) -> ModelConfig:
    data: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                msg = f"Config file must contain a mapping: {config_path}"
                raise ValueError(msg)
            llm = raw.get("llm", {})
            if llm:
                if not isinstance(llm, dict):
                    msg = f"`llm` config must be a mapping: {config_path}"
                    raise ValueError(msg)
                data.update(llm)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    return ModelConfig.from_dict(data)
