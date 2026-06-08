from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "mock"
    model: str = "rule-based"
    temperature: float = 0.0
    api_key_env: str | None = None


class DiagnosticModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockDiagnosticModel:
    def complete(self, prompt: str) -> str:
        return prompt


class ChatModelFactory:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    def create(self) -> DiagnosticModel:
        if self.config.provider == "mock":
            return MockDiagnosticModel()
        msg = (
            "Only the mock provider is implemented in the MVP. "
            "Add a provider-specific DiagnosticModel implementation before "
            f"using provider `{self.config.provider}`."
        )
        raise ValueError(msg)
