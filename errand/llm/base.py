"""The model client interface.

Every LLM call names one Pydantic schema and gets a validated instance of it back (Ollama structured
output: the JSON schema is passed as ``format``). There is no free-form tool calling: the model
proposes slot patches and proposals, and application code decides what actually happens.
"""

from collections.abc import Sequence
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, usd_per_1k_input: float, usd_per_1k_output: float) -> float:
        return (
            self.input_tokens * usd_per_1k_input + self.output_tokens * usd_per_1k_output
        ) / 1000


class ModelRequest(BaseModel):
    """What one call asked the model for. Scripted models keep these so tests can assert exactly
    what reached the model."""

    model_config = ConfigDict(frozen=True)

    purpose: str
    schema_name: str
    messages: tuple[ChatMessage, ...]


class ModelReply(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=True)

    output: T
    raw: str
    model: str
    usage: TokenUsage
    latency_ms: float


class ModelError(Exception):
    """The call produced no usable output. The caller ends the run with a clear message."""


class ModelUnavailable(ModelError):
    """The model server couldn't be reached, or didn't answer in time."""


class ModelOutputError(ModelError):
    """The reply wasn't valid JSON for the requested schema."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class ScriptExhausted(ModelError):
    """A scripted model was called more times than its script covers."""


@runtime_checkable
class ModelClient(Protocol):
    name: str

    def complete(
        self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str
    ) -> ModelReply[T]: ...


def estimate_tokens(text: str) -> int:
    """About four characters per token, for backends that don't report usage."""
    return -(-len(text) // 4)
