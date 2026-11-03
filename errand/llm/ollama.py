"""The live model: qwen2.5:7b on Ollama via langchain_ollama.ChatOllama, structured output only."""

import json
from collections.abc import Sequence
from time import perf_counter
from typing import Self

import httpx
import ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from errand.config import Settings
from errand.llm.base import (
    ChatMessage,
    ModelOutputError,
    ModelReply,
    ModelUnavailable,
    T,
    TokenUsage,
)

_LANGCHAIN_MESSAGE = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


class OllamaModel:
    """Passes the schema's JSON schema as Ollama's ``format``, so decoding is constrained to it,
    then validates the reply with Pydantic anyway.

    Construction makes no network call: a server that is down surfaces as `ModelUnavailable` on the
    first call, not when the app boots.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 512,
        timeout_s: float = 20.0,
    ) -> None:
        self.name = model
        self._chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            client_kwargs={"timeout": timeout_s},
        )

    def close(self) -> None:
        """Close the HTTP connection pool to Ollama (the async client is never opened)."""
        self._chat._client.close()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            num_predict=settings.llm_max_output_tokens,
            timeout_s=settings.llm_request_timeout_s,
        )

    def complete(
        self, messages: Sequence[ChatMessage], schema: type[T], *, purpose: str
    ) -> ModelReply[T]:
        lc_messages: list[BaseMessage] = [
            _LANGCHAIN_MESSAGE[message.role](content=message.content) for message in messages
        ]
        started = perf_counter()
        try:
            reply = self._chat.invoke(lc_messages, format=schema.model_json_schema())
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            raise ModelUnavailable(f"{self.name} on Ollama failed during {purpose}: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000
        raw = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
        try:
            output = schema.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelOutputError(
                f"{self.name} reply for {purpose} doesn't fit {schema.__name__}: {exc}", raw=raw
            ) from exc
        usage = getattr(reply, "usage_metadata", None) or {}
        return ModelReply[schema](
            output=output,
            raw=raw,
            model=self.name,
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            latency_ms=latency_ms,
        )
