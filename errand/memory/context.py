"""Builds the model's context for one call under a fixed token cap, with a manifest of what went in.

Order of priority when the cap bites: the system prompt and the latest message always go in, then
the data the call needs, then relevant facts, then recent turns (newest first), then the rolling
summary of older messages. Facts come only from `MemoryService.relevant`: active, matching the
service, never unconfirmed or forgotten. The summary is built from stored messages, which are
scrubbed when a fact is forgotten.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from errand.llm import ChatMessage
from errand.llm.base import estimate_tokens
from errand.memory.service import KEY_LABELS, display_value
from errand.store import MemoryFact, Store

SUMMARY_LINE_CHARS = 110


class ContextManifest(BaseModel):
    purpose: str
    cap: int
    total_tokens: int
    sections: dict[str, int]
    fact_ids: list[str]
    message_ids: list[int]
    summary_message_ids: list[int]
    dropped: list[str]


class ContextPack(BaseModel):
    messages: list[ChatMessage]
    manifest: ContextManifest


class ContextBuilder:
    def __init__(
        self,
        store: Store,
        *,
        token_cap: int,
        recent_turns: int,
        summary_token_budget: int,
    ) -> None:
        self._store = store
        self._cap = token_cap
        self._recent_turns = recent_turns
        self._summary_budget = summary_token_budget

    def build(
        self,
        *,
        purpose: str,
        system: str,
        latest: str,
        user_id: str,
        thread_id: str,
        latest_message_id: int | None = None,
        facts: Sequence[MemoryFact] = (),
        data: str | None = None,
        recent: bool = True,
        summary: bool = False,
    ) -> ContextPack:
        dropped: list[str] = []
        sections = {"system": estimate_tokens(system), "latest": estimate_tokens(latest)}
        if sections["system"] + 16 > self._cap:
            # Settings refuse such a cap; never exceed it silently (docs/bugs.md #10).
            raise ValueError(
                f"the {purpose} system prompt ({sections['system']} tokens) doesn't fit the"
                f" {self._cap}-token context cap"
            )
        if sections["system"] + sections["latest"] > self._cap:
            # Only a pathological message gets here; keep its end, which carries the request.
            room = max(self._cap - sections["system"], 16) * 4
            latest = "…" + latest[-room:]
            sections["latest"] = estimate_tokens(latest)
            dropped.append("latest:truncated")
        left = self._cap - sections["system"] - sections["latest"]

        data_block = ""
        if data:
            data_block = data if estimate_tokens(data) <= left else data[: left * 4]
            if data_block != data:
                dropped.append("data:truncated")
            sections["data"] = estimate_tokens(data_block)
            left -= sections["data"]

        fact_lines, fact_ids = [], []
        for fact in facts:
            scope = f" (for {fact.service_tag})" if fact.service_tag else ""
            line = f"- {KEY_LABELS.get(fact.key, fact.key)}{scope}: {display_value(fact.value)}"
            cost = estimate_tokens(line) + 1
            if cost > left:
                dropped.append(f"fact:{fact.fact_id}")
                continue
            fact_lines.append(line)
            fact_ids.append(fact.fact_id)
            left -= cost
        sections["facts"] = sum(estimate_tokens(line) + 1 for line in fact_lines)

        turns: list[ChatMessage] = []
        turn_ids: list[int] = []
        if recent and self._recent_turns:
            for message in reversed(
                self._store.messages.recent(
                    user_id, thread_id, self._recent_turns, before_id=latest_message_id
                )
            ):
                cost = estimate_tokens(message.content) + 4
                if cost > left:
                    dropped.append(f"message:{message.message_id}")
                    break
                turns.insert(0, ChatMessage(role=message.role, content=message.content))
                turn_ids.insert(0, message.message_id)
                left -= cost
        sections["recent"] = sum(estimate_tokens(m.content) + 4 for m in turns)

        summary_lines, summary_ids = [], []
        if summary and self._summary_budget:
            room = min(self._summary_budget, left)
            exclude = [*turn_ids, *([latest_message_id] if latest_message_id else [])]
            for message in self._store.messages.history(user_id, exclude_ids=exclude, limit=200):
                if message.role != "user":
                    continue
                text = " ".join(message.content.split())[:SUMMARY_LINE_CHARS]
                line = f"- {message.created_at:%d %b}: {text}"
                cost = estimate_tokens(line) + 1
                if cost > room:
                    dropped.append(f"summary:{message.message_id} and older")
                    break
                summary_lines.insert(0, line)
                summary_ids.insert(0, message.message_id)
                room -= cost
                left -= cost
        sections["summary"] = sum(estimate_tokens(line) + 1 for line in summary_lines)

        system_text = system
        if fact_lines:
            system_text += "\n\nStored preferences (current and confirmed):\n" + "\n".join(
                fact_lines
            )
        if summary_lines:
            system_text += "\n\nEarlier requests from this user (condensed):\n" + "\n".join(
                summary_lines
            )
        messages = [ChatMessage(role="system", content=system_text), *turns]
        if data_block:
            messages.append(ChatMessage(role="user", content=data_block))
        messages.append(ChatMessage(role="user", content=latest))

        manifest = ContextManifest(
            purpose=purpose,
            cap=self._cap,
            total_tokens=sum(sections.values()),
            sections=sections,
            fact_ids=fact_ids,
            message_ids=turn_ids,
            summary_message_ids=summary_ids,
            dropped=dropped,
        )
        return ContextPack(messages=messages, manifest=manifest)
