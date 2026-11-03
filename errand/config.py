"""Runtime settings, read from ``ERRAND_*`` environment variables or a ``.env`` file.

Relative paths are anchored to the project root, so the app behaves the same from any directory.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ERRAND_", env_file=".env", extra="ignore", frozen=True
    )

    # LLM: qwen2.5:7b on Ollama, structured output only (JSON schema passed as `format`).
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    # Ollama silently drops whatever overflows num_ctx, so it must hold the capped context plus the
    # reply (checked below).
    llm_num_ctx: int = Field(8192, ge=1024)
    llm_max_output_tokens: int = Field(512, ge=16)
    llm_request_timeout_s: float = Field(20.0, gt=0)
    # A local model costs nothing; set these to estimate what a hosted equivalent would cost.
    llm_usd_per_1k_input_tokens: float = Field(0.0, ge=0.0)
    llm_usd_per_1k_output_tokens: float = Field(0.0, ge=0.0)

    # Decision model: Vega. None loads vegaml's default checkpoint (0.8B); "4b" the larger one.
    vega_model: str | None = None
    # "auto" follows the paylane router: "mps" on Apple silicon, otherwise let vegaml choose.
    vega_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    vega_confidence_threshold: float = Field(0.85, ge=0.0, le=1.0)

    # Storage. SQLite only: the agent DB (app tables and LangGraph checkpoints) and a separate DB
    # for the mock world.
    agent_db_path: Path = Path("data/agent.db")
    world_db_path: Path = Path("data/world.db")
    traces_dir: Path = Path("traces")
    reports_dir: Path = Path("reports")
    scenarios_dir: Path = Path("scenarios")
    # Offline evals: evals/scenarios/*.yaml, their recordings, and the traces each run writes.
    evals_dir: Path = Path("evals")
    eval_traces_dir: Path = Path("traces/eval")
    static_dir: Path = Path("static")

    # Hard per-run budgets. A run is the handling of one user message or one proactive event.
    # Model calls are Qwen calls; Vega decisions have their own, larger cap.
    max_model_calls: int = Field(6, ge=1)
    max_decisions: int = Field(40, ge=1)
    max_tool_calls: int = Field(12, ge=1)
    turn_timeout_s: float = Field(30.0, gt=0)
    # Kept back for execute + verify, so a run that reaches a side effect can always settle it.
    reserved_tool_calls: int = Field(5, ge=1)
    reserved_time_s: float = Field(8.0, gt=0)
    # Runs nobody is waiting on (proactive follow-ups, reconciliation) get a longer, still hard,
    # wall-clock limit.
    background_timeout_s: float = Field(60.0, gt=0)
    context_token_cap: int = Field(6000, ge=256)

    # Side effects: get_booking_status is retried with these delays when it errors, and a booking
    # whose timeout is verified as "not booked" is retried with the same key up to this many times.
    status_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0)
    max_book_attempts: int = Field(2, ge=1)
    max_cancel_attempts: int = Field(3, ge=1)

    # Context building.
    recent_turns: int = Field(6, ge=0)
    summary_token_budget: int = Field(600, ge=0)
    max_facts_in_context: int = Field(12, ge=0)
    # Providers whose reviews are fetched and rated per search (each costs a tool call).
    providers_to_review: int = Field(4, ge=1)

    # Mock world and reproducibility.
    scenario: str = "default"
    seed: int = 42

    # Load Vega and wake Ollama before serving, so the first turn isn't spent on it.
    warmup_models: bool = True
    # Seed the demo user's memories, history and past bookings into empty databases.
    seed_demo_data: bool = True
    worker_poll_interval_s: float = Field(2.0, gt=0)
    # Strings longer than this are truncated in traces.
    trace_max_str_chars: int = Field(2000, ge=64)

    @field_validator(
        "agent_db_path",
        "world_db_path",
        "traces_dir",
        "reports_dir",
        "scenarios_dir",
        "evals_dir",
        "eval_traces_dir",
        "static_dir",
    )
    @classmethod
    def _anchor_to_project_root(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @model_validator(mode="after")
    def _context_fits_the_model_window(self) -> Self:
        needed = self.context_token_cap + self.llm_max_output_tokens
        if self.llm_num_ctx < needed:
            raise ValueError(
                f"llm_num_ctx={self.llm_num_ctx} cannot hold context_token_cap="
                f"{self.context_token_cap} plus llm_max_output_tokens={self.llm_max_output_tokens}"
                f" ({needed}); Ollama would truncate the prompt without saying so"
            )
        from errand.llm.prompts import MIN_CONTEXT_ROOM_TOKENS, largest_system_prompt_tokens

        prompt = largest_system_prompt_tokens()
        if self.context_token_cap < prompt + MIN_CONTEXT_ROOM_TOKENS:
            raise ValueError(
                f"context_token_cap={self.context_token_cap} can't hold the largest system prompt"
                f" ({prompt} tokens) plus {MIN_CONTEXT_ROOM_TOKENS} for the user's message; use at"
                f" least {prompt + MIN_CONTEXT_ROOM_TOKENS}"
            )
        if self.reserved_tool_calls >= self.max_tool_calls:
            raise ValueError("reserved_tool_calls must leave some tool calls for other steps")
        if self.reserved_time_s >= self.turn_timeout_s:
            raise ValueError("reserved_time_s must leave some of turn_timeout_s for other steps")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
