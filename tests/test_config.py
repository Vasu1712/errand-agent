import pytest
from pydantic import ValidationError

from errand.config import PROJECT_ROOT, Settings


def test_defaults_carry_the_spec_budgets_and_models():
    settings = Settings(_env_file=None)
    assert settings.max_model_calls == 6
    assert settings.max_tool_calls == 12
    assert settings.turn_timeout_s == 30
    assert settings.context_token_cap == 6000
    assert settings.llm_model == "qwen2.5:7b"
    assert 0 < settings.vega_confidence_threshold <= 1


def test_environment_overrides_use_the_errand_prefix(monkeypatch):
    monkeypatch.setenv("ERRAND_MAX_TOOL_CALLS", "9")
    monkeypatch.setenv("ERRAND_VEGA_CONFIDENCE_THRESHOLD", "0.7")
    monkeypatch.setenv("ERRAND_STATUS_BACKOFF_S", "[0.1, 0.2]")
    settings = Settings(_env_file=None)
    assert (settings.max_tool_calls, settings.vega_confidence_threshold) == (9, 0.7)
    assert settings.status_backoff_s == (0.1, 0.2)


@pytest.mark.parametrize(
    "override",
    [{"max_tool_calls": 5, "reserved_tool_calls": 5}, {"turn_timeout_s": 8, "reserved_time_s": 8}],
)
def test_the_execute_reserve_must_leave_room_for_other_steps(override):
    with pytest.raises(ValidationError, match="must leave"):
        Settings(_env_file=None, **override)


def test_relative_paths_are_anchored_to_the_project_root():
    settings = Settings(_env_file=None, traces_dir="traces")
    assert settings.traces_dir == PROJECT_ROOT / "traces"
    assert settings.agent_db_path == PROJECT_ROOT / "data" / "agent.db"
    assert settings.agent_db_path != settings.world_db_path


def test_the_model_window_must_hold_the_capped_context_and_the_reply():
    with pytest.raises(ValidationError, match="truncate"):
        Settings(_env_file=None, llm_num_ctx=4096, context_token_cap=6000)


@pytest.mark.parametrize(
    "override",
    [
        {"max_model_calls": 0},
        {"max_tool_calls": -1},
        {"turn_timeout_s": 0},
        {"vega_confidence_threshold": 1.5},
    ],
)
def test_out_of_range_values_are_rejected(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_settings_are_immutable():
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.max_tool_calls = 99


def test_a_context_cap_smaller_than_the_largest_prompt_is_refused():
    # Regression for docs/bugs.md #10: 300 was accepted, and the builder then sent 363 tokens,
    # with the user's message cut to its last 64 characters.
    with pytest.raises(ValueError, match="largest system prompt"):
        Settings(_env_file=None, context_token_cap=300)
