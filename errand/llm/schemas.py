"""What Qwen may return. Every call names one of these schemas; nothing else is accepted.

The model never gets a tool. The most it can do is propose: a slot patch, a pick among options code
put in front of it, a memory command code will validate, or a conversational reply.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Service = Literal[
    "plumber", "electrician", "ac_service", "carpenter", "cleaning", "painter", "pest_control"
]
When = Literal["today", "tomorrow", "this_week", "next_week", "this_weekend", "weekday", "date"]
Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TimeWindow = Literal["morning", "afternoon", "evening", "any"]

MEMORY_KEYS = (
    "preferred_area",
    "earliest_time",
    "preferred_time_window",
    "technician",
    "budget_max_inr",
)
MemoryKey = Literal[
    "preferred_area", "earliest_time", "preferred_time_window", "technician", "budget_max_inr"
]


class SlotPatch(BaseModel):
    """Details the latest user message states. None means "not mentioned", never "clear it"."""

    model_config = ConfigDict(extra="ignore")

    service: Service | None = None
    area: str | None = None
    # The day(s) wanted: for a new booking, or the new day of a reschedule.
    when: When | None = None
    when_weekday: Weekday | None = None
    when_date: str | None = Field(None, description="YYYY-MM-DD")
    # Only for pointing at an existing booking ("my Thursday booking").
    booking_when: When | None = None
    booking_weekday: Weekday | None = None
    booking_date: str | None = Field(None, description="YYYY-MM-DD")
    time_window: TimeWindow | None = None
    earliest_time: str | None = Field(None, description="HH:MM, 24-hour")
    budget_max_inr: int | None = Field(None, ge=0)
    no_budget_limit: bool | None = None
    same_provider_as_last_time: bool | None = None
    provider_name: str | None = None
    problem: str | None = None


class PickProposal(BaseModel):
    """The user's answer to a list of options, as a proposal code will check."""

    action: Literal["propose_booking", "change_details", "none"]
    option_id: int | None = None


class MemoryCommand(BaseModel):
    op: Literal["inspect", "remember", "forget"]
    key: MemoryKey | None = None
    value: str | None = None
    service: Service | None = None


class ReplyText(BaseModel):
    text: str = Field(max_length=800)
