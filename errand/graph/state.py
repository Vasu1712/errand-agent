"""The explicit workflow state, checkpointed per thread in the agent DB.

Values stay JSON-native (strings, numbers, dicts, lists), so a checkpoint never has to deserialize
one of our classes. Side-effect truth lives in the ledger, not here: ``execution`` is only a
pointer to a ledger row.
"""

from pydantic import BaseModel


class Phase:
    """Where the current task stands. Mirrored to the tasks table."""

    IDLE = "idle"
    COLLECTING = "collecting"  # asked for missing details
    CONFIRMING_FACT = "confirming_fact"  # asked a yes/no question about an inferred preference
    CHOOSING_BOOKING = "choosing_booking"  # asked which booking to cancel or move
    PRESENTING = "presenting"  # options shown, waiting for a pick
    AWAITING_APPROVAL = "awaiting_approval"  # summary shown, waiting for a clear yes
    EXECUTING = "executing"  # approval granted, side effect in progress
    DONE = "done"  # verified outcome
    UNCONFIRMED = "unconfirmed"  # outcome unknown; the reconciler will settle it
    FAILED = "failed"
    ABANDONED = "abandoned"  # the user moved on to another request

    FINISHED = (DONE, FAILED, ABANDONED, IDLE)


class AgentState(BaseModel):
    thread_id: str
    user_id: str
    run_id: str = ""

    # This turn.
    user_message: str = ""
    message_id: int | None = None
    confirm: dict | None = None  # structured confirmation from the UI: approval id + hash
    proactive: dict | None = None  # set when the worker opens a replacement task
    goto: str = ""
    reply: str = ""
    notice: str = ""
    stop: dict | None = None  # set by a node's error boundary
    resume: dict | None = None  # where "continue" picks up a stopped message: node and message
    # The reply lists stored memory, guesses included: it is for the user, never model context.
    reply_private: bool = False

    # Errands the user asked for in the same message as the current one, still to do.
    queued: list[dict] = []  # [{intent, text}]
    queue_reminded: bool = False

    # The task.
    task_id: str | None = None
    intent: str | None = None  # book | reschedule | cancel
    phase: str = Phase.IDLE
    slots: dict = {}
    slot_sources: dict = {}  # slot -> "user" or "memory:<fact id>"
    asked: list[str] = []
    pending: dict | None = None  # what the last question was about
    candidates: list[dict] = []  # search results for the current details
    review_scores: dict = {}  # provider id -> {score, flagged}
    options: list[dict] = []
    excluded: list[str] = []  # "provider|slot start" pairs ruled out (taken, conflict, budget)
    excluded_providers: list[str] = []
    notes: list[str] = []  # code-written remarks for the next reply
    pick: int | None = None
    # The provider of a summary the user changed ("make it Thursday"): if it has a slot that
    # fits the new details, that is proposed again for a fresh yes instead of a new list.
    keep_provider: str | None = None
    hold: dict | None = None
    approval: dict | None = None  # {approval_id, material_hash, action, summary}
    target: dict | None = None  # the booking being cancelled or moved
    new_slot: dict | None = None
    execution: dict | None = None
