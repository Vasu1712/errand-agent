"""Where each node may go next: the flowchart's edges, as data.

Nodes name their next step in ``goto``; the router checks it against this map, so a node can only
take an edge the design allows. Every node may also go to ``finish`` (its error boundary does).
"""

from collections.abc import Callable

from errand.graph.state import AgentState

ROUTES: dict[str, tuple[str, ...]] = {
    "load_context": ("route_intent",),
    "route_intent": (
        "extract_slots",
        "find_booking",
        "memory_ops",
        "smalltalk",
        "clarify",
        "search",
        "check_slots",
        "confirm_fact",
        "parse_pick",
        "approval_check",
    ),
    "extract_slots": ("check_slots",),
    "check_slots": ("search",),
    "confirm_fact": ("check_slots", "extract_slots"),
    "search": ("rank_options",),
    "rank_options": ("present_options", "build_summary"),
    "present_options": ("parse_pick",),
    "parse_pick": (
        "build_summary",
        "extract_slots",
        "memory_ops",
        "find_booking",
        "present_options",
    ),
    # After the yes: hold, then book; a different quote goes back for a fresh yes.
    "hold_slot": ("execute", "build_summary", "rank_options"),
    "build_summary": ("await_approval",),
    "await_approval": ("approval_check",),
    "approval_check": (
        "hold_slot",
        "execute",
        "await_approval",
        "build_summary",
        "extract_slots",
        "find_booking",
    ),
    "execute": ("verify", "build_summary"),
    "verify": ("rank_options", "build_summary", "plan_reschedule"),
    "find_booking": ("build_summary", "plan_reschedule"),
    "plan_reschedule": ("build_summary",),
    "memory_ops": (),
    "smalltalk": (),
    "clarify": (),
}


def router(source: str) -> Callable[[AgentState], str]:
    allowed = {*ROUTES[source], "finish"}

    def route(state: AgentState) -> str:
        target = "finish" if state.stop else state.goto
        if target not in allowed:
            raise ValueError(f"{source} has no edge to {target!r}")
        return target

    route.__name__ = f"after_{source}"
    return route
