"""A keyword-rule stand-in for Vega, for offline tests and evals.

Deliberately simple and deterministic. Anything it can't place gets a low confidence, which
exercises the same clarify and re-confirm paths a hesitant Vega answer would.
"""

import re

from errand.decide.scripted import ScriptedDecider
from errand.gateway.untrusted import injection_hint

_SERVICE_WORDS = (
    r"plumb|tap|leak|pipe|electric|wiring|switch|\bac\b|air ?con|carpent|door|clean|paint|pest"
)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def intent(text: str) -> tuple[str, float]:
    if _has(
        text, r"\b(remember|forget|what do you (know|remember)|preferences?|update my|correct my)\b"
    ):
        return "memory_command", 0.97
    if _has(text, r"\b(cancel|call (it )?off|scrap|drop)\b"):
        return "cancel", 0.97
    if _has(text, r"\b(move|reschedule|shift|postpone|push)\b"):
        return "reschedule", 0.96
    if _has(text, rf"\b(book|find|need|get me|search|look for)\b|{_SERVICE_WORDS}"):
        return "book", 0.95
    if _has(text, r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening)|bye|cool|ok)\b"):
        return "smalltalk", 0.95
    return "book", 0.4


def approval(text: str) -> tuple[str, float]:
    if _has(text, r"\b(no|nope|don'?t|not|stop|instead|cheaper|change)\b"):
        return "no", 0.95
    if _has(text, r"\b(yes|yep|yeah|sure|go ahead|confirm(ed)?|book it|do it|ok(ay)?|please do)\b"):
        return "yes", 0.97
    return "unsure", 0.6


def review_quality(text: str) -> tuple[str, float]:
    if _has(text, r"late|rude|mess|extra|did not show|terrible|worse|waste|overpriced"):
        if _has(text, r"\b(but|though|fine|good)\b"):
            return "mixed", 0.85
        return "negative", 0.9
    if _has(text, r"\b(but|though|okay|average|second visit|pricier)\b"):
        return "mixed", 0.85
    return "positive", 0.9


def injection(text: str) -> tuple[str, float]:
    return ("injection", 0.95) if injection_hint(text) else ("benign", 0.9)


def rule_decider() -> ScriptedDecider:
    return ScriptedDecider(
        {
            "intent": intent,
            "approval": approval,
            "review_quality": review_quality,
            "injection": injection,
        }
    )
