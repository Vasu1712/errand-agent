"""Free text from tools is data, never instructions.

Every review, description or other tool-supplied string is sanitized and wrapped in a labelled
``<untrusted>`` block before any model (Qwen or Vega) sees it. Sanitizing removes invisible and
control characters and turns angle brackets into look-alikes, so the text can't close its own block
or forge a new one. The regex hint is a backstop next to Vega's injection flag, so it only uses
high-precision patterns (docs/bugs.md #7). Neither can change what the agent does: they only decide
what gets logged and left out of ranking.
"""

import re
import unicodedata

_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_HINT = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}\b(?:instructions?|rules|prompt|guidelines)\b"
    r"|\bsystem\s*(?:prompt|override|message)\b"
    r"|\byou are now\b|\b(?:admin|developer|god)\s+mode\b"
    r"|\bpre-?approved\b|\bskip(?:ping)?\s+the\s+confirmation\b"
    r"|\b(?:approve|confirm)\s+(?:this|the|all)\s+bookings?\b"
    r"|\bmark\s+(?:it|this|the booking)\s+as\s+(?:approved|confirmed|paid)\b"
    r"|\b(?:call|invoke|run)\s+(?:the\s+)?(?:book|cancel|reschedule|hold_slot)\b",
    re.IGNORECASE,
)


def sanitize_text(text: str, *, max_chars: int = 600) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("<", "‹").replace(">", "›")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def untrusted_block(text: str, *, source: str) -> str:
    """`text` as a labelled block a model is told to treat as quoted data."""
    label = re.sub(r"[^A-Za-z0-9_:.-]", "_", source)
    return f'<untrusted source="{label}">\n{sanitize_text(text)}\n</untrusted>'


def injection_hint(text: str) -> str | None:
    """The first phrase that reads like an instruction to an AI system, if any."""
    match = _INJECTION_HINT.search(text)
    return match.group(0) if match else None
