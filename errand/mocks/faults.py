"""A view of the mock world whose tools break on cue, as a misbehaving provider API would.

Each fault names a tool and which call to it fails (counted in the world DB, so across runs and
restarts): the call raises, answers with unreadable JSON, or answers with the wrong shape. An
``after`` fault lets the call take effect first, so only its reply is lost.
"""

import json
from typing import Any

from errand.contracts import ToolName

TOOLS = frozenset(tool.value for tool in ToolName)


class FaultyTools:
    def __init__(self, world: Any) -> None:
        self._world = world

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._world, name)
        if name not in TOOLS:
            return attr
        faults = [f for f in self._world.scenario.faults if f.tool == name]
        if not faults:
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            n = self._world._bump("fault", name)
            fault = next((f for f in faults if f.call == n), None)
            if fault is None:
                return attr(*args, **kwargs)
            if fault.after:
                attr(*args, **kwargs)
            if fault.kind == "exception":
                raise RuntimeError(f"{name}: the provider's client crashed (simulated)")
            if fault.kind == "malformed_json":
                raise json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0)
            return {"status": "ok", "data": None}  # wrong_type: not the contract's model

        return call
