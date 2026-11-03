"""The suite's own guarantees: no network, and booting the app loads no model runtime."""

import socket
import subprocess
import sys

import pytest

from errand.config import PROJECT_ROOT


def test_network_connections_fail_inside_tests():
    # Port 11434 is Ollama's: even a server running on this machine is out of reach.
    with pytest.raises(RuntimeError, match="network access attempted"):
        socket.create_connection(("127.0.0.1", 11434), timeout=0.5)


def test_importing_the_app_loads_no_model_runtime():
    probe = (
        "import sys, errand.api, errand.decide, errand.llm.ollama; "
        "print(sorted(m for m in ('torch', 'transformers', 'vegaml') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
