from datetime import datetime

import pytest
from pydantic import BaseModel

from errand.contracts import IST
from errand.ids import is_valid_id
from errand.trace import EventKind, TraceEvent, Tracer, new_run_id, read_trace, sanitize


def test_each_run_writes_its_own_jsonl_file(tmp_path):
    for run_id in ("run_a", "run_b"):
        with Tracer(run_id, tmp_path) as tracer:
            tracer.emit(EventKind.RUN_START, message=f"hello from {run_id}")
            tracer.emit(EventKind.RUN_END, outcome="ok")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["run_a.jsonl", "run_b.jsonl"]
    lines = (tmp_path / "run_a.jsonl").read_text(encoding="utf-8").splitlines()
    events = [TraceEvent.model_validate_json(line) for line in lines]
    assert [(event.seq, event.kind, event.run_id) for event in events] == [
        (1, "run.start", "run_a"),
        (2, "run.end", "run_a"),
    ]
    assert read_trace(tmp_path, "run_a") == events


def test_a_span_records_latency_results_and_errors(tmp_path):
    with Tracer("run_span", tmp_path) as tracer:
        with tracer.span(
            EventKind.TOOL_CALL, tool="search_providers", args={"area": "HSR"}
        ) as extra:
            extra["result"] = {"providers": 3}
        with pytest.raises(TimeoutError), tracer.span(EventKind.TOOL_CALL, tool="book"):
            raise TimeoutError("no response after 5s")

    ok, failed = read_trace(tmp_path, "run_span")
    assert ok.latency_ms is not None and ok.latency_ms >= 0
    assert ok.data == {
        "tool": "search_providers",
        "args": {"area": "HSR"},
        "result": {"providers": 3},
    }
    assert failed.data == {"tool": "book", "error": "TimeoutError: no response after 5s"}


def test_sanitize_cuts_long_text_redacts_secrets_and_dumps_models():
    class Hold(BaseModel):
        hold_id: str
        expires_at: datetime

    hold = Hold(hold_id="hold_1", expires_at=datetime(2026, 11, 3, 10, 10, tzinfo=IST))
    clean = sanitize(
        {
            "review": "x" * 50,
            "api_key": "sk-live-123",
            "input_tokens": 12,
            "hold": hold,
            "labels": {"b", "a"},
        },
        max_str=30,
    )
    assert clean == {
        "review": "x" * 30 + "…[+20 chars]",
        "api_key": "[redacted]",
        "input_tokens": 12,
        "hold": {"hold_id": "hold_1", "expires_at": "2026-11-03T10:10:00+05:30"},
        "labels": ["a", "b"],
    }


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a/b", "", ".hidden", "-x", "x" * 65])
def test_run_ids_that_could_escape_the_traces_dir_are_rejected(tmp_path, bad_id):
    with pytest.raises(ValueError):
        Tracer(bad_id, tmp_path)
    with pytest.raises(ValueError):
        read_trace(tmp_path, bad_id)


def test_a_run_id_names_exactly_one_run(tmp_path):
    Tracer("run_once", tmp_path).close()
    with pytest.raises(FileExistsError):
        Tracer("run_once", tmp_path)


def test_the_reader_skips_a_partially_written_last_line(tmp_path):
    with Tracer("run_live", tmp_path) as tracer:
        tracer.emit(EventKind.RUN_START)
    with (tmp_path / "run_live.jsonl").open("a", encoding="utf-8") as trace_file:
        trace_file.write('{"run_id": "run_live", "seq": 2, "ki')
    assert [event.seq for event in read_trace(tmp_path, "run_live")] == [1]


def test_new_run_ids_are_valid_and_unique():
    run_ids = {new_run_id() for _ in range(200)}
    assert len(run_ids) == 200
    assert all(is_valid_id(run_id) for run_id in run_ids)
