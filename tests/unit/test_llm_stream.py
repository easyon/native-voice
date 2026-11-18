from __future__ import annotations

from native_voice.models.llm.client import ToolCallAccumulator


def test_accumulates_llamacpp_fragments():
    # Shape captured from llama-server b12125 with --jinja.
    chunks = [
        {"index": 0, "id": "abc123", "type": "function",
         "function": {"name": "get_current_time", "arguments": "{"}},
        {"index": 0, "function": {"arguments": '"timezone"'}},
        {"index": 0, "function": {"arguments": ': "Asia/Shanghai"}'}},
    ]
    acc = ToolCallAccumulator()
    for chunk in chunks:
        acc.feed([chunk])
    calls = acc.calls()
    assert len(calls) == 1
    assert calls[0].id == "abc123"
    assert calls[0].name == "get_current_time"
    assert calls[0].arguments == '{"timezone": "Asia/Shanghai"}'


def test_accumulates_parallel_calls():
    acc = ToolCallAccumulator()
    acc.feed([
        {"index": 0, "id": "a", "function": {"name": "web_search", "arguments": "{\"query\": \"1+1\"}"}},
        {"index": 1, "id": "b", "function": {"name": "get_current_time", "arguments": "{}"}},
    ])
    calls = acc.calls()
    assert [call.id for call in calls] == ["a", "b"]
    assert [call.name for call in calls] == ["web_search", "get_current_time"]


def test_defaults_and_nameless_slots():
    acc = ToolCallAccumulator()
    acc.feed([{"index": 0, "function": {"arguments": "{}"}}])  # no name -> ignored
    acc.feed([{"id": "x", "function": {"name": "web_search"}}])  # no arguments -> {}
    calls = acc.calls()
    assert len(calls) == 1
    assert calls[0].arguments == "{}"
    assert calls[0].id == "x"
