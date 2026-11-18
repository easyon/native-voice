from __future__ import annotations

import asyncio

import httpx
import pytest

from native_voice.conversation import tools
from native_voice.conversation.engine import _runaway_repetition
from native_voice.conversation.tools import (
    Tool,
    ToolRegistry,
    _check_public_url,
    _html_to_text,
    _parse_rss,
    build_tool_registry,
    get_current_time,
    web_fetch,
    web_search,
)


def test_current_time_uses_timezone():
    assert "Asia/Shanghai" in get_current_time("Asia/Shanghai")
    assert "当前时间" in get_current_time("")


def test_current_time_rejects_unknown_zone():
    with pytest.raises(ValueError):
        get_current_time("Mars/Olympus")


def test_build_registry_ignores_unknown_names():
    registry = build_tool_registry(["web_search", "does_not_exist", ""])
    assert len(registry) == 1
    assert registry.get("web_search") is not None


def test_build_registry_empty():
    registry = build_tool_registry([])
    assert not registry
    assert registry.specs() == []


def test_default_registry_has_web_tools():
    registry = build_tool_registry(["get_current_time", "web_search", "web_fetch"])
    assert set(registry.tools) == {"get_current_time", "web_search", "web_fetch"}


async def test_registry_call_ok():
    registry = build_tool_registry(["get_current_time"])
    ok, text = await registry.call("get_current_time", '{"timezone": "Asia/Shanghai"}')
    assert ok is True
    assert "Asia/Shanghai" in text


async def test_registry_unknown_tool():
    registry = build_tool_registry(["get_current_time"])
    ok, text = await registry.call("nope", "{}")
    assert ok is False
    assert "未知工具" in text


async def test_registry_bad_arguments():
    registry = build_tool_registry(["get_current_time"])
    ok, text = await registry.call("get_current_time", "not json")
    assert ok is False
    assert "JSON" in text
    ok, text = await registry.call("get_current_time", '{"wrong": 1}')
    assert ok is False
    assert "执行失败" in text


async def test_registry_async_handler_and_failure():
    async def good(value: int) -> dict:
        await asyncio.sleep(0)
        return {"value": value}

    def bad() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(Tool("good", "d", {"type": "object", "properties": {}}, good))
    registry.register(Tool("bad", "d", {"type": "object", "properties": {}}, bad))
    ok, text = await registry.call("good", '{"value": 3}')
    assert ok is True
    assert text == '{"value": 3}'
    ok, text = await registry.call("bad", "{}")
    assert ok is False
    assert "boom" in text


async def test_registry_timeout():
    async def slow() -> str:
        await asyncio.sleep(0.5)
        return "late"

    registry = ToolRegistry()
    registry.register(Tool("slow", "d", {"type": "object", "properties": {}}, slow, timeout_s=0.05))
    ok, text = await registry.call("slow", "{}")
    assert ok is False
    assert "超时" in text


def test_runaway_repetition_guard():
    assert not _runaway_repetition("")
    assert not _runaway_repetition("现在是下午三点，天气不错。")
    assert not _runaway_repetition("好的。" * 2)
    assert _runaway_repetition("现在是22点36分，星期二。" * 3)
    assert _runaway_repetition("哈" * 60)


# ------------------------------------------------------------- web helpers
def test_check_public_url_rejects_internal_targets():
    for url in (
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://user:pass@93.184.216.34/",
        "http:///nohost",
    ):
        with pytest.raises(ValueError):
            _check_public_url(url)


def test_check_public_url_allows_public_ip():
    assert _check_public_url("http://93.184.216.34/page") == "http://93.184.216.34/page"


def test_html_to_text_strips_scripts_and_keeps_title():
    html = (
        "<html><head><title>页面标题</title><style>.x{}</style></head>"
        "<body><script>alert(1)</script><p>第一段</p><p>第二段</p></body></html>"
    )
    text = _html_to_text(html)
    assert "页面标题" in text
    assert "第一段" in text
    assert "第二段" in text
    assert "alert" not in text
    assert ".x{}" not in text


def test_parse_rss_extracts_items_and_strips_html():
    xml = """<?xml version="1.0"?><rss><channel>
      <item><title>标题一</title><link>https://example.com/1</link>
        <description>&lt;b&gt;摘要&lt;/b&gt; 内容</description></item>
      <item><title>标题二</title><link>https://example.com/2</link>
        <description>第二条</description></item>
    </channel></rss>"""
    items = _parse_rss(xml, 5)
    assert len(items) == 2
    assert items[0][0] == "标题一"
    assert items[0][1] == "https://example.com/1"
    assert "摘要" in items[0][2]


def _patch_client(monkeypatch, handler):
    def factory(follow_redirects: bool = False):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr(tools, "_http_client", factory)


async def test_web_fetch_returns_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title>标题</title></head><body><p>网页正文</p></body></html>",
        )

    _patch_client(monkeypatch, handler)
    text = await web_fetch("http://93.184.216.34/article")
    assert "标题" in text
    assert "网页正文" in text


async def test_web_fetch_rejects_redirect_to_private(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError):
        await web_fetch("http://93.184.216.34/article")


async def test_web_fetch_rejects_large_and_binary(monkeypatch):
    def big(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 600000)

    _patch_client(monkeypatch, big)
    with pytest.raises(ValueError):
        await web_fetch("http://93.184.216.34/big")

    def binary(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"png")

    _patch_client(monkeypatch, binary)
    with pytest.raises(ValueError):
        await web_fetch("http://93.184.216.34/image.png")


async def test_web_search_formats_results(monkeypatch):
    xml = """<?xml version="1.0"?><rss><channel>
      <item><title>结果一</title><link>https://example.com/1</link>
        <description>摘要一</description></item>
      <item><title>结果二</title><link>https://example.com/2</link>
        <description>摘要二</description></item>
    </channel></rss>"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "q=%E5%A4%A9%E6%B0%94" in str(request.url)
        return httpx.Response(200, headers={"content-type": "application/xml"}, text=xml)

    _patch_client(monkeypatch, handler)
    text = await web_search("天气", 2)
    assert "结果一" in text
    assert "https://example.com/2" in text
    assert "摘要一" in text


async def test_web_search_rejects_empty_query():
    with pytest.raises(ValueError):
        await web_search("  ")
