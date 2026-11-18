"""Function-calling tools exposed to the conversation LLM.

A tool is a JSON-schema description plus a cheap Python callable.  The model
sees the schema, the engine executes the callable inside the gateway process
and feeds the result back as a ``tool`` message before asking for the spoken
answer.

Keep handlers fast and side-effect free: they run while the caller is waiting
for audio.  Slow or remote work should be wrapped in its own timeout and,
ideally, land behind an HTTP client.  ``web_fetch`` only allows public
http(s) URLs so the model cannot reach the gateway's own internal services.
"""
from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import os
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger(__name__)

Handler = Callable[..., Any] | Callable[..., Awaitable[Any]]

# Tool output is fed back into the prompt; cap it so a chatty tool cannot eat
# the context budget of the next generation.
MAX_RESULT_CHARS = 2000

_USER_AGENT = "Mozilla/5.0 (compatible; native-voice/0.1)"
_MAX_BYTES = 512 * 1024
_MAX_FETCH_CHARS = 4000
_REDIRECTS = {301, 302, 303, 307, 308}
_SEARCH_ENDPOINT = os.environ.get(
    "NV_WEB_SEARCH_URL", "https://cn.bing.com/search"
).strip()
_CJK = re.compile("[" + chr(0x3400) + "-" + chr(0x9FFF) + "]")


@dataclass
class Tool:
    """One callable exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    timeout_s: float = 5.0

    def spec(self) -> dict[str, Any]:
        """OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    """Name -> tool lookup used by the conversation engine."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def __bool__(self) -> bool:
        return bool(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    async def call(self, name: str, arguments: str | dict[str, Any] | None) -> tuple[bool, str]:
        """Run a tool. Returns ``(ok, text_for_the_model)`` and never raises."""
        tool = self.tools.get(name)
        if tool is None:
            return False, f"未知工具：{name}"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return False, f"工具参数不是合法 JSON 对象：{exc}"
        if not isinstance(args, dict):
            return False, "工具参数必须是 JSON 对象"
        try:
            result = await asyncio.wait_for(_invoke(tool, args), timeout=tool.timeout_s)
        except asyncio.TimeoutError:
            return False, f"工具 {name} 超时（{tool.timeout_s:g} 秒）"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("tool %s failed: %r", name, exc)
            return False, f"工具 {name} 执行失败：{exc}"
        return True, _as_text(result)[:MAX_RESULT_CHARS]


async def _invoke(tool: Tool, args: dict[str, Any]) -> Any:
    result = tool.handler(**args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "完成"
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------- built-ins
def get_current_time(timezone: str = "") -> str:
    """Local date/time, optionally in a specific IANA timezone."""
    tz = None
    if timezone:
        try:
            tz = ZoneInfo(timezone)
        except Exception as exc:
            raise ValueError(f"未知时区 {timezone!r}，请使用 IANA 时区名，例如 Asia/Shanghai") from exc
    now = datetime.now(tz)
    weekday = "一二三四五六日"[now.weekday()]
    zone = timezone or now.strftime("%Z") or "本地时间"
    return f"当前时间：{now:%Y-%m-%d %H:%M:%S}，星期{weekday}，时区 {zone}"


def _http_client(follow_redirects: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(8.0, connect=4.0),
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=follow_redirects,
    )


# ------------------------------------------------------------------- fetch
def _check_public_url(url: str) -> str:
    """Reject non-http(s), credentials, and any address that is not global."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("只支持 http/https URL")
    if parsed.username or parsed.password:
        raise ValueError("URL 不能包含用户名或密码")
    host = parsed.hostname
    if not host:
        raise ValueError("URL 缺少主机名")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析主机：{host}") from exc
    addresses = {info[4][0].split("%")[0] for info in infos}
    if not addresses:
        raise ValueError(f"无法解析主机：{host}")
    for raw in addresses:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise ValueError(f"非法地址：{raw}") from exc
        # Link-local covers the cloud metadata endpoint; is_global rejects
        # loopback, private, reserved and multicast ranges.
        if ip.is_link_local or not ip.is_global:
            raise ValueError("不允许访问内网、回环或保留地址")
    return url


class _TextExtractor(HTMLParser):
    """Collect visible text (plus <title>) from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg", "canvas"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "tr", "td"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "canvas") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html or "")
    except Exception:
        return _collapse(html or "")
    title = _collapse(" ".join(parser.title_parts))
    body = _collapse("".join(parser.parts))
    if title and body and not body.startswith(title):
        return f"{title}\n{body}"
    return body or title


def _one_line(text: str, limit: int = 180) -> str:
    return _collapse(text).replace("\n", " ")[:limit]


async def web_fetch(url: str) -> str:
    """Fetch a public web page and return its readable text."""
    current = (url or "").strip()
    if not current:
        raise ValueError("URL 为空")
    async with _http_client() as client:
        for _ in range(4):
            _check_public_url(current)
            try:
                async with client.stream("GET", current) as resp:
                    if resp.status_code in _REDIRECTS:
                        location = resp.headers.get("location")
                        if not location:
                            raise ValueError("重定向缺少 Location")
                        current = urljoin(current, location)
                        continue
                    resp.raise_for_status()
                    content_type = (resp.headers.get("content-type") or "").lower()
                    if content_type and not any(
                        kind in content_type
                        for kind in ("text/", "html", "xml", "json")
                    ):
                        raise ValueError(f"不支持的内容类型：{content_type.split(';')[0]}")
                    body = b""
                    async for chunk in resp.aiter_bytes():
                        body += chunk
                        if len(body) > _MAX_BYTES:
                            raise ValueError("响应内容过大")
                    charset = resp.encoding or "utf-8"
            except httpx.TimeoutException as exc:
                raise ValueError("请求超时") from exc
            except httpx.HTTPStatusError as exc:
                raise ValueError(f"HTTP {exc.response.status_code}") from exc
            except httpx.HTTPError as exc:
                raise ValueError(f"请求失败：{str(exc) or type(exc).__name__}") from exc
            text = body.decode(charset, "replace")
            if "html" in content_type:
                text = _html_to_text(text)
            text = _collapse(text)
            if not text:
                return "页面没有可读文本内容。"
            return text[:_MAX_FETCH_CHARS]
    raise ValueError("重定向次数过多")


# ------------------------------------------------------------------ search
def _parse_rss(xml_text: str, limit: int) -> list[tuple[str, str, str]]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ValueError(f"搜索结果解析失败：{exc}") from exc
    items: list[tuple[str, str, str]] = []
    for node in root.iter("item"):
        title = _one_line(node.findtext("title") or "", 120)
        link = (node.findtext("link") or "").strip()
        snippet = _one_line(_html_to_text(node.findtext("description") or ""), 180)
        if title and link:
            items.append((title, link, snippet))
        if len(items) >= limit:
            break
    return items


async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web through Bing's RSS endpoint."""
    text = (query or "").strip()
    if not text:
        raise ValueError("搜索关键词为空")
    try:
        limit = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        limit = 5
    url = f"{_SEARCH_ENDPOINT}?format=rss&q={quote_plus(text)}"
    if _CJK.search(text):
        # Without a market hint Bing can answer a Chinese query with unrelated
        # localized results (observed on non-CN egress IPs).
        url += "&mkt=zh-CN"
    try:
        async with _http_client(follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            items = _parse_rss(resp.text, limit)
    except httpx.TimeoutException as exc:
        raise ValueError("搜索请求超时") from exc
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"搜索服务返回 HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"搜索请求失败：{str(exc) or type(exc).__name__}") from exc
    if not items:
        return f"没有找到与“{text}”相关的结果。"
    lines = []
    for index, (title, link, snippet) in enumerate(items, 1):
        lines.append(f"{index}. {title}\n{link}" + (f"\n{snippet}" if snippet else ""))
    return "\n".join(lines)


BUILTIN_TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            name="get_current_time",
            description=(
                "获取当前的日期、时间和星期。你不知道现在的时间，凡涉及时间、"
                "日期或星期的问题都必须调用此工具，不要猜测。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名，例如 Asia/Shanghai；留空表示服务器本地时区。",
                    }
                },
                "required": [],
            },
            handler=get_current_time,
        ),
        Tool(
            name="web_search",
            description=(
                "在互联网上搜索信息。凡是你不知道、不确定、或可能过时的内容"
                "（新闻、事实、日期、价格等），都必须调用搜索，不要凭记忆回答。"
                "返回若干条标题、链接和摘要。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {
                        "type": "integer",
                        "description": "返回条数，1-10，默认 5",
                    },
                },
                "required": ["query"],
            },
            handler=web_search,
            timeout_s=12.0,
        ),
        Tool(
            name="web_fetch",
            description=(
                "读取一个公开网页的正文文本。需要查看搜索结果中的具体页面，"
                "或用户直接给出链接时调用。只支持 http/https 公网地址。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要读取的网页 URL"},
                },
                "required": ["url"],
            },
            handler=web_fetch,
            timeout_s=12.0,
        ),
    )
}


def build_tool_registry(names: list[str] | None) -> ToolRegistry:
    """Build a registry from configured names; unknown names are ignored."""
    registry = ToolRegistry()
    for raw in names or []:
        name = str(raw).strip()
        if not name:
            continue
        tool = BUILTIN_TOOLS.get(name)
        if tool is None:
            log.warning("unknown tool %r in llm.tools; available: %s", name, sorted(BUILTIN_TOOLS))
            continue
        registry.register(tool)
    return registry
