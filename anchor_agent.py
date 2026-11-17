"""anchor_agent.py — one autonomous turn against an OpenAI-compatible upstream, with tools (v1.29).

Everything in Anchor so far answered a person: the proxy wraps their messages, the MCP server
serves a host that relays them. A heartbeat, a journal, a room where the AI talks to itself —
those need the AI to take a turn when nobody wrote anything. This module is that turn:

    result = run_turn(upstream, system, user, tools, tool_exec)

- ``upstream``  : Upstream(url, key, model) — any /v1/chat/completions that speaks function calling
                  (OpenAI, Ollama, vLLM, LM Studio, OpenRouter, a gateway …). Same env as the proxy:
                  ANCHOR_UPSTREAM_URL / ANCHOR_UPSTREAM_KEY / ANCHOR_UPSTREAM_MODEL.
- ``tools``     : MCP-style tool dicts (name / description / inputSchema) — the same list anchor_mcp
                  serves — converted here to the OpenAI function format.
- ``tool_exec`` : callable(name, args) -> str | dict, executed for every tool call the model makes;
                  anchor_mcp.create_server()'s handle_tool fits directly.

The loop runs until the model stops calling tools or ``max_turns`` is reached, and returns the
final text plus a trace of every call (name, args, result preview) so the caller can log honestly
what happened. No model configured → ``NoUpstream`` with a plain message; nothing is faked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime


class NoUpstream(RuntimeError):
    pass


class UpstreamError(RuntimeError):
    pass


def mcp_tools_to_openai(tools: list) -> list:
    out = []
    for t in tools or []:
        params = t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": {
            "name": t["name"], "description": (t.get("description") or "")[:1024], "parameters": params}})
    return out


class Upstream:
    """Thin client. ``transport`` (for tests) replaces the HTTP call: callable(payload) -> response dict."""

    def __init__(self, url: str = None, key: str = None, model: str = None, transport=None, timeout: float = 300.0):
        self.url = (url or os.getenv("ANCHOR_UPSTREAM_URL", "")).rstrip("/")
        self.key = key if key is not None else os.getenv("ANCHOR_UPSTREAM_KEY", "")
        self.model = model or os.getenv("ANCHOR_UPSTREAM_MODEL", "")
        self.transport = transport
        self.timeout = timeout

    def ready(self) -> bool:
        return bool(self.transport) or bool(self.url and self.model)

    def chat(self, messages: list, tools: list = None, temperature: float = None) -> dict:
        if not self.ready():
            raise NoUpstream("no upstream model configured: set ANCHOR_UPSTREAM_URL and ANCHOR_UPSTREAM_MODEL "
                             "(ANCHOR_UPSTREAM_KEY if the endpoint needs one)")
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if self.transport:
            return self.transport(payload)
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = httpx.post(f"{self.url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        if r.status_code != 200:
            raise UpstreamError(f"upstream {r.status_code}: {r.text[:300]}")
        return r.json()


def _tool_result_text(res) -> str:
    if isinstance(res, str):
        return res
    try:
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return str(res)


def run_turn(upstream: Upstream, system: str, user: str, tools: list, tool_exec,
             max_turns: int = 8, temperature: float = None, max_tool_result_chars: int = 6000) -> dict:
    """One autonomous turn. Returns {"text": final text, "trace": [...], "turns": n, "stopped": reason}."""
    oai_tools = mcp_tools_to_openai(tools)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    trace, texts, stopped = [], [], "done"
    for turn in range(1, max_turns + 1):
        resp = upstream.chat(messages, oai_tools or None, temperature)
        try:
            msg = resp["choices"][0]["message"]
        except Exception:
            raise UpstreamError(f"unexpected upstream reply: {str(resp)[:300]}")
        content = msg.get("content")
        if isinstance(content, list):   # some servers return parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content and content.strip():
            texts.append(content.strip())
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": content or "", "tool_calls": calls})
        for c in calls:
            fn = (c.get("function") or {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                args = {"_raw": raw_args}
            started = datetime.now().isoformat(timespec="seconds")
            try:
                res = tool_exec(name, args)
                ok = True
            except Exception as e:
                res, ok = f"tool error: {type(e).__name__}: {e}", False
            text = _tool_result_text(res)
            if len(text) > max_tool_result_chars:
                text = text[:max_tool_result_chars] + f"\n…[{len(text) - max_tool_result_chars} more chars cut]"
            trace.append({"at": started, "tool": name, "args": args, "ok": ok, "result_preview": text[:200]})
            messages.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{turn}_{name}", "content": text})
        if turn == max_turns:
            stopped = "max_turns"
    return {"text": "\n\n".join(texts).strip(), "trace": trace, "turns": turn, "stopped": stopped}
