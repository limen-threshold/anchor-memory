"""LLM abstraction layer for Anchor Memory.

Anchor's background passes (dream pass, concept linking, dedup) call out to
an LLM. Earlier versions wrote `anthropic.Anthropic()` directly, which:
  - Forced every user to have an Anthropic API key
  - Made provider switching require source edits
  - Hid the actual model id and cost behind hardcoded strings

This module abstracts that into a single `LLM` interface with provider
implementations. Code calling LLMs should depend on the interface, not on
the SDK.

Supported providers:
  - anthropic       — Claude (default model: claude-haiku-4-5-20251001)
  - openai          — OpenAI (default model: gpt-5-nano)
  - google          — Gemini (default model: gemini-2.5-flash)
  - openai-compat   — DeepSeek, GLM/Zhipu, Ollama, LM Studio, etc.

Resolution order for `get_default_llm()`:
  1. Explicit `llm=` argument passed by caller
  2. ANCHOR_LLM env: provider/model spec like "anthropic/claude-haiku-4-5-20251001"
  3. ~/.anchor/config.yaml (if exists)
  4. Fallback: if ANTHROPIC_API_KEY is set, use Anthropic + Haiku (cheap default)
  5. Raise ConfigError with instructions

Spend tracking: every call() invocation increments per-day spend in
~/.anchor/spend.jsonl. Daily caps from config.yaml are enforced — if today's
spend exceeds the cap, raise SpendCapExceeded BEFORE making the call.
"""
from __future__ import annotations
import os
import re
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

CONFIG_DIR = Path.home() / ".anchor"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
SPEND_PATH = CONFIG_DIR / "spend.jsonl"


# Price table per million tokens (USD). Approximate, used only for spend
# estimation — actual billing is between user and provider. Updated 2026-05.
PRICES_PER_M = {
    # provider:model -> (input, output)
    "anthropic:claude-haiku-4-5-20251001": (1.0, 5.0),
    "anthropic:claude-sonnet-4-6": (3.0, 15.0),
    "anthropic:claude-opus-4-6": (15.0, 75.0),
    "anthropic:claude-opus-4-7": (15.0, 75.0),
    "openai:gpt-5-nano": (0.05, 0.40),
    "openai:gpt-5-mini": (0.25, 2.00),
    "openai:gpt-5": (1.25, 10.00),
    "google:gemini-2.5-flash": (0.075, 0.30),
    "google:gemini-2.5-pro": (1.25, 5.00),
    # openai-compat models are user-supplied; fall back to 0 (caller knows their own pricing)
}


class AnchorLLMError(Exception):
    """Base for LLM layer errors."""


class ConfigError(AnchorLLMError):
    """No LLM configured. User needs to run `anchor init` or set ANCHOR_LLM env."""


class SpendCapExceeded(AnchorLLMError):
    """Today's spend would exceed the configured daily cap."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""


class LLM:
    """Provider-agnostic LLM interface. Subclass and implement `_call_raw`."""

    provider: str = "unknown"

    def __init__(self, model: str, api_key: Optional[str] = None,
                 endpoint: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint

    def call(self, system: str, user: str, max_tokens: int = 1024,
             temperature: float = 0.5, cache_ttl: str = "5m") -> LLMResponse:
        """Top-level call with spend tracking and cap enforcement.

        Args:
            cache_ttl: Provider-specific. For Anthropic: "5m" (default) or "1h".
                Use "1h" for long-running background passes where the same
                system prompt fires many times across more than 5 minutes —
                dream pass, concept_link backfill, etc. 1h write is ~2x the
                cost of 5m write, but read is the same cheap price; on any
                pass running more than ~10 minutes, 1h wins.
                Ignored by providers that auto-manage cache (OpenAI, Google).
        """
        _check_daily_cap()
        resp = self._call_raw(system, user, max_tokens, temperature, cache_ttl)
        _record_spend(resp)
        return resp

    def _call_raw(self, system: str, user: str, max_tokens: int,
                  temperature: float, cache_ttl: str = "5m") -> LLMResponse:
        raise NotImplementedError

    def _price(self, in_tok: int, out_tok: int,
               cache_read_tok: int = 0, cache_write_tok: int = 0,
               cache_ttl: str = "5m") -> float:
        """Estimate $ cost given token counts.

        Anthropic pricing:
        - input_no_cache: base price (in_tok already excludes cache_read/write)
        - cache_read:     0.1x base
        - cache_write_5m: 1.25x base
        - cache_write_1h: 2x base

        For other providers the cache_* fields are 0 (provider auto-managed).
        """
        key = f"{self.provider}:{self.model}"
        rates = PRICES_PER_M.get(key, (0.0, 0.0))
        in_rate, out_rate = rates
        cw_mult = 2.0 if cache_ttl == "1h" else 1.25
        cost = (
            in_tok * in_rate
            + out_tok * out_rate
            + cache_read_tok * in_rate * 0.1
            + cache_write_tok * in_rate * cw_mult
        ) / 1_000_000
        return cost


class AnthropicLLM(LLM):
    provider = "anthropic"

    def _call_raw(self, system, user, max_tokens, temperature, cache_ttl="5m"):
        try:
            import anthropic
        except ImportError as e:
            raise ConfigError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        # Wrap system as a cache_control block. Anthropic requires:
        #   - prompt >= 1024 tokens to actually cache (smaller prompts pass
        #     through transparently with no cache effect)
        #   - same prompt fires within TTL to get a cache hit
        # We always wrap; if the prompt is small the SDK silently no-ops the
        # cache. This costs nothing extra to do unconditionally.
        cache_block = {"type": "ephemeral"}
        if cache_ttl == "1h":
            cache_block["ttl"] = "1h"
        system_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": cache_block,
        }] if system else []
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        # Capture cache hits/writes for spend tracking (Anthropic-only fields)
        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return LLMResponse(
            text=text, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok, cache_read, cache_write, cache_ttl),
            provider=self.provider, model=self.model,
        )


class OpenAILLM(LLM):
    """OpenAI native (gpt-5-nano, gpt-5, etc.)."""
    provider = "openai"

    def _call_raw(self, system, user, max_tokens, temperature, cache_ttl="5m"):
        # OpenAI auto-caches prompts >1024 tokens; no explicit cache_control
        # needed. cache_ttl arg is accepted for interface uniformity, ignored.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ConfigError(
                "openai SDK not installed. Run: pip install openai"
            ) from e
        client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **openai_token_kwargs(self.model, max_tokens, temperature),
        )
        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        return LLMResponse(
            text=text, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok),
            provider=self.provider, model=self.model,
        )


class GoogleLLM(LLM):
    provider = "google"

    def _call_raw(self, system, user, max_tokens, temperature, cache_ttl="5m"):
        # Gemini has implicit context caching; cache_ttl ignored.
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ConfigError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            ) from e
        genai.configure(api_key=self.api_key or os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )
        resp = model.generate_content(
            user,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        text = resp.text or ""
        # Gemini reports usage in resp.usage_metadata
        in_tok = getattr(resp.usage_metadata, "prompt_token_count", 0)
        out_tok = getattr(resp.usage_metadata, "candidates_token_count", 0)
        return LLMResponse(
            text=text, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok),
            provider=self.provider, model=self.model,
        )


class OpenAICompatLLM(LLM):
    """OpenAI-compatible endpoint: DeepSeek, GLM/Zhipu, Ollama, LM Studio, etc.

    Requires endpoint URL and (for cloud providers) API key. Local providers
    like Ollama don't need a real key — pass anything.
    """
    provider = "openai-compat"

    def _call_raw(self, system, user, max_tokens, temperature, cache_ttl="5m"):
        # DeepSeek / GLM caches automatically; Ollama / LM Studio have no
        # remote billing. cache_ttl accepted for interface uniformity, ignored.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ConfigError(
                "openai SDK required for openai-compat. Run: pip install openai"
            ) from e
        if not self.endpoint:
            raise ConfigError("openai-compat requires endpoint URL")
        client = OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.endpoint,
        )
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        in_tok = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
        out_tok = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        return LLMResponse(
            text=text, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=0.0,  # User-managed pricing
            provider=self.provider, model=self.model,
        )


class CallableLLM(LLM):
    """Caller-supplied function. Use to wire in MCP sampling or anything else.

    Signature: fn(system: str, user: str, max_tokens: int, temperature: float) -> str
    """
    provider = "callable"

    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn = fn
        self.model = name
        self.api_key = None
        self.endpoint = None

    def _call_raw(self, system, user, max_tokens, temperature, cache_ttl="5m"):
        # User-supplied function manages its own caching; cache_ttl ignored.
        text = self.fn(system, user, max_tokens, temperature)
        return LLMResponse(text=text, provider=self.provider, model=self.model)


# ──────── config loading ────────


def openai_token_kwargs(model: str, max_tokens: int, temperature: float) -> dict:
    """Chat Completions keyword arguments that the given OpenAI model actually accepts.

    v1.25: the reasoning models (gpt-5 family, o1/o3/o4) reject ``max_tokens`` (they want
    ``max_completion_tokens``) and reject any non-default ``temperature``. With the README's
    recommended ``gpt-5-nano`` every curator / intent call returned HTTP 400, and the caller's
    broad ``except`` turned that into "model configured, nothing stored". Reasoning models also
    spend part of the budget on hidden reasoning, so the completion cap gets headroom.
    """
    m = (model or "").lower()
    if re.match(r"^(gpt-5|o[1-9])", m):
        return {"max_completion_tokens": int(max_tokens) * 2 + 2000}
    return {"max_tokens": max_tokens, "temperature": temperature}


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        import yaml
    except ImportError as e:
        # v1.25: a config file that cannot be read is an error, not an empty config —
        # silently returning {} made a configured model look like "No LLM configured".
        raise ConfigError(
            f"{CONFIG_PATH} exists but pyyaml is not installed. Run: pip install pyyaml"
        ) from e
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def _parse_env_spec(spec: str) -> tuple[str, str]:
    """ANCHOR_LLM='anthropic/claude-haiku-4-5-20251001' -> ('anthropic', 'claude-...')"""
    if "/" not in spec:
        raise ConfigError(f"ANCHOR_LLM must be 'provider/model', got: {spec}")
    provider, model = spec.split("/", 1)
    return provider.strip(), model.strip()


def get_default_llm(override: Optional[LLM] = None) -> LLM:
    """Resolve the LLM to use. See module docstring for resolution order."""
    if override is not None:
        return override

    env_spec = os.getenv("ANCHOR_LLM")
    if env_spec:
        provider, model = _parse_env_spec(env_spec)
        return _build_llm(provider, model)

    cfg = _load_config()
    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("provider"):
        return _build_llm(
            llm_cfg["provider"],
            llm_cfg.get("model", _default_model(llm_cfg["provider"])),
            api_key=llm_cfg.get("api_key"),
            endpoint=llm_cfg.get("endpoint"),
        )

    # Fallback: Anthropic + Haiku if key is in env
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model="claude-haiku-4-5-20251001")

    raise ConfigError(
        "No LLM configured for Anchor.\n\n"
        "Options:\n"
        "  1. Run `python -m anchor_init` to set up (recommended).\n"
        "  2. Set ANCHOR_LLM env: e.g. ANCHOR_LLM='anthropic/claude-haiku-4-5-20251001'\n"
        "  3. Edit ~/.anchor/config.yaml directly.\n"
        "  4. Set ANTHROPIC_API_KEY to auto-use Anthropic Haiku.\n\n"
        "Anchor's store/search/hebbian/emotion features work WITHOUT an LLM.\n"
        "Only dream pass, concept linking, and search_multi (with intent split)\n"
        "need this configured."
    )


def _default_model(provider: str) -> str:
    return {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-5-nano",
        "google": "gemini-2.5-flash",
    }.get(provider, "")


def _build_llm(provider: str, model: str, api_key: Optional[str] = None,
               endpoint: Optional[str] = None) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model=model, api_key=api_key)
    if provider == "openai":
        return OpenAILLM(model=model, api_key=api_key)
    if provider == "google":
        return GoogleLLM(model=model, api_key=api_key)
    if provider == "openai-compat":
        return OpenAICompatLLM(model=model, api_key=api_key, endpoint=endpoint)
    raise ConfigError(f"Unknown provider: {provider}")


# ──────── spend tracking ────────


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _record_spend(resp: LLMResponse) -> None:
    if not resp.cost_usd and not resp.input_tokens:
        return
    CONFIG_DIR.mkdir(exist_ok=True)
    entry = {
        "ts": time.time(),
        "date": _today(),
        "provider": resp.provider,
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": resp.cost_usd,
    }
    try:
        with SPEND_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def today_spend_usd() -> float:
    if not SPEND_PATH.exists():
        return 0.0
    today = _today()
    total = 0.0
    try:
        with SPEND_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("date") == today:
                        total += e.get("cost_usd", 0.0)
                except Exception:
                    continue
    except Exception:
        return 0.0
    return total


def _check_daily_cap() -> None:
    cfg = _load_config()
    cap = cfg.get("safety", {}).get("max_cost_per_day_usd")
    if cap is None:
        return
    spent = today_spend_usd()
    if spent >= cap:
        raise SpendCapExceeded(
            f"Today's Anchor LLM spend is ${spent:.2f}, daily cap is ${cap:.2f}.\n"
            f"Raise the cap in ~/.anchor/config.yaml or wait for tomorrow."
        )


def estimate_cost(provider: str, model: str, input_tokens: int,
                  output_tokens: int) -> float:
    """Pre-call cost estimator. Used by dream_pass to show a heads-up."""
    rates = PRICES_PER_M.get(f"{provider}:{model}", (0.0, 0.0))
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def session_spend_summary() -> dict:
    """Read all spend.jsonl entries and return totals by date + provider."""
    if not SPEND_PATH.exists():
        return {"total": 0.0, "by_date": {}, "by_provider": {}}
    total = 0.0
    by_date: dict = {}
    by_provider: dict = {}
    try:
        with SPEND_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                c = e.get("cost_usd", 0.0)
                total += c
                d = e.get("date", "unknown")
                p = e.get("provider", "unknown")
                by_date[d] = by_date.get(d, 0.0) + c
                by_provider[p] = by_provider.get(p, 0.0) + c
    except Exception:
        pass
    return {"total": total, "by_date": by_date, "by_provider": by_provider}
