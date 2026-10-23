"""OpenRouter text completions with provider-isolated, parameter-aware caching."""

import json
import math
import os
import re
import threading
import warnings
from pathlib import Path
from typing import Optional

import platformdirs
from openai import APIConnectionError, APIStatusError, OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .base import CachedEngine, EngineLM


class EmptyOpenRouterResponseError(RuntimeError):
    """The provider returned no final answer (reasoning alone is not an answer)."""


class UnexpectedOpenRouterReasoningError(RuntimeError):
    """The provider returned reasoning despite an explicit off request."""


TRACE_SCHEMA = "openrouter-generation-trace-v1"


def _plain(value):
    """JSON-safe copy of an SDK object (usage, reasoning details)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _normalize_reasoning(raw):
    """Return (readable reasoning text or None, status)."""
    if raw is None:
        return None, "absent"
    if isinstance(raw, str):
        return (raw, "available") if raw.strip() else (None, "absent")
    if isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            item = _plain(item)
            text = item if isinstance(item, str) else (
                (item.get("text") or item.get("summary") or item.get("content"))
                if isinstance(item, dict) else None
            )
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return ("\n".join(parts), "available") if parts else (None, "unreadable")
    return None, "unreadable"


def _handle_empty_exhaustion(retry_state):
    engine = retry_state.args[0]
    cache_key = engine._request_cache_key(
        retry_state.args[1], **retry_state.kwargs
    )
    engine._save_cache(cache_key, "")
    if engine.capture_traces:
        # Keep the last empty attempt's reasoning: it shows where the budget went.
        pending = getattr(engine._trace_local, "pending_empty", None) or {}
        engine._trace_local.trace = {
            "schema": TRACE_SCHEMA, **pending, "content": "",
            "status": "empty_after_retries", "attempts": retry_state.attempt_number,
        }
    warnings.warn(
        f"OpenRouter returned no final answer after {retry_state.attempt_number} "
        "attempts; cached an empty string so the deterministic evaluator "
        "records an incorrect answer.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ""


class ChatOpenRouter(EngineLM, CachedEngine):
    """Call OpenRouter using OPENROUTER_API_KEY, independently of the critic key.

    Transport retries belong to the OpenAI SDK; only successful responses with
    missing final content are retried here (five attempts, then cached empty).
    Exhausted transient errors may be scored as zero but are never cached.
    Authentication, billing, model, and other configuration errors propagate.

    reasoning_effort pins low/medium/high independently of the provider default.
    It cannot be combined with reasoning_enabled=False.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "openai/gpt-oss-20b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 2000,
        request_timeout: float = 300.0,
        max_retries: int = 5,
        reasoning_enabled=None,
        transient_error_policy: str = "raise",
        provider_sort: str = "provider-default",
        provider: str = "auto",
        reasoning_effort: Optional[str] = None,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")
        if not isinstance(model_string, str) or not model_string.strip():
            raise ValueError("model_string must be a non-empty OpenRouter model ID")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        try:
            request_timeout = float(request_timeout)
        except (TypeError, ValueError) as error:
            raise ValueError("request_timeout must be a positive number") from error
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be a positive number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if reasoning_enabled is not None and not isinstance(reasoning_enabled, bool):
            raise ValueError("reasoning_enabled must be True, False, or None")
        if reasoning_effort is not None and reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium, high, or None")
        if reasoning_effort is not None and reasoning_enabled is False:
            raise ValueError("reasoning_effort cannot be combined with reasoning_enabled=False")
        if transient_error_policy not in {"raise", "score-zero"}:
            raise ValueError("transient_error_policy must be 'raise' or 'score-zero'")
        if provider_sort not in {"provider-default", "throughput", "latency", "price"}:
            raise ValueError(
                "provider_sort must be provider-default, throughput, latency, or price"
            )
        if not isinstance(provider, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*", provider
        ):
            raise ValueError("provider must be 'auto' or a lowercase OpenRouter provider slug")

        self.model_string = model_string
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self.transient_error_policy = transient_error_policy
        self.provider_sort = provider_sort
        self.provider = provider
        self.skipped_transient_requests = 0
        self._cache_bypass_once = set()
        # Opt-in generation traces (reasoning, finish_reason, usage) for fresh
        # completions only. Off by default: requests, cache keys and cached
        # values are unchanged either way; traces are never persisted.
        self.capture_traces = False
        self._trace_local = threading.local()
        # Explicit credentials and URL keep OPENAI_API_KEY / OPENAI_BASE_URL
        # settings for the critic from affecting this target model.
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        # Hash the model ID for a portable cache directory with no path traversal.
        model_key = self._hash_prompt(model_string)
        cache_path = Path(platformdirs.user_cache_dir("textgrad")) / (
            f"cache_openrouter_{model_key}.db"
        )
        super().__init__(cache_path=str(cache_path))

    def _request_cache_key(
        self, prompt, system_prompt=None, temperature=0, max_tokens=None, top_p=0.99
    ):
        """Include provider and every submitted answer-affecting setting."""
        return json.dumps(
            {
                "schema": "openrouter-chat-v1",
                "base_url": self.BASE_URL,
                "model": self.model_string,
                "system_prompt": system_prompt if system_prompt else self.system_prompt,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "top_p": top_p,
                "seed": 42,
                "reasoning_enabled": self.reasoning_enabled,
                # Preserve old default/on/off keys; explicit effort is isolated.
                **(
                    {"reasoning_effort": self.reasoning_effort}
                    if self.reasoning_effort is not None else {}
                ),
                "transient_error_policy": self.transient_error_policy,
                # Omit the default to preserve existing default-routing keys.
                **(
                    {"provider_sort": self.provider_sort}
                    if self.provider_sort != "provider-default" else {}
                ),
                **({"provider": self.provider} if self.provider != "auto" else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def bypass_cache_once(
        self, prompt, system_prompt=None, temperature=0, max_tokens=None, top_p=0.99
    ):
        """Refresh one matching request, as required by sampled evaluation."""
        self._cache_bypass_once.add(self._request_cache_key(
            prompt, system_prompt, temperature, max_tokens, top_p
        ))

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=None, top_p=0.99
    ):
        system_prompt = system_prompt if system_prompt else self.system_prompt
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        cache_key = self._request_cache_key(
            prompt, system_prompt, temperature, max_tokens, top_p
        )
        if cache_key not in self._cache_bypass_once:
            cached = self._check_cache(cache_key)
            if cached is not None:
                if self.capture_traces:
                    # A cached string has no recorded provenance: never attach reasoning.
                    self._trace_local.trace = {
                        "schema": TRACE_SCHEMA, "content": cached, "status": "cache_hit",
                    }
                return cached

        request = dict(
            model=self.model_string,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=42,
            frequency_penalty=0,
            presence_penalty=0,
        )
        extra_body = {}
        reasoning_request = {}
        if self.reasoning_enabled is not None:
            reasoning_request["enabled"] = self.reasoning_enabled
        if self.reasoning_effort is not None:
            reasoning_request["effort"] = self.reasoning_effort
        if reasoning_request:
            extra_body["reasoning"] = reasoning_request
        provider_request = {}
        if self.provider_sort != "provider-default":
            provider_request["sort"] = self.provider_sort
        if self.provider != "auto":
            provider_request.update(only=[self.provider], allow_fallbacks=False)
        if provider_request:
            extra_body["provider"] = provider_request
        if extra_body:
            request["extra_body"] = extra_body
        response = self.client.chat.completions.create(**request)
        # A provider may deliver an error envelope even with HTTP 200. Fail
        # explicitly instead of misclassifying it as a cacheable empty answer.
        if getattr(response, "error", None) is not None:
            raise RuntimeError("OpenRouter returned an error envelope instead of a completion")
        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        reasoning = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning_details", None)
        )
        if self.reasoning_enabled is False and reasoning:
            raise UnexpectedOpenRouterReasoningError(
                "OpenRouter returned reasoning even though reasoning was explicitly disabled"
            )
        trace = None
        if self.capture_traces:
            reasoning_text, reasoning_status = _normalize_reasoning(reasoning)
            trace = {
                "reasoning": reasoning_text,
                "reasoning_status": reasoning_status,
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": _plain(getattr(response, "usage", None)),
                "response_id": getattr(response, "id", None),
            }
        if content is None or (isinstance(content, str) and not content.strip()):
            if trace is not None:
                self._trace_local.pending_empty = trace
            raise EmptyOpenRouterResponseError(
                "OpenRouter returned no final answer "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r})"
            )
        if not isinstance(content, str):
            raise TypeError("OpenRouter final content must be a string")
        self._save_cache(cache_key, content)
        if trace is not None:
            self._trace_local.trace = {
                "schema": TRACE_SCHEMA, **trace, "content": content, "status": "fresh",
            }
        self._cache_bypass_once.discard(cache_key)
        return content

    def generate_with_trace(self, prompt, **kwargs):
        """Return ``(content, trace)`` from one call on this thread.

        Reasoning is attached only to a completion fetched by this very call
        (passed back through thread-local state, never looked up by key), so
        concurrent or repeated rollouts of one request cannot exchange traces.
        Cache hits carry no reasoning.
        """
        local = self._trace_local
        local.trace = None
        local.pending_empty = None
        content = self(prompt, **kwargs)
        trace = getattr(local, "trace", None)
        if not isinstance(trace, dict) or trace.get("content") != content:
            trace = {"schema": TRACE_SCHEMA, "content": content, "status": "unavailable"}
        return content, dict(trace)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_trace_local", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._trace_local = threading.local()

    @retry(
        retry=retry_if_exception_type(EmptyOpenRouterResponseError),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(5),
        retry_error_callback=_handle_empty_exhaustion,
    )
    def _call_with_retries(self, prompt, **kwargs):
        try:
            return self.generate(prompt, **kwargs)
        except (APIConnectionError, APIStatusError) as error:
            status = getattr(error, "status_code", None)
            transient = isinstance(error, APIConnectionError) or (
                status in {408, 409, 429}
                or (isinstance(status, int) and status >= 500)
            )
            if self.transient_error_policy != "score-zero" or not transient:
                raise
            self.skipped_transient_requests += 1
            if self.capture_traces:
                self._trace_local.trace = {
                    "schema": TRACE_SCHEMA, "content": "", "status": "transport_error",
                }
            warnings.warn(
                "OpenRouter request exhausted its SDK retry budget; returning "
                "an empty answer so this sample is scored as incorrect "
                f"(error_type={type(error).__name__}, status_code={status!r}, "
                f"skipped_count={self.skipped_transient_requests}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return ""

    def __call__(self, prompt, **kwargs):
        cache_key = self._request_cache_key(prompt, **kwargs)
        try:
            return self._call_with_retries(prompt, **kwargs)
        finally:
            # A refresh must bypass the old answer during all empty retries,
            # then restore ordinary cache reads after success or failure.
            self._cache_bypass_once.discard(cache_key)
