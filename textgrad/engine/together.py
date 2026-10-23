try:
    from together import Together
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install together`, and add 'TOGETHER_API_KEY' to your environment variables.")

import json
import math
import os
import warnings

import platformdirs
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import EngineLM, CachedEngine


class EmptyTogetherResponseError(RuntimeError):
    """Raised when Together returns a completion without answer content."""


class UnexpectedTogetherReasoningError(RuntimeError):
    """Raised when Together returns reasoning despite an explicit off request."""


def _is_transient_together_error(error):
    """Return whether the exhausted SDK error is safe to score as unavailable."""
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    return error.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _handle_retry_exhaustion(retry_state):
    """Turn repeated content-less completions into one auditable failed answer."""
    error = retry_state.outcome.exception()
    if not isinstance(error, EmptyTogetherResponseError):
        raise error

    engine = retry_state.args[0]
    prompt = retry_state.args[1]
    system_prompt = retry_state.kwargs.get("system_prompt")
    system_prompt = system_prompt if system_prompt else engine.system_prompt
    cache_key = engine._request_cache_key(
        prompt,
        system_prompt=system_prompt,
        temperature=retry_state.kwargs.get("temperature", 0),
        max_tokens=retry_state.kwargs.get("max_tokens"),
        top_p=retry_state.kwargs.get("top_p", 0.99),
    )
    engine._save_cache(cache_key, "")
    warnings.warn(
        f"{error} after {retry_state.attempt_number} attempts; cached an empty "
        "string so the deterministic evaluator records an incorrect answer "
        "instead of aborting the experiment.",
        RuntimeWarning,
        stacklevel=2,
    )
    return ""


class ChatTogether(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_TRANSIENT_ERROR_POLICY = "raise"
    DEFAULT_LEGACY_CACHE_READ = False

    def __init__(
        self,
        model_string="meta-llama/Llama-3-70b-chat-hf",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_tokens=DEFAULT_MAX_TOKENS,
        request_timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        reasoning_enabled=None,
        transient_error_policy=DEFAULT_TRANSIENT_ERROR_POLICY,
        legacy_cache_read=DEFAULT_LEGACY_CACHE_READ,
    ):
        """
        :param model_string:
        :param system_prompt:
        :param max_tokens: Maximum number of tokens in a completion.
        :param request_timeout: Per-attempt Together request timeout in seconds.
        :param max_retries: SDK retries for timeouts, connection errors, and HTTP 429/5xx.
        :param reasoning_enabled: Explicit Together reasoning toggle, or None for provider default.
        :param transient_error_policy: Raise exhausted transient errors or return an empty answer.
        :param legacy_cache_read: Read pre-v2 prompt-only cache keys without writing them.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_together_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if os.getenv("TOGETHER_API_KEY") is None:
            raise ValueError("Please set the TOGETHER_API_KEY environment variable if you'd like to use OpenAI models.")
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
        if transient_error_policy not in {"raise", "score-zero"}:
            raise ValueError(
                "transient_error_policy must be 'raise' or 'score-zero'"
            )
        if not isinstance(legacy_cache_read, bool):
            raise ValueError("legacy_cache_read must be True or False")

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.reasoning_enabled = reasoning_enabled
        self.transient_error_policy = transient_error_policy
        self.legacy_cache_read = legacy_cache_read
        self.skipped_transient_requests = 0
        self.legacy_cache_hits = 0
        self._cache_bypass_once = set()
        self.client = Together(
            api_key=os.getenv("TOGETHER_API_KEY"),
            timeout=self.request_timeout,
            # The SDK honors Retry-After headers for 429/5xx responses. Keep
            # transient transport retries here; the outer retry below handles
            # only successful HTTP responses whose answer content is missing.
            max_retries=self.max_retries,
        )
        self.model_string = model_string
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens

    def _request_cache_key(
        self,
        prompt,
        system_prompt=None,
        temperature=0,
        max_tokens=None,
        top_p=0.99,
    ):
        """Return a cache key that includes all answer-affecting request settings."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        max_tokens_arg = self.max_tokens if max_tokens is None else max_tokens
        return json.dumps(
            {
                "schema": "together-chat-v2",
                "model": self.model_string,
                "system_prompt": sys_prompt_arg,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens_arg,
                "top_p": top_p,
                "seed": 42,
                "reasoning_enabled": self.reasoning_enabled,
                "transient_error_policy": getattr(
                    self, "transient_error_policy", "raise"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _legacy_request_cache_key(self, prompt, system_prompt=None):
        """Return the prompt-only key used before together-chat-v2."""
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        return sys_prompt_arg + prompt

    def bypass_cache_once(
        self,
        prompt,
        system_prompt=None,
        temperature=0,
        max_tokens=None,
        top_p=0.99,
    ):
        """Bypass both current and legacy cache reads for the next matching call."""
        cache_key = self._request_cache_key(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        if not hasattr(self, "_cache_bypass_once"):
            self._cache_bypass_once = set()
        self._cache_bypass_once.add(cache_key)

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=None, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        max_tokens_arg = self.max_tokens if max_tokens is None else max_tokens
        cache_key = self._request_cache_key(
            prompt,
            system_prompt=sys_prompt_arg,
            temperature=temperature,
            max_tokens=max_tokens_arg,
            top_p=top_p,
        )

        bypass_cache = cache_key in getattr(self, "_cache_bypass_once", set())
        if bypass_cache:
            self._cache_bypass_once.discard(cache_key)
        else:
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none
            if getattr(self, "legacy_cache_read", False):
                legacy_cache_key = self._legacy_request_cache_key(
                    prompt,
                    system_prompt=sys_prompt_arg,
                )
                cache_or_none = self._check_cache(legacy_cache_key)
                if cache_or_none is not None:
                    self.legacy_cache_hits = getattr(self, "legacy_cache_hits", 0) + 1
                    return cache_or_none

        request_kwargs = dict(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens_arg,
            top_p=top_p,
            seed=42,
        )
        if self.reasoning_enabled is not None:
            request_kwargs["reasoning"] = {"enabled": self.reasoning_enabled}
        response = self.client.chat.completions.create(**request_kwargs)

        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        reasoning = getattr(message, "reasoning", None)
        if reasoning is None:
            reasoning = getattr(message, "reasoning_content", None)
        if self.reasoning_enabled is False and reasoning:
            raise UnexpectedTogetherReasoningError(
                f"Together model {self.model_string!r} returned reasoning even "
                "though reasoning was explicitly disabled"
            )
        if content is None:
            finish_reason = getattr(choice, "finish_reason", None)
            raise EmptyTogetherResponseError(
                f"Together model {self.model_string!r} returned content=None "
                f"(finish_reason={finish_reason!r}, "
                f"reasoning_present={bool(reasoning)})"
            )

        self._save_cache(cache_key, content)
        return content

    @retry(
        retry=retry_if_exception_type(EmptyTogetherResponseError),
        wait=wait_random_exponential(min=1, max=5),
        stop=stop_after_attempt(5),
        retry_error_callback=_handle_retry_exhaustion,
    )
    def __call__(self, prompt, **kwargs):
        try:
            return self.generate(prompt, **kwargs)
        except Exception as error:
            if (
                getattr(self, "transient_error_policy", "raise") == "score-zero"
                and _is_transient_together_error(error)
            ):
                self.skipped_transient_requests = (
                    getattr(self, "skipped_transient_requests", 0) + 1
                )
                warnings.warn(
                    "Together request exhausted its SDK retry budget; returning "
                    "an empty answer so this sample is scored as incorrect and "
                    "the experiment can continue "
                    f"(error_type={error.__class__.__name__}, "
                    f"status_code={getattr(error, 'status_code', None)!r}, "
                    f"skipped_count={self.skipped_transient_requests}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return ""
            raise
