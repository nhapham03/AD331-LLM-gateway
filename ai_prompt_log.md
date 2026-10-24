# AI Prompt & Audit Log

## Tool used

GitHub Copilot Chat in VS Code, Ask mode (model: gpt-6-luna), Oct 24, 2026

## Prompt

I adapted the assignment's sample prompt, replacing the retired Claude 3.5 model names with current ones (`claude-haiku-5-5`, `claude-sonnet-5-5`):

```
You are an expert Python AI Infrastructure Engineer. I need an asynchronous, streaming multi-provider SDK wrapper supporting both OpenAI (gpt-4o-mini, gpt-4o) and Anthropic (claude-haiku-5-5, claude-sonnet-5-5).

Requirements:
- Environment: Python 3.12, typed with Pydantic v2.
- A unified class `UnifiedLLMGateway` with an async generator method:
  `async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:`
- Use official async clients (`AsyncOpenAI` and `AsyncAnthropic`).

Normalization:
- A Pydantic schema `StreamChunk` with: `delta_text: str`, `provider: str`, `model: str`, `is_final: bool`, `telemetry: Optional[ChunkTelemetry]`.
- Hide all differences in chunk structure between OpenAI and Anthropic from the consumer.

Token & Cost Telemetry:
- A static pricing dictionary with USD rates per 1,000,000 input and output tokens.
- OpenAI: enable `stream_options={"include_usage": True}` and use tiktoken for prompt token verification.
- Anthropic: capture `message_start` for input tokens and output tokens from stream events.
- On the final chunk (`is_final=True`), provide input tokens, output tokens, total latency (seconds), Time-to-First-Token (TTFT), tokens per second, and total request cost (USD).

Error Handling:
- Raise typed custom exceptions (`GatewayProviderError`, `RateLimitExceededError`, `AuthenticationError`) for rate limits, invalid keys, and connection errors.
- Never leak raw API keys or unhandled stack traces in logs or outputs.

Provide code split into `schemas.py`, `pricing.py`, and `gateway.py`.
```

Follow-up message (Copilot first refused because Ask mode is read-only and it wanted to edit files directly):

```
Don't modify any files. Write the complete code for schemas.py, pricing.py, and gateway.py directly here in the chat, as three separate code blocks, one per file.
```

## Original AI output (unedited)

### schemas.py

```python
from pydantic import BaseModel, ConfigDict, Field


class ChunkTelemetry(BaseModel):
    """Request-level token and latency metrics, emitted on the final chunk."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_latency_seconds: float = Field(ge=0)
    ttft_seconds: float = Field(ge=0)
    tokens_per_second: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    usage_is_estimated: bool


class StreamChunk(BaseModel):
    """A provider-independent text chunk."""

    model_config = ConfigDict(frozen=True)

    delta_text: str
    provider: str
    model: str
    is_final: bool
    telemetry: ChunkTelemetry | None = None
```

### pricing.py

```python
from typing import TypedDict


class TokenPricing(TypedDict):
    input_usd_per_million: float
    output_usd_per_million: float


# Prices are USD per 1,000,000 tokens. Confirm these rates and model IDs
# against your provider accounts before relying on them for billing.
MODEL_PRICING: dict[str, TokenPricing] = {
    "gpt-4o-mini": {
        "input_usd_per_million": 0.15,
        "output_usd_per_million": 0.60,
    },
    "gpt-4o": {
        "input_usd_per_million": 2.50,
        "output_usd_per_million": 10.00,
    },
    "claude-haiku-5-5": {
        "input_usd_per_million": 1.00,
        "output_usd_per_million": 5.00,
    },
    "claude-sonnet-5-5": {
        "input_usd_per_million": 3.00,
        "output_usd_per_million": 15.00,
    },
}

OPENAI_MODELS = frozenset({"gpt-4o-mini", "gpt-4o"})
ANTHROPIC_MODELS = frozenset({"claude-haiku-5-5", "claude-sonnet-5-5"})
```

### gateway.py

```python
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import anthropic
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from pricing import ANTHROPIC_MODELS, MODEL_PRICING, OPENAI_MODELS
from schemas import ChunkTelemetry, StreamChunk


class GatewayProviderError(Exception):
    """A sanitized error communicating with an LLM provider."""


class RateLimitExceededError(GatewayProviderError):
    """The provider rejected a request because of rate limits."""


class AuthenticationError(GatewayProviderError):
    """The provider rejected credentials or no API key was configured."""


class UnifiedLLMGateway:
    def __init__(
        self,
        openai_client: AsyncOpenAI | None = None,
        anthropic_client: AsyncAnthropic | None = None,
    ) -> None:
        self._openai_client = openai_client
        self._anthropic_client = anthropic_client

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        if model not in MODEL_PRICING:
            raise ValueError(f"Unsupported model: {model}")

        if model in OPENAI_MODELS:
            async for chunk in self._stream_openai(prompt, model, temperature):
                yield chunk
            return

        if model in ANTHROPIC_MODELS:
            async for chunk in self._stream_anthropic(prompt, model, temperature):
                yield chunk
            return

        raise ValueError(f"Unsupported model: {model}")

    async def _stream_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        started_at = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        estimated_input_tokens = self._count_tokens(prompt, model)

        try:
            client = self._get_openai_client()
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for event in stream:
                usage = event.usage
                if usage is not None:
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens

                if not event.choices:
                    continue

                delta = event.choices[0].delta.content
                if not isinstance(delta, str) or not delta:
                    continue

                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now

                output_parts.append(delta)
                yield StreamChunk(
                    delta_text=delta,
                    provider="openai",
                    model=model,
                    is_final=False,
                )

        except Exception as exc:
            self._raise_provider_error("openai", exc)

        elapsed = time.perf_counter() - started_at
        output_text = "".join(output_parts)
        usage_is_estimated = input_tokens is None or output_tokens is None

        if input_tokens is None:
            input_tokens = estimated_input_tokens
        if output_tokens is None:
            output_tokens = self._count_tokens(output_text, model)

        yield self._final_chunk(
            provider="openai",
            model=model,
            started_at=started_at,
            first_token_at=first_token_at,
            elapsed=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_input_tokens=estimated_input_tokens,
            usage_is_estimated=usage_is_estimated,
        )

    async def _stream_anthropic(
        self,
        prompt: str,
        model: str,
        temperature: float,
    ) -> AsyncGenerator[StreamChunk, None]:
        started_at = time.perf_counter()
        first_token_at: float | None = None
        output_parts: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        estimated_input_tokens = self._count_tokens(prompt, model)

        try:
            client = self._get_anthropic_client()
            stream = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )

            async for event in stream:
                if event.type == "message_start":
                    usage = event.message.usage
                    input_tokens = usage.input_tokens

                elif event.type == "message_delta":
                    usage = event.usage
                    if usage.output_tokens is not None:
                        output_tokens = usage.output_tokens

                elif event.type == "content_block_delta":
                    text = getattr(event.delta, "text", None)
                    if not isinstance(text, str) or not text:
                        continue

                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now

                    output_parts.append(text)
                    yield StreamChunk(
                        delta_text=text,
                        provider="anthropic",
                        model=model,
                        is_final=False,
                    )

        except Exception as exc:
            self._raise_provider_error("anthropic", exc)

        elapsed = time.perf_counter() - started_at
        output_text = "".join(output_parts)
        usage_is_estimated = input_tokens is None or output_tokens is None

        if input_tokens is None:
            input_tokens = estimated_input_tokens
        if output_tokens is None:
            output_tokens = self._count_tokens(output_text, model)

        yield self._final_chunk(
            provider="anthropic",
            model=model,
            started_at=started_at,
            first_token_at=first_token_at,
            elapsed=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_input_tokens=estimated_input_tokens,
            usage_is_estimated=usage_is_estimated,
        )

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise AuthenticationError("OpenAI API key is not configured.")
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self) -> AsyncAnthropic:
        if self._anthropic_client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise AuthenticationError("Anthropic API key is not configured.")
            self._anthropic_client = AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    @staticmethod
    def _count_tokens(text: str, model: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                encoding = tiktoken.get_encoding("o200k_base")
            except ValueError:
                encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    @staticmethod
    def _final_chunk(
        *,
        provider: str,
        model: str,
        started_at: float,
        first_token_at: float | None,
        elapsed: float,
        input_tokens: int,
        output_tokens: int,
        estimated_input_tokens: int,
        usage_is_estimated: bool,
    ) -> StreamChunk:
        pricing = MODEL_PRICING[model]
        cost = (
            input_tokens * pricing["input_usd_per_million"]
            + output_tokens * pricing["output_usd_per_million"]
        ) / 1_000_000

        ttft = (
            max(0.0, first_token_at - started_at)
            if first_token_at is not None
            else elapsed
        )
        generation_seconds = max(0.0, elapsed - ttft)
        tokens_per_second = (
            output_tokens / generation_seconds if generation_seconds > 0 else 0.0
        )

        return StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_latency_seconds=elapsed,
                ttft_seconds=ttft,
                tokens_per_second=tokens_per_second,
                total_cost_usd=cost,
                estimated_input_tokens=estimated_input_tokens,
                usage_is_estimated=usage_is_estimated,
            ),
        )

    @staticmethod
    def _raise_provider_error(provider: str, exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            raise exc from None

        if provider == "openai":
            if isinstance(exc, anthropic.APIError):
                raise GatewayProviderError(
                    "Unexpected provider error while communicating with OpenAI."
                ) from None

            from openai import (
                APIConnectionError as OpenAIConnectionError,
                APIError as OpenAIAPIError,
                AuthenticationError as OpenAIAuthenticationError,
                RateLimitError as OpenAIRateLimitError,
            )

            if isinstance(exc, OpenAIRateLimitError):
                raise RateLimitExceededError("OpenAI rate limit exceeded.") from None
            if isinstance(exc, OpenAIAuthenticationError):
                raise AuthenticationError("OpenAI authentication failed.") from None
            if isinstance(exc, OpenAIConnectionError):
                raise GatewayProviderError(
                    "Could not connect to OpenAI."
                ) from None
            if isinstance(exc, OpenAIAPIError):
                raise GatewayProviderError(
                    "OpenAI returned an API error."
                ) from None

        else:
            if isinstance(exc, anthropic.RateLimitError):
                raise RateLimitExceededError(
                    "Anthropic rate limit exceeded."
                ) from None
            if isinstance(exc, anthropic.AuthenticationError):
                raise AuthenticationError(
                    "Anthropic authentication failed."
                ) from None
            if isinstance(exc, anthropic.APIConnectionError):
                raise GatewayProviderError(
                    "Could not connect to Anthropic."
                ) from None
            if isinstance(exc, anthropic.APIError):
                raise GatewayProviderError(
                    "Anthropic returned an API error."
                ) from None

        raise GatewayProviderError(
            f"Unexpected error while communicating with {provider}."
        ) from None
```

## Audit findings

**Summary:** I ran the AI's first draft against the real APIs with a small script (`try_gateway.py`) that streams the same prompt from both providers and prints the telemetry. OpenAI worked on the first try. Anthropic failed every time. I found 6 issues by running the code, reading the tracebacks, adding a temporary debug print to the error handler, and comparing the pricing table with the official pricing pages.

| # | Finding | Category | Impact |
|---|---|---|---|
| 1 | Error messages too vague to debug | Error handling | Failures could not be diagnosed |
| 2 | tiktoken estimate ignores chat overhead | Telemetry | Token estimate 41% off (Test 2 needs ±10%) |
| 3 | Catch-all hides the gateway's own bugs | Error handling | Real cause completely hidden |
| 4 | Outdated Anthropic SDK parameter | Compatibility | Every Anthropic request crashed |
| 5 | Cost not rounded | Telemetry | Floating-point noise in cost |
| 6 | Outdated Claude prices | Telemetry | Every Claude cost was wrong (Haiku 10× too high) |

---

### Finding 1: Error messages are too vague to debug

- **What I saw:** The first Anthropic failure only said `GatewayProviderError: Unexpected error while communicating with anthropic.` A later failure only said `Anthropic returned an API error.`
- **Why it's a problem:** The prompt asked the AI not to leak API keys. The AI over-corrected and threw away *all* useful information, including the error type. A good error hides secrets but keeps safe details. Exception type names like `BadRequestError` can never contain a key.
- **Before (`_raise_provider_error`, same for OpenAI and Anthropic):**

```python
raise GatewayProviderError(
    "Anthropic returned an API error."
) from None
```

- **After:**

```python
raise GatewayProviderError(
    f"Anthropic returned an API error ({type(exc).__name__})."
) from None
```

The same fix was applied to the OpenAI message: `f"OpenAI returned an API error ({type(exc).__name__})."`

---

### Finding 2: tiktoken prompt estimate ignores chat-format overhead

- **What I saw:** For the same prompt, OpenAI reported `input_tokens=17`, but the tiktoken estimate was `10` (41% off). The assignment's Test 2 requires ±10%. For Anthropic, the estimate was also `10` while Anthropic reported `23`.
- **Why it's a problem:**
  - **OpenAI:** `_count_tokens` only counts the raw prompt text. OpenAI adds hidden tokens to wrap every chat message: 3 per message, plus 1 for the role ("user"), plus 3 to start the reply. That's 7 extra tokens for one message, and 10 + 7 = 17, which matches exactly.
  - **Anthropic:** tiktoken is OpenAI's tokenizer. `_count_tokens` doesn't recognize Claude model names, so it silently falls back to OpenAI's `o200k_base` encoding. That means it was counting Claude's tokens with OpenAI's ruler. Anthropic's own count from the `message_start` event is the real number.
- **Before (`_stream_openai` and `_stream_anthropic`):**

```python
estimated_input_tokens = self._count_tokens(prompt, model)
```

- **After (`_stream_openai`):**

```python
estimated_input_tokens = self._count_tokens(prompt, model) + 7
```

- **After (`_stream_anthropic`):**

```python
estimated_input_tokens = None
...
if input_tokens is None:
    input_tokens = self._count_tokens(prompt, model)  # rough fallback only
```

The fallback line had to change too. Before, it copied `estimated_input_tokens` into `input_tokens`, which would now be `None` and crash the cost calculation.

- **Supporting changes so `None` is allowed:**

```python
# schemas.py — before
estimated_input_tokens: int = Field(ge=0)
# schemas.py — after
estimated_input_tokens: int | None = None

# gateway.py, _final_chunk parameter — before
estimated_input_tokens: int,
# after
estimated_input_tokens: int | None,
```

---

### Finding 3: Catch-all hides the gateway's own bugs as "provider errors"

- **What I saw:** In `_raise_provider_error`, Anthropic's own error types (rate limit, auth, connection, and the general `anthropic.APIError`) are all checked first. The error still fell through to the final generic message.
- **Why it's a problem:** `anthropic.APIError` catches every error that comes from Anthropic's servers. Falling past it meant the failure was a bug inside the gateway's own code, not Anthropic, but it was mislabeled as a communication error. `raise ... from None` also erases the original traceback, so the real cause was completely hidden.
- **How I found the real cause:** I added a temporary debug line at the top of `_raise_provider_error`:

```python
print("DEBUG real error:", type(exc).__name__, "-", exc)
```

It revealed a `TypeError` (see Finding 4). I removed the debug line after fixing the bug.

- **Before:**

```python
raise GatewayProviderError(
    f"Unexpected error while communicating with {provider}."
) from None
```

- **After:**

```python
raise GatewayProviderError(
    f"Unexpected {type(exc).__name__} while communicating with {provider}."
) from None
```

---

### Finding 4: AI used an outdated Anthropic SDK parameter

- **What I saw:** `TypeError: AsyncMessages.create() got an unexpected keyword argument 'temperature'`
- **Why it's a problem:** The Anthropic Python SDK 1.x removed `temperature` as a direct argument to `messages.create()`. The AI wrote code for the older SDK, so every Anthropic request crashed before it was even sent.
- **Before (`_stream_anthropic`):**

```python
stream = await client.messages.create(
    model=model,
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    stream=True,
)
```

- **Attempt 1:** The SDK's migration path is to send the field through `extra_body`:

```python
extra_body={"temperature": temperature},
```

Result: the request now reached Anthropic, but the server rejected it with `400 BadRequestError: temperature: Extra inputs are not permitted`. Current Claude models don't accept a temperature setting at all.

- **Final fix:** Removed temperature from the Anthropic request entirely. Temperature now only applies to OpenAI (documented in the README).

```python
stream = await client.messages.create(
    model=model,
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)
```

- **Lesson:** AI assistants are trained on older code and don't know about recent library or API changes. Always run the code against the versions you actually installed, and read the real error instead of guessing.

---

### Finding 5: Cost not rounded, causing floating-point noise

- **What I saw:** `total_cost_usd=3.4949999999999996e-05` instead of `0.000035`
- **Why it's a problem:** Computers can't store some decimal numbers exactly, so tiny errors creep into the math. The assignment requires micro-cent accuracy using `round(cost, 6)`, and the AI's draft didn't round.
- **Before (`_final_chunk`):**

```python
cost = (
    input_tokens * pricing["input_usd_per_million"]
    + output_tokens * pricing["output_usd_per_million"]
) / 1_000_000
```

- **After:**

```python
cost = round(
    (
        input_tokens * pricing["input_usd_per_million"]
        + output_tokens * pricing["output_usd_per_million"]
    ) / 1_000_000,
    6,
)
```

---

### Finding 6: AI used outdated prices for both Claude models

- **What I saw:** Claude Haiku cost $0.000428 for 23 input + 81 output tokens. That only matches $1.00 / $5.00 per 1M tokens, which is the price of the older Claude Haiku 4.5. The AI used older-generation prices for both new 5.5 models.
- **Why it's a problem:** The rubric requires cost verified against official published pricing. The AI can't know current prices, and its own code comment admitted it: *"Confirm these rates and model IDs against your provider accounts before relying on them for billing."*
- **Before (`pricing.py`):**

```python
"claude-haiku-5-5": {
    "input_usd_per_million": 1.00,
    "output_usd_per_million": 5.00,
},
"claude-sonnet-5-5": {
    "input_usd_per_million": 3.00,
    "output_usd_per_million": 15.00,
},
```

- **After:**

```python
"claude-haiku-5-5": {
    "input_usd_per_million": 0.10,
    "output_usd_per_million": 0.50,
},
"claude-sonnet-5-5": {
    "input_usd_per_million": 2.00,
    "output_usd_per_million": 10.00,
},
```

- **Verified:** claude.com/pricing (API tab) and the OpenAI developer pricing page, Oct 24, 2026. OpenAI prices (`gpt-4o-mini` $0.15/$0.60, `gpt-4o` $2.50/$10.00) were already correct.

---

## Result after all fixes

|                         | gpt-4o-mini | claude-haiku-5-5 |
| ----------------------- | ----------- | ---------------- |
| Input tokens            | 17          | 23               |
| Estimated input tokens  | 17          | None             |
| Output tokens           | 52          | 90               |
| TTFT                    | 1.90 s      | 0.92 s           |
| Tokens/sec              | 90.10       | 205.60           |
| Cost                    | $0.000034   | $0.000047        |

Both providers now stream through the same `UnifiedLLMGateway.stream()` method, with accurate token estimates, rounded costs, and current official prices.