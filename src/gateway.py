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
        estimated_input_tokens = self._count_tokens(prompt, model) + 7

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
        estimated_input_tokens = None

        try:
            client = self._get_anthropic_client()
            stream = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
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
            input_tokens = self._count_tokens(prompt, model)
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
        estimated_input_tokens: int | None,
        usage_is_estimated: bool,
    ) -> StreamChunk:
        pricing = MODEL_PRICING[model]
        cost = round(
    (
        input_tokens * pricing["input_usd_per_million"]
        + output_tokens * pricing["output_usd_per_million"]
    ) / 1_000_000,
    6,
)

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
                    f"OpenAI returned an API error ({type(exc).__name__})."
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
                    f"Anthropic returned an API error ({type(exc).__name__})."
                ) from None
        raise GatewayProviderError(
            f"Unexpected {type(exc).__name__} while communicating with {provider}."
        ) from None