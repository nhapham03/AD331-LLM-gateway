import asyncio
import os
from types import SimpleNamespace as ns

import pytest
import tiktoken
from dotenv import load_dotenv

from gateway import UnifiedLLMGateway
from schemas import StreamChunk

load_dotenv()


# ---------- Fake providers (no network, no cost) ----------

async def fake_stream(events):
    for event in events:
        await asyncio.sleep(0.01)  # pretend network delay
        yield event


class FakeOpenAI:
    def __init__(self, events):
        async def create(**kwargs):
            return fake_stream(events)
        self.chat = ns(completions=ns(create=create))


class FakeAnthropic:
    def __init__(self, events):
        async def create(**kwargs):
            return fake_stream(events)
        self.messages = ns(create=create)


# Shaped exactly like each real API's stream
OPENAI_EVENTS = [
    ns(usage=None, choices=[ns(delta=ns(content="Hello"))]),
    ns(usage=None, choices=[ns(delta=ns(content=" world"))]),
    ns(usage=ns(prompt_tokens=20, completion_tokens=10), choices=[]),  # final usage chunk
]

ANTHROPIC_EVENTS = [
    ns(type="message_start", message=ns(usage=ns(input_tokens=20))),
    ns(type="content_block_delta", delta=ns(text="Hello")),
    ns(type="content_block_delta", delta=ns(text=" world")),
    ns(type="message_delta", usage=ns(output_tokens=10)),
    ns(type="message_stop"),
]


async def collect(model):
    gateway = UnifiedLLMGateway(
        openai_client=FakeOpenAI(OPENAI_EVENTS),
        anthropic_client=FakeAnthropic(ANTHROPIC_EVENTS),
    )
    return [chunk async for chunk in gateway.stream("Say hello", model)]


# ---------- Test 1: Unified stream protocol ----------

@pytest.mark.parametrize(
    "model, provider",
    [("gpt-4o-mini", "openai"), ("claude-haiku-5-5", "anthropic")],
)
async def test_unified_stream_protocol(model, provider):
    chunks = await collect(model)

    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert all(c.provider == provider and c.model == model for c in chunks)
    assert "".join(c.delta_text for c in chunks) == "Hello world"
    assert chunks[-1].is_final
    assert sum(c.is_final for c in chunks) == 1


# ---------- Test 2: Token count validation (real OpenAI call) ----------

FIFTY_WORDS = (
    "Large language models generate text one token at a time. A gateway hides "
    "differences between providers so applications can switch models easily. It "
    "also measures latency, throughput, and cost for every request. Good engineering "
    "means verifying generated code, testing edge cases, and protecting secrets such "
    "as API keys from leaks."
)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs a real OpenAI key")
async def test_prompt_token_count_matches_tiktoken():
    assert len(FIFTY_WORDS.split()) == 50

    chunks = [c async for c in UnifiedLLMGateway().stream(FIFTY_WORDS, "gpt-4o-mini")]
    telemetry = chunks[-1].telemetry

    # tiktoken count + 7 tokens of OpenAI chat-format overhead (audit Finding 2)
    expected = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(FIFTY_WORDS)) + 7

    assert telemetry.estimated_input_tokens == expected
    assert abs(telemetry.input_tokens - expected) <= 0.10 * telemetry.input_tokens


# ---------- Test 3: Telemetry sanity ----------

@pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-haiku-5-5"])
async def test_telemetry_sanity(model):
    final = (await collect(model))[-1]
    t = final.telemetry

    assert final.is_final and t is not None
    assert t.total_cost_usd > 0
    assert t.tokens_per_second > 0
    assert t.ttft_seconds * 1000 > 0  # TTFT in milliseconds