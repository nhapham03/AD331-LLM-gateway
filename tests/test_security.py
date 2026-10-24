import traceback
from types import SimpleNamespace as ns

import pytest

from gateway import AuthenticationError, GatewayProviderError, UnifiedLLMGateway

FAKE_SECRET = "sk-test-SECRET-123456789"
MODELS = [("gpt-4o-mini", "OPENAI_API_KEY"), ("claude-haiku-5-5", "ANTHROPIC_API_KEY")]


async def drain(gateway, model):
    async for _ in gateway.stream("hi", model):
        pass


# ---------- Test 4: Missing or malformed keys ----------

@pytest.mark.parametrize("model, env_var", MODELS)
async def test_missing_key_raises_authentication_error(monkeypatch, model, env_var):
    monkeypatch.delenv(env_var, raising=False)  # pretend the key isn't set

    with pytest.raises(AuthenticationError, match="not configured"):
        await drain(UnifiedLLMGateway(), model)


@pytest.mark.parametrize("model, env_var", MODELS)
async def test_malformed_key_raises_authentication_error(monkeypatch, model, env_var):
    monkeypatch.setenv(env_var, FAKE_SECRET)  # real API rejects it (free, needs internet)

    with pytest.raises(AuthenticationError) as err:
        await drain(UnifiedLLMGateway(), model)

    assert FAKE_SECRET not in str(err.value)


# ---------- Test 5: Secret redaction ----------

class LeakyClient:
    """A fake client whose error message contains the API key (worst case)."""

    def __init__(self):
        async def create(**kwargs):
            raise RuntimeError(f"request failed with api_key={FAKE_SECRET}")
        self.chat = ns(completions=ns(create=create))
        self.messages = ns(create=create)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-haiku-5-5"])
async def test_errors_never_contain_api_key(model, capsys):
    gateway = UnifiedLLMGateway(openai_client=LeakyClient(), anthropic_client=LeakyClient())

    with pytest.raises(GatewayProviderError) as err:
        await drain(gateway, model)

    full_traceback = "".join(traceback.format_exception(err.value))
    printed = capsys.readouterr()

    for text in (str(err.value), repr(err.value), full_traceback, printed.out, printed.err):
        assert FAKE_SECRET not in text