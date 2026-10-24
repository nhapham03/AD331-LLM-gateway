import asyncio
import threading
import time

import streamlit as st
from dotenv import load_dotenv

from gateway import AuthenticationError, GatewayProviderError, UnifiedLLMGateway
from pricing import ANTHROPIC_MODELS, OPENAI_MODELS

load_dotenv()
st.set_page_config(page_title="LLM Gateway", page_icon="⚡", layout="wide")


@st.cache_resource
def get_runtime():
    """One background event loop + one gateway, reused on every click."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop, UnifiedLLMGateway()


loop, gateway = get_runtime()


def stream_sync(prompt: str, model: str, temperature: float):
    """Pull chunks one at a time from the async gateway into Streamlit."""
    agen = gateway.stream(prompt, model, temperature)
    while True:
        try:
            yield asyncio.run_coroutine_threadsafe(agen.__anext__(), loop).result()
        except StopAsyncIteration:
            break


# ---------- Sidebar: settings + telemetry ----------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Anthropic"])
    models = sorted(OPENAI_MODELS if provider == "OpenAI" else ANTHROPIC_MODELS)
    model = st.selectbox("Model", models)
    temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.7, 0.1,
        disabled=(provider == "Anthropic"),
        help="Current Claude models don't accept a temperature setting.",
    )

    st.divider()
    st.header("Live telemetry")
    ttft_box = st.empty()
    speed_box = st.empty()
    tokens_box = st.empty()
    cost_box = st.empty()
    ttft_box.metric("TTFT", "—")
    speed_box.metric("Speed", "—")
    tokens_box.metric("Tokens (prompt / completion)", "—")
    cost_box.metric("Cost (USD)", "—")

# ---------- Main area: prompt + streaming output ----------
st.title("⚡ Multi-Provider LLM Gateway")
prompt = st.text_area("Prompt", "Explain what an API is in 2 sentences.")

if st.button("Send", type="primary") and prompt.strip():
    output_box = st.empty()
    text = ""
    chunks = 0
    started = time.perf_counter()
    first_at = None

    try:
        for chunk in stream_sync(prompt, model, temperature):
            if chunk.is_final:
                t = chunk.telemetry
                ttft_box.metric("TTFT", f"{t.ttft_seconds * 1000:.0f} ms")
                speed_box.metric("Speed", f"{t.tokens_per_second:.1f} tokens/s")
                tokens_box.metric(
                    "Tokens (prompt / completion)",
                    f"{t.input_tokens} / {t.output_tokens}",
                )
                cost_box.metric("Cost (USD)", f"${t.total_cost_usd:.6f}")
            else:
                now = time.perf_counter()
                if first_at is None:
                    first_at = now
                    ttft_box.metric("TTFT", f"{(now - started) * 1000:.0f} ms")
                text += chunk.delta_text
                chunks += 1
                output_box.markdown(text + "▌")
                gen_time = now - first_at
                if gen_time > 0:
                    speed_box.metric("Speed (live)", f"~{chunks / gen_time:.0f} chunks/s")
        output_box.markdown(text)
    except AuthenticationError as e:
        st.error(f"🔑 Authentication problem: {e}")
    except GatewayProviderError as e:
        st.error(f"⚠️ Provider error: {e}")