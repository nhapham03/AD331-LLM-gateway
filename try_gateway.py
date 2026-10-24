import asyncio
from dotenv import load_dotenv
load_dotenv()  # load keys from .env before the gateway starts

from src.gateway import UnifiedLLMGateway

async def main():
    gw = UnifiedLLMGateway()
    for model in ["gpt-4o-mini", "claude-haiku-5-5"]:
        print(f"\n=== {model} ===")
        async for chunk in gw.stream("Explain what an API is in 2 sentences.", model):
            print(chunk.delta_text, end="", flush=True)
            if chunk.is_final:
                print("\n\nTELEMETRY:", chunk.telemetry)

asyncio.run(main())