import asyncio, os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

load_dotenv()  # reads your .env file

async def main():
    oa = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    r1 = await oa.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi in 3 words"}],
    )
    print("OpenAI:", r1.choices[0].message.content)

    an = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    r2 = await an.messages.create(
        model="claude-haiku-5-5", max_tokens=50,
        messages=[{"role": "user", "content": "Say hi in 3 words"}],
    )
    print("Anthropic:", r2.content[0].text)

asyncio.run(main())