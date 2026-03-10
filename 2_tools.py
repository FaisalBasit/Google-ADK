# ---------------------------------------------------------
# GOOGLE ADK — Demo 2: Tools
# ---------------------------------------------------------
# Tools give agents "Superpowers" — the ability to call Python functions.
# You'll learn:
# 1. How to turn ANY Python function into a tool.
# 2. Why docstrings are critical for Agent reasoning.
# 3. How to pass tools to an Agent.
#
# Run:  python 2_tools.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Tool Definitions ──────────────────────────────────────

def get_weather(city: str) -> str:
    """Get the current weather for a specific city.

    Args:
        city: The name of the city (e.g., 'London', 'Tokyo').
    """
    return f"The weather in {city} is currently 25°C and sunny with a light breeze."


def calculate_tax(amount: float, rate: float = 0.1) -> float:
    """Calculate the tax amount for a given transaction total.

    Args:
        amount: The total monetary amount before tax.
        rate: The tax rate as a decimal (e.g., 0.05 for 5%). Defaults to 0.1 (10%).
    """
    return amount * rate


# ── Agent ─────────────────────────────────────────────────
weather_agent = Agent(
    name="weather_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    description="An agent capable of checking weather and performing financial calculations.",
    instruction=(
        "You are a helpful travel and finance assistant. "
        "If a user asks about the weather, use the 'get_weather' tool. "
        "If a user asks about tax or total costs, use the 'calculate_tax' tool. "
        "Always answer based on the tool results."
    ),
    tools=[get_weather, calculate_tax],
)

# ── Hardcoded questions ───────────────────────────────────
QUESTIONS = [
    "What is the weather in Tokyo right now?",
    "I bought something for $250. How much tax do I owe at a 7% rate?",
]

APP_NAME = "toolbox_app"
USER_ID = "user_1"
SESSION_ID = "session_tools"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=weather_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 2: Tools (get_weather, calculate_tax)")
    print("=" * 60)
    print(f"Agent : {weather_agent.name}")
    print(f"Model : {weather_agent.model}")
    print(f"Tools : get_weather, calculate_tax")
    print("=" * 60)

    for idx, question in enumerate(QUESTIONS, 1):
        print(f"\n📝 QUESTION {idx}:\n   {question}")
        print(f"\n[{weather_agent.model} is reasoning with tools...]\n")

        content = types.Content(role="user", parts=[types.Part(text=question)])
        final_response = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text

        print(f"🤖 ANSWER:\n{final_response if final_response else '[No response]'}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
