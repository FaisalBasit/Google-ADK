# ---------------------------------------------------------
# GOOGLE ADK — Demo 7: Session & Memory (Multi-Turn)
# ---------------------------------------------------------
# Agents need memory to handle multi-turn conversations.
# You'll learn:
# 1. How InMemorySessionService preserves conversation history.
# 2. How an agent remembers earlier messages within a session.
# 3. Multi-turn auto-demo with hardcoded messages.
#
# Run:  python 7_session_state.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Agent ─────────────────────────────────────────────────
chat_agent = Agent(
    name="chat_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "You are a memory assistant. Remember every detail the user tells you. "
        "When the user asks about something they mentioned earlier, recall it accurately."
    ),
)

# ── Hardcoded multi-turn conversation ────────────────────
MESSAGES = [
    "Hi! My name is Faisal and my favorite color is blue.",
    "I work as a machine learning engineer and I love hiking.",
    "What is my name and what do I do for a living?",
    "What is my favorite color and hobby?",
]

APP_NAME = "memory_demo_app"
USER_ID = "user_1"
SESSION_ID = "session_memory"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=chat_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 7: Session & Memory (Multi-Turn)")
    print("=" * 60)
    print(f"Agent   : {chat_agent.name}")
    print(f"Model   : {chat_agent.model}")
    print(f"Memory  : InMemorySessionService (same session across turns)")
    print("=" * 60)

    for idx, message in enumerate(MESSAGES, 1):
        print(f"\n👤 TURN {idx}:\n   {message}")
        print(f"\n[{chat_agent.model} processing...]\n")

        content = types.Content(role="user", parts=[types.Part(text=message)])
        final_response = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text

        print(f"🤖 AGENT:\n{final_response if final_response else '[No response]'}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
