# ---------------------------------------------------------
# GOOGLE ADK — Demo 1: Specialized Agent (Basic)
# ---------------------------------------------------------
# Demonstrates a specialized agent answering a hardcoded question.
# You'll learn:
# 1. How to define a specialized agent (English Teacher).
# 2. How to create a session and runner.
# 3. How to send a query and receive the agent's response.
#
# Run:  python 1_agent.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Agent Definition ──────────────────────────────────────
english_teacher = Agent(
    name="english_teacher",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    description="An expert English language and writing teacher.",
    instruction=(
        "You are an expert English Teacher. "
        "Your expertise covers grammar, writing style, composition, and literary analysis. "
        "Answer questions about English language and writing with clarity and examples. "
        "Be educational, encouraging, and precise."
    ),
)

# ── Hardcoded question ────────────────────────────────────
QUESTION = (
    "What are the 5 most important principles of clear and effective English writing? "
    "Please explain each one briefly with an example."
)

# ── Constants ─────────────────────────────────────────────
APP_NAME = "english_teacher_app"
USER_ID = "user_1"
SESSION_ID = "session_001"


async def main():
    # --- Session & Runner setup ---
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=english_teacher,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Display ---
    print("=" * 60)
    print("GOOGLE ADK DEMO 1: Specialized Agent")
    print("=" * 60)
    print(f"Agent : {english_teacher.name}")
    print(f"Model : {english_teacher.model}")
    print("=" * 60)
    print(f"\n📝 QUESTION:\n   {QUESTION}")
    print(f"\n[{english_teacher.model} is thinking...]\n")

    # --- Send the question to the agent ---
    content = types.Content(role="user", parts=[types.Part(text=QUESTION)])

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    # --- Print the answer ---
    print(f"👨‍🏫 {english_teacher.name.upper()} ANSWERS:\n")
    print(final_response if final_response else "[No response received]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
