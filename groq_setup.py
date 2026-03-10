# ---------------------------------------------------------
# GOOGLE ADK — Demo: Model Flexibility (LiteLLM / Groq)
# ---------------------------------------------------------
# Google ADK is model-agnostic. You can switch models without changing logic.
# You'll learn:
# 1. Native Gemini support.
# 2. How LiteLLM enables 200+ other models.
# 3. The 'provider/model' naming convention.
#
# Run:  python groq_setup.py
#
# NOTE: This demo runs with Gemini by default.
#       To use Groq, set GROQ_API_KEY in your .env file and uncomment
#       the groq_agent section below.

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Native Gemini agent ──────────────────────────────────
gemini_agent = Agent(
    name="gemini_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction="Answer concisely and informatively.",
)

# ── LiteLLM-backed agent (Groq example — uncomment to use) ──
# Requires GROQ_API_KEY in .env
# groq_agent = Agent(
#     name="groq_agent",
#     model=LiteLlm(model="groq/llama-3.1-8b-instant"),
#     instruction="Be extremely concise.",
# )

# ── Hardcoded question ───────────────────────────────────
QUESTION = "Explain the difference between supervised and unsupervised learning in 3 sentences."

APP_NAME = "model_flex_app"
USER_ID = "user_1"
SESSION_ID = "session_flex"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=gemini_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO: Model Flexibility (LiteLLM)")
    print("=" * 60)
    print(f"Active Model : {gemini_agent.model}")
    print(f"LiteLLM      : Available (uncomment groq_agent to test)")
    print("=" * 60)
    print(f"\n📝 QUESTION:\n   {QUESTION}")
    print(f"\n[{gemini_agent.model} is thinking...]\n")

    content = types.Content(role="user", parts=[types.Part(text=QUESTION)])
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"🤖 ANSWER:\n{final_response if final_response else '[No response]'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
