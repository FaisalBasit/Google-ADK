# ---------------------------------------------------------
# GOOGLE ADK — Demo 6: Guardrails (Callbacks)
# ---------------------------------------------------------
# Callbacks let you monitor and control agent behavior at every step.
# You'll learn:
# 1. How to hook into the Agent lifecycle with callback FUNCTIONS.
# 2. How before_model_callback can block unsafe requests.
# 3. How callbacks are used for logging and safety.
#
# Run:  python 6_guardrails.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Blocked keywords (safety guardrail) ──────────────────
BLOCKED_WORDS = ["hack", "exploit", "bypass security"]


def safety_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """before_model_callback — inspects the latest user message.

    If a blocked word is found, returns a canned LlmResponse that short-circuits
    the LLM call entirely. Otherwise returns None to let the call proceed.
    """
    # Get the last user message text from the request
    last_user_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                last_user_text = content.parts[0].text or ""
                break

    for word in BLOCKED_WORDS:
        if word.lower() in last_user_text.lower():
            print(f"  🚫 GUARDRAIL BLOCKED: detected '{word}' in request!")
            # Return a canned response to block the LLM call
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                "I'm sorry, but I can't assist with that request. "
                                "It was blocked by the safety guardrail."
                            )
                        )
                    ],
                )
            )

    print("  ✅ GUARDRAIL PASSED: request is safe.")
    return None  # Let the LLM proceed normally


# ── Agent with guardrail callback ────────────────────────
guarded_agent = Agent(
    name="guarded_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    description="An agent demonstrating safety guardrails through callbacks.",
    instruction="You are a helpful, security-conscious assistant. Answer user questions clearly.",
    before_model_callback=safety_guardrail,
)

# ── Hardcoded questions (one safe, one blocked) ──────────
QUESTIONS = [
    "What is the capital of France?",
    "How do I hack into a WiFi network?",
]

APP_NAME = "guarded_app"
USER_ID = "user_1"


async def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=guarded_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 6: Guardrails (before_model_callback)")
    print("=" * 60)
    print(f"Agent   : {guarded_agent.name}")
    print(f"Model   : {guarded_agent.model}")
    print(f"Blocked : {BLOCKED_WORDS}")
    print("=" * 60)

    for idx, question in enumerate(QUESTIONS, 1):
        session_id = f"session_guard_{idx}"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        )

        print(f"\n📝 QUESTION {idx}:\n   {question}")
        print(f"\n[Checking guardrail...]")

        content = types.Content(role="user", parts=[types.Part(text=question)])
        final_response = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text

        print(f"\n🤖 ANSWER:\n{final_response if final_response else '[No response]'}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
