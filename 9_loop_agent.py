# ---------------------------------------------------------
# GOOGLE ADK — Demo 9: Loop Agent (Iterative Refinement)
# ---------------------------------------------------------
# LoopAgent repeats a cycle of agents until quality is met.
# You'll learn:
# 1. How to define a LoopAgent for iterative refinement.
# 2. The Write → Review → Revise cycle.
# 3. How max_iterations prevents infinite loops.
#
# Run:  python 9_loop_agent.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Writer ────────────────────────────────────────────────
draft_writer = Agent(
    name="draft_writer",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "Task: Write or revise a short, engaging paragraph (3-5 sentences) "
        "about the given topic.\n\n"
        "- On the FIRST pass: write a clean initial draft.\n"
        "- On SUBSEQUENT passes: you will see reviewer feedback in the "
        "conversation history. Address every point and produce an improved version.\n\n"
        "Output ONLY the paragraph text — no labels, no preamble."
    ),
)

# ── Reviewer ──────────────────────────────────────────────
draft_reviewer = Agent(
    name="draft_reviewer",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "Task: Review the most recent draft paragraph critically.\n\n"
        "- If the draft is clear, accurate, engaging, and well-written:\n"
        "  Output EXACTLY: APPROVED\n\n"
        "- If it needs improvement:\n"
        "  Output EXACTLY: NEEDS REVISION: <specific, actionable feedback>\n\n"
        "Be a demanding but fair reviewer. Push for quality."
    ),
)

# ── Loop Agent ────────────────────────────────────────────
refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[draft_writer, draft_reviewer],
    max_iterations=3,
    description="Iterative Write → Review loop that refines content quality.",
)

# ── Hardcoded topic ──────────────────────────────────────
TOPIC = "Why learning to code is one of the most valuable skills in the 21st century"

APP_NAME = "loop_app"
USER_ID = "user_1"
SESSION_ID = "session_loop"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=refinement_loop, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 9: Loop Agent (Write → Review → Revise)")
    print("=" * 60)
    print(f"Pattern     : LoopAgent (max 3 iterations)")
    print(f"Writer      : {draft_writer.name}")
    print(f"Reviewer    : {draft_reviewer.name}")
    print("=" * 60)
    print(f"\n📝 TOPIC:\n   {TOPIC}")
    print(f"\n[Loop started: write → review → repeat...]\n")

    content = types.Content(role="user", parts=[types.Part(text=TOPIC)])
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"✍️  FINAL OUTPUT AFTER REFINEMENT:\n")
    print(final_response if final_response else "[No response received]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
