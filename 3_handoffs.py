# ---------------------------------------------------------
# GOOGLE ADK — Demo 3: Multi-Agent Handoffs (Dynamic)
# ---------------------------------------------------------
# A coordinator agent dynamically decides which specialist to
# hand off to based on the user's request.
# You'll learn:
# 1. How to define specialized sub-agents.
# 2. How to use a coordinator that hands off to sub-agents at runtime.
# 3. How the LLM decides which agent should handle the request.
#
# Run:  python 3_handoffs.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

MODEL = LiteLlm(model="openrouter/google/gemini-2.0-flash-001")

# ── Specialists ───────────────────────────────────────────
researcher = Agent(
    name="researcher",
    model=MODEL,
    description="Handles research requests — finding facts, data, and information about a topic.",
    instruction=(
        "You are a research specialist. Find the most important facts about "
        "the user's topic. Be concise, factual, and well-organized."
    ),
)

writer = Agent(
    name="writer",
    model=MODEL,
    description="Handles writing requests — crafting summaries, essays, paragraphs, and creative text.",
    instruction=(
        "You are a writing specialist. Take the user's request and produce "
        "a beautiful, engaging piece of writing."
    ),
)

# ── Coordinator (decides who to hand off to) ─────────────
coordinator = Agent(
    name="coordinator",
    model=MODEL,
    instruction=(
        "You are a coordinator. Based on the user's request, decide which "
        "specialist should handle it:\n"
        "- If the user wants facts, data, or information → hand off to 'researcher'.\n"
        "- If the user wants writing, summaries, or creative text → hand off to 'writer'.\n\n"
        "Do NOT answer directly. Always delegate to the appropriate specialist."
    ),
    sub_agents=[researcher, writer],
)

# ── Hardcoded queries ────────────────────────────────────
QUERIES = [
    "Find the key facts about the Transformer architecture in AI.",
    "Write a short engaging paragraph about the future of space exploration.",
]

APP_NAME = "handoff_app"
USER_ID = "user_1"
SESSION_ID = "session_handoff"


async def main():
    session_service = InMemorySessionService()

    print("=" * 60)
    print("GOOGLE ADK DEMO 3: Dynamic Agent Handoff")
    print("=" * 60)
    print(f"Coordinator : {coordinator.name}")
    print(f"Specialists : {researcher.name}, {writer.name}")
    print("=" * 60)

    for i, query in enumerate(QUERIES):
        sid = f"{SESSION_ID}_{i}"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=sid,
        )
        runner = Runner(
            agent=coordinator, app_name=APP_NAME, session_service=session_service,
        )

        print(f"\n📝 QUERY {i+1}:\n   {query}")
        print(f"\n[Coordinator is deciding who to hand off to...]\n")

        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response = ""
        agent_name = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=sid, new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                    agent_name = event.author

        print(f"🤖 Handled by: {agent_name}")
        print(f"📄 Response:\n")
        print(final_response if final_response else "[No response received]")
        print("\n" + "-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
