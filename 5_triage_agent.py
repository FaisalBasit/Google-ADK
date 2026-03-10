# ---------------------------------------------------------
# GOOGLE ADK — Demo 5: Triage / Routing Agent
# ---------------------------------------------------------
# A triage agent classifies queries and routes them to specialist agents.
# You'll learn:
# 1. How TRIAGE works as a routing pattern with real handoffs.
# 2. How an LLM agent dynamically delegates to sub-agents.
# 3. Specialist agents handle and respond to routed queries.
#
# Run:  python 5_triage_agent.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

MODEL = LiteLlm(model="openrouter/google/gemini-2.0-flash-001")

# ── Specialist Agents ────────────────────────────────────
billing_agent = Agent(
    name="billing_agent",
    model=MODEL,
    description="Handles BILLING queries — invoices, payments, pricing, subscriptions, and charges.",
    instruction=(
        "You are a billing support specialist. Help the user with their "
        "billing issue. Be empathetic, clear, and provide actionable steps."
    ),
)

technical_agent = Agent(
    name="technical_agent",
    model=MODEL,
    description="Handles TECHNICAL queries — debugging, code errors, installation, and system issues.",
    instruction=(
        "You are a technical support specialist. Help the user resolve their "
        "technical issue. Provide clear, step-by-step solutions."
    ),
)

general_agent = Agent(
    name="general_agent",
    model=MODEL,
    description="Handles GENERAL queries — office hours, company info, and anything not billing or technical.",
    instruction=(
        "You are a general support agent. Answer the user's question "
        "helpfully and concisely."
    ),
)

# ── Triage Router ────────────────────────────────────────
triage_agent = Agent(
    name="triage_agent",
    model=MODEL,
    instruction=(
        "You are a triage router. Classify the user's query and delegate to "
        "the right specialist:\n"
        "- BILLING questions (invoices, payments, pricing) → hand off to 'billing_agent'\n"
        "- TECHNICAL questions (code, debugging, installation) → hand off to 'technical_agent'\n"
        "- Everything else → hand off to 'general_agent'\n\n"
        "Do NOT answer directly. Always delegate to the appropriate specialist."
    ),
    sub_agents=[billing_agent, technical_agent, general_agent],
)

# ── Hardcoded queries ────────────────────────────────────
QUERIES = [
    "I was charged twice for my subscription last month.",
    "My Python script throws a ModuleNotFoundError when I try to import pandas.",
    "What are your office hours on weekends?",
]

APP_NAME = "triage_app"
USER_ID = "user_1"


async def main():
    session_service = InMemorySessionService()

    print("=" * 60)
    print("GOOGLE ADK DEMO 5: Triage / Routing Agent")
    print("=" * 60)
    print(f"Router      : {triage_agent.name}")
    print(f"Specialists : {billing_agent.name}, {technical_agent.name}, {general_agent.name}")
    print("=" * 60)

    for idx, query in enumerate(QUERIES, 1):
        session_id = f"session_triage_{idx}"
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        )
        runner = Runner(
            agent=triage_agent, app_name=APP_NAME, session_service=session_service,
        )

        print(f"\n📝 QUERY {idx}:\n   {query}")
        print(f"\n[Triage is routing...]\n")

        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response = ""
        agent_name = ""
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                    agent_name = event.author

        print(f"🤖 Routed to: {agent_name}")
        print(f"📄 Response:\n\n{final_response if final_response else '[No response]'}")
        print("\n" + "-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
