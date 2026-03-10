# ---------------------------------------------------------
# GOOGLE ADK — Demo 4: Structured Output
# ---------------------------------------------------------
# Agents can return data in a specific JSON format.
# You'll learn:
# 1. How to define a response schema with a dict.
# 2. How to use output_schema with ADK.
# 3. How to parse the structured JSON from the agent's response.
#
# Run:  python 4_structured_output.py

import asyncio
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Schema (Pydantic model for output_schema) ────────────
class PersonInfo(BaseModel):
    name: str
    age: int
    hobbies: list[str]

# ── Agent ─────────────────────────────────────────────────
# When output_schema is set, the agent CANNOT use tools.
structured_agent = Agent(
    name="structured_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    description="An agent that extracts data into a structured format.",
    instruction=(
        "Identify the person's name, age, and hobbies from the text. "
        "Return the data exactly in the requested JSON schema."
    ),
    output_schema=PersonInfo,
)

# ── Hardcoded input ──────────────────────────────────────
INPUT_TEXT = (
    "My name is Sarah Chen and I'm 29 years old. In my free time I love "
    "hiking in the mountains, painting watercolors, and playing chess online."
)

APP_NAME = "structured_app"
USER_ID = "user_1"
SESSION_ID = "session_struct"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=structured_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 4: Structured Output")
    print("=" * 60)
    print(f"Agent  : {structured_agent.name}")
    print(f"Model  : {structured_agent.model}")
    print(f"Schema : name (str), age (int), hobbies (list[str])")
    print("=" * 60)
    print(f"\n📝 INPUT TEXT:\n   {INPUT_TEXT}")
    print(f"\n[{structured_agent.model} is extracting data...]\n")

    content = types.Content(role="user", parts=[types.Part(text=INPUT_TEXT)])
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print("📦 EXTRACTED JSON:\n")
    if final_response:
        try:
            parsed = json.loads(final_response)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(final_response)
    else:
        print("[No response received]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
