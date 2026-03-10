# ---------------------------------------------------------
# GOOGLE ADK — Demo 10: Human-in-the-Loop (Approval Gate)
# ---------------------------------------------------------
# The agent PAUSES and requests human approval before critical actions.
# You'll learn:
# 1. How to implement a Human-in-the-Loop gate using a tool.
# 2. Why high-stakes decisions need human oversight.
# 3. How the agent adapts based on approval / rejection.
#
# NOTE: This is the ONE file that keeps an interactive yes/no prompt.
#
# Run:  python 10_human_in_loop.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)


# ── Approval tool (interactive) ──────────────────────────
def human_approval(action: str) -> str:
    """Request explicit human approval before executing a critical action.

    ALWAYS call this tool BEFORE performing any of the following:
    - Financial transactions (transfers, payments, purchases)
    - Data deletion or account changes
    - Sending emails or messages on behalf of the user

    Args:
        action: A clear description of the action that requires approval.
    """
    print(f"\n{'=' * 52}")
    print(f"  ⚠  HUMAN APPROVAL REQUIRED")
    print(f"{'=' * 52}")
    print(f"  Proposed Action: {action}")
    print(f"{'=' * 52}")
    decision = input("  Approve this action? (yes / no): ").strip().lower()

    if decision in ("yes", "y"):
        return f"APPROVED. You may now proceed with: {action}"
    return (
        f"REJECTED by human. Do NOT execute: {action}. "
        f"Inform the user their request was declined and ask what they would like to do instead."
    )


# ── Agent ─────────────────────────────────────────────────
approval_agent = Agent(
    name="approval_agent",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    description="A financial assistant that always seeks human approval before critical actions.",
    instruction=(
        "Role: Financial & Operations Assistant.\n\n"
        "CRITICAL RULE — You MUST call the human_approval tool before executing ANY of:\n"
        "- Transferring money or making payments\n"
        "- Deleting data, files, or accounts\n"
        "- Sending emails or messages on the user's behalf\n\n"
        "Workflow:\n"
        "1. Understand the request.\n"
        "2. Call human_approval with a clear description.\n"
        "3. If APPROVED → proceed and confirm.\n"
        "4. If REJECTED → inform the user and ask what to do instead.\n\n"
        "Never skip the approval step."
    ),
    tools=[human_approval],
)

# ── Hardcoded request (the yes/no prompt still appears) ──
REQUEST = "Transfer $500 from my checking account to Alice's savings account."

APP_NAME = "approval_app"
USER_ID = "user_1"
SESSION_ID = "session_approval"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=approval_agent, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 10: Human-in-the-Loop (Approval Gate)")
    print("=" * 60)
    print(f"Agent   : {approval_agent.name}")
    print(f"Model   : {approval_agent.model}")
    print(f"Pattern : Agent → Propose → Human Approves/Rejects → Proceed/Stop")
    print("=" * 60)
    print(f"\n📝 REQUEST:\n   {REQUEST}")
    print(f"\n[{approval_agent.model} is processing...]\n")

    content = types.Content(role="user", parts=[types.Part(text=REQUEST)])
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"\n🤖 AGENT RESPONSE:\n{final_response if final_response else '[No response]'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
