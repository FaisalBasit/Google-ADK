# ---------------------------------------------------------
# GOOGLE ADK — Demo 8: Parallel Agent (Fan-Out)
# ---------------------------------------------------------
# Multiple agents work simultaneously — the Fan-out Pattern.
# You'll learn:
# 1. How to run multiple agents in parallel with ParallelAgent.
# 2. How to merge parallel results with a downstream summarizer.
# 3. The difference between Sequential (A→B→C) and Parallel (A,B,C at once).
#
# Run:  python 8_parallel_agent.py

import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(override=True)

# ── Parallel Workers (Fan-out) ───────────────────────────
google_searcher = Agent(
    name="google_searcher",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "You are a web researcher. Find the most relevant and recent facts "
        "about the given topic from general sources."
    ),
)

news_searcher = Agent(
    name="news_searcher",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "You are a news analyst. Find the latest headlines, developments, "
        "and breaking news about the given topic."
    ),
)

paper_searcher = Agent(
    name="paper_searcher",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "You are an academic researcher. Find relevant research papers, "
        "studies, and expert opinions about the given topic."
    ),
)

# All three run simultaneously on the same input
parallel_search = ParallelAgent(
    name="parallel_search",
    sub_agents=[google_searcher, news_searcher, paper_searcher],
    description="Fans out to three specialised searchers simultaneously.",
)

# ── Summarizer (combines parallel outputs) ───────────────
summarizer = Agent(
    name="summarizer",
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    instruction=(
        "You will receive research findings from three specialised agents:\n"
        "1. A general web researcher\n"
        "2. A news analyst\n"
        "3. An academic researcher\n\n"
        "Synthesise all their findings into a single, well-structured summary. "
        "Use clear sections (e.g., Key Facts, Latest News, Research Insights)."
    ),
)

# ── Full pipeline: Parallel → Summarizer ─────────────────
research_pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[parallel_search, summarizer],
    description="Fan-out: 3 parallel searches → combined summary.",
)

# ── Hardcoded topic ──────────────────────────────────────
TOPIC = "Recent breakthroughs in quantum computing"

APP_NAME = "parallel_app"
USER_ID = "user_1"
SESSION_ID = "session_parallel"


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID,
    )
    runner = Runner(
        agent=research_pipeline, app_name=APP_NAME, session_service=session_service,
    )

    print("=" * 60)
    print("GOOGLE ADK DEMO 8: Parallel Agent (Fan-Out → Summarize)")
    print("=" * 60)
    print(f"Pattern  : ParallelAgent → Summarizer")
    print(f"Workers  : {google_searcher.name} | {news_searcher.name} | {paper_searcher.name}")
    print(f"Combiner : {summarizer.name}")
    print("=" * 60)
    print(f"\n📝 TOPIC:\n   {TOPIC}")
    print(f"\n[All 3 agents are searching in parallel...]\n")

    content = types.Content(role="user", parts=[types.Part(text=TOPIC)])
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"📊 COMBINED RESEARCH SUMMARY:\n")
    print(final_response if final_response else "[No response received]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
