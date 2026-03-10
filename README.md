# Google Agent Development Kit (ADK) Practice

This project contains a series of examples demonstrating the core concepts of the **Google Agent Development Kit (ADK)**, an open-source framework for building multi-agent systems.

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:
   Copy `.env` and add your `GOOGLE_API_KEY`.

3. **Run Examples**:
   Execute the scripts in order to learn the pillars of ADK.

## Project Structure

| File | Description | Pillar |
|------|-------------|--------|
| `1_agent.py` | Basic Hello World agent setup | BUILD |
| `2_tools.py` | Defining and using Python functions as tools | BUILD |
| `3_handoffs.py` | Sequential multi-agent handoffs | INTERACT |
| `4_structured_output.py`| Using Pydantic for structured JSON responses | BUILD |
| `5_triage_agent.py` | Workflow patterns and triage logic | INTERACT |
| `6_guardrails.py` | Lifecycle hooks and callbacks | EVALUATE |
| `7_session_state.py` | Session management and stateful memory | INTERACT |
| `groq_setup.py` | Model flexibility (External LLMs via LiteLLM) | DEPLOY |

## Launching the Dev UI

To test and debug your agents interactively in the browser:
```bash
adk web
```

---
*Created for Google ADK Seminar*
