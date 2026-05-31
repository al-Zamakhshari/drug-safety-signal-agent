"""Entry point — analyze drug safety signals locally."""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()


async def run(drug_name: str):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
    from agent.agent import root_agent, _tracing_active

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="drug_safety_agent", user_id="cli_user"
    )
    runner = Runner(
        agent=root_agent,
        app_name="drug_safety_agent",
        session_service=session_service,
    )

    print(f"\nAnalyzing safety signals for: {drug_name}")
    print("─" * 60 + "\n")

    message = Content(parts=[Part(text=f"Analyze safety signals for: {drug_name}")])
    async for event in runner.run_async(
        user_id="cli_user", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text, flush=True)

    if _tracing_active:
        _tracing_active.force_flush()


def main():
    drug = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Drug name: ").strip()
    if not drug:
        print("Usage: uv run python main.py <drug_name>")
        sys.exit(1)
    asyncio.run(run(drug))


if __name__ == "__main__":
    main()
