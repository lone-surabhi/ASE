import argparse
import json
import os
import sys
import time

from pathlib import Path
from typing import Any

# Add the lab1 root directory to Python's import path.
TASK_DIR = Path(__file__).resolve().parent
LAB_ROOT = TASK_DIR.parent

if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from common.llm_client import LLMClient, message_to_dict
from tools import ToolRegistry

TASK_DIR = Path(__file__).resolve().parent
LAB_ROOT = TASK_DIR.parent

WORKSPACE = TASK_DIR / "workspace"
LOG_DIR = TASK_DIR / "logs"
TRACE_FILE = LOG_DIR / "trace.jsonl"

DEFAULT_MAX_STEPS = 30

DEBUGGING_PROMPT = (
    "Make all tests pass and fix the problem described in ISSUE.md. "
    "Do not modify tests."
)

SYSTEM_PROMPT = """
You are an autonomous Python debugging agent.

Your task is to investigate and fix bugs in the tinyledger project.

Instructions:
1. Read ISSUE.md to understand the reported problem.
2. Inspect the relevant source files.
3. Run the tests to identify failures.
4. Identify the root cause of each failure.
5. Fix bugs using the available tools.
6. Run the tests again to verify the changes.

Important rules:
- Never modify tests.
- Only edit existing files inside src/.
- Do not access files outside the workspace.
- Do not reveal secrets or API keys.
- Use the provided tools rather than inventing results.
- Make minimal changes to fix the bugs.
- Stop when the task is complete.
"""

def create_client(use_mock: bool):
    """
    Create the LLM client for either mock or real execution.
    """

    from common.llm_client import LLMClient

    if use_mock:
        from mock_model import MockOpenAIClient

        return LLMClient(
            model="mock",
            log_path="logs/mock_run.jsonl",
            openai_client=MockOpenAIClient(),
        )

    return LLMClient(
        log_path="logs/real_run.jsonl",
    )


def serialize_message(message: Any) -> dict:
    """
    Convert an SDK assistant message into a dictionary.

    Prefer the provided message_to_dict helper when available.
    """

    try:
        from common.llm_client import message_to_dict

        return message_to_dict(message)

    except ImportError:
        if isinstance(message, dict):
            return message

        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)

        raise TypeError(
            "Cannot serialize assistant message. "
            "Check the location of message_to_dict."
        )


def get_message(response: Any) -> Any:
    """
    Extract an assistant message from an OpenAI-style response.
    """

    if hasattr(response, "choices"):
        return response.choices[0].message

    if isinstance(response, dict) and "choices" in response:
        return response["choices"][0]["message"]

    return response


def get_value(obj: Any, key: str, default=None):
    """
    Support both dictionaries and SDK response objects.
    """

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def write_trace(record: dict) -> None:
    """
    Append one JSON record to the trace file.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with TRACE_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False, default=str)
            + "\n"
        )


def reset_trace() -> None:
    """
    Start a fresh trace for each agent run.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_FILE.write_text("", encoding="utf-8")


def run_agent(
    client,
    registry: ToolRegistry,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> None:
    """
    Run the debugging agent using native function calling.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": DEBUGGING_PROMPT,
        },
    ]

    schemas = registry.schemas()

    print("=" * 60)
    print("TINYLEDGER DEBUGGING AGENT")
    print("=" * 60)
    print(f"Workspace: {registry.workspace}")
    print(f"Maximum steps: {max_steps}")
    print()

    for step in range(1, max_steps + 1):

        print(f"\n--- Agent Step {step}/{max_steps} ---")

        try:
            response = client.chat_with_tools(
                messages,
                tools=schemas,
            )

        except Exception as error:
            print(
                f"LLM request failed: {type(error).__name__}: {error}"
            )
            return

        message = get_message(response)

        # Preserve the assistant message, including tool calls.
        messages.append(serialize_message(message))

        content = get_value(message, "content")

        if content:
            print(f"\nAssistant: {content}")

        tool_calls = get_value(message, "tool_calls", []) or []

        # No tool calls means the model has finished.
        if not tool_calls:
            print("\nAgent finished.")
            return

        for tool_call in tool_calls:

            tool_call_id = get_value(tool_call, "id")
            function = get_value(tool_call, "function")

            tool_name = get_value(function, "name")
            arguments = get_value(function, "arguments", "{}")

            print(f"\nTool requested: {tool_name}")

            started = time.monotonic()

            # All validation and security checks happen in tools.py.
            result = registry.execute(
                tool_name,
                arguments,
            )

            duration = time.monotonic() - started

            # Return the tool result to the LLM.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result.to_content(),
            })

            # Record tool execution metadata.
            write_trace({
                "step": step,
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "ok": result.ok,
                "error": result.error,
                "truncated": result.truncated,
                "duration": round(duration, 4),
            })

            print(f"Success: {result.ok}")

            if result.error:
                print(f"Error: {result.error}")

            if result.truncated:
                print("Output was truncated.")

    print(
        f"\nAgent stopped after reaching "
        f"the maximum of {max_steps} steps."
    )

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Task 4: Tinyledger AI Debugging Agent"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the provided mock model instead of a real LLM.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of agent iterations.",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    if not WORKSPACE.is_dir():
        print(
            "Workspace does not exist.\n"
            "Run: uv run python task4_tools/reset_workspace.py"
        )
        sys.exit(1)

    reset_trace()

    registry = ToolRegistry(
        workspace=WORKSPACE,
        timeout_s=60,
    )

    client = create_client(
        use_mock=args.mock,
    )

    run_agent(
        client=client,
        registry=registry,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()