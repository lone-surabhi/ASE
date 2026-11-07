from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any

import jsonschema

from tool_contract import (
    ToolRegistryBase,
    ToolResult,
    DEFAULT_TIMEOUT_S,
    MAX_OUTPUT_CHARS,
    MAX_READ_LINES,
    MAX_SEARCH_RESULTS,
    MAX_LIST_ENTRIES,
)


class ToolRegistry(ToolRegistryBase):
    """
    Secure tool registry for the Task 4 debugging agent.

    Provides:
        list_files
        read_file
        search_code
        run_tests
        edit_file
        run_cli

    All operations are restricted to the workspace.
    """

    def __init__(
        self,
        workspace: Path,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        super().__init__(workspace, timeout_s)

        self.workspace = Path(workspace).resolve()
        self.timeout_s = timeout_s

    def schemas(self) -> list[dict]:

        def tool(name, description, properties, required=None):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required or [],
                        "additionalProperties": False,
                    },
                },
            }

        return [
            tool(
                "list_files",
                "List files inside the workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional filename glob pattern.",
                    },
                },
            ),

            tool(
                "read_file",
                "Read numbered lines from a workspace file.",
                {
                    "path": {"type": "string"},
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
                ["path"],
            ),

            tool(
                "search_code",
                "Search workspace files using a regular expression.",
                {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                    },
                },
                ["pattern"],
            ),

            tool(
                "run_tests",
                "Run pytest safely and return structured results.",
                {
                    "target": {
                        "type": "string",
                        "description": "Optional test file or test name.",
                    },
                },
            ),

            tool(
                "edit_file",
                "Replace one unique string in an existing source file.",
                {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                ["path", "old_str", "new_str"],
            ),

            tool(
                "run_cli",
                "Run the tinyledger CLI with restricted arguments.",
                {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                ["args"],
            ),
        ]

    def execute(self, name: str, arguments: str) -> ToolResult:
        """
        Validate and execute a tool call.

        Never raises an exception to the caller.
        """

        started = time.monotonic()

        try:
            available = {
                schema["function"]["name"]: schema["function"]
                for schema in self.schemas()
            }

            if name not in available:
                return ToolResult(
                    ok=False,
                    error=f"Unknown tool: {name}",
                )

            try:
                args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return ToolResult(
                    ok=False,
                    error="Invalid JSON arguments.",
                )

            try:
                jsonschema.validate(
                    instance=args,
                    schema=available[name]["parameters"],
                )
            except jsonschema.ValidationError as error:
                return ToolResult(
                    ok=False,
                    error=f"Invalid arguments: {error.message}",
                )

            handlers = {
                "list_files": self._list_files,
                "read_file": self._read_file,
                "search_code": self._search_code,
                "run_tests": self._run_tests,
                "edit_file": self._edit_file,
                "run_cli": self._run_cli,
            }

            result = handlers[name](**args)

            if not isinstance(result, ToolResult):
                result = ToolResult(
                    ok=False,
                    error="Invalid tool result.",
                )

            return self._limit_result(result)

        except Exception as error:
            return self._limit_result(
                ToolResult(
                    ok=False,
                    error=f"Tool execution failed: {type(error).__name__}",
                )
            )

    def _safe_path(
        self,
        path: str,
        write: bool = False,
    ) -> Path:
        """
        Resolve a relative path and ensure it remains
        inside the permitted workspace.
        """

        requested = Path(path)

        if requested.is_absolute():
            raise ValueError("Absolute paths are not allowed.")

        if ".." in requested.parts:
            raise ValueError("Parent directory traversal is not allowed.")

        resolved = (self.workspace / requested).resolve()

        if not resolved.is_relative_to(self.workspace):
            raise ValueError("Path escapes the workspace.")

        if write:
            source_root = (self.workspace / "src").resolve()

            if not source_root.is_relative_to(self.workspace):
                raise ValueError("Invalid source directory.")

            if not resolved.is_relative_to(source_root):
                raise ValueError(
                    "Writes are only allowed inside src/."
                )

            if not resolved.is_file():
                raise ValueError(
                    "Only existing source files may be edited."
                )

        return resolved

    def _limit_result(self, result: ToolResult) -> ToolResult:
        """
        Ensure serialized tool output stays within the limit.
        """

        content = result.to_content()

        if len(content) <= MAX_OUTPUT_CHARS:
            return result

        result.truncated = True

        if result.ok:
            serialized = (
                result.output
                if isinstance(result.output, str)
                else json.dumps(result.output, ensure_ascii=False)
            )

            result.output = serialized[: MAX_OUTPUT_CHARS - 500]

        else:
            result.error = str(result.error)[: MAX_OUTPUT_CHARS - 500]

        while len(result.to_content()) > MAX_OUTPUT_CHARS:
            if result.ok:
                result.output = result.output[:-100]
            else:
                result.error = result.error[:-100]

        return result
    
    def _list_files(
        self,
        path: str = ".",
        pattern: str | None = None,
    ) -> ToolResult:

        directory = self._safe_path(path)

        if not directory.is_dir():
            return ToolResult(
                ok=False,
                error="Directory does not exist.",
            )

        files = []
        truncated = False

        for item in sorted(directory.rglob("*")):

            if not item.is_file():
                continue

            try:
                resolved = item.resolve()

                if not resolved.is_relative_to(self.workspace):
                    continue

                relative = item.relative_to(self.workspace)

                if pattern and not relative.match(pattern):
                    continue

                files.append(relative.as_posix())

                if len(files) >= MAX_LIST_ENTRIES:
                    truncated = True
                    break

            except (OSError, ValueError):
                continue

        return ToolResult(
            ok=True,
            output=files,
            truncated=truncated,
        )

    def _read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> ToolResult:

        file_path = self._safe_path(path)

        if not file_path.is_file():
            return ToolResult(
                ok=False,
                error="File does not exist.",
            )

        if end_line is not None and end_line < start_line:
            return ToolResult(
                ok=False,
                error="end_line must be >= start_line.",
            )

        if end_line is None:
            end_line = start_line + MAX_READ_LINES - 1

        end_line = min(
            end_line,
            start_line + MAX_READ_LINES - 1,
        )

        lines = []
        truncated = False

        with file_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for number, line in enumerate(file, start=1):

                if number < start_line:
                    continue

                if number > end_line:
                    truncated = True
                    break

                lines.append(f"{number}: {line.rstrip()}")

        return ToolResult(
            ok=True,
            output="\n".join(lines),
            truncated=truncated,
        )
    
    def _search_code(
        self,
        pattern: str,
        glob: str | None = None,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> ToolResult:

        try:
            regex = re.compile(pattern)
        except re.error:
            return ToolResult(
                ok=False,
                error="Invalid regular expression.",
            )

        matches = []
        truncated = False

        for file_path in sorted(self.workspace.rglob("*")):

            if not file_path.is_file():
                continue

            try:
                resolved = file_path.resolve()

                if not resolved.is_relative_to(self.workspace):
                    continue

                relative = file_path.relative_to(self.workspace)

                if glob and not relative.match(glob):
                    continue

                with file_path.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as file:

                    for number, line in enumerate(file, start=1):

                        if regex.search(line):

                            matches.append(
                                f"{relative.as_posix()}:{number}: "
                                f"{line.rstrip()}"
                            )

                            if len(matches) >= max_results:
                                truncated = True
                                break

                if truncated:
                    break

            except (OSError, UnicodeError):
                continue

        return ToolResult(
            ok=True,
            output="\n".join(matches),
            truncated=truncated,
        )

    def _safe_environment(self) -> dict[str, str]:
        """
        Create a minimal environment for subprocesses.

        API keys and other inherited secrets are excluded.
        """

        environment = {
            "PATH": os.environ.get(
                "PATH",
                "/usr/bin:/bin",
            ),
            "PYTHONPATH": str(self.workspace / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONNOUSERSITE": "1",
        }

        return environment

    def _run_process(self, command: list[str]) -> ToolResult:

        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                env=self._safe_environment(),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_s,
                shell=False,
            )

            return ToolResult(
                ok=True,
                output={
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                error="Process timed out.",
            )

        except OSError:
            return ToolResult(
                ok=False,
                error="Could not start subprocess.",
            )

    def _run_tests(
        self,
        target: str | None = None,
    ) -> ToolResult:

        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "junitxml",
        ]

        report = None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                report = Path(temp_dir) / "results.xml"

                command.append(f"--junitxml={report}")

                if target:

                    test_path = target.split("::", 1)[0]

                    if not test_path.startswith("tests/"):
                        return ToolResult(
                            ok=False,
                            error="Only tests/ targets are allowed.",
                        )

                    resolved = self._safe_path(test_path)

                    tests_root = (self.workspace / "tests").resolve()

                    if not resolved.is_relative_to(tests_root):
                        return ToolResult(
                            ok=False,
                            error="Invalid test target.",
                        )

                    if not resolved.is_file():
                        return ToolResult(
                            ok=False,
                            error="Test file does not exist.",
                        )

                    command.append(target)

                else:
                    command.append("tests")

                process = self._run_process(command)

                if not process.ok:
                    return process

                output = process.output

                if not report.exists():
                    return ToolResult(
                        ok=False,
                        error="Pytest did not produce a report.",
                    )

                root = ET.parse(report).getroot()

                cases = root.findall(".//testcase")

                passed = 0
                failed = 0
                skipped = 0
                failures = []

                for case in cases:

                    failure = case.find("failure")
                    error = case.find("error")
                    skip = case.find("skipped")

                    if failure is not None or error is not None:

                        failed += 1

                        detail = failure if failure is not None else error

                        failures.append({
                            "name": case.get("name"),
                            "class": case.get("classname"),
                            "message": (detail.get("message") or "")[:500],
                            "traceback": (detail.text or "")[:1000],
                        })

                    elif skip is not None:
                        skipped += 1

                    else:
                        passed += 1

                return ToolResult(
                    ok=True,
                    output={
                        "passed": passed,
                        "failed": failed,
                        "skipped": skipped,
                        "failures": failures,
                        "exit_code": output["exit_code"],
                    },
                )

        except (OSError, ET.ParseError):
            return ToolResult(
                ok=False,
                error="Could not parse test results.",
            )
    
    def _edit_file(
        self,
        path: str,
        old_str: str,
        new_str: str,
    ) -> ToolResult:

        file_path = self._safe_path(path, write=True)

        if not old_str:
            return ToolResult(
                ok=False,
                error="old_str cannot be empty.",
            )

        original = file_path.read_text(
            encoding="utf-8",
        )

        count = original.count(old_str)

        if count != 1:
            return ToolResult(
                ok=False,
                error=(
                    "old_str must match exactly once. "
                    f"Found {count} matches."
                ),
            )

        updated = original.replace(
            old_str,
            new_str,
            1,
        )

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )

        file_path.write_text(
            updated,
            encoding="utf-8",
        )

        return ToolResult(
            ok=True,
            output=diff,
        )
    
    def _run_cli(
        self,
        args: list[str],
    ) -> ToolResult:

        if not args:
            return ToolResult(
                ok=False,
                error="CLI arguments cannot be empty.",
            )

        # Only allow known tinyledger commands.
        allowed_commands = {
            "report",
            "balance",
            "settle",
        }

        if args[0] not in allowed_commands:
            return ToolResult(
                ok=False,
                error="CLI command is not allowed.",
            )

        # Reject path traversal in arguments.
        for argument in args:

            if argument.startswith("-"):
                continue

            if ".." in Path(argument).parts:
                return ToolResult(
                    ok=False,
                    error="Unsafe CLI argument.",
                )

        command = [
            sys.executable,
            "-m",
            "tinyledger",
            *args,
        ]

        return self._run_process(command)