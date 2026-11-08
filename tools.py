"""Bounded, validated tools for the tinyledger debugging agent."""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import jsonschema

from tool_contract import (
    DEFAULT_TIMEOUT_S,
    MAX_LIST_ENTRIES,
    MAX_OUTPUT_CHARS,
    MAX_READ_LINES,
    MAX_SEARCH_RESULTS,
    ToolRegistryBase,
    ToolResult,
)


class ToolRegistry(ToolRegistryBase):
    """Expose a small set of validated operations inside one workspace."""

    CLI_COMMANDS = frozenset({"report", "balance", "settle"})

    def __init__(self, workspace: Path, timeout_s: float = DEFAULT_TIMEOUT_S):
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
        """Validate a model request, execute it, and return a bounded result."""
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_code": self._search_code,
            "run_tests": self._run_tests,
            "edit_file": self._edit_file,
            "run_cli": self._run_cli,
        }
        schemas = {
            entry["function"]["name"]: entry["function"]["parameters"]
            for entry in self.schemas()
        }
        if name not in handlers:
            return self._limit_result(ToolResult(ok=False, error=f"Unknown tool: {name}"))

        try:
            params = json.loads(arguments)
        except (ValueError, TypeError):
            return self._limit_result(ToolResult(ok=False, error="Invalid JSON arguments."))

        try:
            jsonschema.validate(params, schemas[name])
        except jsonschema.ValidationError as exc:
            return self._limit_result(
                ToolResult(ok=False, error=f"Invalid arguments: {exc.message}")
            )

        try:
            result = handlers[name](**params)
            if not isinstance(result, ToolResult):
                result = ToolResult(ok=False, error="Invalid tool result.")
        except Exception as exc:
            result = ToolResult(
                ok=False, error=f"Tool execution failed: {type(exc).__name__}"
            )
        return self._limit_result(result)

    def _safe_path(self, path: str, write: bool = False) -> Path:
        """Resolve a workspace-relative path, rejecting escapes and unsafe writes."""
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
                raise ValueError("Writes are only allowed inside src/.")
            if not resolved.is_file():
                raise ValueError("Only existing source files may be edited.")
        return resolved

    @staticmethod
    def _limit_result(result: ToolResult) -> ToolResult:
        """Limit serialized responses to the contract's maximum length."""
        if len(result.to_content()) <= MAX_OUTPUT_CHARS:
            return result

        result.truncated = True
        if result.ok:
            value = result.output
            result.output = (
                value if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )[: max(0, MAX_OUTPUT_CHARS - 500)]
        else:
            result.error = str(result.error)[: max(0, MAX_OUTPUT_CHARS - 500)]

        # ToolResult serialization adds wrapper fields to the output length.
        while len(result.to_content()) > MAX_OUTPUT_CHARS:
            if result.ok and result.output:
                result.output = result.output[:-100]
            elif result.error:
                result.error = result.error[:-100]
            else:
                break
        return result

    def _workspace_files(self):
        """Yield regular files whose resolved targets stay inside the workspace."""
        for candidate in sorted(self.workspace.rglob("*")):
            try:
                if candidate.is_file() and candidate.resolve().is_relative_to(self.workspace):
                    yield candidate, candidate.relative_to(self.workspace)
            except (OSError, ValueError):
                continue

    def _list_files(self, path: str = ".", pattern: str | None = None) -> ToolResult:
        directory = self._safe_path(path)
        if not directory.is_dir():
            return ToolResult(ok=False, error="Directory does not exist.")

        names = []
        truncated = False
        for candidate in sorted(directory.rglob("*")):
            try:
                if not candidate.is_file() or not candidate.resolve().is_relative_to(self.workspace):
                    continue
                relative = candidate.relative_to(self.workspace)
                if pattern and not relative.match(pattern):
                    continue
                names.append(relative.as_posix())
                if len(names) >= MAX_LIST_ENTRIES:
                    truncated = True
                    break
            except (OSError, ValueError):
                continue
        return ToolResult(ok=True, output=names, truncated=truncated)

    def _read_file(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> ToolResult:
        file_path = self._safe_path(path)
        if not file_path.is_file():
            return ToolResult(ok=False, error="File does not exist.")
        if end_line is not None and end_line < start_line:
            return ToolResult(ok=False, error="end_line must be >= start_line.")

        last_line = min(
            end_line if end_line is not None else start_line + MAX_READ_LINES - 1,
            start_line + MAX_READ_LINES - 1,
        )
        lines = []
        truncated = False
        with file_path.open(encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream, start=1):
                if number < start_line:
                    continue
                if number > last_line:
                    truncated = True
                    break
                lines.append(f"{number}: {line.rstrip()}")
        return ToolResult(ok=True, output="\n".join(lines), truncated=truncated)

    def _search_code(
        self, pattern: str, glob: str | None = None,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> ToolResult:
        try:
            expression = re.compile(pattern)
        except re.error:
            return ToolResult(ok=False, error="Invalid regular expression.")

        matches = []
        for candidate, relative in self._workspace_files():
            if glob and not relative.match(glob):
                continue
            try:
                with candidate.open(encoding="utf-8", errors="replace") as stream:
                    for number, line in enumerate(stream, start=1):
                        if expression.search(line):
                            matches.append(f"{relative.as_posix()}:{number}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                return ToolResult(ok=True, output="\n".join(matches), truncated=True)
            except (OSError, UnicodeError):
                continue
        return ToolResult(ok=True, output="\n".join(matches), truncated=False)

    def _safe_environment(self) -> dict[str, str]:
        """Pass only the environment required by the permitted subprocess."""
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(self.workspace / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONNOUSERSITE": "1",
        }

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
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="Process timed out.")
        except OSError:
            return ToolResult(ok=False, error="Could not start subprocess.")
        return ToolResult(
            ok=True,
            output={
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    def _test_target(self, target: str) -> str:
        """Validate a pytest node identifier against the tests directory."""
        test_file = target.split("::", 1)[0]
        if not test_file.startswith("tests/"):
            raise ValueError("Only tests/ targets are allowed.")
        resolved = self._safe_path(test_file)
        tests_root = (self.workspace / "tests").resolve()
        if not resolved.is_relative_to(tests_root):
            raise ValueError("Invalid test target.")
        if not resolved.is_file():
            raise ValueError("Test file does not exist.")
        return target

    @staticmethod
    def _test_summary(report: Path, exit_code: int) -> dict[str, Any]:
        cases = ET.parse(report).getroot().findall(".//testcase")
        summary: dict[str, Any] = {
            "passed": 0, "failed": 0, "skipped": 0,
            "failures": [], "exit_code": exit_code,
        }
        for case in cases:
            failure = case.find("failure")
            error = case.find("error")
            if failure is not None or error is not None:
                summary["failed"] += 1
                detail = failure if failure is not None else error
                summary["failures"].append({
                    "name": case.get("name"),
                    "class": case.get("classname"),
                    "message": (detail.get("message") or "")[:500],
                    "traceback": (detail.text or "")[:1000],
                })
            elif case.find("skipped") is not None:
                summary["skipped"] += 1
            else:
                summary["passed"] += 1
        return summary

    def _run_tests(self, target: str | None = None) -> ToolResult:
        try:
            selected = self._test_target(target) if target else "tests"
            with tempfile.TemporaryDirectory() as temp_dir:
                report = Path(temp_dir) / "results.xml"
                command = [
                    sys.executable, "-m", "pytest", "-q", "-p", "junitxml",
                    f"--junitxml={report}", selected,
                ]
                process = self._run_process(command)
                if not process.ok:
                    return process
                if not report.exists():
                    return ToolResult(ok=False, error="Pytest did not produce a report.")
                summary = self._test_summary(report, process.output["exit_code"])
                return ToolResult(ok=True, output=summary)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        except (OSError, ET.ParseError):
            return ToolResult(ok=False, error="Could not parse test results.")

    def _edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        file_path = self._safe_path(path, write=True)
        if not old_str:
            return ToolResult(ok=False, error="old_str cannot be empty.")

        original = file_path.read_text(encoding="utf-8")
        occurrences = original.count(old_str)
        if occurrences != 1:
            return ToolResult(
                ok=False,
                error=f"old_str must match exactly once. Found {occurrences} matches.",
            )

        updated = original.replace(old_str, new_str, 1)
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        file_path.write_text(updated, encoding="utf-8")
        return ToolResult(ok=True, output=diff)

    def _run_cli(self, args: list[str]) -> ToolResult:
        if not args:
            return ToolResult(ok=False, error="CLI arguments cannot be empty.")
        if args[0] not in self.CLI_COMMANDS:
            return ToolResult(ok=False, error="CLI command is not allowed.")
        if any(".." in Path(argument).parts for argument in args):
            return ToolResult(ok=False, error="Unsafe CLI argument.")
        return self._run_process([sys.executable, "-m", "tinyledger", *args])
