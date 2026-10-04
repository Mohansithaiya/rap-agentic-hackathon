"""Controlled execution of agent tools."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Structured outcome of a tool execution request."""

    success: bool
    result: Any = None
    error: str | None = None


class ToolEvent(BaseModel):
    """In-memory trace entry for a tool execution request."""

    tool_name: str
    success: bool
    error: str | None = None


class ToolCallLimitError(RuntimeError):
    """Raised internally when the run has exhausted its tool call budget."""


class Harness:
    """Execute callables with a per-run call limit and event trace."""

    def __init__(self, max_tool_calls: int = 5) -> None:
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        self.max_tool_calls = max_tool_calls
        self.call_count = 0
        self.events: list[ToolEvent] = []

    def execute(
        self,
        tool: Callable[..., Any],
        *args: Any,
        tool_name: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute ``tool`` and return a result without propagating tool errors."""
        name = tool_name or getattr(tool, "__name__", tool.__class__.__name__)

        if self.call_count >= self.max_tool_calls:
            error = f"Maximum tool calls reached ({self.max_tool_calls})"
            self.events.append(ToolEvent(tool_name=name, success=False, error=error))
            return ToolResult(success=False, error=error)

        self.call_count += 1
        try:
            result = tool(*args, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.events.append(ToolEvent(tool_name=name, success=False, error=error))
            return ToolResult(success=False, error=error)

        self.events.append(ToolEvent(tool_name=name, success=True))
        return ToolResult(success=True, result=result)
