"""Allowlisted document-tool dispatcher with isolated per-question runs."""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from . import tools


class DocumentToolEvent(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    call_number: int  # Monotonic trace sequence number.
    success: bool
    tool_call_number: int | None = None  # Budget sequence; present for execution or blocked call.
    blocked: bool = False
    result_metadata: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class DocumentToolOutcome(BaseModel):
    success: bool
    result: Any = None
    error: str | None = None


@dataclass
class DocumentRun:
    """Mutable state owned by exactly one question key; never reset in place."""

    question: str
    call_count: int = 0
    trace: list[DocumentToolEvent] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    MAX_CALLS = 6
    _allowed_names = frozenset({"list_documents", "list_headings", "get_page", "search_keyword"})

    @property
    def remaining_calls(self) -> int:
        with self._lock:
            return self.MAX_CALLS - self.call_count

    def call(self, tool_name: str, arguments: dict[str, object]) -> DocumentToolOutcome:
        with self._lock:
            sequence = len(self.trace) + 1
            if self.call_count >= self.MAX_CALLS:
                error = "Maximum document-tool calls reached (6)"
                self.trace.append(DocumentToolEvent(tool_name=tool_name, arguments=arguments,
                    call_number=sequence, tool_call_number=self.MAX_CALLS + 1, success=False,
                    blocked=True, error=error))
                return DocumentToolOutcome(success=False, error=error)
            if tool_name not in self._allowed_names:
                self.trace.append(DocumentToolEvent(tool_name=tool_name, arguments=arguments,
                    call_number=sequence, success=False, error="Tool is not allowed"))
                return DocumentToolOutcome(success=False, error="Tool is not allowed")
            if tool_name == "get_page" and any(
                event.success and event.tool_name == "get_page" and event.arguments == arguments
                for event in self.trace
            ):
                self.trace.append(DocumentToolEvent(tool_name=tool_name, arguments=arguments,
                    call_number=sequence, success=False, error="Duplicate page read avoided"))
                return DocumentToolOutcome(success=False, error="Duplicate page read avoided")

            self.call_count += 1  # Execution attempt consumes budget even if it raises.
            budget_number = self.call_count
            try:
                result = getattr(tools, tool_name)(**arguments)
                metadata: dict[str, object] = {}
                if isinstance(result, BaseModel):
                    data = result.model_dump()
                    metadata = {key: value for key, value in data.items() if key != "text"}
                    if "text" in data:
                        metadata["text_length"] = len(str(data["text"]))
                self.trace.append(DocumentToolEvent(tool_name=tool_name, arguments=arguments,
                    call_number=sequence, tool_call_number=budget_number, success=True,
                    result_metadata=metadata))
                return DocumentToolOutcome(success=True, result=result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.trace.append(DocumentToolEvent(tool_name=tool_name, arguments=arguments,
                    call_number=sequence, tool_call_number=budget_number, success=False, error=error))
                return DocumentToolOutcome(success=False, error=error)

    def record_agent_error(self, error: str) -> None:
        with self._lock:
            self.trace.append(DocumentToolEvent(tool_name="agent_decision", arguments={},
                call_number=len(self.trace) + 1, success=False, error=error))

    def record_answer_error(self, error: str) -> None:
        with self._lock:
            self.trace.append(DocumentToolEvent(tool_name="final_answer", arguments={},
                call_number=len(self.trace) + 1, success=False, error=error))


class DocumentHarness:
    MAX_CALLS = 6

    def __init__(self) -> None:
        self._runs: dict[str, DocumentRun] = {}
        self._lock = RLock()

    def new_question(self, question: str) -> DocumentRun:
        """Return one stable run per question; calling again cannot reset its budget."""
        with self._lock:
            if question not in self._runs:
                self._runs[question] = DocumentRun(question)
            return self._runs[question]
