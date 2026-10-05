"""Budgeted, tool-mediated document question answering."""

import os
import re
from typing import Protocol

from pydantic import BaseModel, Field

from .harness import DocumentHarness
from .llm import OpenAIDocumentLLM


class DocumentAction(BaseModel):
    """One structured decision from the planning model."""

    action_type: str = Field(description="tool or finish")
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    reason: str = ""


class EvidenceState(BaseModel):
    """Evidence gathered exclusively from the public document tools."""

    pages_read: list[dict[str, object]] = Field(default_factory=list)
    keyword_searches: list[dict[str, object]] = Field(default_factory=list)
    headings_results: list[dict[str, object]] = Field(default_factory=list)
    document_metadata: list[dict[str, object]] = Field(default_factory=list)
    tool_trace: list[dict[str, object]] = Field(default_factory=list)


class DocumentLLM(Protocol):
    def choose_action(self, question: str, evidence: EvidenceState, remaining_calls: int) -> DocumentAction: ...
    def answer(self, question: str, evidence: EvidenceState) -> str: ...


class DocumentAgent:
    def __init__(self, harness: DocumentHarness, llm: DocumentLLM | None = None) -> None:
        self._harness = harness
        self._llm = llm

    @staticmethod
    def _validated_answer(answer: str, question: str, evidence: EvidenceState) -> str:
        if answer.strip().rstrip(".").casefold() == "insufficient information":
            return "Insufficient information."
        pages = {int(page["page_number"]): str(page.get("text", ""))
                 for page in evidence.pages_read if "page_number" in page}
        if not pages:
            return "Insufficient information."
        cited = [int(n) for n in re.findall(r"\[(?:p\.|page)\s*(\d+)\]", answer, re.IGNORECASE)]
        if not cited or any(number not in pages for number in cited):
            return "Insufficient information."
        words = lambda value: set(re.findall(r"[a-z0-9]+", value.casefold()))
        stop = {"the", "and", "for", "with", "that", "this", "was", "were", "are", "is", "of", "in", "on", "to", "a", "an", "it", "as", "by", "from", "what", "which", "who", "when", "where", "how", "do", "does", "did", "be", "been", "has", "have", "had", "i", "we", "they", "you", "their", "its"}
        question_terms = words(question) - stop
        cited_text = " ".join(pages[number] for number in cited)
        evidence_terms = words(cited_text)
        answer_terms = words(re.sub(r"\[(?:p\.|page)\s*\d+\]", "", answer, flags=re.IGNORECASE)) - stop
        if not question_terms.intersection(evidence_terms) or not answer_terms or not answer_terms.issubset(evidence_terms):
            return "Insufficient information."
        return answer

    def run(self, question: str) -> "DocumentResponse":
        llm = self._llm
        if llm is None:
            if not (os.getenv("DOCUMENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
                raise RuntimeError("Set DOCUMENT_LLM_API_KEY (or OPENAI_API_KEY) to use the document agent")
            llm = OpenAIDocumentLLM()
        evidence = EvidenceState()
        run = self._harness.new_question(question)
        for _ in range(7):
            try:
                action = llm.choose_action(question, evidence, run.remaining_calls)
            except Exception as exc:
                run.record_agent_error(type(exc).__name__ + ": " + str(exc))
                break
            if action.action_type != "tool":
                break
            outcome = run.call(action.tool_name or "", action.arguments)
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            if outcome.success:
                value = outcome.result
                data = value.model_dump() if isinstance(value, BaseModel) else {}
                if action.tool_name == "get_page":
                    evidence.pages_read.append(data)
                elif action.tool_name == "search_keyword":
                    evidence.keyword_searches.append(data)
                elif action.tool_name == "list_headings":
                    evidence.headings_results.append(data)
                elif action.tool_name == "list_documents":
                    evidence.document_metadata = data.get("documents", [])
            if run.remaining_calls == 0:
                break
        evidence.tool_trace = [event.model_dump() for event in run.trace]
        try:
            answer = self._validated_answer(llm.answer(question, evidence), question, evidence)
        except Exception as exc:
            answer = "Insufficient information."
            run.record_answer_error(type(exc).__name__ + ": " + str(exc))
        return DocumentResponse(answer=answer, evidence=evidence, trace=run.trace.copy(),
                                calls_made=run.call_count, remaining_calls=run.remaining_calls)


class DocumentResponse(BaseModel):
    answer: str
    evidence: EvidenceState
    trace: list[object]
    calls_made: int
    remaining_calls: int
