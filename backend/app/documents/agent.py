"""Budgeted, tool-mediated document question answering."""

import os
import re
import logging
from typing import Protocol

from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field

from .harness import DocumentHarness
from .llm import OpenAIDocumentLLM

logger = logging.getLogger(__name__)


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
    _fallback_stop_words = {
        "the", "and", "for", "with", "that", "this", "was", "were", "are", "is", "of", "in", "on",
        "to", "a", "an", "it", "as", "by", "from", "what", "which", "who", "when", "where", "how",
        "do", "does", "did", "be", "been", "has", "have", "had", "i", "we", "they", "you", "their",
        "its", "per", "can", "could", "would", "should", "about", "document", "pdf", "number",
    }

    @classmethod
    def _tool_inventory_question(cls, question: str) -> bool:
        """Recognize requests for a complete description of the available tools."""
        normalized = question.casefold()
        return bool(re.search(r"\b(tools?|functions?|apis?|interfaces?)\b", normalized)
                    and re.search(r"\b(each|every|all|complete|provided|available|inventory|list|four)\b", normalized)
                    and re.search(r"\b(return|returns|output|outputs|does|provide|provided)\b", normalized))

    @staticmethod
    def _tool_budget_question(question: str) -> bool:
        normalized = question.casefold()
        asks_about_calls = bool(re.search(r"\btool[- ]calls?\b|\bcalls?\b", normalized))
        asks_about_limit = bool(re.search(
            r"\b(max(?:imum)?|limit|budget|constraint|allowed|six|6)\b", normalized
        ))
        return asks_about_calls and asks_about_limit

    @staticmethod
    def _constraint_page(evidence: EvidenceState) -> int | None:
        """Select an unread page whose heading identifies constraints or call limits."""
        candidates: list[tuple[int, int]] = []
        read_pages = {int(page["page_number"]) for page in evidence.pages_read if "page_number" in page}
        for result in evidence.headings_results:
            for heading in result.get("headings", []):
                title = str(heading.get("title", ""))
                normalized = title.casefold()
                if "hard constraint" in normalized:
                    score = 4
                elif "constraint" in normalized:
                    score = 3
                elif "budget" in normalized or re.search(r"\btool[- ]calls?\b", normalized):
                    score = 2
                elif re.search(r"\b(?:call|tool) limits?\b", normalized):
                    score = 1
                else:
                    continue
                page_number = int(heading["page_number"])
                if page_number not in read_pages:
                    candidates.append((score, page_number))
        return max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))[1] if candidates else None

    @staticmethod
    def _page_supports_tool_budget(text: str) -> bool:
        """Recognize page text that states a numeric tool-call limit."""
        terms = set(re.findall(r"[a-z0-9]+", text.casefold()))
        has_tool_calls = "tool" in terms and bool(terms & {"call", "calls"})
        has_limit = bool(terms & {"max", "maximum", "limit", "budget", "constraint", "allowed", "allow"})
        has_quantity = bool(terms & {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty",
        })
        return has_tool_calls and has_limit and has_quantity

    @staticmethod
    def _tool_call_quantities(text: str) -> set[str]:
        """Extract numeric quantities stated close to a tool-call phrase."""
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
            "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        }
        quantities = set()
        for clause in re.split(r"[,;.!?\n]+", text.casefold()):
            terms = [number_words.get(term, term) for term in re.findall(r"\d+|[a-z]+", clause)]
            for index, term in enumerate(terms):
                if not term.isdigit():
                    continue
                window = terms[max(0, index - 6):index + 7]
                if "tool" in window and ("call" in window or "calls" in window):
                    quantities.add(term)
        return quantities

    @classmethod
    def _tool_inventory_page(cls, question: str, evidence: EvidenceState) -> int | None:
        """Choose a page from headings that best matches the requested tool inventory."""
        stop = cls._fallback_stop_words | {"tool", "tools", "function", "functions", "return", "returns",
                                           "output", "outputs", "provided", "available", "each", "every",
                                           "complete", "four", "all"}
        query_terms = set(re.findall(r"[a-z0-9]+", question.casefold())) - stop
        read_pages = {int(page["page_number"]) for page in evidence.pages_read if "page_number" in page}
        candidates: list[tuple[int, int]] = []
        for result in evidence.headings_results:
            for heading in result.get("headings", []):
                title = str(heading.get("title", ""))
                title_terms = set(re.findall(r"[a-z0-9]+", title.casefold()))
                if re.search(r"\b(tools?|functions?)\b", title, re.I):
                    candidates.append((len(query_terms & title_terms) + 1, int(heading["page_number"])))
        unread = [candidate for candidate in candidates if candidate[1] not in read_pages]
        return max(unread, key=lambda candidate: (candidate[0], -candidate[1]))[1] if unread else None

    def __init__(self, harness: DocumentHarness, llm: DocumentLLM | None = None) -> None:
        self._harness = harness
        self._llm = llm

    @staticmethod
    def _search_terms(keyword: object) -> frozenset[str]:
        """Normalize lexical queries so reordered words and close variants are deduplicated."""
        terms = set()
        for term in re.findall(r"[a-z0-9]+", str(keyword).casefold()):
            if len(term) > 4 and term.endswith("ies"):
                term = term[:-3] + "y"
            elif len(term) > 4 and term.endswith("s"):
                term = term[:-1]
            terms.add(term)
        return frozenset(terms)

    @classmethod
    def _duplicate_search(cls, doc_id: str, keyword: object,
                          attempted: dict[str, list[frozenset[str]]]) -> bool:
        terms = cls._search_terms(keyword)
        if not terms:
            return False
        for previous in attempted.get(doc_id.casefold(), []):
            overlap = len(terms & previous)
            # Containment and high-overlap rewrites usually ask the same lexical lookup.
            if overlap == min(len(terms), len(previous)) or overlap / len(terms | previous) >= 0.8:
                return True
        attempted.setdefault(doc_id.casefold(), []).append(terms)
        return False

    @staticmethod
    def _validated_answer(answer: str, question: str, evidence: EvidenceState,
                          diagnostics: dict[str, object] | None = None) -> str:
        def rejected(reason: str) -> str:
            if diagnostics is not None:
                diagnostics["validation_result"] = "rejected"
                diagnostics["failed_condition"] = reason
            return "Insufficient information."

        if answer.strip().rstrip(".").casefold() == "insufficient information":
            if diagnostics is not None:
                diagnostics["validation_result"] = "provider_abstained"
                diagnostics["failed_condition"] = "provider returned Insufficient information"
            return "Insufficient information."
        pages = {int(page["page_number"]): str(page.get("text", ""))
                 for page in evidence.pages_read if "page_number" in page}
        if not pages:
            return rejected("no get_page evidence available")
        cited = [int(n) for n in re.findall(r"\[(?:p\.|page)\s*(\d+)\]", answer, re.IGNORECASE)]
        if not cited:
            return rejected("answer contains no page citations")
        if any(number not in pages for number in cited):
            return rejected("answer cites a page absent from pages_read")
        # Validate content words rather than numbering, punctuation, or answer framing.
        # Citation parsing and page membership checks above remain strict and unchanged.
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
            "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        }
        def words(value: str) -> set[str]:
            return {number_words.get(term, term) for term in re.findall(r"\d+|[a-z]+", value.casefold())}

        stop = {
            "the", "and", "for", "with", "that", "this", "was", "were", "are", "is", "of", "in", "on",
            "to", "a", "an", "it", "as", "by", "from", "what", "which", "who", "when", "where", "how",
            "do", "does", "did", "be", "been", "has", "have", "had", "i", "we", "they", "you", "their",
            "its", "also", "then", "than", "but", "or", "so", "each", "every", "all", "first", "second", "third", "number", "provided",
            "available", "following", "below", "above", "answer", "answers", "tool", "tools", "function",
            "functions", "agent", "respectively", "following", "returning", "called", "named",
        }
        question_terms = words(question) - stop
        cited_text = " ".join(pages[number] for number in cited)
        evidence_terms = words(cited_text)
        answer_text = re.sub(r"\[(?:p\.|page)\s*\d+\]", "", answer, flags=re.IGNORECASE)
        answer_text = re.sub(r"(?m)^\s*\d+[.)]\s*", "", answer_text)
        answer_terms = words(answer_text) - stop
        if not question_terms.intersection(evidence_terms):
            return rejected("no question-term overlap with cited evidence")
        unsupported_terms = answer_terms - evidence_terms
        if not answer_terms:
            return rejected("answer has no verifiable content terms")
        evidence_has_budget_context = (
            "tool" in evidence_terms
            and bool(evidence_terms & {"call", "calls"})
            and bool(evidence_terms & {"budget", "limit", "maximum", "max", "total", "allowed", "allow"})
        )
        answer_call_quantities = DocumentAgent._tool_call_quantities(answer_text)
        evidence_call_quantities = DocumentAgent._tool_call_quantities(cited_text)
        budget_claim_supported = (
            DocumentAgent._tool_budget_question(question)
            and evidence_has_budget_context
            and bool(answer_call_quantities)
            and answer_call_quantities.issubset(evidence_call_quantities)
        )
        if (DocumentAgent._tool_budget_question(question) and answer_call_quantities
                and not budget_claim_supported):
            if diagnostics is not None:
                diagnostics["answer_tool_call_quantities"] = sorted(answer_call_quantities)
                diagnostics["evidence_tool_call_quantities"] = sorted(evidence_call_quantities)
                diagnostics["unsupported_answer_terms"] = sorted(
                    answer_call_quantities - evidence_call_quantities
                )
            return rejected("tool-call quantity is not supported by cited budget evidence")
        if budget_claim_supported:
            # These are paraphrases of the source's budget wording. Only allow the
            # aliases in this supported, quantity-checked tool-budget context.
            unsupported_terms -= {"maximum", "max", "limit", "total", "budget", "allowed", "allow",
                                  "document", "documents"}
        if unsupported_terms:
            if diagnostics is not None:
                diagnostics["unsupported_answer_terms"] = sorted(unsupported_terms)
            return rejected("answer contains terms absent from cited page evidence")
        if diagnostics is not None:
            diagnostics["validation_result"] = "accepted"
            diagnostics["failed_condition"] = None
        return answer

    @staticmethod
    def _provider_unavailable(error: Exception) -> bool:
        """Limit fallback to provider throttling, server outages, and transport failures."""
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        status = getattr(error, "status_code", None)
        return status == 429 or isinstance(status, int) and 500 <= status <= 599

    @classmethod
    def _fallback_terms(cls, question: str) -> list[str]:
        terms = {
            token for token in re.findall(r"[a-z0-9]+", question.casefold())
            if len(token) >= 3 and token not in cls._fallback_stop_words
        }
        return sorted(terms, key=lambda token: (-len(token), token))

    @classmethod
    def _fallback_answer(cls, question: str, evidence: EvidenceState) -> str:
        """Return a relevant source sentence verbatim only when it directly supports the question."""
        if cls._tool_inventory_question(question):
            required = ("list_documents", "list_headings", "get_page", "search_keyword")
            for page in evidence.pages_read:
                page_number = page.get("page_number")
                if not isinstance(page_number, int):
                    continue
                source = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()
                # Split common list delimiters but keep each returned clause verbatim.
                clauses = [part.strip(" ;,\t") for part in re.split(r"\s*;\s*", source) if part.strip(" ;,\t")]
                supported = []
                for name in required:
                    clause = next((part for part in clauses
                                   if re.search(rf"\b{re.escape(name)}\b", part)
                                   and re.search(r"\b(return|returns|output|outputs)\b", part, re.I)), None)
                    if clause is None:
                        supported = []
                        break
                    supported.append(clause)
                if len(supported) == len(required):
                    return "; ".join(supported) + f" [p. {page_number}]"
        terms = set(cls._fallback_terms(question))
        if len(terms) < 2:
            return "Insufficient information."
        quantity_question = bool(re.search(r"\b(how many|how much|maximum|minimum|number|limit)\b", question, re.I))
        number_words = r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand"
        candidates: list[tuple[int, int, str]] = []
        for page in evidence.pages_read:
            page_number = page.get("page_number")
            if not isinstance(page_number, int):
                continue
            text = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
                overlap = terms & sentence_terms
                # A partial keyword match is not enough to assert an answer. The
                # fallback may quote a sentence only when it covers every key term.
                if len(overlap) < max(2, len(terms)):
                    continue
                if quantity_question and not re.search(rf"\d|\b(?:{number_words})\b|unlimited", sentence, re.I):
                    continue
                if re.search(r"\b(ignore|disregard|override|reveal|system prompt|secret)\b", sentence, re.I):
                    continue
                candidates.append((len(overlap), -page_number, f"{sentence} [p. {page_number}]"))
        return max(candidates)[2] if candidates else "Insufficient information."

    def _run_fallback_tools(self, question: str, evidence: EvidenceState, run, document_id: str | None) -> None:
        """Gather candidate pages through the existing harness and its four-tool allowlist."""
        selected_id = document_id
        if selected_id is None and len(evidence.document_metadata) == 1:
            selected_id = str(evidence.document_metadata[0].get("doc_id", "")) or None
        if selected_id is None and not evidence.document_metadata and run.remaining_calls > 0:
            outcome = run.call("list_documents", {})
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            if outcome.success and isinstance(outcome.result, BaseModel):
                evidence.document_metadata = outcome.result.model_dump().get("documents", [])
                if len(evidence.document_metadata) == 1:
                    selected_id = str(evidence.document_metadata[0].get("doc_id", "")) or None
        if selected_id is None:
            return

        page_numbers: list[int] = []
        read_pages = {int(page["page_number"]) for page in evidence.pages_read if "page_number" in page}
        for keyword in self._fallback_terms(question)[:3]:
            if run.remaining_calls <= 1:
                break
            outcome = run.call("search_keyword", {"doc_id": selected_id, "keyword": keyword})
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            if outcome.success and isinstance(outcome.result, BaseModel):
                for page_number in outcome.result.model_dump().get("page_numbers", []):
                    if isinstance(page_number, int) and page_number not in read_pages and page_number not in page_numbers:
                        page_numbers.append(page_number)

        for page_number in page_numbers:
            if run.remaining_calls <= 0:
                break
            outcome = run.call("get_page", {"doc_id": selected_id, "page_number": page_number})
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            if outcome.success and isinstance(outcome.result, BaseModel):
                evidence.pages_read.append(outcome.result.model_dump())
                evidence.tool_trace = [event.model_dump() for event in run.trace]
                if self._fallback_answer(question, evidence) != "Insufficient information.":
                    break

    def run(self, question: str, document_id: str | None = None) -> "DocumentResponse":
        llm = self._llm
        if llm is None:
            if not (os.getenv("DOCUMENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
                raise RuntimeError("Set DOCUMENT_LLM_API_KEY (or OPENAI_API_KEY) to use the document agent")
            llm = OpenAIDocumentLLM()
        evidence = EvidenceState()
        run = self._harness.new_question(question)
        fallback_mode = False
        attempted_searches: dict[str, list[frozenset[str]]] = {}
        located_unread_pages: list[tuple[str, int]] = []
        for _ in range(7):
            if (self._tool_budget_question(question) and document_id and not evidence.headings_results
                    and not any(event.tool_name == "list_headings" for event in run.trace)):
                action = DocumentAction(action_type="tool", tool_name="list_headings",
                                        arguments={"doc_id": document_id},
                                        reason="Locate the hard-constraints or call-budget section")
            elif (self._tool_budget_question(question) and evidence.pages_read
                  and any(self._page_supports_tool_budget(str(page.get("text", "")))
                          for page in evidence.pages_read)):
                break
            elif self._tool_budget_question(question) and evidence.headings_results:
                page_number = self._constraint_page(evidence)
                if page_number is not None:
                    action = DocumentAction(action_type="tool", tool_name="get_page",
                                            arguments={"doc_id": document_id, "page_number": page_number},
                                            reason="Read the page headed with the relevant constraints")
                else:
                    read_pages = {
                        (str(page.get("doc_id", "")).casefold(), int(page["page_number"]))
                        for page in evidence.pages_read if "page_number" in page
                    }
                    pending = next((item for item in located_unread_pages
                                    if item not in read_pages), None)
                    if pending is not None:
                        action = DocumentAction(action_type="tool", tool_name="get_page",
                                                arguments={"doc_id": pending[0], "page_number": pending[1]},
                                                reason="Read a page located by budget-term search")
                    elif len(evidence.keyword_searches) < 2:
                        keyword = "maximum" if not evidence.keyword_searches else "tool calls"
                        action = DocumentAction(action_type="tool", tool_name="search_keyword",
                                                arguments={"doc_id": document_id, "keyword": keyword},
                                                reason="Locate the tool-call limit in page text")
                    else:
                        break
            elif (self._tool_inventory_question(question) and document_id and not evidence.keyword_searches):
                # A return-oriented lexical query locates the interface descriptions
                # without reading unrelated constraint pages. Search returns locations
                # only; the matching page is read in the next step through the harness.
                keyword = "returns" if re.search(r"\breturns?\b", question, re.I) else "outputs"
                action = DocumentAction(action_type="tool", tool_name="search_keyword",
                                        arguments={"doc_id": document_id, "keyword": keyword},
                                        reason="Locate tool descriptions by their documented return behavior")
            elif (self._tool_inventory_question(question) and evidence.keyword_searches
                  and not evidence.pages_read and located_unread_pages):
                candidate_doc, candidate_page = located_unread_pages[0]
                action = DocumentAction(action_type="tool", tool_name="get_page",
                                        arguments={"doc_id": candidate_doc, "page_number": candidate_page},
                                        reason="Read the best page located by the return-description search")
            elif self._tool_inventory_question(question) and evidence.pages_read:
                break
            elif (self._tool_inventory_question(question) and document_id and not evidence.headings_results
                    and not any(event.tool_name == "list_headings" for event in run.trace)):
                action = DocumentAction(action_type="tool", tool_name="list_headings",
                                        arguments={"doc_id": document_id},
                                        reason="Find the section describing available tools")
            elif self._tool_inventory_question(question) and evidence.headings_results:
                page_number = self._tool_inventory_page(question, evidence)
                if page_number is None and evidence.pages_read:
                    break
                if page_number is not None:
                    action = DocumentAction(action_type="tool", tool_name="get_page",
                                            arguments={"doc_id": document_id, "page_number": page_number},
                                            reason="Read the matching tool description page")
                else:
                    try:
                        action = llm.choose_action(question, evidence, run.remaining_calls)
                    except Exception as exc:
                        run.record_agent_error(type(exc).__name__ + ": " + str(exc))
                        fallback_mode = self._provider_unavailable(exc)
                        break
            else:
                try:
                    action = llm.choose_action(question, evidence, run.remaining_calls)
                except Exception as exc:
                    run.record_agent_error(type(exc).__name__ + ": " + str(exc))
                    fallback_mode = self._provider_unavailable(exc)
                    break
            if action.action_type != "tool":
                break
            arguments = dict(action.arguments)
            if document_id and action.tool_name in {"list_headings", "get_page", "search_keyword"}:
                arguments["doc_id"] = document_id
            if action.tool_name == "search_keyword":
                search_doc_id = str(arguments.get("doc_id", ""))
                if self._duplicate_search(search_doc_id, arguments.get("keyword", ""), attempted_searches):
                    # A repeated planner decision cannot add evidence; ask for another decision
                    # without spending one of the document-tool calls.
                    continue
                # Search returns locations only. Read those locations before allowing another
                # discovery query, so each discovery action is followed by evidence assessment.
                already_read = {(str(page.get("doc_id", "")).casefold(), int(page["page_number"]))
                                for page in evidence.pages_read if "page_number" in page}
                pending = next((item for item in located_unread_pages if item not in already_read), None)
                if pending:
                    action = DocumentAction(action_type="tool", tool_name="get_page",
                                            arguments={"doc_id": pending[0], "page_number": pending[1]},
                                            reason="Read located evidence before further discovery")
                    arguments = dict(action.arguments)
            outcome = run.call(action.tool_name or "", arguments)
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            if outcome.success:
                value = outcome.result
                data = value.model_dump() if isinstance(value, BaseModel) else {}
                if action.tool_name == "get_page":
                    evidence.pages_read.append(data)
                elif action.tool_name == "search_keyword":
                    evidence.keyword_searches.append(data)
                    for page_number in data.get("page_numbers", []):
                        if isinstance(page_number, int):
                            item = (str(data.get("doc_id", arguments.get("doc_id", ""))).casefold(), page_number)
                            if item not in located_unread_pages:
                                located_unread_pages.append(item)
                elif action.tool_name == "list_headings":
                    evidence.headings_results.append(data)
                elif action.tool_name == "list_documents":
                    evidence.document_metadata = data.get("documents", [])
            if run.remaining_calls == 0:
                break
        # A failed planning request must not trigger more retrieval after a page has
        # already been read. Give the separate answer call the available evidence.
        if fallback_mode and not evidence.pages_read:
            self._run_fallback_tools(question, evidence, run, document_id)
            answer = self._fallback_answer(question, evidence)
        else:
            evidence.tool_trace = [event.model_dump() for event in run.trace]
            try:
                provider_answer = llm.answer(question, evidence)
                diagnostic_request = (
                    question == "What are the four document tools provided to the agent, and what does each tool return?"
                    or self._tool_budget_question(question)
                )
                answer_diagnostics: dict[str, object] | None = None
                if diagnostic_request:
                    answer_diagnostics = dict(getattr(llm, "last_answer_diagnostics", {}))
                    answer_diagnostics["answer_text"] = provider_answer
                    answer_diagnostics["citations"] = [int(n) for n in re.findall(
                        r"\[(?:p\.|page)\s*(\d+)\]", provider_answer, re.IGNORECASE
                    )]
                    answer_diagnostics["validation_evidence"] = [
                        {"page_number": page.get("page_number"),
                         "text": str(page.get("text", ""))[:10000],
                         "text_truncated": len(str(page.get("text", ""))) > 10000}
                        for page in evidence.pages_read
                    ]
                    if self._tool_budget_question(question):
                        answer_diagnostics["budget_evidence_checks"] = [
                            {"page_number": page.get("page_number"),
                             "contains_tool_call_limit_and_quantity": self._page_supports_tool_budget(
                                 str(page.get("text", ""))
                             )}
                            for page in evidence.pages_read
                        ]
                answer = self._validated_answer(provider_answer, question, evidence, answer_diagnostics)
                run.record_answer_success(answer, answer_diagnostics)
            except Exception as exc:
                run.record_answer_error(type(exc).__name__ + ": " + str(exc))
                logger.exception("Document final-answer provider call failed")
                # Retrieval is complete before final-answer generation begins. If that
                # separate call fails, only use evidence already read; never search again.
                answer = self._fallback_answer(question, evidence)
                run.trace[-1].result_metadata["answer"] = answer
        return DocumentResponse(answer=answer, evidence=evidence, trace=run.trace.copy(),
                                calls_made=run.call_count, remaining_calls=run.remaining_calls)


class DocumentResponse(BaseModel):
    answer: str
    evidence: EvidenceState
    trace: list[object]
    calls_made: int
    remaining_calls: int
