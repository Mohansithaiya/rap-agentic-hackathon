# Budgeted Document-Answering Agent

A tool-mediated PDF question-answering agent that retrieves page evidence within a fixed budget and checks answers before returning them.

## Problem

Answer questions about an uploaded PDF with traceable, page-cited evidence while limiting document access and avoiding unsupported claims.

## Key constraints

- PDF content is accessed only through the four registered document tools.
- A question can use at most **6 document-tool calls**. The separate final-answer LLM call does not add a document-tool call.
- PDF text is untrusted input and cannot override the user's request or system instructions.
- Unsupported or insufficient evidence returns **“Insufficient information.”**

## Solution / architecture

The FastAPI service accepts a PDF and a question. The document agent plans retrieval, and every document action goes through an allowlisted harness. The harness enforces the call limit and records a tool trace. After retrieval, a separate LLM call drafts the answer; evidence validation checks its cited pages and factual terms against the retrieved page text.

## Four document tools

- `list_documents()` — returns titles and metadata for registered documents.
- `list_headings(doc_id)` — returns document headings or the available fallback headings.
- `search_keyword(doc_id, keyword)` — returns page numbers containing a case-insensitive literal keyword match.
- `get_page(doc_id, page_number)` — returns the text of one page.

## Budget-aware harness

The harness allows only those four tools, counts each execution attempt toward the six-call limit, avoids duplicate reads of the same page, and records completed, failed, or blocked actions in the trace. The final-answer call is separate from this document-tool budget.

## Evidence validation / “Proof Before Answer”

Answers must cite pages retrieved with `get_page`. Citations to unread pages, missing citations, irrelevant evidence, or answer terms unsupported by cited text are rejected. The agent returns **“Insufficient information.”** when the provider abstains or validation cannot establish support. The UI displays the tool trace, call count, and cited pages.

## Tech stack

Python, FastAPI, Pydantic, PyMuPDF, and the OpenAI-compatible Python client for the configured LLM endpoint. The document chat UI is served by the FastAPI application.

## How to run locally

From the repository root, install the runtime and test dependencies:

```bash
python -m pip install fastapi uvicorn pydantic pymupdf openai python-dotenv pytest
```

Configure `DOCUMENT_LLM_API_KEY` (or `OPENAI_API_KEY`) and, when required by your endpoint, `DOCUMENT_LLM_BASE_URL`. `DOCUMENT_LLM_MODEL` can select the configured model; the code default is `gemini-2.5-flash-lite`.

Start the API:

```bash
python -m uvicorn app.api.main:app --reload
```

Open <http://127.0.0.1:8000/documents>, upload a PDF, and ask a question.

## Example workflow

1. Upload a PDF in the document chat; it is registered in the in-memory document store.
2. Ask a question. The agent uses discovery tools such as `search_keyword` or `list_headings` to locate relevant pages.
3. The agent calls `get_page` for candidate evidence, within the six-call limit.
4. A separate final-answer call drafts a response with page citations; validation checks it against the retrieved page text.
5. Review the answer, cited pages, and tool trace in the UI.

## Testing

Run the test suite from the repository root:

```bash
pytest -q
```

## Known limitations

- Uploaded PDFs and their extracted text are held in memory and are not persisted across server restarts.
- Keyword search is literal and case-insensitive; it does not find semantic matches or synonyms.
- Evidence validation is lexical, so valid paraphrases may be rejected when their wording is not supported directly by the retrieved text.
- The service depends on a reachable, correctly configured LLM endpoint for planning and final-answer generation.
