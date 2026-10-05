"""Per-document PDF storage; page text stays behind page-scoped methods."""

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import fitz

from .models import DocumentMetadata, Heading


class DocumentNotFoundError(LookupError):
    """Raised when a tool receives an unregistered document ID."""


class InvalidPageNumberError(ValueError):
    """Raised when a requested page is outside the document's page range."""


@dataclass
class _StoredDocument:
    metadata: DocumentMetadata
    pages: list[str]
    headings: list[Heading]


class DocumentStore:
    """In-memory registry populated by loading PDFs from paths."""

    def __init__(self) -> None:
        self._documents: dict[str, _StoredDocument] = {}

    def register_pdf(self, pdf_path: str | Path) -> DocumentMetadata:
        """Load and register a PDF, extracting text separately for each page."""
        path = Path(pdf_path)
        with fitz.open(path) as pdf:
            pages = [page.get_text("text") for page in pdf]
            toc = pdf.get_toc()
            title = (pdf.metadata or {}).get("title") or path.stem
            headings = [
                Heading(title=heading_title, level=level, page_number=page_number)
                for level, heading_title, page_number in toc
                if 1 <= page_number <= len(pages)
            ]
            if not headings:
                headings = self._fallback_headings(pages)

        metadata = DocumentMetadata(doc_id=str(uuid4()), title=title, page_count=len(pages))
        self._documents[metadata.doc_id] = _StoredDocument(metadata, pages, headings)
        return metadata

    @staticmethod
    def _fallback_headings(pages: list[str]) -> list[Heading]:
        """Use short, non-empty lines as a modest TOC fallback."""
        result: list[Heading] = []
        for page_number, text in enumerate(pages, start=1):
            for line in text.splitlines():
                candidate = line.strip()
                if candidate and len(candidate) <= 100 and len(candidate.split()) <= 12:
                    result.append(Heading(title=candidate, level=1, page_number=page_number))
        return result

    def list_documents(self) -> list[DocumentMetadata]:
        return [stored.metadata.model_copy() for stored in self._documents.values()]

    def get(self, doc_id: str) -> _StoredDocument:
        try:
            return self._documents[doc_id]
        except KeyError:
            raise DocumentNotFoundError(f"Unknown document ID: {doc_id}") from None

    def get_page(self, doc_id: str, page_number: int) -> str:
        document = self.get(doc_id)
        if isinstance(page_number, bool) or not 1 <= page_number <= len(document.pages):
            raise InvalidPageNumberError(
                f"Invalid page number {page_number}; valid range is 1-{len(document.pages)}"
            )
        return document.pages[page_number - 1]


document_store = DocumentStore()
