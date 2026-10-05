"""Structured document tools over the registered PDF store."""

from pathlib import Path

from .models import DocumentListResult, DocumentMetadata, HeadingsResult, PageResult, SearchResult
from .store import DocumentStore, document_store


def register_pdf(pdf_path: str | Path, store: DocumentStore = document_store) -> DocumentMetadata:
    """Load a PDF into the document store and return metadata only."""
    return store.register_pdf(pdf_path)


def list_documents(store: DocumentStore = document_store) -> DocumentListResult:
    """List registered document metadata without exposing page text."""
    return DocumentListResult(documents=store.list_documents())


def list_headings(doc_id: str, store: DocumentStore = document_store) -> HeadingsResult:
    """Return the outline or fallback headings for a registered document."""
    document = store.get(doc_id)
    return HeadingsResult(doc_id=doc_id, headings=document.headings.copy())


def get_page(doc_id: str, page_number: int, store: DocumentStore = document_store) -> PageResult:
    """Return text from exactly one 1-based page."""
    return PageResult(doc_id=doc_id, page_number=page_number, text=store.get_page(doc_id, page_number))


def search_keyword(doc_id: str, keyword: str, store: DocumentStore = document_store) -> SearchResult:
    """Find case-insensitive lexical matches and return page numbers only."""
    if not keyword:
        raise ValueError("Keyword must not be empty")
    document = store.get(doc_id)
    needle = keyword.casefold()
    matches = [
        page_number
        for page_number, text in enumerate(document.pages, start=1)
        if needle in text.casefold()
    ]
    return SearchResult(doc_id=doc_id, keyword=keyword, page_numbers=matches)
