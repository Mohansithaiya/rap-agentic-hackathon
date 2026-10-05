"""PDF document registration and page-scoped tools."""

from .store import DocumentStore, document_store
from .tools import get_page, list_documents, list_headings, register_pdf, search_keyword

__all__ = ["DocumentStore", "document_store", "register_pdf", "list_documents", "list_headings", "get_page", "search_keyword"]
