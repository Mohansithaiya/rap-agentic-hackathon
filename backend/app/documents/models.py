"""Typed public results for document tools."""

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    page_count: int


class DocumentListResult(BaseModel):
    documents: list[DocumentMetadata]


class Heading(BaseModel):
    title: str
    level: int
    page_number: int


class HeadingsResult(BaseModel):
    doc_id: str
    headings: list[Heading]


class PageResult(BaseModel):
    doc_id: str
    page_number: int
    text: str


class SearchResult(BaseModel):
    doc_id: str
    keyword: str
    page_numbers: list[int]
