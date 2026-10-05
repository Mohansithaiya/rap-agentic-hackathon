import fitz
import pytest

from app.documents import DocumentStore, get_page, list_documents, list_headings, register_pdf, search_keyword
from app.documents.store import DocumentNotFoundError, InvalidPageNumberError


@pytest.fixture
def loaded_store(tmp_path):
    path = tmp_path / "report.pdf"
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Quarterly Report\nRevenue grew steadily")
    second = pdf.new_page()
    second.insert_text((72, 72), "Risk review\nRevenue outlook is positive")
    pdf.set_toc([[1, "Quarterly Report", 1], [1, "Risk review", 2]])
    pdf.save(path)
    pdf.close()
    store = DocumentStore()
    metadata = register_pdf(path, store)
    return store, metadata


def test_loading_pdf_and_list_documents(loaded_store):
    store, metadata = loaded_store
    result = list_documents(store)
    assert result.documents == [metadata]
    assert metadata.title == "report"
    assert metadata.page_count == 2
    assert not hasattr(result, "text")


def test_list_headings_returns_outline_only(loaded_store):
    store, metadata = loaded_store
    result = list_headings(metadata.doc_id, store)
    assert [(heading.title, heading.page_number) for heading in result.headings] == [
        ("Quarterly Report", 1), ("Risk review", 2)
    ]
    assert not hasattr(result, "text")


def test_get_page_returns_exactly_requested_page(loaded_store):
    store, metadata = loaded_store
    result = get_page(metadata.doc_id, 2, store)
    assert result.page_number == 2
    assert "Risk review" in result.text
    assert "Quarterly Report" not in result.text


def test_search_keyword_is_case_insensitive_page_numbers_only(loaded_store):
    store, metadata = loaded_store
    result = search_keyword(metadata.doc_id, "REVENUE", store)
    assert result.page_numbers == [1, 2]
    assert set(result.model_dump()) == {"doc_id", "keyword", "page_numbers"}
    assert "Revenue" not in str(result.model_dump())
    assert search_keyword(metadata.doc_id, "missing", store).page_numbers == []


def test_unknown_document_id_is_controlled(loaded_store):
    store, _ = loaded_store
    with pytest.raises(DocumentNotFoundError, match="Unknown document ID"):
        list_headings("unknown", store)


@pytest.mark.parametrize("page_number", [0, 3, -1])
def test_invalid_page_number_is_controlled(loaded_store, page_number):
    store, metadata = loaded_store
    with pytest.raises(InvalidPageNumberError):
        get_page(metadata.doc_id, page_number, store)
