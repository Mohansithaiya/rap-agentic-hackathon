import fitz
from fastapi.testclient import TestClient

from app.api import main


def make_pdf() -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Annual revenue was 42 million dollars.")
    data = pdf.tobytes()
    pdf.close()
    return data


def test_upload_and_ask_selected_document(monkeypatch):
    client = TestClient(main.app)
    uploaded = client.post("/documents/upload", content=make_pdf(), headers={
        "content-type": "application/pdf", "x-filename": "annual-report.pdf",
    })
    assert uploaded.status_code == 200
    metadata = uploaded.json()
    assert metadata["title"] == "annual-report"
    assert metadata["page_count"] == 1
    seen = {}

    class FakeAgent:
        def __init__(self, harness):
            pass

        def run(self, question, document_id=None):
            seen.update(question=question, document_id=document_id)
            return type("Response", (), {"model_dump": lambda self: {
                "answer": "Revenue was 42 million dollars [p. 1]",
                "calls_made": 2, "remaining_calls": 4,
                "trace": [{"tool_name": "get_page", "success": True, "tool_call_number": 2}],
                "evidence": {},
            }})()

    monkeypatch.setattr(main, "DocumentAgent", FakeAgent)
    result = client.post("/documents/ask", json={
        "question": "What was revenue?", "doc_id": metadata["doc_id"],
    })
    assert result.status_code == 200
    assert seen == {"question": "What was revenue?", "document_id": metadata["doc_id"]}
    assert result.json()["calls_made"] == 2
    assert result.json()["trace"][0]["tool_name"] == "get_page"


def test_upload_rejects_non_pdf():
    client = TestClient(main.app)
    response = client.post("/documents/upload", content=b"not a pdf", headers={
        "content-type": "application/pdf",
    })
    assert response.status_code == 400
