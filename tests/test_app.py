import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import main
from app.rag import Chunk, bm25_rank, chunk_pages, hybrid_search, rrf, tokenize


def make_pdf(pages: list[str]) -> bytes:
    """Tiny valid PDF with one line of text per page."""
    w = PdfWriter()
    for text in pages:
        page = w.add_blank_page(612, 792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class FakeAI:
    embeddings_enabled = False

    def __init__(self):
        self.last_sources = None

    async def embed(self, texts):
        return None

    async def answer(self, question, sources):
        self.last_sources = sources
        return "<think>hidden</think>Refunds take 14 days [1]."


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        main.app.state.ai = FakeAI()
        main.store.sessions.clear()
        main.upload_limit.hits.clear()
        main.question_limit.hits.clear()
        yield c


# --- retrieval ---

def test_chunks_overlap_and_keep_page_numbers():
    chunks = chunk_pages("a.pdf", ["word " * 400, "second page"], size=300, overlap=50)
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert all(len(c.text) <= 300 for c in chunks)
    assert len([c for c in chunks if c.page == 1]) >= 6


def test_bm25_prefers_matching_chunk_and_skips_unrelated():
    chunks = [Chunk("d", 1, t, tokenize(t)) for t in ["refund policy is 14 days", "shipping takes a week", "office hours"]]
    assert bm25_rank("how long is the refund", chunks) == [0]


def test_rrf_rewards_items_ranked_by_both_lists():
    assert rrf([1, 2, 3], [3, 1, 2])[0] == 1


def test_hybrid_uses_vectors_when_keywords_miss():
    chunks = [Chunk("d", 1, "alpha", tokenize("alpha"), [1, 0]), Chunk("d", 2, "beta", tokenize("beta"), [0, 1])]
    assert hybrid_search("synonym", chunks, [0, 1], top_k=1)[0].page == 2


# --- API ---

def test_upload_and_ask_cites_pages(client):
    pdf = make_pdf(["Refunds are processed within 14 days.", "Shipping is free over 50 dollars."])
    r = client.post("/documents", files={"file": ("policy.pdf", pdf, "application/pdf")})
    assert r.status_code == 201 and r.json()["pages"] == 2
    r = client.post("/ask", json={"question": "How long do refunds take?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "Refunds take 14 days [1]."  # think block stripped
    assert body["sources"][0] == {"n": 1, "file": "policy.pdf", "page": 1, "excerpt": "Refunds are processed within 14 days."}
    assert body["search"] == "keyword"


def test_documents_are_private_per_visitor(client):
    client.post("/documents", files={"file": ("a.pdf", make_pdf(["secret"]), "application/pdf")})
    with TestClient(main.app) as other:
        assert other.get("/documents").json() == []
    assert client.get("/documents").json() == ["a.pdf"]


def test_rejects_non_pdf_and_ask_without_docs(client):
    assert client.post("/documents", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 415
    assert client.post("/ask", json={"question": "hi"}).status_code == 400
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_upload_rate_limit(client):
    pdf = make_pdf(["text"])
    codes = [client.post("/documents", files={"file": (f"{i}.pdf", pdf, "application/pdf")}).status_code for i in range(6)]
    assert codes[:5] == [201] * 5 and codes[5] == 429


def test_document_cap_and_delete(client):
    pdf = make_pdf(["text"])
    main.upload_limit.limit = 100
    try:
        for i in range(5):
            client.post("/documents", files={"file": (f"{i}.pdf", pdf, "application/pdf")})
        assert client.post("/documents", files={"file": ("6.pdf", pdf, "application/pdf")}).status_code == 409
        assert client.delete("/documents/0.pdf").status_code == 204
        assert client.delete("/documents/0.pdf").status_code == 404
    finally:
        main.upload_limit.limit = main.settings.uploads_per_window


def test_client_ip_uses_proxy_added_entry(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(main, "settings", SimpleNamespace(trust_forwarded_for=True))
    req = SimpleNamespace(headers={"x-forwarded-for": "6.6.6.6, 1.2.3.4"}, client=SimpleNamespace(host="10.0.0.1"))
    assert main.client_ip(req) == "1.2.3.4"
