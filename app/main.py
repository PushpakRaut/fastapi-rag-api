import io
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .ai import AIClient, AIError
from .config import Settings
from .rag import chunk_pages, hybrid_search
from .store import RateLimiter, Session, SessionStore

settings = Settings()
COOKIE = "sid"
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # reasoning models (qwen3) emit this first


@asynccontextmanager
async def lifespan(app: FastAPI):
    ai = app.state.ai = AIClient(settings)
    yield
    await ai.http.aclose()


app = FastAPI(
    title="Docs Q&A API",
    description="Upload PDFs, then ask questions. Answers use only your documents and cite the source pages. "
    "Retrieval is hybrid: BM25 keyword search plus embeddings, merged with reciprocal rank fusion.",
    version="1.0.0",
    lifespan=lifespan,
)
store = SessionStore(settings.max_sessions, settings.session_idle_seconds)
upload_limit = RateLimiter(settings.uploads_per_window, settings.rate_window_seconds)
question_limit = RateLimiter(settings.questions_per_window, settings.rate_window_seconds)


@app.exception_handler(AIError)
async def ai_error(_: Request, e: AIError):
    return JSONResponse(status_code=502, content={"detail": str(e)})


def client_ip(request: Request) -> str:
    # Only trust a header your proxy sets and overwrites (Cloudflare's CF-Connecting-IP on Render).
    if settings.client_ip_header:
        ip = request.headers.get(settings.client_ip_header, "").strip()
        if ip:
            return ip
    return request.client.host if request.client else "unknown"


def session(request: Request, response: Response) -> Session:
    """Each visitor gets a private document space, keyed by an HttpOnly cookie."""
    sid = request.cookies.get(COOKIE)
    if not sid or len(sid) > 64:
        sid = secrets.token_urlsafe(24)
        response.set_cookie(COOKIE, sid, httponly=True, samesite="strict", secure=settings.secure_cookies)
    return store.get_or_create(sid)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    n: int
    file: str
    page: int
    excerpt: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    search: str


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents(s: Session = Depends(session)) -> list[str]:
    return list(s.docs)


@app.post("/documents", status_code=201)
async def upload(file: UploadFile, request: Request, s: Session = Depends(session)):
    if not upload_limit.allow(client_ip(request)):
        raise HTTPException(429, "Too many uploads. Please wait a few minutes.")
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"PDF is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Please upload a PDF file.")
    name = (file.filename or "document.pdf").rsplit("/", 1)[-1][:100]
    if name not in s.docs and len(s.docs) >= settings.max_docs_per_session:
        raise HTTPException(409, f"You can keep up to {settings.max_docs_per_session} documents. Delete one first.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > settings.max_pages:
            raise HTTPException(413, f"PDF has more than {settings.max_pages} pages.")
        pages = [p.extract_text() or "" for p in reader.pages]
    except PdfReadError:
        raise HTTPException(422, "Could not read this PDF.")
    chunks = chunk_pages(name, pages)
    if not chunks:
        raise HTTPException(422, "No text found. Scanned PDFs need OCR first.")
    vectors = await request.app.state.ai.embed([c.text for c in chunks])
    for c, v in zip(chunks, vectors or []):
        c.vector = v
    s.docs[name] = chunks
    return {"file": name, "pages": len(pages), "chunks": len(chunks)}


@app.delete("/documents/{name}", status_code=204)
def delete(name: str, s: Session = Depends(session)):
    if s.docs.pop(name, None) is None:
        raise HTTPException(404, "No such document.")


@app.post("/ask")
async def ask(q: Question, request: Request, s: Session = Depends(session)) -> Answer:
    if not question_limit.allow(client_ip(request)):
        raise HTTPException(429, "Too many questions. Please wait a few minutes.")
    chunks = s.chunks()
    if not chunks:
        raise HTTPException(400, "Upload a PDF first.")
    ai: AIClient = request.app.state.ai
    qv = await ai.embed([q.question])
    hits = hybrid_search(q.question, chunks, qv[0] if qv else None)
    if not hits:
        return Answer(answer="I could not find anything about that in your documents.", sources=[], search="keyword")
    text = THINK.sub("", await ai.answer(q.question, [h.text for h in hits])).strip()
    sources = [Source(n=n, file=h.doc, page=h.page, excerpt=h.text[:240]) for n, h in enumerate(hits, start=1)]
    return Answer(answer=text, sources=sources, search="hybrid" if qv else "keyword")

