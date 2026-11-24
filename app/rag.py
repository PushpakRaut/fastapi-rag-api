"""Chunking and hybrid retrieval (BM25 keyword search + embedding similarity, merged with reciprocal rank fusion)."""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Chunk:
    doc: str
    page: int
    text: str
    tokens: list[str] = field(default_factory=list)
    vector: list[float] | None = None


def chunk_pages(doc: str, pages: list[str], size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping windows of about `size` characters, breaking on whitespace."""
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                space = text.rfind(" ", start + size // 2, end)
                end = space if space > 0 else end
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(doc, page_no, piece, tokenize(piece)))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def bm25_rank(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indexes of chunks ordered by BM25 score; chunks with no query term are left out."""
    terms = set(tokenize(query))
    if not terms or not chunks:
        return []
    n = len(chunks)
    avg_len = sum(len(c.tokens) for c in chunks) / n or 1
    df = Counter(t for c in chunks for t in set(c.tokens) & terms)
    scores = []
    for i, c in enumerate(chunks):
        tf = Counter(t for t in c.tokens if t in terms)
        s = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(c.tokens) / avg_len))
        if s > 0:
            scores.append((s, i))
    return [i for _, i in sorted(scores, reverse=True)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def vector_rank(query_vector: list[float], chunks: list[Chunk]) -> list[int]:
    scored = [(cosine(query_vector, c.vector), i) for i, c in enumerate(chunks) if c.vector]
    return [i for _, i in sorted(scored, reverse=True)]


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: items ranked well by any list rise to the top."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def hybrid_search(query: str, chunks: list[Chunk], query_vector: list[float] | None, top_k: int = 4) -> list[Chunk]:
    rankings = [bm25_rank(query, chunks)]
    if query_vector:
        rankings.append(vector_rank(query_vector, chunks))
    return [chunks[i] for i in rrf(*rankings)[:top_k]]
