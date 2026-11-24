"""Minimal client for any OpenAI-compatible API (OpenAI, Gemini, Ollama)."""
import httpx

from .config import Settings

SYSTEM_PROMPT = (
    "Answer the question using only the numbered sources below. "
    "Cite sources like [1] or [2] after the sentences they support. "
    "If the sources do not contain the answer, say you could not find it in the documents. "
    "Do not use outside knowledge."
)


class AIError(Exception):
    pass


class AIClient:
    def __init__(self, settings: Settings):
        self.s = settings
        headers = {"Authorization": f"Bearer {settings.ai_api_key}"} if settings.ai_api_key else {}
        self.http = httpx.AsyncClient(base_url=settings.ai_base_url, headers=headers, timeout=60)

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.s.embed_model)

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.embeddings_enabled or not texts:
            return None
        body = {"model": self.s.embed_model, "input": texts}
        if self.s.embed_dimensions:
            body["dimensions"] = self.s.embed_dimensions
        data = (await self._post("/embeddings", body))["data"]
        # Some providers (Gemini) omit "index"; keep input order unless every item has one.
        if all("index" in d for d in data):
            data = sorted(data, key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    async def answer(self, question: str, sources: list[str]) -> str:
        context = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(sources, start=1))
        body = {
            "model": self.s.chat_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
            ],
        }
        data = await self._post("/chat/completions", body)
        return data["choices"][0]["message"]["content"].strip()

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self.http.post(path, json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise AIError("The AI service is unavailable right now. Please try again in a minute.") from e
