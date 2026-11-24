import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Any OpenAI-compatible endpoint: Ollama locally, Gemini or OpenAI when hosted.
    ai_base_url: str = os.getenv("AI_BASE_URL", "http://localhost:11434/v1")
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen3:14b")
    embed_model: str = os.getenv("EMBED_MODEL", "")  # empty = keyword (BM25) search only
    embed_dimensions: int = _int("EMBED_DIMENSIONS", 0)

    max_upload_bytes: int = _int("MAX_UPLOAD_MB", 5) * 1024 * 1024
    max_pages: int = _int("MAX_PAGES", 20)
    max_docs_per_session: int = _int("MAX_DOCS_PER_SESSION", 5)
    max_sessions: int = _int("MAX_SESSIONS", 200)
    session_idle_seconds: int = _int("SESSION_IDLE_MINUTES", 30) * 60
    uploads_per_window: int = _int("UPLOADS_PER_10_MIN", 5)
    questions_per_window: int = _int("QUESTIONS_PER_10_MIN", 30)
    rate_window_seconds: int = 600
    # Header holding the real client IP, set by your proxy (e.g. cf-connecting-ip). Empty = use the socket peer.
    client_ip_header: str = os.getenv("CLIENT_IP_HEADER", "").lower()
    secure_cookies: bool = os.getenv("SECURE_COOKIES", "false").lower() == "true"
