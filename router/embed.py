"""Thin embeddings client. SIE /v1/embeddings; falls back to DashScope if
SIE_API_KEY is unset (see STUB/SKIP in CLAUDE.md). Batches of 10, retry once."""

import httpx

from router import config

BATCH = 10  # DashScope caps embedding input at 10 items; harmless for SIE


class EmbedClient:
    def __init__(self):
        if config.SIE_API_KEY:
            self.backend = "sie"
            self.base_url = config.SIE_BASE_URL
            self.api_key = config.SIE_API_KEY
            self.model = config.SIE_EMBED_MODEL
        else:
            self.backend = "dashscope"
            self.base_url = config.DASHSCOPE_BASE_URL
            self.api_key = config.DASHSCOPE_API_KEY
            self.model = config.DASHSCOPE_EMBED_MODEL
        self.http = httpx.Client(timeout=30)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self._call(texts[i : i + BATCH]))
        return out

    def _call(self, batch: list[str], _retried: bool = False) -> list[list[float]]:
        try:
            r = self.http.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception:
            if _retried:
                raise
            return self._call(batch, _retried=True)


_client: EmbedClient | None = None


def get_client() -> EmbedClient:
    global _client
    if _client is None:
        _client = EmbedClient()
    return _client
