from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashEmbeddingProvider:
    """Deterministic local embedding fallback.

    It is not semantically strong, but keeps skill memory functional when no
    embedding service is configured. Chroma still stores and searches vectors.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in text.lower().replace("\n", " ").split(" ") if token]
        if not tokens:
            tokens = [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        self.client = OpenAI(**kwargs)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider(skill_cfg: dict) -> EmbeddingProvider:
    vector_cfg = skill_cfg.get("vector_store") or {}
    backend = (vector_cfg.get("embedding_backend") or "hash").lower()
    model = vector_cfg.get("embedding_model") or "hash-384"
    if backend == "openai":
        cognition_cfg = skill_cfg.get("_cognition_openai") or {}
        try:
            return OpenAIEmbeddingProvider(
                model=model,
                api_key=cognition_cfg.get("api_key"),
                base_url=cognition_cfg.get("base_url"),
                organization=cognition_cfg.get("organization"),
            )
        except Exception:
            return HashEmbeddingProvider()
    return HashEmbeddingProvider()
