"""Vector similarity retrieval over the local Chroma index.

Embeds the query with the same Gemini embedding model used at ingestion time
(see ingestion/build_index.py), but with task_type="RETRIEVAL_QUERY" rather
than "RETRIEVAL_DOCUMENT" — Gemini's embedding model is asymmetric, so query
and document text are embedded differently for best retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ingestion.build_index import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    source_title: str
    publisher: str
    section_title: str
    url: str
    distance: float
    publication_year: Optional[int] = None
    page: Optional[int] = None


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"No index found at {CHROMA_DIR}. Run `python -m ingestion.build_index` first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def _embed_query(query: str) -> list[float]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Top-k vector similarity search for `query` against the local index."""
    query_embedding = _embed_query(query)
    results = _collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(
            text=text,
            source_id=meta["source_id"],
            source_title=meta["source_title"],
            publisher=meta["publisher"],
            section_title=meta["section_title"],
            url=meta["url"],
            distance=distance,
            publication_year=meta.get("publication_year"),
            page=meta.get("page"),
        )
        for text, meta, distance in zip(documents, metadatas, distances)
    ]
