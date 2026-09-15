"""Vector similarity retrieval over the MongoDB Atlas Vector Search index.

Embeds the query with the same Gemini embedding model used at ingestion time
(see ingestion/build_index.py), but with task_type="RETRIEVAL_QUERY" rather
than "RETRIEVAL_DOCUMENT" — Gemini's embedding model is asymmetric, so query
and document text are embedded differently for best retrieval quality.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ingestion.build_index import (
    EMBEDDING_MODEL,
    MONGODB_COLLECTION,
    MONGODB_DB,
    VECTOR_FIELD,
    VECTOR_INDEX_NAME,
)


@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    source_title: str
    publisher: str
    section_title: str
    url: str
    score: float  # Atlas vectorSearchScore: higher is more similar (unlike Chroma's distance).
    publication_year: Optional[int] = None
    page: Optional[int] = None


@lru_cache(maxsize=1)
def _collection():
    from pymongo import MongoClient

    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is not set (see .env.example). Run `python -m ingestion.build_index` "
            "first to populate the Atlas collection and vector index."
        )
    client = MongoClient(mongodb_uri)
    return client[MONGODB_DB][MONGODB_COLLECTION]


def _embed_query(query: str) -> list[float]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
    except Exception:
        print(f"_embed_query() failed calling Gemini (model={EMBEDDING_MODEL!r}):", flush=True)
        traceback.print_exc()
        raise
    return response.embeddings[0].values


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Top-k vector similarity search for `query` against the Atlas index."""
    print(f"retrieve() called: query={query!r} top_k={top_k}", flush=True)
    query_embedding = _embed_query(query)
    print(f"Query embedding length: {len(query_embedding)}", flush=True)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": query_embedding,
                "numCandidates": max(top_k * 10, 100),
                "limit": top_k,
            }
        },
        {
            "$project": {
                "text": 1,
                "source_id": 1,
                "source_title": 1,
                "publisher": 1,
                "section_title": 1,
                "url": 1,
                "publication_year": 1,
                "page": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    print(f"Running MongoDB query search aggregation pipeline: {pipeline}", flush=True)
    try:
        results = list(_collection().aggregate(pipeline))
    except Exception:
        print("retrieve() failed running the Atlas $vectorSearch aggregation:", flush=True)
        traceback.print_exc()
        raise
    print(f"Atlas returned {len(results)} chunk(s)", flush=True)

    return [
        RetrievedChunk(
            text=doc["text"],
            source_id=doc["source_id"],
            source_title=doc["source_title"],
            publisher=doc["publisher"],
            section_title=doc["section_title"],
            url=doc["url"],
            score=doc["score"],
            publication_year=doc.get("publication_year"),
            page=doc.get("page"),
        )
        for doc in results
    ]
