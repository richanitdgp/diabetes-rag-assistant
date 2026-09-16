"""Build the vector index: parse -> embed -> load into MongoDB Atlas.

A plain script (not a notebook) so it can run unattended — locally, in CI,
or on a schedule — whenever data/raw/manifest.json changes (e.g. after an
annual guideline update via download_sources.py). Ingestion is deliberately
decoupled from deploying the app: it writes to a shared Atlas cluster the
running API only ever reads from, so re-running it is how the knowledge
base gets refreshed, independent of any deploy.

Pipeline:
  1. Read data/raw/manifest.json for sources with status "ok".
  2. Parse each into structure-respecting Chunks (see parsers.py).
  3. Embed chunk text with Gemini's gemini-embedding-001.
  4. Upsert (_id, embedding, text, metadata) into a MongoDB Atlas collection,
     and create its Atlas Vector Search index if it doesn't exist yet.

Usage (run from the repo root, as a module so relative imports resolve):
    python -m ingestion.build_index              # full pipeline
    python -m ingestion.build_index --parse-only  # steps 1-2 only, no API key needed
    python -m ingestion.build_index --chunks-out data/processed/chunks.jsonl

Requires GOOGLE_API_KEY or GEMINI_API_KEY for embedding, and MONGODB_URI for
the Atlas connection (see .env.example) for the embed/load steps. If
MONGODB_COLLECTION already holds vectors from a different embedding model,
drop its vector search index before re-running — an Atlas vector index is
locked to whatever numDimensions it was created with.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable

from ingestion.parsers import Chunk, parse_source

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.json"
DEFAULT_CHUNKS_OUT = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
MONGODB_DB = "diabetes_rag"
MONGODB_COLLECTION = "chunks"
VECTOR_INDEX_NAME = "chunks_vector_index"
VECTOR_FIELD = "embedding"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 100


@traceable(name="parse_all_sources")
def parse_all_sources() -> list[Chunk]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    chunks: list[Chunk] = []
    for source in manifest["sources"]:
        if source.get("status") != "ok":
            print(f"skipping {source['id']}: status={source.get('status')!r}")
            continue
        source_chunks = parse_source(REPO_ROOT / source["local_path"], source)
        print(f"{source['id']}: {len(source_chunks)} chunks")
        chunks.extend(source_chunks)
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            record = {"id": chunk.id, "text": chunk.text, **chunk.metadata()}
            fh.write(json.dumps(record) + "\n")
    print(f"wrote {len(chunks)} chunks to {out_path.relative_to(REPO_ROOT)}")


@traceable(name="embed_chunks", run_type="embedding")
def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    from google import genai
    from google.genai import types

    # genai.Client() with no args reads GOOGLE_API_KEY (or GEMINI_API_KEY as a
    # fallback) from the environment — same pattern as OpenAI() before it.
    client = genai.Client()
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[chunk.text for chunk in batch],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.extend(item.values for item in response.embeddings)
        print(f"embedded {start + len(batch)}/{len(chunks)}")
    return embeddings


@traceable(name="load_into_mongo")
def load_into_mongo(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    from pymongo import MongoClient, ReplaceOne

    mongodb_uri = os.environ["MONGODB_URI"]
    client = MongoClient(mongodb_uri)
    collection = client[MONGODB_DB][MONGODB_COLLECTION]

    operations = [
        ReplaceOne(
            {"_id": chunk.id},
            {"_id": chunk.id, "text": chunk.text, VECTOR_FIELD: embedding, **chunk.metadata()},
            upsert=True,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    result = collection.bulk_write(operations)
    print(
        f"upserted into '{MONGODB_DB}.{MONGODB_COLLECTION}': "
        f"{result.upserted_count} inserted, {result.modified_count} updated"
    )

    _ensure_vector_index(collection, dimensions=len(embeddings[0]))


def _ensure_vector_index(collection, dimensions: int) -> None:
    """Create the Atlas Vector Search index if it doesn't already exist.

    Idempotent, so it's safe to call on every ingestion run. NOTE:
    unverified against a real Atlas cluster as of writing (no cluster
    available in the environment this was written in) — the shape here
    follows pymongo's documented SearchIndexModel/create_search_index API;
    double-check against your pymongo version if this errors.
    """
    from pymongo.operations import SearchIndexModel

    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if VECTOR_INDEX_NAME in existing:
        print(f"vector index '{VECTOR_INDEX_NAME}' already exists, skipping")
        return

    model = SearchIndexModel(
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": VECTOR_FIELD,
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                }
            ]
        },
    )
    collection.create_search_index(model)
    print(
        f"created vector index '{VECTOR_INDEX_NAME}' ({dimensions} dims) — "
        "Atlas builds it asynchronously, so it may take a minute before queries work"
    )


@traceable(name="build_index")
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse sources into chunks (no embedding, no API key needed).",
    )
    parser.add_argument(
        "--chunks-out",
        type=Path,
        default=DEFAULT_CHUNKS_OUT,
        help=f"Where to write the parsed chunks as JSONL (default: {DEFAULT_CHUNKS_OUT.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    load_dotenv()

    chunks = parse_all_sources()
    write_chunks_jsonl(chunks, args.chunks_out)

    if args.parse_only:
        return

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit(
            "GOOGLE_API_KEY / GEMINI_API_KEY is not set (see .env.example). Use --parse-only to skip embedding."
        )
    if not os.environ.get("MONGODB_URI"):
        raise SystemExit("MONGODB_URI is not set (see .env.example). Use --parse-only to skip embedding/loading.")

    embeddings = embed_chunks(chunks)
    load_into_mongo(chunks, embeddings)


if __name__ == "__main__":
    main()
