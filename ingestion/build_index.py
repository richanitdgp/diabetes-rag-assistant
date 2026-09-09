"""Build the local vector index: parse -> embed -> load into Chroma.

A plain script (not a notebook) so it can run unattended — locally, in CI,
or on a schedule — whenever data/raw/manifest.json changes (e.g. after an
annual guideline update via download_sources.py).

Pipeline:
  1. Read data/raw/manifest.json for sources with status "ok".
  2. Parse each into structure-respecting Chunks (see parsers.py).
  3. Embed chunk text with Gemini's gemini-embedding-001.
  4. Upsert (id, embedding, document, metadata) into a persistent local
     Chroma collection at data/chroma/.

Usage (run from the repo root, as a module so relative imports resolve):
    python -m ingestion.build_index              # full pipeline
    python -m ingestion.build_index --parse-only  # steps 1-2 only, no API key needed
    python -m ingestion.build_index --chunks-out data/processed/chunks.jsonl

Requires GOOGLE_API_KEY or GEMINI_API_KEY (see .env.example) for the embed/load
steps. If data/chroma/ already holds vectors from a different embedding
model, delete it before re-running — a Chroma collection is locked to
whatever vector dimensionality its first entries were written with.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion.parsers import Chunk, parse_source

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "manifest.json"
DEFAULT_CHUNKS_OUT = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
CHROMA_COLLECTION = "diabetes-guidelines"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 100


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


def load_into_chroma(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        CHROMA_COLLECTION,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata() for chunk in chunks],
    )
    print(f"upserted {len(chunks)} chunks into '{CHROMA_COLLECTION}' at {CHROMA_DIR.relative_to(REPO_ROOT)}")


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

    embeddings = embed_chunks(chunks)
    load_into_chroma(chunks, embeddings)


if __name__ == "__main__":
    main()
