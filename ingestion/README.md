# ingestion

Document loading, chunking, and embedding scripts for building the vector store.

- `download_sources.py` — fetches raw source documents (v1: ADA + CDC) into
  `data/raw/` and records each retrieval in `data/raw/manifest.json`. See
  `data/README.md` for the manifest format and versioning rationale.
