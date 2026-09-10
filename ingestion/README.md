# ingestion

Document loading, chunking, and embedding scripts for building the vector store.

- `download_sources.py` — fetches raw source documents (v1: CDC; ADA deferred)
  into `data/raw/` and records each retrieval in `data/raw/manifest.json`. See
  `data/README.md` for the manifest format and versioning rationale.
- `parsers.py` — structure-aware parsers (`parse_html`, `parse_pdf`) that turn
  a raw source file into a list of `Chunk`s split along its own document
  structure (CDC: HTML headings; ADA: numbered chapter/recommendation
  headings) rather than fixed-size windows. Each `Chunk` carries citation
  metadata: source, section title, page/URL, publication year.
- `build_index.py` — the runnable pipeline: parse every source in the
  manifest, embed the chunks with Gemini's `gemini-embedding-001`, and
  upsert them into a MongoDB Atlas collection (`diabetes_rag.chunks` by
  default), creating its Atlas Vector Search index on first run if it
  doesn't already exist. Run it directly, or on a schedule (e.g. after
  `download_sources.py` picks up a new guideline year) — it's a separate,
  occasional operation, not something the deployed API triggers itself:

  ```bash
  python -m ingestion.build_index              # full pipeline (needs GOOGLE_API_KEY/GEMINI_API_KEY, MONGODB_URI)
  python -m ingestion.build_index --parse-only  # parse + write data/processed/chunks.jsonl only
  ```

  If the Atlas collection already holds vectors from a different embedding
  model (e.g. left over from testing another provider), drop its vector
  search index first — an Atlas vector index is locked to whatever
  `numDimensions` it was created with, same constraint Chroma had.
