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
  manifest, embed the chunks with `text-embedding-3-small`, and upsert them
  into a local persistent Chroma collection at `data/chroma/`. Run it
  directly, in CI, or on a schedule (e.g. after `download_sources.py` picks
  up a new guideline year):

  ```bash
  python -m ingestion.build_index              # full pipeline (needs OPENAI_API_KEY)
  python -m ingestion.build_index --parse-only  # parse + write data/processed/chunks.jsonl only
  ```
