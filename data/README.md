# data

Raw source documents and sample data for the diabetes RAG assistant.

## `raw/`

Raw, unmodified source documents used for ingestion, one subfolder per
publisher (`raw/ada/`, `raw/cdc/`, ...). Nothing in this tree is hand-edited —
it's exactly what was retrieved from the source.

**v1 scope is ADA + CDC only.** Other guideline bodies (e.g. NICE) are
deferred to a later version so the pipeline can ship an evaluated v1 in
week one instead of stalling on ingestion breadth.

Every file is tracked in [`raw/manifest.json`](raw/manifest.json), which
records, per source: id, title, publisher, category, source URL, local
path, retrieval timestamp, content hash (sha256), and size. This matters
because clinical guidelines are revised annually (e.g. the ADA "Standards
of Care" gets a new edition every December/January) — the manifest is
what lets you tell which version of a document is in the knowledge base
and re-run ingestion when a source updates.

To (re-)fetch sources:

```bash
python ingestion/download_sources.py
```

This re-downloads every source in `ingestion/download_sources.py`'s
`SOURCES` list and updates the manifest in place (existing entries are
overwritten by id, so re-running is safe). A failed download is recorded
in the manifest with `"status": "error"` and the underlying error, rather
than silently skipped — check the manifest after each run.

- Large or sensitive datasets should not be committed directly — see `.gitignore` and add exclusions as needed.
