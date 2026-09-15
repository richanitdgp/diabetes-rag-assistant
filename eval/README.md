# eval

Retrieval and answer quality tests for the RAG pipeline.

## `golden_dataset.yaml`

Hand-curated golden evaluation set: 50 in-scope Q&A pairs grounded in the
actual CDC knowledge base (v1 scope — see `data/README.md`), plus 18
out-of-scope questions (dosing, diagnosis, personalized treatment
decisions, and out-of-corpus/unrelated topics) the assistant must refuse.

Every in-scope pair's `expected_citations` was checked against real output
of `python -m ingestion.build_index --parse-only` — not hand-typed section
titles — so it stays aligned with what `ingestion/parsers.py` actually
produces. Re-verify citations the same way after any change to the CDC
source pages or the HTML parser.

The dataset is versioned (`version` field, semver) and documents its own
schema and grading conventions (`must_include_facts` for in-scope cases,
`expected_refusal_reason` tying each refusal to a specific rule in
`app/generation.py`'s `SYSTEM_PROMPT`) in its header comments.

ADA Standards of Care is deliberately excluded: its license prohibits use
for "text or data mining, machine learning, or similar technologies"
without permission, which rules out RAG use. Don't add it (or any other
copyrighted clinical guideline) to `data/raw/` or this dataset without
confirming that permission first.

Only ~50 in-scope pairs rather than the 50-100 a larger corpus would
support — the two CDC pages (21 chunks) run out of distinct, non-redundant
factual claims before 100. Expand when a properly licensed clinical
guideline is added.

There's no automated runner yet — this is the dataset itself. A harness
that loads it, calls `/ask` for each case, and grades
`must_include_facts`/citations/refusal-vs-answer is a natural next step.
