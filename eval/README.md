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

## `run_ragas.py`

Automated harness that loads `golden_dataset.yaml`, runs every case through
the live pipeline (`app.retrieval.retrieve` + `app.generation.generate_answer`
— not the HTTP API, so no running server is needed), and scores it two ways:

- **Refusal accuracy** (custom scorer, no LLM judge): does the assistant
  answer the in-scope cases and refuse the out-of-scope ones, matching each
  case's `expected_behavior`? Detected by exact string match against
  `app.generation.REFUSAL_MESSAGE`. Reported overall and split by
  in-scope/out-of-scope, with the list of mismatched cases.
- **RAGAS metrics** — faithfulness, context precision, context recall, and
  answer relevancy — computed over the in-scope cases the assistant actually
  answered. Uses OpenAI (`OPENAI_API_KEY`) as RAGAS's judge LLM and
  embeddings model, independent of the app's own Gemini generation model, so
  the pipeline isn't grading itself. `must_include_facts` is joined into a
  reference string for the metrics (context precision, context recall) that
  need a ground truth to compare against.

Results are written as a timestamped JSON file under `results/` (see
`results/README.md`) so scores can be tracked across iterations — that
history is what backs the metrics table in the top-level README.

Requires the same environment as the running API
(`GOOGLE_API_KEY`/`GEMINI_API_KEY`, `MONGODB_URI`) plus `OPENAI_API_KEY` for
the RAGAS judge (see `.env.example`). Note the pinned `langchain*`/`ragas`
versions in `requirements.txt` — see the comment there for why.

```bash
# Full run: refusal-accuracy + RAGAS metrics over all 68 cases
python -m eval.run_ragas

# Refusal-accuracy only, no OPENAI_API_KEY needed (fast iteration)
python -m eval.run_ragas --skip-ragas

# A subset of cases, and/or a different top_k
python -m eval.run_ragas --case-id cdc-001 --case-id refuse-003 --top-k 8
```
