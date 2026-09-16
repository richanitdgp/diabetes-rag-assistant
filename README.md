# diabetes-rag-assistant

A retrieval-augmented generation (RAG) assistant for answering questions about diabetes using a curated knowledge base of medical documents.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and fill in your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
   `OPENAI_API_KEY` is used by `eval/run_ragas.py` as the RAGAS judge model
   (see [Evaluation](#evaluation)), independent of the app's own Gemini
   generation model.

## Usage

1. Fetch sources and build the vector index (needs `GOOGLE_API_KEY`/`GEMINI_API_KEY` and `MONGODB_URI`, pointing at an Atlas cluster with Vector Search support):
   ```bash
   python ingestion/download_sources.py
   python -m ingestion.build_index
   ```
   This is a separate, occasional step, not something the running API does
   itself — re-run it whenever `data/raw/manifest.json` changes.

2. Run the API (needs `GOOGLE_API_KEY`/`GEMINI_API_KEY` and `MONGODB_URI`):
   ```bash
   uvicorn app.main:app --reload
   ```

3. Ask a question:
   ```bash
   curl -s http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is type 2 diabetes?", "top_k": 5}'
   ```
   Returns `{"answer": "...", "sources": [...]}`. The answer is grounded
   strictly in retrieved context, cites `[Source Title — Section Title]`
   inline, and the assistant refuses rather than guesses when retrieval
   doesn't support a confident answer.

## Evaluation

`eval/run_ragas.py` runs the golden set (`eval/golden_dataset.yaml`, 50
in-scope + 18 out-of-scope cases — see `eval/README.md`) through the live
pipeline and scores it: RAGAS metrics (faithfulness, context precision,
context recall, answer relevancy) on the in-scope answers, plus a custom
refusal-accuracy check — does it answer in-scope questions and refuse
out-of-scope ones. Each run writes a timestamped JSON report to `results/`
so scores can be tracked across iterations.

```bash
python -m eval.run_ragas              # full run (needs OPENAI_API_KEY too)
python -m eval.run_ragas --skip-ragas  # refusal-accuracy only, faster
```

Latest results (see `results/` for the full history and per-case detail):

| Metric | Score | Run |
| --- | --- | --- |
| Refusal accuracy (overall) | _run `eval/run_ragas.py` to populate_ | |
| Refusal accuracy (in-scope) | | |
| Refusal accuracy (out-of-scope) | | |
| Faithfulness | | |
| Context precision | | |
| Context recall | | |
| Answer relevancy | | |

### CI gating

`.github/workflows/eval.yml` re-runs the golden-set eval on any pull
request that touches `app/generation.py`, `app/retrieval.py`,
`eval/golden_dataset.yaml`, or `eval/run_ragas.py`, and fails the check if
refusal accuracy drops below 0.95 or faithfulness drops below 0.8 (both
tunable via the workflow's `env:` block). It needs `MONGODB_URI`,
`GOOGLE_API_KEY`, and `OPENAI_API_KEY` set as repository secrets to
actually run — see the workflow file for details and its cost caveat
(every gated PR calls all three APIs over the full 68-case set).

## Observability & guardrails

Every `retrieve()`/`generate_answer()` call is traced via Langfuse
(`app/observability.py`) — latency, retrieved chunks, and Gemini token
usage — when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set; with them
unset, tracing silently no-ops and the app behaves exactly as before. Get a
free project at [cloud.langfuse.com](https://cloud.langfuse.com) or
self-host and set `LANGFUSE_HOST`.

`app/guardrails.py` adds three guardrails to `POST /ask`, checked in order
before any paid API call:

1. **Rate limiting** — per-client, via [slowapi](https://github.com/laurentS/slowapi), default `20/minute` (`RATE_LIMIT`).
2. **Input filtering** — a fast, pre-LLM regex filter that immediately
   refuses the clearest out-of-bounds questions (personalized dosing
   amounts, self-diagnosis phrasing), saving a retrieval + generation call
   on the obvious cases. It's a cost-saving fast path, not a replacement for
   `SYSTEM_PROMPT` rule 5 in `app/generation.py`, which remains the actual
   backstop for every other phrasing — see the comment in
   `app/guardrails.py` for the false-positive check against the golden set.
3. **Daily token budget** — a cap on Gemini generation tokens per UTC day
   (`MAX_DAILY_TOKENS`, default 500,000), enforced before each chat call;
   requests beyond it get HTTP 429 until the next day. Tracked in tokens
   rather than a dollar figure directly, since per-token pricing changes —
   see the comment in `app/guardrails.py`.

All three are in-process (no Redis or other shared store), which is fine
for the single free-tier instance this deploys as (`render.yaml`) but won't
enforce a shared limit/budget across multiple instances — see
`docs/ARCHITECTURE.md` §14 for more detail.
