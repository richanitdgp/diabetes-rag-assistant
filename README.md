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
