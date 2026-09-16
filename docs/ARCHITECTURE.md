# diabetes-rag-assistant — Architecture Guide

Internal engineering reference. A retrieval-augmented generation service that
answers diabetes questions strictly from a curated, citable source corpus —
and refuses rather than guesses when that corpus doesn't support a confident
answer. This doc walks a new engineer through how it's built, end to end.

`FastAPI` · `MongoDB Atlas Vector Search` · `Gemini embeddings + chat` ·
`v1 scope: CDC only` · `Render`

> Snapshot as of commit `574cc36` (2026-09-15) — not a live document; verify
> specifics against current code before relying on them for anything
> load-bearing.

## Contents

1. [Overview](#1-overview)
2. [Two-phase architecture](#2-two-phase-architecture)
3. [Request lifecycle: `POST /ask`](#3-request-lifecycle-post-ask)
4. [Component reference](#4-component-reference)
5. [Data model & citation contract](#5-data-model--citation-contract)
6. [Grounding rules (the system prompt)](#6-grounding-rules-the-system-prompt)
7. [API reference](#7-api-reference)
8. [Environment & deployment](#8-environment--deployment)
9. [Golden eval set](#9-golden-eval-set)
10. [Known limitations & gotchas](#10-known-limitations--gotchas)
11. [Getting started](#11-getting-started)
12. [Gemini usage, step by step](#12-gemini-usage-step-by-step)
13. [OpenAI usage, step by step (RAGAS eval)](#13-openai-usage-step-by-step-ragas-eval)

---

## 1. Overview

The assistant answers questions about diabetes by retrieving relevant
passages from a knowledge base and having an LLM compose an answer **only**
from those passages, with an inline citation for every claim. If the
retrieved passages don't support a confident answer — or the question asks
for something the system should never answer, like a personal dosing
decision — it refuses instead of guessing.

That refusal-first behavior isn't a guardrail bolted on afterward; it's
encoded directly in `app/generation.py`'s system prompt as five
priority-ordered rules (see [§6](#6-grounding-rules-the-system-prompt)), and
it's the main thing the golden eval set ([§9](#9-golden-eval-set)) checks for.

## 2. Two-phase architecture

The single most important thing to internalize before touching this
codebase: ingestion and serving are two separate, decoupled processes that
only ever meet inside MongoDB Atlas.

```mermaid
flowchart LR
    subgraph offline["OFFLINE — run manually / occasionally"]
        direction LR
        A["CDC pages\n(external, HTML)"] -->|fetch| B["download_sources.py\nwrites data/raw/*.html\n+ manifest.json"]
        B -->|parse| C["parsers.py\nHTML → Chunk[]\n(one per heading)"]
        C -->|embed| D["build_index.py\nGemini embed +\nupsert to Atlas"]
    end

    D -->|"upsert by _id"| M[("MongoDB Atlas\ndiabetes_rag.chunks\n+ vector index")]

    subgraph online["ONLINE — always-on FastAPI service"]
        direction LR
        E["Client\nPOST /ask"] --> F["main.py\nFastAPI route"]
        F --> G["retrieval.py\n$vectorSearch"]
        G --> H["generation.py\nGemini chat, grounded"]
    end

    G -->|"query top-k"| M
```

The ingestion pipeline (top lane) never runs as part of a deploy or a
request — it's invoked by hand whenever `data/raw/manifest.json` changes.
The FastAPI service (bottom lane) never writes to Atlas, only reads. The two
lanes share no code path; the only thing connecting them is the Atlas
collection itself.

**Why:** Clinical guidelines like the ADA Standards of Care are revised
annually. Re-embedding and re-upserting on every deploy would be wasteful
and could race concurrent deploys — so ingestion is a deliberate, separate
operation you re-run when a source changes, not a build step.

## 3. Request lifecycle: `POST /ask`

What actually happens between a client's question and the JSON response,
including the branch that skips the LLM entirely.

```mermaid
sequenceDiagram
    participant Client
    participant main.py
    participant retrieval.py as retrieval.py (+ Atlas)
    participant generation.py as generation.py (+ Gemini chat)

    Client->>main.py: {"question", "top_k"}
    main.py->>retrieval.py: retrieve(question, top_k)
    Note over retrieval.py: embed query, task_type=RETRIEVAL_QUERY<br/>$vectorSearch, limit=top_k
    retrieval.py-->>main.py: List[RetrievedChunk]
    main.py->>generation.py: generate_answer(question, chunks)
    alt chunks == []
        generation.py-->>main.py: REFUSAL_MESSAGE (no LLM call)
    else chunks non-empty
        Note over generation.py: build numbered context +<br/>SYSTEM_PROMPT → Gemini, temp=0
        generation.py-->>main.py: grounded, cited answer
    end
```

The refusal path when retrieval returns nothing is a local short-circuit in
`generate_answer()` — no Gemini call happens. Every other refusal (off-topic
question, dosing request, etc.) still retrieves `top_k` chunks as normal and
relies on the LLM following the system prompt's rules to decline. See
[§10](#10-known-limitations--gotchas) for why that matters.

## 4. Component reference

Repository layout and what each module actually does, file by file.

```
diabetes-rag-assistant/
├── app/                      # online: the FastAPI service
│   ├── main.py                # routes: GET /health, POST /ask
│   ├── retrieval.py           # embed query + Atlas $vectorSearch
│   └── generation.py          # grounded prompt + Gemini chat call
├── ingestion/                 # offline: builds the knowledge base
│   ├── download_sources.py    # fetch raw docs → data/raw/, update manifest
│   ├── parsers.py             # raw file → structure-aware Chunk[]
│   └── build_index.py         # parse → embed → upsert to Atlas (runnable)
├── data/
│   └── raw/                   # tracked source docs + manifest.json
│       └── cdc/                # diabetes-basics.html, living-with-diabetes.html
├── eval/
│   └── golden_dataset.yaml    # 68 hand-curated Q&A / refusal cases
├── render.yaml                # Render web service definition
└── requirements.txt
```

### `app/` — online service

| File | Role | Key pieces |
|---|---|---|
| `main.py` | FastAPI entry point | `GET /health` (liveness only, doesn't touch Atlas) · `POST /ask` ({question, top_k=5} → {answer, sources[]}) · RuntimeError/PyMongoError from retrieval → HTTP 503 |
| `retrieval.py` | Vector similarity search | `_embed_query()` — Gemini `gemini-embedding-001`, `task_type=RETRIEVAL_QUERY` (asymmetric vs. document embedding) · `retrieve()` — `$vectorSearch`, `numCandidates=max(top_k×10, 100)` · raises `RuntimeError` if `MONGODB_URI` unset |
| `generation.py` | Grounded answer synthesis | `SYSTEM_PROMPT` — 5 priority-ordered rules, see [§6](#6-grounding-rules-the-system-prompt) · `generate_answer()` — local refusal if chunks empty, else Gemini `gemini-3.5-flash-lite`, temperature=0 |

### `ingestion/` — offline pipeline

| File | Role | Key pieces |
|---|---|---|
| `download_sources.py` | Fetch + manifest | `SOURCES` — v1 active: 2 CDC pages only · `DEFERRED_SOURCES` — ADA Standards of Care, documented, not fetched (see [§10](#10-known-limitations--gotchas)) · writes `data/raw/<publisher>/*.html` + `manifest.json` (id, hash, timestamp) |
| `parsers.py` | Structure-aware chunking | `parse_html()` — one `Chunk` per h1–h4 heading in CDC pages' main content · `parse_pdf()` — numbered-heading chunker written for ADA, currently unused (no PDF source active) |
| `build_index.py` | Runnable pipeline | `parse_all_sources()` — reads manifest, dispatches by content type · `embed_chunks()` — batches of 100, `task_type=RETRIEVAL_DOCUMENT` · `load_into_mongo()` — upsert by chunk id, creates vector index if absent |

## 5. Data model & citation contract

Every retrievable unit is a `Chunk` — the same shape from parsing through
Atlas storage through the citation the model prints.

| Field | Type | Example |
|---|---|---|
| `text` | `str` | `"Type 1 diabetes\n\nType 1 diabetes is thought to be caused by…"` |
| `source_id` | `str` | `cdc-diabetes-basics` |
| `source_title` | `str` | `Diabetes Basics` |
| `section_title` | `str` | `Types — Type 1 diabetes` |
| `url` | `str` | `cdc.gov/diabetes/about/…#type-1` |
| `publication_year` / `page` | `int \| None` | from `<time>` tag (HTML) or PDF page (PDF) |
| `chunk_index` | `int` | position within its source, used to build the stable id |

**Nested section titles.** `parse_html()` joins a subheading to its parent h2
so citations stay unambiguous when a page reuses a title (e.g. "Type 1
diabetes" appears under both "Types" and "Prevention"):

```
h2 "Prevention"
  h3 "Type 1 diabetes"  →  section_title = "Prevention — Type 1 diabetes"
```

**What the model actually cites.** The prompt in `generation.py` requires
every claim to carry `[Source Title — Section Title]` — literally
`chunk.source_title` and `chunk.section_title` joined, e.g.:

```
"…type 1 diabetes stops the body from making insulin
[Diabetes Basics — Types — Type 1 diabetes]"
```

## 6. Grounding rules (the system prompt)

Five priority-ordered rules in `app/generation.py`'s `SYSTEM_PROMPT` — this
is the actual behavioral contract, not a paraphrase.

| # | Rule |
|---|---|
| 1 | Base the answer strictly on the provided context passages — no outside facts, even if believed true. |
| 2 | Every claim carries an inline citation `[Source Title — Section Title]`; never invent one. |
| 3 | If context doesn't support a confident answer, respond with the exact refusal sentence and nothing else. |
| 4 | If context only partially covers the question, answer the supported part and say what's missing. |
| 5 | Never give personalized medical advice (dosing, diagnosis, individual treatment decisions) — even if context discusses the topic generally. |

The exact refusal string (also returned locally when retrieval finds
nothing at all):

```
"I don't have enough information in the retrieved sources to answer that
confidently. Please consult a healthcare professional or the CDC's
diabetes resources directly."
```

## 7. API reference

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/health` | `GET` | none | `{"status":"ok"}` — does not verify Atlas connectivity. |
| `/ask` | `POST` | none | Returns 503 if Atlas is unreachable or `MONGODB_URI` is unset. |

**Request**

```json
POST /ask
{
  "question": "What is type 2 diabetes?",
  "top_k": 5
}
```

**Response**

```json
{
  "answer": "…grounded text with [Source — Section] citations…",
  "sources": [
    {
      "source_title": "Diabetes Basics",
      "publisher": "CDC",
      "section_title": "Types — Type 2 diabetes",
      "url": "https://www.cdc.gov/…"
    }
  ]
}
```

## 8. Environment & deployment

| Variable | Used by | Notes |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | ingestion (embed), app (chat) | google-genai reads either; `GOOGLE_API_KEY` wins if both set. |
| `MONGODB_URI` | ingestion (upsert), app (query) | Atlas cluster must support Vector Search. |
| `OPENAI_API_KEY` | `eval/run_ragas.py` | RAGAS's judge LLM + embeddings (see [§12](#12-gemini-usage-step-by-step)) — independent of the app's own Gemini models, deliberately, so the pipeline isn't grading itself. |
| `ANTHROPIC_API_KEY` | — currently unused | Present in `.env.example`, no code path reads it. |

**Render (`render.yaml`)**: single web service, Python env, free plan.
Build: `pip install -r requirements.txt`. Start: `uvicorn app.main:app
--host 0.0.0.0 --port $PORT`. Health check: `/health`. `GOOGLE_API_KEY` and
`MONGODB_URI` are declared but `sync: false` — set them manually in the
Render dashboard per environment. Ingestion is **not** part of the build
step — run it manually or as a separate scheduled job.

## 9. Golden eval set

`eval/golden_dataset.yaml` — 68 hand-curated cases, every citation checked
against real parser output rather than typed by hand.

- **50 in-scope**: grounded Q&A pairs from the two CDC pages, with
  `expected_citations` and `must_include_facts` per case. Includes 3
  multi-source synthesis cases that span both pages.
- **18 out-of-scope**: must-refuse cases across 5 subtypes — dosing,
  diagnosis, treatment decisions, out-of-corpus specificity, unrelated
  topics — each tagged with which `SYSTEM_PROMPT` rule should fire.

`eval/run_ragas.py` is the automated runner: it calls `retrieve()` +
`generate_answer()` directly (not `/ask`) for every case, grades
refuse-vs-answer behavior against `expected_behavior`, and — for in-scope
cases that were actually answered — scores faithfulness/context
precision/context recall/answer relevancy via RAGAS, using OpenAI as an
independent judge (see [§12](#12-gemini-usage-step-by-step) for how that
interacts with Gemini). Results land as timestamped JSON in `results/`.

## 10. Known limitations & gotchas

Things verified firsthand while building this system that a new engineer
would otherwise rediscover the hard way.

> **Scope.** The knowledge base is **CDC patient-education pages only** — 2
> pages, 21 chunks. ADA Standards of Care was evaluated and deliberately
> excluded: its license prohibits use for "text or data mining, machine
> learning, or similar technologies" without permission, which RAG
> chunking/embedding is. The PDF was downloaded, inspected, and removed
> once that was discovered. `parsers.py`'s `parse_pdf()`/`_ADA_HEADING_RE`
> remain for a future, properly licensed source, but testing against the
> real ADA PDF (before removal) found they mishandle multi-line chapter
> titles and lettered sub-recommendations like "2.2a"/"2.2b" — treat that
> code as unverified, not ready to point at a new source without rework.

> **Retrieval.** `retrieve()` always returns the `top_k` nearest chunks —
> there's no similarity-score cutoff. An off-topic question still gets
> chunks back; refusal for it depends entirely on the LLM following
> `SYSTEM_PROMPT` rule 3/5, not on retrieval coming back empty.

> **Unverified code.** `build_index.py`'s `_ensure_vector_index()` is
> annotated in its own docstring as unverified against a real Atlas
> cluster. Separately, `write_chunks_jsonl()`'s final status print crashes
> (`ValueError` on `Path.relative_to`) if `--chunks-out` points outside the
> repo — reproduced firsthand; chunk writing itself succeeds, only the
> trailing print fails. Minor, not yet fixed.

> **Dependencies.** `requirements.txt` lists `langchain`,
> `langchain-community`, `langchain-openai`, `openai`, and `tiktoken` — none
> are imported anywhere in `app/` or `ingestion/` (verified by grep), so
> they're still dead weight for the running service. `langchain-openai` is
> no longer fully unused, though: `eval/run_ragas.py` imports it to hand
> RAGAS an OpenAI-backed judge/embeddings model, and its version (along with
> `langchain`/`langchain-core`/`langchain-community`) is pinned below their
> 0.4/1.0 lines specifically for `ragas` compatibility — see the comment in
> `requirements.txt`. `openai` and `tiktoken` remain genuinely unused.

## 11. Getting started

Local setup, in order.

1. **Environment:**
   ```bash
   python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
   ```
2. **Secrets:** `cp .env.example .env`, then fill in `GOOGLE_API_KEY` (or
   `GEMINI_API_KEY`) and `MONGODB_URI` (an Atlas cluster with Vector Search
   support).
3. **Fetch sources** (only needed once, or when a source updates):
   ```bash
   python ingestion/download_sources.py
   ```
4. **Build the index** — parses, embeds, and upserts into Atlas, creating
   the vector index on first run:
   ```bash
   python -m ingestion.build_index
   ```
5. **Run the API:**
   ```bash
   uvicorn app.main:app --reload
   ```
6. **Ask it something:**
   ```bash
   curl -s http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is type 2 diabetes?", "top_k": 5}'
   ```
7. **Run the eval harness against the golden set:**
   ```bash
   python -m eval.run_ragas --skip-ragas   # refusal-accuracy only, no OPENAI_API_KEY needed
   python -m eval.run_ragas                # + RAGAS metrics, needs OPENAI_API_KEY
   ```
   See [§9](#9-golden-eval-set) and [§12](#12-gemini-usage-step-by-step).

## 12. Gemini usage, step by step

Gemini is the backbone of the app itself — it shows up in three distinct
calls, split across two phases of the system's lifecycle (offline ingestion
vs. live serving). This walks through all three in order, since §2/§3's
diagrams show *where* they sit but not *why* two of them use the same model
differently.

**Phase 1 — Ingestion (offline, `python -m ingestion.build_index`).** Not
triggered by any request; run by hand whenever `data/raw/manifest.json`
changes.

1. `parse_all_sources()` splits the CDC HTML pages into `Chunk` objects —
   plain HTML parsing, no Gemini call yet.
2. `embed_chunks()` (`ingestion/build_index.py`) sends chunk texts to
   Gemini's **embedding** model, `gemini-embedding-001`, in batches of 100,
   with `task_type="RETRIEVAL_DOCUMENT"`. This is the first Gemini call in
   the system — it turns each CDC chunk into a vector.
3. `load_into_mongo()` upserts each chunk's text + embedding into Atlas and
   creates the vector search index if it doesn't exist yet.

End state: Atlas holds 21 chunks, each with a Gemini-generated vector next
to its text. Gemini's ingestion job is done until a source page changes.

**Phase 2 — Live query, retrieval (`app/retrieval.py`, every `POST /ask`).**

4. `_embed_query()` sends the **question** (not a document) to the same
   `gemini-embedding-001` model, but with `task_type="RETRIEVAL_QUERY"`
   instead of `RETRIEVAL_DOCUMENT`. Gemini's embedding model is asymmetric —
   a question and a passage that mean the same thing are deliberately
   embedded differently for better retrieval — which is why ingestion and
   retrieval use the same model but different `task_type` values, and why
   query vectors can't reuse Phase 1's document embeddings.
5. That query vector goes into Atlas's `$vectorSearch` aggregation, which
   returns the top-k nearest chunks. Gemini's role ends here — the
   nearest-neighbor search itself happens inside Atlas, not Gemini.

**Phase 3 — Live query, generation (`app/generation.py`).**

6. The retrieved chunks + question get wrapped in `SYSTEM_PROMPT` (context-
   only, cite `[Source — Section]`, refuse if unsupported, never give
   personalized medical advice — see [§6](#6-grounding-rules-the-system-prompt))
   and sent as a **chat** call to `gemini-3.5-flash-lite` — a third, separate
   Gemini model from the embedding one.
7. Gemini's response is the final answer. If retrieval returned zero
   chunks, this step never happens — `generate_answer()` short-circuits to
   `REFUSAL_MESSAGE` locally (see [§3](#3-request-lifecycle-post-ask)).

**Summary — three touchpoints, two models:**

| Step | When | Gemini model | `task_type` |
|---|---|---|---|
| Embed corpus chunks | Ingestion (offline) | `gemini-embedding-001` | `RETRIEVAL_DOCUMENT` |
| Embed the question | Every `/ask` call | `gemini-embedding-001` | `RETRIEVAL_QUERY` |
| Generate the answer | Every `/ask` call (unless retrieval was empty) | `gemini-3.5-flash-lite` | n/a (chat) |

**How this connects to the eval harness.** `eval/run_ragas.py` calls
`retrieve()` and `generate_answer()` directly for every golden-set case, so
Phases 2 and 3 above are exactly what runs during an eval pass — Gemini
still does all the embedding and answering. OpenAI only enters afterward,
as an external judge scoring Gemini's output with RAGAS; it never touches
retrieval or generation itself. Gemini builds and answers, OpenAI grades.

## 13. OpenAI usage, step by step (RAGAS eval)

OpenAI never appears in the live `/ask` path — it exists solely inside
`eval/run_ragas.py`, as an external judge scoring what Gemini already
produced (see [§12](#12-gemini-usage-step-by-step)). This walks through
where exactly it's called and why `ragas.evaluate()` needs two different
OpenAI capabilities, not just one.

**Phase 1 — Run the pipeline first (no OpenAI yet).** For every golden-set
case, `run_pipeline_case()` calls the same `retrieve()` +
`generate_answer()` from [§12](#12-gemini-usage-step-by-step) — Gemini
embeds the question, Atlas returns chunks, Gemini generates the answer.
This produces, per case, a question/answer/retrieved-chunks triple with no
OpenAI involvement at all. This phase alone is enough for the
refusal-accuracy scorer, which is why `--skip-ragas` needs no
`OPENAI_API_KEY`.

**Phase 2 — Assemble the RAGAS input (still no API calls).** `run_ragas_eval()`
filters down to in-scope cases the pipeline actually answered (not refused,
not errored), then builds an in-memory table per case:

```
user_input          = the question
response            = Gemini's answer
retrieved_contexts  = the retrieved chunk texts (from Atlas, via Gemini's query embedding)
reference           = the golden set's must_include_facts, joined
```

**Phase 3 — This is where OpenAI is actually called.** Two OpenAI clients
are constructed, both reading `OPENAI_API_KEY` from the environment:

- `ChatOpenAI(model="gpt-4o-mini")` — the **judge**, used for reasoning/verdicts.
- `OpenAIEmbeddings(model="text-embedding-3-small")` — used for one specific
  similarity calculation.

`ragas.evaluate()` then fires the actual calls, one case at a time, per metric:

| Metric | What gets sent to OpenAI | Endpoint |
|---|---|---|
| `faithfulness` | Gemini's answer → chat call splitting it into atomic claims; each claim + retrieved chunk text → chat call asking "is this claim supported?" | Chat completions |
| `context_precision` | Question + reference + each retrieved chunk → chat call asking "was this chunk actually useful?" | Chat completions |
| `context_recall` | Reference text + retrieved chunks → chat call asking "is each part of the reference backed by these chunks?" | Chat completions |
| `answer_relevancy` | Gemini's answer → chat call generating a few hypothetical questions it would answer; those questions + the original question → **embeddings** call, compared by cosine similarity locally | Chat completions **and** embeddings |

`answer_relevancy` is the only metric needing both capabilities —
generate-then-embed — which is why `evaluate()` takes both an `llm=` and an
`embeddings=` argument even though the other three metrics only ever use
the chat endpoint.

**Phase 4 — Collect scores (no more API calls).** `result.to_pandas()`
returns one row per case, one column per metric. `run_ragas_eval()`
averages each column for the aggregate score and keeps per-row values for
the JSON report's `cases[].ragas_scores`. Pure local computation from here
— the report is written to `results/` (see `results/README.md`).

**Why OpenAI and not Gemini for judging.** Two reasons, covered in more
detail in `eval/README.md`:

1. **Avoiding the model grading its own homework.** If Gemini judged its
   own answers, its own stylistic quirks would tend to look "correct" to
   itself — a known bias in LLM-as-judge setups. An independent model keeps
   the score honest.
2. **Plumbing.** RAGAS's wrappers (`LangchainLLMWrapper`,
   `LangchainEmbeddingsWrapper`) expect LangChain-compatible clients. The
   app talks to Gemini directly through the raw `google-genai` SDK, with no
   LangChain in that path at all — wiring Gemini into RAGAS would mean
   adding yet another LangChain provider package on top of the
   `langchain-community`/`ragas` version pin already needed (see
   [§10](#10-known-limitations--gotchas)). `langchain-openai` was already a
   dependency and `OPENAI_API_KEY` was already provisioned in
   `.env.example`, unused, so OpenAI was the path of least resistance.

**One privacy note.** Because `faithfulness`/`context_precision`/
`context_recall` send retrieved chunk text to OpenAI as part of the judging
prompt, CDC page content leaves the app's infra and goes to OpenAI's API
during evaluation. Harmless here since the corpus is public CDC text, but
worth remembering if the corpus ever contains anything sensitive.
