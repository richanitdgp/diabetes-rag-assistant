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

## Usage

1. Fetch sources and build the vector index (needs `GOOGLE_API_KEY`/`GEMINI_API_KEY`):
   ```bash
   python ingestion/download_sources.py
   python -m ingestion.build_index
   ```

2. Run the API (needs `GOOGLE_API_KEY`/`GEMINI_API_KEY`):
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
