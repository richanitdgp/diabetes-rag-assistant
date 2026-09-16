import traceback

from fastapi import FastAPI, HTTPException
from langsmith import traceable
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from app.generation import generate_answer
from app.retrieval import retrieve

app = FastAPI(title="diabetes-rag-assistant")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceRef(BaseModel):
    source_title: str
    publisher: str
    section_title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@traceable(name="answer_query")
def ask(request: AskRequest) -> AskResponse:
    """Retrieve top-k relevant chunks and generate a context-grounded answer."""
    try:
        chunks = retrieve(request.question, top_k=request.top_k)
    except (RuntimeError, PyMongoError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # TEMPORARY debug aid: surface the real exception in the response body
        # instead of a bare "Internal Server Error", so root-causing a live
        # deploy issue doesn't require digging through platform logs. Revert
        # to a generic message once the underlying issue is found — returning
        # raw exception text to API clients isn't something to leave in place.
        print("Unexpected error in retrieve():", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"retrieve() failed: {exc!r}") from exc

    try:
        answer = generate_answer(request.question, chunks)
    except Exception as exc:
        print("Unexpected error in generate_answer():", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate_answer() failed: {exc!r}") from exc

    sources = [
        SourceRef(
            source_title=chunk.source_title,
            publisher=chunk.publisher,
            section_title=chunk.section_title,
            url=chunk.url,
        )
        for chunk in chunks
    ]
    return AskResponse(answer=answer, sources=sources)
