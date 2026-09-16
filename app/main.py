import traceback

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.generation import REFUSAL_MESSAGE, generate_answer
from app.guardrails import RATE_LIMIT, BudgetExceededError, limiter, out_of_bounds_reason
from app.observability import record_guardrail_block
from app.retrieval import retrieve

app = FastAPI(title="diabetes-rag-assistant")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
@limiter.limit(RATE_LIMIT)
def ask(request: Request, payload: AskRequest) -> AskResponse:
    """Retrieve top-k relevant chunks and generate a context-grounded answer.

    Guardrails, checked in order before any paid API call: per-client rate
    limit (`RATE_LIMIT`, via slowapi — see the 429 raised by
    `_rate_limit_exceeded_handler` if exceeded), a pre-LLM filter for the
    clearest out-of-bounds questions (`app.guardrails.out_of_bounds_reason`),
    and a daily Gemini token budget (`app.guardrails.token_budget`, checked
    inside `generate_answer()`).
    """
    reason = out_of_bounds_reason(payload.question)
    if reason is not None:
        record_guardrail_block("out_of_bounds_filter", payload.question, reason)
        return AskResponse(answer=REFUSAL_MESSAGE, sources=[])

    try:
        chunks = retrieve(payload.question, top_k=payload.top_k)
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
        answer = generate_answer(payload.question, chunks)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
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
