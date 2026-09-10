from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
def ask(request: AskRequest) -> AskResponse:
    """Retrieve top-k relevant chunks and generate a context-grounded answer."""
    try:
        chunks = retrieve(request.question, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = generate_answer(request.question, chunks)
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
