from fastapi import FastAPI

app = FastAPI(title="diabetes-rag-assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
