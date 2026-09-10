"""Grounded answer generation from retrieved context.

This is the v1 baseline: a strict system prompt that forces the model to
answer only from retrieved chunks, cite them inline, and refuse when the
context doesn't support a confident answer. Wire this up end-to-end and get
it working before spending effort on retrieval quality (chunking, top_k,
reranking, etc.) — see eval/ for the harness that will measure that.
"""

from __future__ import annotations

from app.retrieval import RetrievedChunk

CHAT_MODEL = "gemini-3.5-flash-lite"

REFUSAL_MESSAGE = (
    "I don't have enough information in the retrieved sources to answer that "
    "confidently. Please consult a healthcare professional or the CDC's "
    "diabetes resources directly."
)

SYSTEM_PROMPT = f"""You are a diabetes information assistant. You answer questions \
ONLY using the numbered context passages given to you in the user message — \
never from your own knowledge. You are not a substitute for medical advice.

Rules, in strict priority order:
1. Base your answer strictly on the provided context passages. Do not add \
facts, figures, or recommendations that are not stated in the context, even \
if you believe them to be true.
2. Every factual claim in your answer must carry an inline citation in the \
form [Source Title — Section Title], naming the exact passage it came from. \
Do not invent citations or cite a passage that was not provided.
3. If the context does not contain enough information to answer the question \
confidently, do not guess or extrapolate. Respond with exactly this sentence \
and nothing else: "{REFUSAL_MESSAGE}"
4. If the context only partially covers the question, answer the part it \
supports (with citations) and explicitly state which part you cannot answer \
from the given sources.
5. Never give personalized medical advice (dosing, diagnosis, treatment \
decisions for a specific person). Point the user to a healthcare \
professional for anything requiring individual medical judgment, even if the \
context discusses it in general terms.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] Source: {chunk.source_title} — {chunk.section_title}\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Call the chat model with a strict, context-only system prompt.

    Refuses locally (no LLM call) when retrieval returned nothing at all —
    there is no context for the model to even attempt to ground an answer in.
    """
    if not chunks:
        return REFUSAL_MESSAGE

    from google import genai
    from google.genai import types

    user_message = (
        f"Context passages:\n\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context passages above, with an inline "
        "citation [Source Title — Section Title] for every claim."
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    return response.text or REFUSAL_MESSAGE
