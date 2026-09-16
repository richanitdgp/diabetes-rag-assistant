"""Langfuse tracing, shared by retrieval.py, generation.py, and main.py.

Uses Langfuse's OTel-based client (`start_as_current_observation`, not the
older `.trace()/.span()` API from Langfuse SDK v2). Requires
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and optionally
`LANGFUSE_HOST`, default `https://cloud.langfuse.com`) to actually send
traces — with no keys set, the client logs one warning and every call
becomes a no-op (verified against the installed SDK), so tracing is
opt-in and never blocks the app from running without it.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_langfuse():
    from langfuse import Langfuse

    return Langfuse()


def record_guardrail_block(name: str, question: str, reason: str) -> None:
    """Log a pre-LLM guardrail refusal as its own observation, so blocked
    requests still show up in Langfuse even though nothing downstream
    (retrieval, generation) ever ran for them."""
    langfuse = get_langfuse()
    with langfuse.start_as_current_observation(
        name=name, as_type="guardrail", input=question, output=reason
    ):
        pass
