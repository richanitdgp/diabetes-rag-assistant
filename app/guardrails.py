"""In-process request guardrails for `POST /ask`: rate limiting, a daily
token-budget cap, and a pre-LLM filter for clearly out-of-bounds questions.

All three are intentionally simple and in-process: the deployed API is a
single free-tier instance (see render.yaml), not a fleet, so an in-memory
limiter/counter is enough for v1 and needs no extra infrastructure (Redis,
etc.). Swap slowapi's storage backend and TokenBudget's state for something
shared before ever running more than one instance — otherwise each instance
enforces its own separate limit/budget.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address

# --- Rate limiting ----------------------------------------------------------
# slowapi's decorator looks for a parameter literally named `request` typed
# as starlette.Request on the route function — see app/main.py's `ask()`.
RATE_LIMIT = os.environ.get("RATE_LIMIT", "20/minute")
limiter = Limiter(key_func=get_remote_address)


# --- Daily token budget ------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised when today's Gemini token budget is already used up."""


class TokenBudget:
    """Tracks Gemini chat token usage against a daily cap, reset at UTC
    midnight.

    Tracks *tokens* rather than a dollar figure directly: Gemini's per-token
    price changes over time and isn't something to hardcode here — verify
    current pricing at https://ai.google.dev/gemini-api/docs/pricing before
    deriving a dollar estimate from `MAX_DAILY_TOKENS`. This only covers
    generation tokens (`app/generation.py`), not embedding tokens, which are
    comparatively negligible per request.
    """

    def __init__(self, max_daily_tokens: int):
        self.max_daily_tokens = max_daily_tokens
        self._lock = threading.Lock()
        self._day = self._today()
        self._used = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_if_new_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._used = 0

    def check(self) -> None:
        """Raise BudgetExceededError if today's budget is already spent."""
        with self._lock:
            self._reset_if_new_day()
            if self._used >= self.max_daily_tokens:
                raise BudgetExceededError(
                    f"Daily token budget ({self.max_daily_tokens}) exhausted; resets at UTC midnight."
                )

    def record(self, total_tokens: int) -> None:
        with self._lock:
            self._reset_if_new_day()
            self._used += max(total_tokens, 0)

    @property
    def used_today(self) -> int:
        with self._lock:
            self._reset_if_new_day()
            return self._used


token_budget = TokenBudget(max_daily_tokens=int(os.environ.get("MAX_DAILY_TOKENS", 500_000)))


# --- Pre-LLM out-of-bounds filter -------------------------------------------
#
# A fast, cost-saving path for the *clearest* out-of-bounds questions
# (personalized dosing amounts, self-diagnosis phrasing) — not a
# replacement for SYSTEM_PROMPT rule 5 in app/generation.py, which remains
# the actual backstop for every other phrasing. Deliberately conservative:
# a false negative here still gets refused by the LLM; a false positive
# would incorrectly refuse a legitimate question. Every pattern below is
# checked against eval/golden_dataset.yaml before being added — 0 false
# positives against the 50 in-scope cases, catching 8 of the 18 refuse
# cases (the dosing-amount and self-diagnosis subtypes) pre-LLM.
_OUT_OF_BOUNDS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bhow (many|much) (units?|mg|ml|milligrams?)\b.*\b(insulin|metformin|medicat)", re.I),
        "personalized dosing amount",
    ),
    (
        re.compile(r"\b(increase|decrease|adjust|skip|stop|switch)\b.*\b(insulin|dose|dosage|medication)\b", re.I),
        "personalized dosing/medication decision",
    ),
    (
        re.compile(r"\bwhat dose\b|\bstarting dose\b|\bright dose\b", re.I),
        "personalized dosing amount",
    ),
    (
        re.compile(r"\bdo i have\b.*\bdiabetes\b", re.I),
        "self-diagnosis",
    ),
    (
        re.compile(r"\bcould i have\b.*\bdiabetes\b|\bam i diabetic\b", re.I),
        "self-diagnosis",
    ),
]


def out_of_bounds_reason(question: str) -> Optional[str]:
    """Return a short reason string if `question` is clearly out-of-bounds,
    else None. See the module docstring above for what this does and
    doesn't catch."""
    for pattern, reason in _OUT_OF_BOUNDS_PATTERNS:
        if pattern.search(question):
            return reason
    return None
