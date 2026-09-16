"""RAGAS evaluation + refusal-accuracy scoring for the golden dataset.

Runs every case in `eval/golden_dataset.yaml` through the live pipeline
(`app.retrieval.retrieve` + `app.generation.generate_answer`), then scores it
two ways:

1. Refusal accuracy (custom scorer, no external judge needed): does the
   assistant answer the in-scope cases and refuse the out-of-scope cases,
   exactly as `expected_behavior` says? Refusal is detected by exact match
   against `app.generation.REFUSAL_MESSAGE`, since that's a fixed string the
   system prompt requires verbatim.
2. RAGAS metrics (faithfulness, context precision, context recall, answer
   relevancy) on the in-scope cases the assistant actually answered. RAGAS
   needs an LLM judge and an embedding model; this uses OpenAI
   (`OPENAI_API_KEY`) for both, independent of the app's own Gemini
   generation model, so scoring the pipeline doesn't grade itself.

Results are written as a timestamped JSON file under `results/` so scores
can be tracked across iterations (see `results/README.md`).

Requires the same environment as the running API (GOOGLE_API_KEY/
GEMINI_API_KEY, MONGODB_URI) plus OPENAI_API_KEY for the RAGAS judge.

Usage (run from the repo root, as a module so relative imports resolve):
    python -m eval.run_ragas
    python -m eval.run_ragas --case-id cdc-001 --case-id refuse-003
    python -m eval.run_ragas --skip-ragas   # refusal-accuracy only, no OPENAI_API_KEY needed
    python -m eval.run_ragas --top-k 8 --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = REPO_ROOT / "eval" / "golden_dataset.yaml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

RAGAS_METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    expected_behavior: str
    actual_behavior: Optional[str] = None
    behavior_correct: Optional[bool] = None
    answer: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    ragas_scores: Optional[dict[str, float]] = None
    error: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "behavior_correct": self.behavior_correct,
        }
        if self.answer is not None:
            d["answer"] = self.answer
        if self.sources:
            d["sources"] = self.sources
        if self.ragas_scores is not None:
            d["ragas_scores"] = self.ragas_scores
        if self.error is not None:
            d["error"] = self.error
        return d


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def run_pipeline_case(case: dict[str, Any], top_k: int) -> tuple[CaseResult, list[Any]]:
    """Call the live retrieve -> generate pipeline for one golden-set case."""
    from app.generation import REFUSAL_MESSAGE, generate_answer
    from app.retrieval import retrieve

    result = CaseResult(
        id=case["id"],
        category=case["category"],
        question=case["question"],
        expected_behavior=case["expected_behavior"],
    )
    try:
        chunks = retrieve(case["question"], top_k=top_k)
        answer = generate_answer(case["question"], chunks)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result, []

    result.answer = answer
    result.sources = [f"{c.source_title} — {c.section_title}" for c in chunks]
    result.actual_behavior = "refuse" if answer.strip() == REFUSAL_MESSAGE else "answer"
    result.behavior_correct = result.actual_behavior == result.expected_behavior
    return result, chunks


def compute_refusal_accuracy(case_results: list[CaseResult]) -> dict[str, Any]:
    """Does the assistant answer in-scope cases and refuse out-of-scope ones?"""
    scored = [c for c in case_results if c.behavior_correct is not None]
    in_scope = [c for c in scored if c.category == "in_scope"]
    out_of_scope = [c for c in scored if c.category == "out_of_scope"]

    def rate(cases: list[CaseResult]) -> Optional[float]:
        return sum(1 for c in cases if c.behavior_correct) / len(cases) if cases else None

    failures = [
        {
            "id": c.id,
            "category": c.category,
            "question": c.question,
            "expected_behavior": c.expected_behavior,
            "actual_behavior": c.actual_behavior,
        }
        for c in scored
        if not c.behavior_correct
    ]
    errored_ids = [c.id for c in case_results if c.error is not None]

    return {
        "overall_accuracy": rate(scored),
        "in_scope_answer_accuracy": rate(in_scope),
        "out_of_scope_refusal_accuracy": rate(out_of_scope),
        "num_scored": len(scored),
        "num_errored": len(errored_ids),
        "errored_case_ids": errored_ids,
        "failures": failures,
    }


def run_ragas_eval(
    case_results: list[CaseResult],
    contexts_by_id: dict[str, list[str]],
    cases_by_id: dict[str, dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, dict[str, Optional[float]]]]:
    """Score faithfulness/context precision/context recall/answer relevancy.

    Only covers in-scope cases the pipeline actually answered (a correct or
    incorrect refusal has no "answer" to grade against retrieved context).
    Uses OpenAI as RAGAS's judge LLM + embeddings, kept separate from the
    app's own Gemini generation model.
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    records: list[dict[str, Any]] = []
    record_ids: list[str] = []
    for cr in case_results:
        if cr.category != "in_scope" or cr.actual_behavior != "answer" or cr.error:
            continue
        contexts = contexts_by_id.get(cr.id)
        if not contexts:
            continue
        case = cases_by_id[cr.id]
        reference = " ".join(case.get("must_include_facts", []))
        records.append(
            {
                "user_input": cr.question,
                "response": cr.answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )
        record_ids.append(cr.id)

    if not records:
        return None, {}

    dataset = Dataset.from_list(records)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    df = result.to_pandas()

    per_case: dict[str, dict[str, Optional[float]]] = {}
    for record_id, (_, row) in zip(record_ids, df.iterrows()):
        per_case[record_id] = {
            m: (float(row[m]) if m in row and row[m] == row[m] else None) for m in RAGAS_METRIC_NAMES
        }

    aggregate: dict[str, Any] = {}
    for m in RAGAS_METRIC_NAMES:
        if m in df.columns:
            series = df[m].dropna()
            aggregate[m] = float(series.mean()) if len(series) else None
    aggregate["num_cases_scored"] = len(records)

    return aggregate, per_case


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question (default: 5).")
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Restrict the run to this case id (repeatable). Default: run every case.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Only compute refusal-accuracy; skip RAGAS metrics (no OPENAI_API_KEY needed).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write the timestamped JSON report into (default: results/).",
    )
    args = parser.parse_args(argv)

    dataset = load_golden_dataset()
    cases = dataset["cases"]
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"warning: unknown case id(s) ignored: {sorted(missing)}", flush=True)

    print(f"Running {len(cases)} case(s) through the live pipeline (top_k={args.top_k})...", flush=True)

    case_results: list[CaseResult] = []
    contexts_by_id: dict[str, list[str]] = {}
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:80]!r}", flush=True)
        cr, chunks = run_pipeline_case(case, top_k=args.top_k)
        case_results.append(cr)
        if chunks:
            contexts_by_id[cr.id] = [c.text for c in chunks]
        if cr.error:
            print(f"  ERROR: {cr.error}", flush=True)
        else:
            marker = "OK" if cr.behavior_correct else "MISMATCH"
            print(f"  expected={cr.expected_behavior} actual={cr.actual_behavior} [{marker}]", flush=True)

    refusal_accuracy = compute_refusal_accuracy(case_results)

    ragas_aggregate: Optional[dict[str, Any]] = None
    if args.skip_ragas:
        print("Skipping RAGAS metrics (--skip-ragas).", flush=True)
    else:
        print("Running RAGAS metrics on answered in-scope cases...", flush=True)
        cases_by_id = {c["id"]: c for c in cases}
        try:
            ragas_aggregate, ragas_per_case = run_ragas_eval(case_results, contexts_by_id, cases_by_id)
        except Exception as exc:
            print("RAGAS evaluation failed:", flush=True)
            traceback.print_exc()
            ragas_aggregate = {"error": f"{type(exc).__name__}: {exc}"}
            ragas_per_case = {}
        for cr in case_results:
            if cr.id in ragas_per_case:
                cr.ragas_scores = ragas_per_case[cr.id]

    counts = {
        "total": len(case_results),
        "in_scope": sum(1 for c in case_results if c.category == "in_scope"),
        "out_of_scope": sum(1 for c in case_results if c.category == "out_of_scope"),
    }

    now = datetime.now(timezone.utc)
    report = {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_version": dataset.get("version"),
        "top_k": args.top_k,
        "counts": counts,
        "refusal_accuracy": refusal_accuracy,
        "ragas": ragas_aggregate,
        "cases": [cr.to_json() for cr in case_results],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"eval_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\nWrote {out_path}", flush=True)
    print(
        "Refusal accuracy: "
        f"overall={refusal_accuracy['overall_accuracy']} "
        f"in_scope={refusal_accuracy['in_scope_answer_accuracy']} "
        f"out_of_scope={refusal_accuracy['out_of_scope_refusal_accuracy']}",
        flush=True,
    )
    if ragas_aggregate:
        print(f"RAGAS: {ragas_aggregate}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
