# results

Timestamped output of `eval/run_ragas.py`, one JSON file per run
(`eval_<UTC timestamp>.json`). Committed on purpose so score trends across
iterations (retrieval/prompt/chunking changes) are visible in git history —
diff two files to see what moved.

Each file contains:

- `dataset_version` / `top_k` — which golden-set version and retrieval
  setting produced this run.
- `refusal_accuracy` — overall/in-scope/out-of-scope refusal-vs-answer
  accuracy against `eval/golden_dataset.yaml`, plus the list of mismatched
  cases.
- `ragas` — aggregate faithfulness / context precision / context recall /
  answer relevancy over the in-scope cases that were actually answered
  (`null` if the run used `--skip-ragas`).
- `cases` — full per-case detail: question, expected vs. actual behavior,
  the generated answer, cited sources, and (when scored) that case's own
  RAGAS metrics.

See `eval/README.md` for how to run it.
