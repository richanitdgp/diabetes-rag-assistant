"""Download raw source documents for the diabetes RAG knowledge base.

v1 scope is deliberately narrow: ADA + CDC only. NICE (and any other
guideline body) is deferred to a later version so v1 can ship an
evaluated pipeline in week one instead of stalling on ingestion breadth.

Guidelines are updated annually (e.g. "Standards of Care in Diabetes"
gets a new year every December/January), so every run is recorded in
data/raw/manifest.json with the source, URL, and retrieval date. Re-run
this script each year and diff the manifest to see what changed.

Usage:
    python ingestion/download_sources.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

USER_AGENT = (
    "diabetes-rag-assistant-ingestion/1.0 "
    "(+https://github.com/richanitdgp/diabetes-rag-assistant)"
)


@dataclass
class Source:
    id: str
    title: str
    publisher: str
    url: str
    category: str  # "clinical_guideline" | "patient_education"
    local_path: Path
    notes: str = ""


# v1 sources: ADA + CDC only. See module docstring.
#
# NOTE ON URLS: these were located via web search from a sandboxed session
# whose network egress policy blocks diabetesjournals.org, professional.diabetes.org,
# diabetes.org, and cdc.gov, so they could not be verified by an actual fetch here.
# The script detects content-type at download time and stores whatever is
# actually returned (see `status`/`content_type` in the manifest); if a URL
# below turns out to be a landing page rather than a direct file, update it
# once you can browse the site and re-run.
SOURCES: list[Source] = [
    Source(
        id="ada-standards-of-care-2026",
        title="Standards of Care in Diabetes—2026 (Abridged for Primary Care Professionals)",
        publisher="American Diabetes Association (ADA)",
        url="https://diabetesjournals.org/docm-care/article/1/3/487/164620/Standards-of-Care-in-Diabetes-2026-Abridged-for",
        category="clinical_guideline",
        local_path=RAW_DIR / "ada" / "standards-of-care-2026-abridged.pdf",
        notes=(
            "Abridged Standards of Care, chosen for v1 as a single compact PDF "
            "with recommendations substantively the same as the complete "
            "Standards of Care (Diabetes Care, Vol 49, Supplement 1). "
            "Full multi-chapter edition: "
            "https://diabetesjournals.org/care/issue/49/Supplement_1"
        ),
    ),
    Source(
        id="cdc-diabetes-basics",
        title="Diabetes Basics",
        publisher="Centers for Disease Control and Prevention (CDC)",
        url="https://www.cdc.gov/diabetes/about/index.html",
        category="patient_education",
        local_path=RAW_DIR / "cdc" / "diabetes-basics.html",
    ),
    Source(
        id="cdc-living-with-diabetes",
        title="Living with Diabetes",
        publisher="Centers for Disease Control and Prevention (CDC)",
        url="https://www.cdc.gov/diabetes/living-with/index.html",
        category="patient_education",
        local_path=RAW_DIR / "cdc" / "living-with-diabetes.html",
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source) -> dict:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    entry = {
        "id": source.id,
        "title": source.title,
        "publisher": source.publisher,
        "category": source.category,
        "source_url": source.url,
        "local_path": str(source.local_path.relative_to(REPO_ROOT)),
        "retrieved_at": retrieved_at,
        "notes": source.notes,
    }
    try:
        resp = requests.get(
            source.url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        return entry

    source.local_path.parent.mkdir(parents=True, exist_ok=True)
    source.local_path.write_bytes(resp.content)

    entry["status"] = "ok"
    entry["final_url"] = resp.url
    entry["content_type"] = resp.headers.get("Content-Type", "")
    entry["size_bytes"] = len(resp.content)
    entry["sha256"] = sha256_of(source.local_path)
    return entry


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"sources": []}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    manifest = load_manifest()
    by_id = {entry["id"]: entry for entry in manifest["sources"]}

    for source in SOURCES:
        print(f"Fetching {source.id} <- {source.url}")
        entry = download(source)
        by_id[entry["id"]] = entry
        if entry["status"] == "ok":
            print(f"  ok: {entry['size_bytes']} bytes -> {entry['local_path']}")
        else:
            print(f"  ERROR: {entry['error']}")

    manifest["sources"] = list(by_id.values())
    save_manifest(manifest)
    print(f"\nManifest written to {MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
