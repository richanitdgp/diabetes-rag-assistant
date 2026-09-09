"""Structure-aware parsers: raw source files -> a flat list of Chunks.

Each source type is split along its own natural document structure rather
than fixed-size windows, so a chunk boundary always lands on something a
human would call a section:

- CDC pages (HTML): one chunk per heading (<h2>/<h3>/<h4>) within the page's
  main content, using the heading text as the chunk's section title.
- ADA Standards of Care (PDF): one chunk per numbered chapter/recommendation
  heading (e.g. "2. Diagnosis and Classification", "2.1 ...").

Every Chunk carries the metadata needed to cite it back to its source:
source name, section title, page/URL, and publication year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

# Section headings that are page furniture (nav/newsletter signup), not
# content, and would otherwise become junk chunks.
_CDC_BOILERPLATE_HEADINGS = {"on this page", "get email updates"}


@dataclass
class Chunk:
    text: str
    source_id: str
    source_title: str
    publisher: str
    section_title: str
    url: str
    publication_year: Optional[int] = None
    page: Optional[int] = None
    chunk_index: int = 0

    def metadata(self) -> dict:
        """Metadata dict suitable for a Chroma `metadatas` entry (no None values — Chroma rejects them)."""
        meta = {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "publisher": self.publisher,
            "section_title": self.section_title,
            "url": self.url,
            "chunk_index": self.chunk_index,
        }
        if self.publication_year is not None:
            meta["publication_year"] = self.publication_year
        if self.page is not None:
            meta["page"] = self.page
        return meta

    @property
    def id(self) -> str:
        return f"{self.source_id}::{self.chunk_index}"


def parse_html(path: Path, source: dict) -> list[Chunk]:
    """Split a CDC-style HTML page into one chunk per heading in its main content.

    CDC pages wrap each heading and its body in separate sibling divs (e.g.
    ``.dfe-section__header`` / ``.dfe-section__body``), so a chunk's content
    isn't necessarily a DOM sibling of its heading. To stay robust to that,
    this flattens the content area to plain text (``get_text``, which
    preserves document order regardless of wrapper nesting) and splits on
    each heading's own text as a line, rather than walking the DOM tree.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    content = soup.find(id="content") or soup.find("main") or soup
    year = _cdc_publication_year(soup)
    base_url = source.get("final_url") or source["source_url"]

    headings = [h for h in content.find_all(["h1", "h2", "h3", "h4"]) if h.get_text(strip=True)]
    lines = [line for line in content.get_text("\n", strip=True).split("\n") if line.strip()]

    # Locate each heading's line by scanning forward from the previous match,
    # so repeated titles (e.g. "Type 1 diabetes" under two different h2s)
    # resolve to their correct, distinct occurrences in document order.
    positions: list[tuple[int, Tag]] = []
    search_start = 0
    for heading in headings:
        title = heading.get_text(" ", strip=True)
        try:
            idx = lines.index(title, search_start)
        except ValueError:
            continue
        positions.append((idx, heading))
        search_start = idx + 1

    chunks: list[Chunk] = []
    parent_h2 = ""
    for i, (idx, heading) in enumerate(positions):
        title = heading.get_text(" ", strip=True)
        if title.lower() in _CDC_BOILERPLATE_HEADINGS:
            continue
        if heading.name == "h2":
            parent_h2 = title

        end = positions[i + 1][0] if i + 1 < len(positions) else len(lines)
        body = " ".join(lines[idx + 1 : end]).strip()
        if not body:
            continue

        section_title = title if heading.name in ("h1", "h2") or title == parent_h2 else f"{parent_h2} — {title}"
        anchor = heading.get("id")
        url = f"{base_url}#{anchor}" if anchor else base_url

        chunks.append(
            Chunk(
                text=f"{title}\n\n{body}",
                source_id=source["id"],
                source_title=source["title"],
                publisher=source["publisher"],
                section_title=section_title,
                url=url,
                publication_year=year,
                chunk_index=len(chunks),
            )
        )

    return chunks


def _cdc_publication_year(soup: BeautifulSoup) -> Optional[int]:
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        match = re.search(r"\d{4}", time_tag["datetime"])
        if match:
            return int(match.group())

    meta = soup.find("meta", attrs={"name": "cdc:last_updated"})
    if meta and meta.get("content"):
        match = re.search(r"\d{4}", meta["content"])
        if match:
            return int(match.group())

    return None


# Matches ADA-style numbered headings, e.g. "2. Diagnosis and Classification
# of Diabetes" (chapter) or "2.1 Screening for Type 2 Diabetes" (recommendation).
_ADA_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-Z][A-Za-z0-9 ,/&()'-]{3,100})$")


def parse_pdf(path: Path, source: dict) -> list[Chunk]:
    """Split the ADA Standards of Care PDF into one chunk per numbered chapter/recommendation heading.

    NOTE: unverified against a real ADA PDF as of writing (ADA is currently
    deferred — see ingestion/download_sources.py). The heuristic below
    targets ADA's documented numbering convention ("N. Chapter Title",
    "N.N Recommendation"); re-check chunk boundaries against a real
    extraction once the source is downloaded, and adjust _ADA_HEADING_RE
    if the PDF's line breaks don't match this pattern.
    """
    reader = PdfReader(str(path))
    year_match = re.search(r"20\d{2}", source["title"])
    year = int(year_match.group()) if year_match else None
    base_url = source.get("final_url") or source["source_url"]

    chunks: list[Chunk] = []
    current_heading = source["title"]
    current_page = 1
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(
                Chunk(
                    text=f"{current_heading}\n\n{text}",
                    source_id=source["id"],
                    source_title=source["title"],
                    publisher=source["publisher"],
                    section_title=current_heading,
                    url=base_url,
                    publication_year=year,
                    page=current_page,
                    chunk_index=len(chunks),
                )
            )

    for page_number, page in enumerate(reader.pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if not line:
                continue
            match = _ADA_HEADING_RE.match(line)
            if match:
                flush()
                current_heading = line
                current_page = page_number
                current_lines = []
            else:
                current_lines.append(line)

    flush()
    return chunks


def parse_source(local_path: Path, source: dict) -> list[Chunk]:
    """Dispatch to the right parser based on the source's recorded content type."""
    content_type = source.get("content_type", "")
    if "html" in content_type or local_path.suffix == ".html":
        return parse_html(local_path, source)
    if "pdf" in content_type or local_path.suffix == ".pdf":
        return parse_pdf(local_path, source)
    raise ValueError(f"Don't know how to parse {source['id']} (content_type={content_type!r})")
