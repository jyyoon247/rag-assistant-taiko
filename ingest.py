#!/usr/bin/env python3
"""
Ingest anonymized taiko interview documents (.docx) into RAG-ready JSONL chunks.

Each document is expected to look like:

    DocumentID: 20191104_P101_G101_email
    Date: 2019-11-04
    Interviewee: P101
    Collegiate Taiko Group Name: G101
    Message Type: Email
    School Affiliation: S101
    Location: West

    Q: question text
    A: answer text (may span several paragraphs, lists, or a table)
    Q: ...

Output: one chunk per Q+A pair, with normalized metadata and a short context
header prepended to the text so the embedding model and the LLM know whose
words they are reading.

Usage:
    pip install python-docx
    python ingest.py "RAG Documents" --out chunks.jsonl --report ingest_report.md \
        [--names private_names.txt]

--names is an optional PRIVATE file (never commit it) with one real name or
alias per line. Every chunk is searched for them; any hit is reported as a leak.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:
    sys.exit("python-docx is required: pip install python-docx")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map every header spelling seen in the documents to one canonical field name.
HEADER_ALIASES = {
    "documentid": "doc_id",
    "document id": "doc_id",
    "date": "date",
    "interviewee": "participant",
    "collegiate taiko group name": "group",
    "collegiate taiko group": "group",
    "group name": "group",
    "group": "group",
    "message type": "source_type",
    "school affiliation": "school",
    "location": "region",
}
REQUIRED_FIELDS = ["doc_id", "date", "participant", "group", "source_type", "school", "region"]

# US Census regions. Change this set if you adopt a different scheme,
# but use ONE scheme for every document.
ALLOWED_REGIONS = {"Northeast", "Midwest", "South", "West"}
ALLOWED_SOURCE_TYPES = {"email", "interview"}

ID_PATTERN = re.compile(r"\b[PGSH]\d{3}\b")          # e.g. P101, G101, S101, H101
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_PATTERN = re.compile(r"\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")


# ---------------------------------------------------------------------------
# Reading the .docx in document order (paragraphs AND tables)
# ---------------------------------------------------------------------------

def iter_blocks(document):
    """Yield ('p', text) and ('table', rows) in the order they appear."""
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield "p", Paragraph(child, document).text
        elif tag == "tbl":
            rows = []
            for row in Table(child, document).rows:
                cells = [clean(c.text) for c in row.cells]
                if any(cells):
                    rows.append(cells)
            yield "table", rows


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]+", " ", text).strip()


def table_to_text(rows: list[list[str]]) -> str:
    """Render a table as pipe-separated lines (readable by LLMs, greppable)."""
    out = []
    for i, row in enumerate(rows):
        # python-docx repeats merged cells; drop consecutive duplicates
        dedup = [c for j, c in enumerate(row) if j == 0 or c != row[j - 1]]
        out.append("| " + " | ".join(dedup) + " |")
        if i == 0:
            out.append("|" + "---|" * len(dedup))
    return "\n".join(out)


def join_line(prev: str, line: str) -> str:
    """Join a continuation paragraph: repair hard line wraps, keep real breaks."""
    if not prev:
        return line
    wrapped = not re.search(r"[.!?:;)\]\"]$", prev) and line[:1].islower()
    list_item = re.match(r"^(\d+[.)]|[-*•])\s", line)
    if wrapped and not list_item:
        return prev + " " + line
    return prev + "\n" + line


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_date(value: str, warn) -> str:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    warn(f"Unrecognized date format: {value!r}")
    return value


def parse_document(path: Path):
    warnings: list[str] = []
    warn = warnings.append
    document = docx.Document(str(path))

    meta: dict[str, str] = {}
    pairs: list[dict] = []
    current: dict | None = None
    state = "header"  # header -> q -> a

    for kind, content in iter_blocks(document):
        if kind == "table":
            if current is None:
                warn("Table found before the first question; skipped.")
                continue
            if state == "q":  # a table right after a question is its answer
                warn(f"Q{len(pairs)}: answer has no 'A:' prefix (table follows the question).")
                state = "a"
            current["answer"] = join_line(current["answer"], table_to_text(content))
            continue

        line = clean(content)
        if not line:
            continue

        if state == "header" and not line.startswith("Q:"):
            m = re.match(r"^([^:]{2,40}):\s*(.*)$", line)
            if not m:
                warn(f"Unparsed header line: {line!r}")
                continue
            key = HEADER_ALIASES.get(m.group(1).strip().lower())
            if key is None:
                warn(f"Unknown header field {m.group(1)!r}; add it to HEADER_ALIASES.")
                continue
            meta[key] = m.group(2).strip()
            continue

        if line.startswith("Q:"):
            current = {"question": line[2:].strip(), "answer": ""}
            pairs.append(current)
            state = "q"
        elif line.startswith("A:"):
            if current is None:
                warn("'A:' found before any 'Q:'; skipped.")
                continue
            if current["answer"]:
                warn(f"Q{len(pairs)}: second 'A:' line; appended to the same answer.")
            current["answer"] = join_line(current["answer"], line[2:].strip())
            state = "a"
        elif state == "q":
            # Either the question wraps onto a new line, or the answer is missing its 'A:'.
            if current["question"].rstrip().endswith("?") and not line[:1].islower():
                warn(f"Q{len(pairs)}: answer has no 'A:' prefix: {line[:60]!r}")
                current["answer"] = line
                state = "a"
            else:
                current["question"] = join_line(current["question"], line)
        else:  # continuation of an answer
            if line.endswith("?") and re.match(r"^(\d+[.)]\s+)?[A-Z]", line) and len(line) > 25:
                warn(f"Q{len(pairs)}: possible question without 'Q:' prefix inside an answer: "
                     f"{line[:70]!r}")
            current["answer"] = join_line(current["answer"], line)

    # ---- validate metadata -------------------------------------------------
    for field in REQUIRED_FIELDS:
        if not meta.get(field):
            warn(f"Missing header field: {field}")
    if "date" in meta:
        meta["date"] = normalize_date(meta["date"], warn)
    if "source_type" in meta:
        meta["source_type"] = meta["source_type"].lower()
        if meta["source_type"] not in ALLOWED_SOURCE_TYPES:
            warn(f"Unexpected Message Type: {meta['source_type']!r}")
    if meta.get("region") and meta["region"] not in ALLOWED_REGIONS:
        warn(f"Location {meta['region']!r} is not one of {sorted(ALLOWED_REGIONS)}")
    if " " in path.name:
        warn(f"File name contains a space: {path.name!r}")
    if meta.get("doc_id") and meta["doc_id"] != path.stem.replace(" ", ""):
        warn(f"DocumentID {meta['doc_id']!r} does not match file name {path.stem!r}")
    for i, pair in enumerate(pairs, 1):
        if not pair["answer"]:
            warn(f"Q{i}: no answer text.")

    return meta, pairs, warnings


# ---------------------------------------------------------------------------
# Chunks and leak checks
# ---------------------------------------------------------------------------

def make_chunks(meta: dict, pairs: list[dict], source_file: str) -> list[dict]:
    header = (f"[Group {meta.get('group', '?')} | interviewee {meta.get('participant', '?')} | "
              f"school {meta.get('school', '?')} | region {meta.get('region', '?')} | "
              f"{meta.get('source_type', '?')}, {meta.get('date', '?')}]")
    chunks = []
    for i, pair in enumerate(pairs, 1):
        body = f"Q: {pair['question']}\nA: {pair['answer']}"
        own = {meta.get("participant"), meta.get("group"), meta.get("school")}
        mentions = sorted(set(ID_PATTERN.findall(body)) - own)
        chunks.append({
            "chunk_id": f"{meta.get('doc_id', Path(source_file).stem)}_q{i:02d}",
            "doc_id": meta.get("doc_id"),
            "qa_index": i,
            "participant": meta.get("participant"),
            "group": meta.get("group"),
            "school": meta.get("school"),
            "region": meta.get("region"),
            "source_type": meta.get("source_type"),
            "date": meta.get("date"),
            "question": pair["question"],
            "answer": pair["answer"],
            "mentions": mentions,          # other pseudonymized people/groups referenced
            "text": f"{header}\n{body}",   # what gets embedded and shown to the LLM
            "source_file": source_file,
        })
    return chunks


def load_names(path: Path | None) -> list[str]:
    if not path:
        return []
    return [n.strip() for n in path.read_text(encoding="utf-8").splitlines()
            if n.strip() and not n.startswith("#")]


def leak_check(chunk: dict, names: list[str]) -> list[str]:
    text = chunk["question"] + "\n" + chunk["answer"]
    hits = []
    for name in names:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.I):
            hits.append(f"real name/alias {name!r}")
    hits += [f"URL {u!r}" for u in URL_PATTERN.findall(text)]
    hits += [f"email address {e!r}" for e in EMAIL_PATTERN.findall(text)]
    hits += [f"phone number {p!r}" for p in PHONE_PATTERN.findall(text)]
    return hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("ingest_report.md"))
    ap.add_argument("--names", type=Path, help="private file of real names/aliases to scan for")
    args = ap.parse_args()

    files = sorted(p for p in args.input_dir.glob("*.docx") if not p.name.startswith("~$"))
    if not files:
        sys.exit(f"No .docx files found in {args.input_dir}")
    names = load_names(args.names)

    all_chunks, report = [], ["# Ingestion report", ""]
    total_warnings = total_leaks = 0
    for path in files:
        meta, pairs, warnings = parse_document(path)
        chunks = make_chunks(meta, pairs, path.name)
        leaks = [f"{c['chunk_id']}: {h}" for c in chunks for h in leak_check(c, names)]
        all_chunks += chunks
        total_warnings += len(warnings)
        total_leaks += len(leaks)

        report += [f"## {path.name}", "",
                   f"- Q+A chunks: {len(chunks)}",
                   f"- Metadata: " + ", ".join(f"{k}={meta.get(k)}" for k in REQUIRED_FIELDS),
                   f"- Mentions other IDs: " + (", ".join(sorted({m for c in chunks for m in c['mentions']})) or "none")]
        report += ["", "**Format warnings**", ""] + ([f"- {w}" for w in warnings] or ["- none"])
        report += ["", "**Possible leaks**", ""] + ([f"- {l}" for l in leaks] or ["- none found"])
        report.append("")

    with args.out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = (f"{len(files)} documents -> {len(all_chunks)} chunks | "
               f"{total_warnings} format warnings | {total_leaks} possible leaks"
               + ("" if names else " (no --names file given; only URL/email/phone checked)"))
    report[1:1] = [summary, ""]
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(summary)
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
