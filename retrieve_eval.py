#!/usr/bin/env python3
"""
Retrieval evaluation for the taiko RAG project. Runs fully locally.

Methods compared over chunks.jsonl:
  - bm25:          keyword search
  - embedding:     semantic search with a local embedding model
  - hybrid:        both rankings merged with reciprocal rank fusion (RRF)
  - hybrid_rerank: hybrid top-N re-scored by a local cross-encoder (with --rerank)

Options:
  --split dev|test|all   which questions to score (default: dev). The split is
                         created once in eval_split.csv (stratified by question
                         type, fixed seed) and reused on every run. Experiment on
                         dev; run test only at the end.
  --group-filter         if a question names a group (e.g. G101), rank that group's
                         chunks first, then everything else.
  --rerank               add the hybrid_rerank method.

Every run writes retrieval_results_<tag>.csv and retrieval_summary_<tag>.md and
appends one line per method to experiments_log.csv.

Setup (once):
    python3 -m venv .venv && source .venv/bin/activate
    pip install sentence-transformers rank-bm25

Typical sequence (from the rag_pipeline folder):
    python retrieve_eval.py --split dev                          # baseline
    python retrieve_eval.py --split dev --group-filter           # improvement 1
    python retrieve_eval.py --split dev --rerank                 # improvement 2
    python retrieve_eval.py --split dev --group-filter --rerank  # both
    python retrieve_eval.py --split test --group-filter --rerank # final, once

Models download from Hugging Face on first use and are cached; after that the
script runs offline. No document text is sent anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    sys.exit("Missing packages. Run: pip install sentence-transformers rank-bm25")

# bge embedding models retrieve better when short queries carry this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60                      # standard constant for reciprocal rank fusion
GROUP_ID = re.compile(r"\bG\d{3}\b")
SPLIT_SEED = 42
TEST_FRACTION = 1 / 3


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    chunks = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not chunks:
        sys.exit(f"No chunks in {path}")
    return chunks


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gold"] = [c.strip() for c in r.get("source_chunks", "").split(";") if c.strip()]
    return rows


def short_id(chunk: dict) -> str:
    """20191104_P101_G101_email_q07 -> G101_q07 (easier to read in results)."""
    return f"{chunk.get('group') or chunk['doc_id']}_q{chunk['qa_index']:02d}"


# ---------------------------------------------------------------------------
# Dev/test split: created once, then reused so every run scores the same set
# ---------------------------------------------------------------------------

def load_or_create_split(path: Path, questions: list[dict]) -> dict[str, str]:
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            split = {r["qid"]: r["split"] for r in csv.DictReader(f)}
        missing = [q["qid"] for q in questions if q["qid"] not in split]
        if missing:
            sys.exit(f"{path} has no split for {missing}. Add them by hand "
                     f"(qid,split) or delete {path} to regenerate.")
        return split

    rng = random.Random(SPLIT_SEED)
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q["qid"])
    split = {}
    for qids in by_type.values():
        qids = sorted(qids)
        rng.shuffle(qids)
        n_test = round(len(qids) * TEST_FRACTION) if len(qids) >= 3 else 0
        for i, qid in enumerate(qids):
            split[qid] = "test" if i < n_test else "dev"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "split"])
        for q in questions:
            w.writerow([q["qid"], q["type"], split[q["qid"]]])
    n_test = sum(v == "test" for v in split.values())
    print(f"Created {path}: {len(split) - n_test} dev / {n_test} test questions "
          f"(stratified by type). It will be reused on every run.")
    return split


# ---------------------------------------------------------------------------
# Retrieval methods: each returns chunk indices ranked best-first
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    # Lowercase words and IDs like g006 / p010; drop very short noise tokens.
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


class BM25Retriever:
    def __init__(self, texts: list[str]):
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def rank(self, query: str) -> list[int]:
        scores = self.index.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores, kind="stable")]


class EmbeddingRetriever:
    def __init__(self, texts: list[str], model_name: str):
        print(f"Loading embedding model {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        t0 = time.time()
        self.doc_emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        print(f"Embedded {len(texts)} chunks in {time.time() - t0:.1f}s "
              f"(device: {self.model.device})")

    def rank(self, query: str) -> list[int]:
        q = self.model.encode([self.prefix + query], normalize_embeddings=True)[0]
        scores = self.doc_emb @ q  # cosine similarity (vectors are normalized)
        return [int(i) for i in np.argsort(-scores, kind="stable")]


class Reranker:
    """Cross-encoder: reads the question and each chunk together and scores relevance."""

    def __init__(self, texts: list[str], model_name: str, top_n: int):
        print(f"Loading reranker {model_name} ...")
        self.model = CrossEncoder(model_name)
        self.texts = texts
        self.top_n = top_n

    def rerank(self, query: str, ranking: list[int]) -> list[int]:
        head, tail = ranking[:self.top_n], ranking[self.top_n:]
        scores = self.model.predict([(query, self.texts[i]) for i in head], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [head[int(j)] for j in order] + tail


def rrf_fuse(*rankings: list[int]) -> list[int]:
    """Reciprocal rank fusion: score = sum of 1 / (RRF_K + rank) across rankings."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] += 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda i: -scores[i])


def group_first(ranking: list[int], query: str, chunk_groups: list[str]) -> list[int]:
    """If the query names groups, move those groups' chunks to the front (order kept)."""
    named = set(GROUP_ID.findall(query))
    if not named:
        return ranking
    inside = [i for i in ranking if chunk_groups[i] in named]
    outside = [i for i in ranking if chunk_groups[i] not in named]
    return inside + outside


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(ranked_ids: list[str], gold: list[str], ks: list[int]) -> dict:
    gold_set = set(gold)
    out = {}
    for k in ks:
        top = set(ranked_ids[:k])
        out[f"recall@{k}"] = len(top & gold_set) / len(gold_set)
        out[f"hit@{k}"] = 1.0 if top & gold_set else 0.0
    first = next((r for r, cid in enumerate(ranked_ids, 1) if cid in gold_set), None)
    out["first_gold_rank"] = first
    out["rr"] = 1.0 / first if first else 0.0
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("eval_questions.csv"))
    ap.add_argument("--split-file", type=Path, default=Path("eval_split.csv"))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--group-filter", action="store_true",
                    help="rank chunks of groups named in the question first")
    ap.add_argument("--rerank", action="store_true", help="add the hybrid_rerank method")
    ap.add_argument("--rerank-top", type=int, default=20,
                    help="how many hybrid results the reranker re-scores (default 20)")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-base",
                    help="cross-encoder model (bge-reranker-base is ~1.1 GB; a smaller option is "
                         "cross-encoder/ms-marco-MiniLM-L-6-v2, ~90 MB)")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="embedding model, e.g. nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--tag", help="name for this run's output files (default: built from options)")
    ap.add_argument("--log", type=Path, default=Path("experiments_log.csv"))
    args = ap.parse_args()
    ks = sorted(args.k)
    tag = args.tag or "_".join([args.split] + (["gf"] if args.group_filter else [])
                               + (["rr"] if args.rerank else []) or ["base"])
    if tag == args.split:
        tag += "_base"

    chunks = load_chunks(args.chunks)
    all_questions = load_questions(args.questions)
    split = load_or_create_split(args.split_file, all_questions)
    questions = [q for q in all_questions if args.split == "all" or split[q["qid"]] == args.split]

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    groups = [c.get("group") or "" for c in chunks]
    short = {c["chunk_id"]: short_id(c) for c in chunks}

    unknown = sorted({g for q in questions for g in q["gold"] if g not in short})
    if unknown:
        sys.exit(f"eval_questions.csv references chunks not in {args.chunks}: {unknown}")

    bm25 = BM25Retriever(texts)
    emb = EmbeddingRetriever(texts, args.model)
    reranker = Reranker(texts, args.reranker_model, args.rerank_top) if args.rerank else None

    methods = ["bm25", "embedding", "hybrid"] + (["hybrid_rerank"] if reranker else [])
    rows, skipped = [], []
    t0 = time.time()
    for q in questions:
        if not q["gold"]:
            skipped.append(q["qid"])
            continue
        query = q["question"]
        b, e = bm25.rank(query), emb.rank(query)
        h = rrf_fuse(b, e)
        rankings = {"bm25": b, "embedding": e, "hybrid": h}
        if args.group_filter:
            rankings = {m: group_first(r, query, groups) for m, r in rankings.items()}
        if reranker:
            # Rerank the (possibly group-filtered) hybrid list, then keep named groups first.
            rr = reranker.rerank(query, rankings["hybrid"])
            rankings["hybrid_rerank"] = group_first(rr, query, groups) if args.group_filter else rr
        for m in methods:
            ranked_ids = [ids[i] for i in rankings[m]]
            s = score(ranked_ids, q["gold"], ks)
            rows.append({
                "qid": q["qid"], "type": q["type"], "split": split[q["qid"]], "method": m,
                **s,
                "n_gold": len(q["gold"]),
                "gold": ";".join(short[g] for g in q["gold"]),
                f"top{ks[0]}": ";".join(short[c] for c in ranked_ids[:ks[0]]),
                "question": query,
            })
    elapsed = time.time() - t0
    if not rows:
        sys.exit("No questions with gold chunks in this split.")

    out_csv = Path(f"retrieval_results_{tag}.csv")
    out_md = Path(f"retrieval_summary_{tag}.md")

    # ---- per-question results --------------------------------------------
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary by method and question type -------------------------------
    types = sorted({r["type"] for r in rows})
    metric_cols = [f"recall@{k}" for k in ks] + ["rr"]
    n_scored = len(rows) // len(methods)
    options = (f"split={args.split} | group_filter={'on' if args.group_filter else 'off'} | "
               f"rerank={'on (' + args.reranker_model + ', top ' + str(args.rerank_top) + ')' if reranker else 'off'}")
    lines = [
        f"# Retrieval results: {tag}",
        "",
        f"{options}",
        "",
        f"Embedding model: `{args.model}` | chunks: {len(chunks)} | questions scored: {n_scored} "
        f"| skipped (no gold chunks): {', '.join(skipped) or 'none'} "
        f"| search time: {1000 * elapsed / max(n_scored, 1):.0f} ms/question (all methods)",
        "",
        "MRR = mean reciprocal rank of the first correct chunk (1.0 = always ranked first).",
        "",
    ]
    overall = {}
    for metric in metric_cols:
        label = "MRR" if metric == "rr" else metric
        lines += [f"## {label}", "", "| type | n | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for t in types + ["ALL"]:
            sel = [r for r in rows if t == "ALL" or r["type"] == t]
            n = len(sel) // len(methods)
            vals = [mean([r[metric] for r in sel if r["method"] == m]) for m in methods]
            if t == "ALL":
                for m, v in zip(methods, vals):
                    overall.setdefault(m, {})[label] = v
            name = f"**{t}**" if t == "ALL" else t
            lines.append(f"| {name} | {n} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
        lines.append("")

    # ---- misses worth looking at (best method, single-answer question types) ----
    best = methods[-1]
    k0 = ks[0]
    misses = [r for r in rows if r["method"] == best
              and r["type"] in ("factual", "narrative") and r[f"recall@{k0}"] < 1]
    lines += [f"## {best} misses on factual/narrative questions (recall@{k0} < 1)", ""]
    if misses:
        lines += [f"| qid | question | gold | first correct rank | top {k0} retrieved |",
                  "|---|---|---|---|---|"]
        for r in misses:
            lines.append(f"| {r['qid']} | {r['question']} | {r['gold']} | "
                         f"{r['first_gold_rank']} | {r[f'top{k0}']} |")
    else:
        lines.append("None.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- experiment log (one line per method per run) ----------------------
    new_log = not args.log.exists()
    with args.log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        labels = [f"recall@{k}" for k in ks] + ["MRR"]
        if new_log:
            w.writerow(["timestamp", "tag", "split", "group_filter", "rerank", "embedding_model",
                        "method", "n_questions"] + labels)
        for m in methods:
            w.writerow([datetime.now().isoformat(timespec="seconds"), tag, args.split,
                        args.group_filter, args.reranker_model if reranker else "", args.model,
                        m, n_scored] + [f"{overall[m][l]:.3f}" for l in labels])

    print()
    print("\n".join(lines))
    print(f"\nWrote {out_csv}, {out_md}; appended to {args.log}")


if __name__ == "__main__":
    main()
