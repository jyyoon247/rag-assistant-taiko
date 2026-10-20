#!/usr/bin/env python3
"""
Answer generation + scoring for the taiko RAG project. Runs fully locally.

Strategies (all scored the same way, so runs are directly comparable):

  default            plain RAG: hybrid search + group filter, top-k chunks -> answer
  --router           send cross-group questions ("which groups...", "how many...")
                     down an aggregate path; everything else uses plain RAG
    --aggregate-mode per_group   (default) for each group, retrieve that group's
                                 top chunks and extract its finding; then one
                                 final call combines the findings
    --aggregate-mode table       answer from the structured table (groups.csv)
  --follow-refs      if retrieved chunks mention another group in the collection
                     (e.g. G101's chunk mentions G102), also retrieve that group's
                     most relevant chunks

Scores per question:
  correctness  LLM judge vs your gold answer (correct 1 / partial 0.5 / incorrect 0)
  faithful     judge: every claim supported by the cited evidence
  abstention   NOT FOUND on unanswerable questions, and only those
  citations    gold chunks cited (recall) / cited chunks that were gold (precision)
  route        which path the router chose (routing accuracy is reported)

Human grades are never lost: any grade you entered in an earlier answers_*.csv is
copied into the new file wherever the model gave the identical answer to the same
question. Only new or changed answers need grading.

Setup (once): install Ollama (https://ollama.com/download), then: ollama pull qwen2.5:7b

Examples (from the rag_pipeline folder, venv active, Ollama running):
    python generate_eval.py --split dev                                    # plain RAG
    python generate_eval.py --split dev --router                           # per-group
    python generate_eval.py --split dev --router --aggregate-mode table    # table
    python generate_eval.py --split dev --router --follow-refs             # + references
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from retrieve_eval import (BM25Retriever, EmbeddingRetriever, Reranker, group_first,
                           load_chunks, load_or_create_split, load_questions, rrf_fuse,
                           short_id)

GROUP_ID = re.compile(r"\bG\d{3}\b")

ANSWER_SYSTEM = """You answer questions about collegiate taiko groups using ONLY the numbered
excerpts provided. Each excerpt is an interview Q&A labeled like [C1], with a header naming the
group (e.g. G004) and interviewee (e.g. P010).

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge or guess.
- Pay attention to which group each excerpt is about; do not mix up groups.
- Cite every excerpt you used by its label, e.g. ["C2", "C5"].
- If the excerpts do not contain the answer, set "found" to false and answer "NOT FOUND".
- Pseudonyms like G001, P003, S002 are the only names available; never invent real names.

Respond with JSON only:
{"answer": "<concise answer>", "citations": ["C1"], "found": true}"""

MAP_SYSTEM = """You help answer a question about ALL collegiate taiko groups by reporting what
the interview excerpts say about ONE group. All excerpts below are about group {group}.

Rules:
- Report only what the excerpts state that is relevant to the question: include exact numbers,
  dates, names (pseudonyms like P003) and pieces. Do not answer for other groups.
- If the excerpts say nothing relevant, set "found" to false and finding to "NOT STATED".
- Cite the excerpts you used by label, e.g. ["C1"].

Respond with JSON only:
{{"finding": "<what the excerpts say about {group} for this question>", "citations": ["C1"], "found": true}}"""

REDUCE_SYSTEM = """You answer a question about collegiate taiko groups from per-group findings.
Each line gives one group's finding, extracted from that group's interview.

Rules:
- Consider EVERY group listed before answering; do not skip any.
- A finding of NOT STATED means the interview did not say. Never assume a value for it; mention
  it as not stated when the question is about all groups.
- Count and compare carefully. Use only the findings.
- List the groups whose findings you used in "groups_used".
- If no finding answers the question, set "found" to false and answer "NOT FOUND".

Respond with JSON only:
{"answer": "<concise answer>", "groups_used": ["G001"], "found": true}"""

TABLE_SYSTEM = """You answer questions about collegiate taiko groups using ONLY the table below.
Each row is one group; values come from interviews. "not stated" means the interview did not say.

Rules:
- Read EVERY row before answering; do not skip any group.
- Use only what the table states. Never assume a value that says "not stated".
- Count and compare carefully.
- List the groups (rows) you used in "groups_used".
- If the table does not contain the answer, set "found" to false and answer "NOT FOUND".

Respond with JSON only:
{"answer": "<concise answer>", "groups_used": ["G001"], "found": true}"""

JUDGE_SYSTEM = """You grade answers from a question-answering system against a reference answer
written by the researcher who conducted the interviews.

Grade "correctness":
- "correct": matches the reference in substance (wording may differ; extra correct detail is fine)
- "partial": some of the reference is right but something important is missing or wrong
- "incorrect": wrong, contradicts the reference, or answers a different question
If the reference says the information is not stated/unknown, the system is "correct" only if it
says it could not find the answer.

Grade "faithful": true if every claim in the system answer is supported by the evidence the
system cited (or the system said NOT FOUND); false if it states anything the evidence does not support.

Respond with JSON only:
{"correctness": "correct|partial|incorrect", "faithful": true, "reason": "<one sentence>"}"""

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
NOT_FOUND = re.compile(r"\bnot\s+found\b|not (stated|mentioned|provided|in the excerpts)", re.I)
# Cross-group cues for the router (used only when the question names no group).
AGG_SUBJECT = re.compile(r"\bgroups?\b|\bfound(?:ing|ers?)\b|\bmembers?\b", re.I)
AGG_CUE = re.compile(r"\b(which|how many|each|every|all|across|total|range|largest|smallest|"
                     r"oldest|newest|youngest|most|least|any)\b", re.I)
TABLE_SKIP_COLS = {"source_chunks", "needs_review", "doc_id"}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_chat(url: str, model: str, system: str, user: str, num_ctx: int) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": "json",
        "stream": False,
        # temperature 0 for repeatable runs; num_ctx raised because Ollama's default
        # context window is small and would silently cut off the excerpts.
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {url} ({e}). Is the Ollama app running?")
    elapsed = time.time() - t0
    if "error" in body:
        sys.exit(f"Ollama error: {body['error']} (did you run `ollama pull {model}`?)")
    text = body.get("message", {}).get("content", "")
    try:
        out = json.loads(text)
        return (out if isinstance(out, dict) else {"_raw": text}), elapsed
    except json.JSONDecodeError:
        return {"_raw": text}, elapsed


def check_model(url: str, model: str):
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as resp:
            names = {m["name"] for m in json.loads(resp.read()).get("models", [])}
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {url} ({e}). Open the Ollama app first.")
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"Model {model!r} not found in Ollama. Run: ollama pull {model}")


def as_list(value) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, args, chunks):
        self.a = args
        self.texts = [c["text"] for c in chunks]
        self.ids = [c["chunk_id"] for c in chunks]
        self.idx = {cid: i for i, cid in enumerate(self.ids)}
        self.groups = [c.get("group") or "" for c in chunks]
        self.mentions = [set(m for m in c.get("mentions", []) if GROUP_ID.fullmatch(m)) for c in chunks]
        self.all_groups = sorted({g for g in self.groups if g})
        self.bm25 = BM25Retriever(self.texts)
        self.emb = EmbeddingRetriever(self.texts, args.embed_model)
        self.reranker = Reranker(self.texts, args.reranker_model, 20) if args.rerank else None
        self.table_rows = self._load_table(args.table) if args.router and args.aggregate_mode == "table" else {}

    # ---- retrieval ---------------------------------------------------------
    def rank(self, query: str, only_group: str | None = None) -> list[int]:
        ranking = rrf_fuse(self.bm25.rank(query), self.emb.rank(query))
        if not self.a.no_group_filter:
            ranking = group_first(ranking, query, self.groups)
        if self.reranker:
            ranking = self.reranker.rerank(query, ranking)
            if not self.a.no_group_filter:
                ranking = group_first(ranking, query, self.groups)
        if only_group:
            ranking = [i for i in ranking if self.groups[i] == only_group]
        return ranking

    def route(self, query: str) -> str:
        if not self.a.router:
            return "rag"
        if not GROUP_ID.search(query) and AGG_SUBJECT.search(query) and AGG_CUE.search(query):
            return self.a.aggregate_mode
        return "rag"

    # ---- strategies --------------------------------------------------------
    def answer_rag(self, query: str) -> dict:
        top = self.rank(query)[:self.a.top_k]
        followed = []
        if self.a.follow_refs:
            present = {self.groups[i] for i in top}
            named = set(GROUP_ID.findall(query))
            referenced = set().union(*(self.mentions[i] for i in top[:5])) - present - named
            for g in sorted(referenced & set(self.all_groups)):
                extra = [i for i in self.rank(query, only_group=g) if i not in top][:self.a.refs_per_group]
                followed += extra
        context = top + followed
        label = {f"C{n}": self.ids[i] for n, i in enumerate(context, 1)}
        excerpts = "\n\n".join(f"[C{n}] {self.texts[i]}" for n, i in enumerate(context, 1))
        out, secs = ollama_chat(self.a.ollama_url, self.a.llm_model, ANSWER_SYSTEM,
                                f"Excerpts:\n\n{excerpts}\n\nQuestion: {query}", self.a.num_ctx)
        cited = [label[c] for c in as_list(out.get("citations")) if c in label]
        return {"answer": str(out.get("answer", out.get("_raw", ""))).strip(),
                "found": out.get("found"), "cited": cited, "context": [self.ids[i] for i in context],
                "evidence": "\n\n".join(self.texts[self.idx[c]] for c in cited),
                "calls": 1, "seconds": secs,
                "detail": f"followed refs: {', '.join(short_ids(self, [self.ids[i] for i in followed]))}"
                          if followed else ""}

    def answer_per_group(self, query: str) -> dict:
        findings, cited_by_group, context, secs = [], {}, [], 0.0
        for g in self.all_groups:
            top = self.rank(query, only_group=g)[:self.a.per_group_k]
            context += [self.ids[i] for i in top]
            label = {f"C{n}": self.ids[i] for n, i in enumerate(top, 1)}
            excerpts = "\n\n".join(f"[C{n}] {self.texts[i]}" for n, i in enumerate(top, 1))
            out, s = ollama_chat(self.a.ollama_url, self.a.llm_model, MAP_SYSTEM.format(group=g),
                                 f"Excerpts about {g}:\n\n{excerpts}\n\nQuestion about all groups: {query}",
                                 self.a.num_ctx)
            secs += s
            finding = str(out.get("finding", out.get("_raw", ""))).strip() or "NOT STATED"
            if out.get("found") is False:
                finding = "NOT STATED"
            findings.append(f"{g}: {finding}")
            cited_by_group[g] = [label[c] for c in as_list(out.get("citations")) if c in label] \
                if finding != "NOT STATED" else []
        out, s = ollama_chat(self.a.ollama_url, self.a.llm_model, REDUCE_SYSTEM,
                             "Findings:\n" + "\n".join(findings) + f"\n\nQuestion: {query}", self.a.num_ctx)
        secs += s
        used = [g for g in as_list(out.get("groups_used")) if g in cited_by_group] or list(cited_by_group)
        cited = [c for g in used for c in cited_by_group[g]]
        return {"answer": str(out.get("answer", out.get("_raw", ""))).strip(),
                "found": out.get("found"), "cited": cited, "context": context,
                "evidence": "Per-group findings:\n" + "\n".join(findings) + "\n\nCited excerpts:\n\n"
                            + "\n\n".join(self.texts[self.idx[c]] for c in cited),
                "calls": len(self.all_groups) + 1, "seconds": secs,
                "detail": " | ".join(findings)}

    def answer_table(self, query: str) -> dict:
        header = list(next(iter(self.table_rows.values())).keys())
        table = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join(
            "| " + " | ".join(str(r[h]).replace("|", "/") for h in header) + " |"
            for r in self.table_rows.values())
        out, secs = ollama_chat(self.a.ollama_url, self.a.llm_model, TABLE_SYSTEM,
                                f"Table:\n\n{table}\n\nQuestion: {query}", self.a.num_ctx)
        used = [g for g in as_list(out.get("groups_used")) if g in self.table_rows]
        rows = [self.table_rows[g] for g in (used or self.table_rows)]
        evidence = "Table rows used:\n" + "\n".join(
            "; ".join(f"{h}: {r[h]}" for h in header) for r in rows)
        return {"answer": str(out.get("answer", out.get("_raw", ""))).strip(),
                "found": out.get("found"), "cited": [], "context": [], "evidence": evidence,
                "calls": 1, "seconds": secs, "detail": f"table rows used: {', '.join(used) or 'all'}"}

    def _load_table(self, path: Path) -> dict:
        if not path.exists():
            sys.exit(f"--aggregate-mode table needs {path} (the structured groups table).")
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "group" not in rows[0]:
            sys.exit(f"{path} must have a 'group' column.")
        return {r["group"]: {k: (v or "not stated") for k, v in r.items() if k not in TABLE_SKIP_COLS}
                for r in rows}


def short_ids(p: Pipeline, chunk_ids: list[str]) -> list[str]:
    return [SHORT[c] for c in chunk_ids]


SHORT: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Grade carry-over
# ---------------------------------------------------------------------------

def collect_grades() -> dict:
    """(qid, model_answer, gold_answer) -> human_grade from every answers_*.csv in the
    folder (including an earlier version of the file about to be written). Keying on the
    gold answer too means a corrected answer key never inherits a stale grade."""
    grades = {}
    for name in sorted(glob.glob("answers_*.csv")):
        with open(name, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("human_grade") or "").strip():
                    key = (r["qid"], r["model_answer"], r.get("gold_answer", ""))
                    grades.setdefault(key, r["human_grade"].strip())
    return grades


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, default=Path("chunks.jsonl"))
    ap.add_argument("--questions", type=Path, default=Path("eval_questions.csv"))
    ap.add_argument("--split-file", type=Path, default=Path("eval_split.csv"))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--types", help="only these question types, comma-separated (e.g. aggregate,multi_doc)")
    ap.add_argument("--limit", type=int, help="only run the first N questions (smoke test)")
    ap.add_argument("--top-k", type=int, default=10, help="chunks given to the LLM in plain RAG (default 10)")
    ap.add_argument("--no-group-filter", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--router", action="store_true", help="route cross-group questions to the aggregate path")
    ap.add_argument("--aggregate-mode", choices=["per_group", "table"], default="per_group")
    ap.add_argument("--per-group-k", type=int, default=3, help="chunks per group in per_group mode (default 3)")
    ap.add_argument("--table", type=Path, default=Path("groups.csv"))
    ap.add_argument("--follow-refs", action="store_true", help="also retrieve groups mentioned in retrieved chunks")
    ap.add_argument("--refs-per-group", type=int, default=2)
    ap.add_argument("--llm-model", default="qwen2.5:7b")
    ap.add_argument("--judge-model", help="model used for grading (default: same as --llm-model)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--tag", help="name for output files (default built from options)")
    ap.add_argument("--log", type=Path, default=Path("generation_log.csv"))
    args = ap.parse_args()
    judge_model = args.judge_model or args.llm_model
    use_gf = not args.no_group_filter

    safe = lambda m: m.replace(":", "-").replace("/", "-")
    tag = args.tag or "_".join(
        [args.split, safe(args.llm_model), f"k{args.top_k}"]
        + (["gf"] if use_gf else []) + (["rr"] if args.rerank else [])
        + ([f"router-{'pg' if args.aggregate_mode == 'per_group' else 'table'}"] if args.router else [])
        + (["refs"] if args.follow_refs else [])
        + ([f"judge-{safe(judge_model)}"] if judge_model != args.llm_model else [])
        + ([args.types.replace(",", "-")] if args.types else []))

    check_model(args.ollama_url, args.llm_model)
    if judge_model != args.llm_model:
        check_model(args.ollama_url, judge_model)

    chunks = load_chunks(args.chunks)
    SHORT.update({c["chunk_id"]: short_id(c) for c in chunks})
    all_q = load_questions(args.questions)
    split = load_or_create_split(args.split_file, all_q)
    questions = [q for q in all_q if args.split == "all" or split[q["qid"]] == args.split]
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        questions = [q for q in questions if q["type"] in wanted]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit("No questions selected.")
    unknown = sorted({g for q in questions for g in q["gold"] if g not in SHORT})
    if unknown:
        sys.exit(f"eval_questions.csv references chunks not in {args.chunks}: {unknown}")

    pipe = Pipeline(args, chunks)
    strategy = {"rag": pipe.answer_rag, "per_group": pipe.answer_per_group, "table": pipe.answer_table}

    rows = []
    print(f"\nAnswering {len(questions)} questions with {args.llm_model} (judge: {judge_model}); "
          f"router: {args.aggregate_mode if args.router else 'off'}; follow refs: "
          f"{'on' if args.follow_refs else 'off'} ...")
    for n, q in enumerate(questions, 1):
        query = q["question"]
        route = pipe.route(query)
        res = strategy[route](query)
        answer = res["answer"]
        said_not_found = (res["found"] is False) or bool(NOT_FOUND.search(answer) and len(answer) < 40)

        gold = set(q["gold"])
        is_unanswerable = q["type"] == "unanswerable"
        scorable_cits = gold and not is_unanswerable and route != "table"
        cited = res["cited"]
        cit_recall = len(gold & set(cited)) / len(gold) if scorable_cits else None
        cit_precision = len(gold & set(cited)) / len(cited) if scorable_cits and cited else None
        retrieved_gold = len(gold & set(res["context"])) / len(gold) if gold and route != "table" else None
        abstention_ok = said_not_found if is_unanswerable else not said_not_found

        judge_user = (f"Question: {query}\n\nReference answer: {q['gold_answer']}\n\n"
                      f"System answer: {answer}\n\nEvidence the system cited:\n\n"
                      f"{res['evidence'] or '(no evidence cited)'}")
        verdict, judge_s = ollama_chat(args.ollama_url, judge_model, JUDGE_SYSTEM, judge_user, args.num_ctx)
        correctness = str(verdict.get("correctness", "")).lower().strip()
        if correctness not in CORRECTNESS_SCORE:
            correctness = "unparsed"

        rows.append({
            "qid": q["qid"], "type": q["type"], "split": split[q["qid"]], "route": route,
            "question": query, "gold_answer": q["gold_answer"], "model_answer": answer,
            "said_not_found": said_not_found, "abstention_ok": abstention_ok,
            "judge_correctness": correctness, "judge_faithful": verdict.get("faithful"),
            "judge_reason": verdict.get("reason", verdict.get("_raw", "")),
            "human_grade": "",
            "citations": ";".join(SHORT[c] for c in cited),
            "gold_chunks": ";".join(SHORT[g] for g in q["gold"]),
            "gold_retrieved": retrieved_gold, "citation_recall": cit_recall,
            "citation_precision": cit_precision,
            "retrieved": ";".join(SHORT[c] for c in res["context"]),
            "llm_calls": res["calls"], "answer_seconds": round(res["seconds"], 1),
            "judge_seconds": round(judge_s, 1), "detail": res["detail"],
        })
        mark = {"correct": "✓", "partial": "~", "incorrect": "✗"}.get(correctness, "?")
        print(f"  [{n}/{len(questions)}] {q['qid']:4} {route:9} {mark} {res['seconds']:5.1f}s  {answer[:70]}")

    # ---- carry over human grades from any earlier file with the same answer ----
    out_csv, out_md = Path(f"answers_{tag}.csv"), Path(f"answers_{tag}.md")
    grades = collect_grades()
    carried = 0
    for r in rows:
        g = grades.get((r["qid"], r["model_answer"], r["gold_answer"]))
        if g:
            r["human_grade"] = g
            carried += 1
    need = [r["qid"] for r in rows if not r["human_grade"]]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary -------------------------------------------------------------
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v, pct=True):
        return "–" if v is None else (f"{100 * v:.0f}%" if pct else f"{v:.1f}")

    def summarize(sel):
        return {
            "n": len(sel),
            "correctness": avg([CORRECTNESS_SCORE.get(r["judge_correctness"]) for r in sel]),
            "faithful": avg([1.0 if r["judge_faithful"] is True else 0.0 if r["judge_faithful"] is False else None
                             for r in sel]),
            "abstention": avg([1.0 if r["abstention_ok"] else 0.0 for r in sel]),
            "gold_retrieved": avg([r["gold_retrieved"] for r in sel]),
            "cit_recall": avg([r["citation_recall"] for r in sel]),
            "cit_precision": avg([r["citation_precision"] for r in sel]),
            "seconds": avg([r["answer_seconds"] for r in sel]),
        }

    types = sorted({r["type"] for r in rows})
    overall = summarize(rows)
    lines = [
        f"# Answer evaluation: {tag}",
        "",
        f"LLM: `{args.llm_model}` | judge: `{judge_model}` | top-k: {args.top_k} | "
        f"group filter: {'on' if use_gf else 'off'} | rerank: {'on' if args.rerank else 'off'} | "
        f"router: {args.aggregate_mode if args.router else 'off'} | follow refs: "
        f"{'on' if args.follow_refs else 'off'} | split: {args.split} | questions: {len(rows)}",
        "",
        "- **correct**: judge score vs your gold answer (correct 1, partial 0.5, incorrect 0). "
        "Judge is unreliable on list answers: use judge_agreement.py with your grades.",
        "- **faithful**: judge says every claim is supported by the cited evidence",
        "- **abstain ok**: said NOT FOUND on unanswerable questions, and didn't on answerable ones",
        "- **gold in context**: share of gold chunks the model saw (– for table answers)",
        "- **cit. recall / precision**: gold chunks cited / cited chunks that were gold",
        "",
        "| type | n | correct | faithful | abstain ok | gold in context | cit. recall | cit. precision | sec/answer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, s in [(t, summarize([r for r in rows if r["type"] == t])) for t in types] + [("**ALL**", overall)]:
        lines.append(f"| {t} | {s['n']} | {fmt(s['correctness'])} | {fmt(s['faithful'])} | "
                     f"{fmt(s['abstention'])} | {fmt(s['gold_retrieved'])} | {fmt(s['cit_recall'])} | "
                     f"{fmt(s['cit_precision'])} | {fmt(s['seconds'], pct=False)} |")
    if args.router:
        lines += ["", "**Routing** (questions sent to the aggregate path, by type):", ""]
        for t in types:
            sel = [r for r in rows if r["type"] == t]
            sent = [r["qid"] for r in sel if r["route"] != "rag"]
            lines.append(f"- {t}: {len(sent)}/{len(sel)}" + (f" ({', '.join(sent)})" if sent else ""))
        missed = [r["qid"] for r in rows if r["type"] == "aggregate" and r["route"] == "rag"]
        if missed:
            lines.append(f"- aggregate questions the router missed: {', '.join(missed)}")
    lines += ["", f"**Human grades:** carried over {carried} from earlier files (identical answers); "
              f"{len(need)} still need grading" + (f": {', '.join(need)}" if need else "") + "."]
    wrong = [r for r in rows if r["judge_correctness"] != "correct"]
    lines += ["", "## Not fully correct per the judge (check these by hand)", ""]
    if wrong:
        cell = lambda s: str(s).replace("|", "/").replace("\n", " ")
        lines += ["| qid | route | grade | question | model answer | gold answer | judge reason |",
                  "|---|---|---|---|---|---|---|"]
        for r in wrong:
            lines.append(f"| {r['qid']} | {r['route']} | {r['judge_correctness']} | {cell(r['question'])} | "
                         f"{cell(r['model_answer'])} | {cell(r['gold_answer'])} | {cell(r['judge_reason'])} |")
    else:
        lines.append("None.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    new_log = not args.log.exists()
    with args.log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp", "tag", "split", "llm_model", "judge_model", "top_k", "group_filter",
                        "rerank", "router", "follow_refs", "n", "correctness", "faithful", "abstention_ok",
                        "citation_recall", "citation_precision", "sec_per_answer"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), tag, args.split, args.llm_model,
                    judge_model, args.top_k, use_gf, args.rerank,
                    args.aggregate_mode if args.router else "off", args.follow_refs, overall["n"]]
                   + [("" if overall[k] is None else f"{overall[k]:.3f}")
                      for k in ("correctness", "faithful", "abstention", "cit_recall", "cit_precision", "seconds")])

    print()
    print("\n".join(lines[:lines.index("## Not fully correct per the judge (check these by hand)")]))
    print(f"\nWrote {out_csv}, {out_md}; appended to {args.log}")


if __name__ == "__main__":
    main()
