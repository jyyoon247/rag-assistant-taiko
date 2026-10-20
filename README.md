# RAG Research Assistant — Taiko Edition

A fully local, privacy-preserving retrieval-augmented generation (RAG) assistant over
interview research data, with a rigorous evaluation harness. Everything runs on a laptop
(built and tested on a MacBook Pro M3 Pro, 36 GB): no document text leaves the machine.

Write-up: **[Creating a Personal RAG Research Assistant — Taiko Edition](https://www.joshyoon.com/creating-a-personal-rag-research-assistant-taiko-edition/)**

> **About the data.** The system was built for a private corpus of anonymized email interviews
> with collegiate taiko groups. That corpus is research data and is **not** included. This repo
> ships with a **fully fictional synthetic corpus** in the same format (groups G101–G108), so
> every step runs end to end. Results on the synthetic data will differ from those reported below.

## What it does

```
.docx interviews ──► ingest.py ──► Q&A chunks (+ metadata, context headers, leak checks)
                                        │
question ──► router ─┬─ single group / narrative ──► hybrid search (BM25 + embeddings, RRF)
                     │                                + group filter [+ reranker] [+ follow refs]
                     │                                        └──► local LLM answers with citations
                     └─ cross-group ("which groups…", "how many…")
                            ├─ per-group: retrieve each group → extract its finding → combine
                            └─ table: answer from a structured per-group table
                                        │
                     answer ──► LLM judge + your human grade ──► metrics, agreement, paired tests
```

- **Retrieval:** BM25 keyword search, `bge-small-en-v1.5` embeddings, hybrid via reciprocal rank
  fusion, metadata filtering by group, optional `bge-reranker-base` cross-encoder.
- **Generation:** Qwen2.5-7B-Instruct via [Ollama](https://ollama.com), JSON-mode structured
  output, temperature 0, citations to source chunks, explicit "NOT FOUND" abstention.
- **Routing:** cross-group questions go to a per-group map-reduce path (or a structured table);
  optional reference-following when one group's interview mentions another.
- **Evaluation:** recall@k and MRR for retrieval; correctness, faithfulness, abstention and
  citation precision/recall for answers; an LLM judge validated against human grades
  (agreement, Cohen's kappa); bootstrap confidence intervals; paired comparisons with a
  sign test; stratified dev/test split.

No LangChain or LlamaIndex: the pipeline is plain Python so every step is inspectable.

## Results (private corpus)

5 interviews → 51 chunks; 66 evaluation questions across 6 types (factual, narrative,
aggregate, multi-document, unanswerable, privacy), split 2/3 dev, 1/3 test. Accuracy uses
**human grades** (correct 1, partial 0.5, incorrect 0).

**Retrieval** (hybrid search):

| Setup | Dev recall@5 | Dev MRR | Test recall@5 | Test MRR |
|---|---|---|---|---|
| Hybrid baseline | 0.83 | 0.77 | 0.96 | 0.89 |
| + group filter | 0.86 | 0.86 | 0.96 | 0.96 |
| + group filter + reranker | 0.92 | 0.89 | 0.97 | 0.90 |

The group filter improved ranking on both splits; the reranker helped on dev but not on test
(at ~50× the search time), so the final setup uses hybrid + group filter.

**Answers, held-out test set (32 questions):**

| Setup | Aggregate (12) | Multi-doc (7) | Overall (32) |
|---|---|---|---|
| Plain RAG | 42% | 71% | 70% (95% CI 56–84%) |
| + router, per-group | 54% | 86% | 78% (64–91%) |
| + router, per-group + references | 54% | 86% | 78% (64–91%) |
| + router, table | 50% | 86% | 77% (62–89%) |

Factual, narrative and unanswerable questions scored 94–100%; cross-group aggregate questions
were the weak spot. Per-group routing improved accuracy on both dev (+3 points) and test
(+8 points); across all 66 questions it won on 6, lost on 3 and tied on 57 (sign test
p ≈ 0.5), so the gain is consistent but not yet statistically conclusive. The LLM judge
agreed with human grades 91–94% of the time on dev but only 75–81% on the harder test set,
mostly by over-crediting incomplete lists.

## Quickstart (synthetic data)

Requirements: Python 3.9+, [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/jyyoon247/rag-assistant-taiko.git
cd rag-assistant-taiko
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b

# 1. (optional) regenerate the synthetic corpus; it is already included
python make_synthetic.py

# 2. run the pipeline from the data folder
cd data/synthetic
python ../../ingest.py docs                                   # -> chunks.jsonl, ingest_report.md
python ../../retrieve_eval.py --split dev --group-filter      # retrieval metrics
python ../../generate_eval.py --split dev                     # plain RAG answers + judge
python ../../generate_eval.py --split dev --router            # per-group routing
python ../../generate_eval.py --split dev --router --aggregate-mode table
```

Optional human grading: fill in the `human_grade` column of an `answers_*.csv`
(correct / partial / incorrect), then:

```bash
python ../../judge_agreement.py answers_dev_qwen2.5-7b_k10_gf.csv
python ../../compare_runs.py answers_dev_qwen2.5-7b_k10_gf.csv answers_dev_qwen2.5-7b_k10_gf_router-pg.csv
```

Grades carry over automatically to later runs wherever the answer (and gold answer) is unchanged.
Models download once from Hugging Face / Ollama and are cached; after that everything runs offline.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Parse .docx interviews (header + `Q:`/`A:` pairs, lists, tables) into JSONL chunks; validate metadata; scan for leaks (`--names` takes a private list of real names) |
| `retrieve_eval.py` | BM25 / embedding / hybrid retrieval, group filter, reranker; recall@k, MRR; dev/test split |
| `generate_eval.py` | Answer generation via Ollama; router, per-group, table and reference-following strategies; LLM judge; summaries and logs |
| `judge_agreement.py` | Human vs. judge agreement, Cohen's kappa, bootstrap CIs, disagreement report |
| `compare_runs.py` | Paired comparison of two runs: wins/losses/ties, paired bootstrap interval, sign test |
| `make_synthetic.py` | Generates the fictional corpus, structured table, eval questions (gold answers computed from the same facts) and split |
| `data/synthetic/` | The generated synthetic corpus |

## Privacy design

- Real names of people, groups and schools are replaced with consistent pseudonyms
  (P###, G###, S###); the mapping is kept separately and never indexed.
- `ingest.py --names private_names.txt` flags any remaining real name, alias, URL, email or
  phone number before indexing.
- Embedding, reranking, generation and judging all run locally; no interview text is sent
  to an external API.

## License

Code: [MIT](LICENSE). The synthetic data is fictional and released under the same license.
Models are downloaded from their original sources and are subject to their own licenses.
