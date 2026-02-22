#!/usr/bin/env python3
"""
Build a lightweight BM25 index for life-common corpus.

Input format supports:
  - {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], ...}
  - {"text": "..."}

Output:
  rag/life_common_it_bm25.json
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_docs(path: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if isinstance(obj, dict) and "messages" in obj:
                user = ""
                assistant = ""
                for m in obj["messages"]:
                    role = str(m.get("role", "")).lower()
                    content = str(m.get("content", ""))
                    if role == "user":
                        user = content
                    elif role == "assistant":
                        assistant = content
                text = f"Domanda: {user}\nRisposta: {assistant}".strip()
                category = str(obj.get("meta", {}).get("category", "unknown"))
            else:
                text = str(obj.get("text", "")).strip()
                category = str(obj.get("category", "unknown"))

            if text:
                key = text.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                docs.append({"text": text, "category": category})
    return docs


def build_bm25(docs: List[Dict[str, str]]) -> Dict[str, object]:
    k1 = 1.5
    b = 0.75

    term_postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    doc_lens: List[int] = []
    doc_freq: Dict[str, int] = {}

    for i, doc in enumerate(docs):
        tokens = tokenize(doc["text"])
        doc_lens.append(len(tokens))
        tf = Counter(tokens)
        for term, freq in tf.items():
            term_postings[term].append((i, freq))
        for term in tf.keys():
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_docs = len(docs)
    avg_dl = (sum(doc_lens) / n_docs) if n_docs else 0.0

    postings_json = {
        term: [[doc_id, freq] for doc_id, freq in plist]
        for term, plist in term_postings.items()
    }

    return {
        "meta": {
            "engine": "bm25",
            "k1": k1,
            "b": b,
            "n_docs": n_docs,
            "avg_doc_len": avg_dl,
        },
        "docs": docs,
        "doc_lens": doc_lens,
        "doc_freq": doc_freq,
        "postings": postings_json,
    }


def save_index(index: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=True)


def load_index(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bm25_search(index: Dict[str, object], query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    meta = index["meta"]
    docs = index["docs"]
    doc_lens = index["doc_lens"]
    doc_freq = index["doc_freq"]
    postings = index["postings"]

    k1 = float(meta["k1"])
    b = float(meta["b"])
    n_docs = int(meta["n_docs"])
    avg_dl = float(meta["avg_doc_len"]) if float(meta["avg_doc_len"]) > 0 else 1.0

    scores: Dict[int, float] = defaultdict(float)
    q_terms = tokenize(query)

    for term in q_terms:
        plist = postings.get(term, [])
        if not plist:
            continue
        df = int(doc_freq.get(term, 0))
        idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5))) if df > 0 else 0.0

        for doc_id, tf in plist:
            dl = float(doc_lens[doc_id])
            denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
            score = idf * ((tf * (k1 + 1.0)) / denom) if denom != 0 else 0.0
            scores[int(doc_id)] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[: max(1, min(top_k, len(docs)))]


def print_hits(index: Dict[str, object], hits: List[Tuple[int, float]]) -> None:
    docs = index["docs"]
    for rank, (doc_id, score) in enumerate(hits, start=1):
        doc = docs[doc_id]
        text = doc["text"].replace("\n", " ")
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        print(f"[{rank}] score={score:.4f} category={doc.get('category','unknown')}")
        print(f"     {snippet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/search BM25 index for life_common dataset.")
    parser.add_argument("--input", default="datasets/life_common_it.jsonl", help="Corpus JSONL path.")
    parser.add_argument(
        "--index-out",
        default="rag/life_common_it_bm25.json",
        help="Output BM25 index path.",
    )
    parser.add_argument("--query", default=None, help="Optional one-shot query.")
    parser.add_argument("--top-k", type=int, default=5, help="Top results to return.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building and only search existing index.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive query mode after build/load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    index_path = Path(args.index_out)

    if not args.skip_build:
        docs = load_docs(input_path)
        index = build_bm25(docs)
        save_index(index, index_path)
        print(f"Index built: {index_path}")
        print(f"Docs: {index['meta']['n_docs']}, avg_len: {index['meta']['avg_doc_len']:.2f}")
    else:
        index = load_index(index_path)
        print(f"Index loaded: {index_path}")

    if args.query:
        hits = bm25_search(index, args.query, top_k=args.top_k)
        print(f"Query: {args.query}")
        print_hits(index, hits)

    if args.interactive:
        print("Interactive mode. Type 'exit' to stop.")
        while True:
            q = input("query> ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            hits = bm25_search(index, q, top_k=args.top_k)
            print_hits(index, hits)


if __name__ == "__main__":
    main()
