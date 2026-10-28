"""
Measure retrieval quality for each search mode (vector, keyword, hybrid).
Only the search runs, not the chat model, so this is fast.

Usage:
    python eval.py
    python eval.py eval/questions.json --k 4 --show-misses

A retrieved chunk counts as relevant when its text contains one of the question's "expect" terms.
This is an approximation: it checks that the search finds the right topic, not that the answer is correct.

Metrics:
    Hit@k      share of questions with at least one relevant chunk in the top k
    MRR        mean of 1 / (rank of the first relevant chunk); 0 when there is none
    Precision  share of the top-k chunks that are relevant
"""
import argparse
import json
import time
from pathlib import Path

from retrieval import MODES, Retriever


def is_relevant(text: str, expect: list) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in expect)


def evaluate(retriever: Retriever, questions: list, mode: str, k: int) -> dict:
    hits = rr = precision = 0.0
    misses = []
    start = time.perf_counter()
    for q in questions:
        results = retriever.search(q["question"], q.get("component"), mode=mode, k=k)
        flags = [is_relevant(r["text"], q["expect"]) for r in results]
        if any(flags):
            hits += 1
            rr += 1 / (flags.index(True) + 1)
        else:
            misses.append(q["question"])
        precision += sum(flags) / k
    n = len(questions)
    return {
        "mode": mode,
        "hit": hits / n,
        "mrr": rr / n,
        "precision": precision / n,
        "seconds": (time.perf_counter() - start) / n,
        "misses": misses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="?", default="eval/questions.json")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    retriever = Retriever()
    if retriever.count() == 0:
        print("No documents are loaded. Run ingest.py first.")
        return

    print(f"{len(questions)} questions, top {args.k} chunks, {retriever.count()} chunks in the database\n")
    print(f"{'Mode':<8} {'Hit@' + str(args.k):>7} {'MRR':>7} {'Precision':>10} {'s/query':>8}")
    for mode in MODES:
        r =evaluate(retriever, questions, mode, args.k)
        print(f"{mode:<8} {r['hit']:>7.0%} {r['mrr']:>7.2f} {r['precision']:>10.0%} {r['seconds']:>8.2f}")
        if args.show_misses and r["misses"]:
            for q in r["misses"]:
                print(f"         miss: {q}")


if __name__ == "__main__":
    main()
