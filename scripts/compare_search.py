"""
Measure retrieval accuracy and speed of the document index — run once per index branch
and compare the JSON outputs. Retrieval only (no LLM calls); each case makes one
embedding call.

Cases are generated from the live index (no labelling needed):
    full          exact number as stored                 expect: exact, doc ranked 1st
    loose         lower-case, spaces instead of hyphens  expect: exact, doc ranked 1st
    last_segment  only the trailing number ("4110")      expect: partial, doc in top 5
    title         title only, no number                  expect: doc in top 5 (title search)
    missing       same series, number that doesn't exist expect: nearest, never exact
plus optional --cases CSV with columns: query,expected_document_number

Usage (repo root, same env vars as the Function, e.g. in .env):
    python scripts/compare_search.py --sample 50 --out compare_<branch>.json
"""
import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from RAG.config import Config  # noqa: E402
from RAG.services.rag_service import RAGService  # noqa: E402
from RAG.utils.analytics import EVENT_PREFIX  # noqa: E402


class _EventCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith(EVENT_PREFIX):
            self.events.append(json.loads(msg[len(EVENT_PREFIX):]))


def _ms(start):
    return round((time.perf_counter() - start) * 1000)


def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def generate_cases(index, sample, seed):
    entries = list(index.entries)
    random.Random(seed).shuffle(entries)
    existing = {e.norm for e in index.entries}
    cases = []
    for e in entries[:sample]:
        num = e.document_number
        cases.append({"type": "full", "query": num, "expected": num, "expect_match": "exact"})
        cases.append({"type": "loose", "query": num.lower().replace("-", " "), "expected": num, "expect_match": "exact"})
        m = re.search(r"(\d+)$", num)
        if m:
            cases.append({"type": "last_segment", "query": m.group(1), "expected": num, "expect_match": "partial"})
            width = len(m.group(1))
            for k in range(1, 50):
                candidate = num[:m.start()] + str(int(m.group(1)) + k).zfill(width)
                if re.sub(r"[^A-Z0-9]", "", candidate.upper()) not in existing:
                    cases.append({"type": "missing", "query": candidate, "expected": None, "expect_match": "nearest"})
                    break
        if len(e.title) >= 4:
            cases.append({"type": "title", "query": e.title, "expected": num, "expect_match": None})
    return cases


def load_csv_cases(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{"type": "custom", "query": r["query"], "expected": r.get("expected_document_number") or None,
                 "expect_match": None} for r in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description="Compare document index retrieval")
    parser.add_argument("--sample", type=int, default=50, help="Documents to generate cases from (default 50)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (keep the same across branches)")
    parser.add_argument("--cases", type=str, default=None, help="Optional CSV: query,expected_document_number")
    parser.add_argument("--out", type=str, default="compare_search.json", help="Output JSON path")
    args = parser.parse_args()

    collector = _EventCollector()
    logging.basicConfig(level=logging.WARNING)
    analytics_logger = logging.getLogger("rag.analytics")
    analytics_logger.setLevel(logging.INFO)
    analytics_logger.addHandler(collector)

    config = Config()

    # Cold start: service construction + first index load
    t = time.perf_counter()
    service = RAGService(config)
    init_ms = _ms(t)
    index = service.document_index
    t = time.perf_counter()
    index.ensure_fresh()
    cold_load_ms = _ms(t)

    # Refresh (what happens every index_refresh_s)
    index._last_check = time.time() - config.index_refresh_s - 1
    t = time.perf_counter()
    index.ensure_fresh()
    refresh_ms = _ms(t)
    index_events = [e for e in collector.events if e.get("event") == "index_load"]

    # Drift: active documents in the main collection vs entries in the index
    main_docs = service.chromadb_client.collection.get(
        where={"$and": [{"active": True}, {"chunk_index": 0}]}, include=["metadatas"]
    )
    main_nums = {m.get("document_number") for m in main_docs["metadatas"] if m.get("document_number")}
    index_nums = {e.document_number for e in index.entries}

    cases = generate_cases(index, args.sample, args.seed)
    if args.cases:
        cases += load_csv_cases(args.cases)

    results = []
    for case in cases:
        event, timings = {}, {}
        t = time.perf_counter()
        r = service.retrieve(case["query"], event, timings)
        ms = _ms(t)
        ranked = list(dict.fromkeys(d["metadata"].get("document_number") for d in r["docs"]))
        match_type = r["match"].match_type
        expected = case["expected"]
        results.append({
            **case,
            "ms": ms,
            "timings_ms": timings,
            "match_type": match_type,
            "match_ok": case["expect_match"] is None or match_type == case["expect_match"],
            "false_exact": case["type"] == "missing" and match_type == "exact",
            "fast_path": service._is_lookup(case["query"], r["match"]),
            "hit1": expected is not None and ranked[:1] == [expected],
            "hit5": expected is not None and expected in ranked[:5],
            "rank": ranked.index(expected) + 1 if expected in ranked else None,
            "top5": ranked[:5],
            "title_top_score": event.get("title_top_score"),
        })

    summary = {}
    for ctype in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == ctype]
        with_expected = [r for r in rows if r["expected"]]
        timing_keys = sorted({k for r in rows for k in r["timings_ms"]})
        summary[ctype] = {
            "n": len(rows),
            "hit@1": round(sum(r["hit1"] for r in with_expected) / len(with_expected), 3) if with_expected else None,
            "hit@5": round(sum(r["hit5"] for r in with_expected) / len(with_expected), 3) if with_expected else None,
            "match_ok": round(sum(r["match_ok"] for r in rows) / len(rows), 3),
            "false_exact": sum(r["false_exact"] for r in rows),
            "fast_path": sum(r["fast_path"] for r in rows),
            "match_types": {mt: sum(r["match_type"] == mt for r in rows) for mt in ("exact", "partial", "nearest", "none")},
            "ms_p50": _pct([r["ms"] for r in rows], 50),
            "ms_p95": _pct([r["ms"] for r in rows], 95),
            "timings_p50_ms": {k: _pct([r["timings_ms"][k] for r in rows if k in r["timings_ms"]], 50) for k in timing_keys},
        }

    report = {
        "index_source": index.source.name,
        "index_entries": len(index.entries),
        "init_ms": init_ms,
        "cold_load_ms": cold_load_ms,
        "refresh_ms": refresh_ms,
        "index_load_events": index_events,
        "drift": {
            "main_active_documents": len(main_nums),
            "missing_from_index": sorted(main_nums - index_nums),
            "extra_in_index": sorted(index_nums - main_nums),
        },
        "summary": summary,
        "cases": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Index source: {report['index_source']}  entries: {report['index_entries']}  "
          f"cold load: {cold_load_ms} ms  refresh: {refresh_ms} ms  init: {init_ms} ms")
    print(f"Drift: {len(report['drift']['missing_from_index'])} missing from index, "
          f"{len(report['drift']['extra_in_index'])} extra (main has {len(main_nums)} active docs)")
    print(f"\n{'case':<14}{'n':>5}{'hit@1':>8}{'hit@5':>8}{'match ok':>10}{'false exact':>13}{'fast':>6}{'p50 ms':>8}{'p95 ms':>8}")
    for ctype, s in summary.items():
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"{ctype:<14}{s['n']:>5}{fmt(s['hit@1']):>8}{fmt(s['hit@5']):>8}{s['match_ok']:>10.2f}"
              f"{s['false_exact']:>13}{s['fast_path']:>6}{s['ms_p50']:>8}{s['ms_p95']:>8}")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
