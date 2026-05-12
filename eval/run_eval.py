"""Run the agent against ``queries.jsonl`` and emit a metrics summary.

Usage:

    python eval/run_eval.py                     # run all 40 queries
    python eval/run_eval.py --category math     # only the math queries
    python eval/run_eval.py --limit 5           # smoke test
    python eval/run_eval.py --dry-run           # parse only, no API calls

Outputs:
    eval/results/<timestamp>/traces.jsonl       (one line per query)
    eval/results/<timestamp>/summary.json       (aggregate metrics)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from research_agent.agent import Agent  # noqa: E402

QUERIES_PATH = ROOT / "eval" / "queries.jsonl"


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _summarise(traces: list[dict]) -> dict:
    tool_usage: Counter[str] = Counter()
    per_tool_latency: dict[str, list[float]] = defaultdict(list)
    durations: list[float] = []
    success = 0
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0})

    for t in traces:
        cat = t["query_meta"].get("category", "unknown")
        by_category[cat]["n"] += 1
        if t["final_answer"] and not t["final_answer"].startswith("[error]"):
            success += 1
            by_category[cat]["ok"] += 1
        durations.append(t["duration_sec"])
        for ev in t["events"]:
            if ev["kind"] == "tool_use":
                tool_usage[ev["data"]["name"]] += 1
            elif ev["kind"] == "tool_result":
                per_tool_latency[ev["data"]["tool_name"]].append(ev["elapsed_ms"])

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"count": 0}
        xs_sorted = sorted(xs)
        return {
            "count": len(xs),
            "mean": sum(xs) / len(xs),
            "median": xs_sorted[len(xs) // 2],
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
        }

    return {
        "n_queries": len(traces),
        "n_success": success,
        "success_rate": success / len(traces) if traces else 0.0,
        "duration_sec": _stats(durations),
        "tool_usage": dict(tool_usage),
        "tool_latency_ms": {k: _stats(v) for k, v in per_tool_latency.items()},
        "by_category": dict(by_category),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["factual", "math", "code", "mixed"],
                        help="Restrict to one category")
    parser.add_argument("--limit", type=int, help="Cap number of queries (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call the LLM; just validate the file")
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    queries = _load_queries(Path(args.queries))
    if args.category:
        queries = [q for q in queries if q.get("category") == args.category]
    if args.limit:
        queries = queries[: args.limit]

    print(f"Loaded {len(queries)} queries.")
    if args.dry_run:
        return 0

    out_dir = ROOT / "eval" / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"

    agent = Agent()
    traces: list[dict] = []
    for i, q in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {q['id']} ({q['category']}): {q['query']}")
        t0 = time.time()
        trace = agent.run(q["query"])
        elapsed = time.time() - t0
        record = trace.to_dict()
        record["query_meta"] = q
        traces.append(record)
        with traces_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        n_tools = len(trace.tool_calls())
        print(f"    -> {n_tools} tool call(s), {elapsed:.1f}s")

    summary = _summarise(traces)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote results to {out_dir}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
