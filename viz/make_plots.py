"""Generate the three required visualisations from an eval traces file.

Usage:

    python viz/make_plots.py eval/results/<timestamp>/traces.jsonl

Outputs three PNGs into ``viz/output/``:

    1. tool_usage_bar.png   - frequency of each tool across all queries
    2. tool_latency_box.png - latency distribution per tool (boxplot)
    3. sample_code_plot.png - a representative figure produced by the
                              code execution tool (decoded from base64)
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"


def _load(traces_path: Path) -> list[dict]:
    out = []
    with traces_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def plot_tool_usage(traces: list[dict], out_path: Path) -> None:
    counts: Counter[str] = Counter()
    for t in traces:
        for ev in t["events"]:
            if ev["kind"] == "tool_use":
                counts[ev["data"]["name"]] += 1

    if not counts:
        print("No tool_use events found; skipping tool_usage_bar.")
        return

    names = sorted(counts.keys())
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color=["#4c72b0", "#dd8452", "#55a868"][: len(names)])
    ax.set_title("Tool usage frequency")
    ax.set_ylabel("Number of calls")
    ax.set_xlabel("Tool")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05 * max(values), str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_tool_latency(traces: list[dict], out_path: Path) -> None:
    by_tool: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        for ev in t["events"]:
            if ev["kind"] == "tool_result":
                by_tool[ev["data"]["tool_name"]].append(ev["elapsed_ms"])

    if not by_tool:
        print("No tool_result events found; skipping tool_latency_box.")
        return

    names = sorted(by_tool.keys())
    data = [by_tool[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=names, showmeans=True)
    ax.set_title("Per-tool latency")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def extract_sample_code_plot(traces: list[dict], out_path: Path) -> None:
    for t in traces:
        for ev in t["events"]:
            if ev["kind"] != "tool_result":
                continue
            if ev["data"].get("tool_name") != "code_exec_tool":
                continue
            figs = ev["data"].get("result", {}).get("figures", []) or []
            for fig in figs:
                b64 = fig.get("image_base64")
                if isinstance(b64, str) and b64:
                    out_path.write_bytes(base64.b64decode(b64))
                    print(f"Wrote {out_path} (from query: {t['query'][:60]!r})")
                    return
    print("No matplotlib figure found in any code_exec_tool result.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", help="Path to traces.jsonl from an eval run")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = _load(Path(args.traces))
    print(f"Loaded {len(traces)} traces.")

    plot_tool_usage(traces, out_dir / "tool_usage_bar.png")
    plot_tool_latency(traces, out_dir / "tool_latency_box.png")
    extract_sample_code_plot(traces, out_dir / "sample_code_plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
