"""Command-line interface for the agent.

Examples:

    llm-agent "What is the GDP of Texas divided by its population?"
    llm-agent --repl
    llm-agent --query "Plot sin(x) from 0 to 2*pi" --save-figures ./out
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import Agent
from .tracing import Trace

DEFAULT_TRACE_FILE = Path("traces") / "runs.jsonl"

def _print_trace(trace: Trace, console: Console) -> None:
    table = Table(title=f"Reasoning trace ({trace.run_id})", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("kind", style="magenta")
    table.add_column("detail")
    table.add_column("ms", justify="right", style="yellow")

    for i, ev in enumerate(trace.events):
        if ev.kind == "user":
            detail = ev.data.get("text", "")[:80]
        elif ev.kind == "model":
            detail = f"stop={ev.data.get('stop_reason')} (tokens in/out: "
            detail += f"{ev.data.get('input_tokens', 0)}/{ev.data.get('output_tokens', 0)})"
        elif ev.kind == "tool_use":
            args_preview = str(ev.data.get("input", {}))[:60]
            detail = f"{ev.data.get('name')} <- {args_preview}"
        elif ev.kind == "tool_result":
            r = ev.data.get("result", {})
            detail = f"{ev.data.get('tool_name')} -> ok={r.get('ok')}"
        elif ev.kind == "final":
            detail = ev.data.get("text", "")[:80]
        else:
            detail = str(ev.data)[:80]
        table.add_row(str(i), ev.kind, detail, f"{ev.elapsed_ms:.0f}")

    console.print(table)

def _save_figures(trace: Trace, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ev in trace.events:
        if ev.kind != "tool_result":
            continue
        result = ev.data.get("result", {})
        for fig in result.get("figures", []) or []:
            path = out_dir / f"{trace.run_id}_{fig['filename']}"
            path.write_bytes(base64.b64decode(fig["image_base64"]))
            n += 1
    return n

def _run_query(agent: Agent, query: str, console: Console,
               trace_path: Path, save_figures: Path | None) -> Trace:
    console.print(Panel.fit(query, title="Query", border_style="cyan"))
    with console.status("Thinking..."):
        trace = agent.run(query)
    console.print(Panel.fit(trace.final_answer or "(no answer)",
                            title="Answer", border_style="green"))
    _print_trace(trace, console)
    trace.append_jsonl(trace_path)
    if save_figures is not None:
        n = _save_figures(trace, save_figures)
        if n:
            console.print(f"[dim]Saved {n} figure(s) to {save_figures}[/dim]")
    duration = (trace.ended_at or 0) - trace.started_at
    console.print(
        f"[dim]Run {trace.run_id} | {len(trace.tool_calls())} tool call(s) | "
        f"{trace.total_input_tokens} in / {trace.total_output_tokens} out tokens | "
        f"{duration:.2f}s total[/dim]"
    )
    return trace

def _repl(agent: Agent, console: Console, trace_path: Path,
          save_figures: Path | None) -> None:
    console.print("[bold]llm-agent REPL[/bold] -- type 'exit' or Ctrl-D to quit")
    while True:
        try:
            query = console.input("[bold cyan]>>> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return
        try:
            _run_query(agent, query, console, trace_path, save_figures)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")

def main(argv: list[str] | None = None) -> int:
    # Load environment variables
    load_dotenv()
    # Build a parser for the arguments
    # Five flags: positional query, --query, --repl, --trace-file, --save-figures, --max-steps.
    parser = argparse.ArgumentParser(prog="llm-agent",
                                     description="Multi-tool LLM agent")
    parser.add_argument("query", nargs="?", help="Single user query to run")
    parser.add_argument("--query", dest="query_opt", help="Alternate way to pass a query")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")
    parser.add_argument("--trace-file", default=str(DEFAULT_TRACE_FILE),
                        help=f"Where to append run traces (default: {DEFAULT_TRACE_FILE})")
    parser.add_argument("--save-figures", metavar="DIR",
                        help="Directory to write any matplotlib figures to as PNG")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override agent max_steps")
    args = parser.parse_args(argv)

    # Check if the Anthropic API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write("ANTHROPIC_API_KEY is not set; copy .env.example to .env\n")
        return 2

    console = Console()
    agent_kwargs: dict = {}
    if args.max_steps is not None:
        agent_kwargs["max_steps"] = args.max_steps
    agent = Agent(**agent_kwargs)
    trace_path = Path(args.trace_file)
    save_figures = Path(args.save_figures) if args.save_figures else None

    query = args.query or args.query_opt
    if args.repl or not query:
        _repl(agent, console, trace_path, save_figures)
    else:
        _run_query(agent, query, console, trace_path, save_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())