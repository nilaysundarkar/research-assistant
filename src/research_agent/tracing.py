"""Structured tracing for an agent run.

Every interesting event during an agent run (model call, tool call,
tool result, final answer) is recorded as a dict in a ``Trace`` object.
At the end of a run the trace can be serialised to JSON or appended to
a JSONL log file. The eval harness and the visualisation script read
these traces to compute tool-usage stats and per-tool latency.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """A single event in the agent's reasoning trace."""

    kind: str  # "user", "model", "tool_use", "tool_result", "final", "error"
    data: dict[str, Any]
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trace:
    """Collected events plus aggregate stats for a single user query."""

    query: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[TraceEvent] = field(default_factory=list)
    final_answer: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def add(self, kind: str, data: dict[str, Any], elapsed_ms: float = 0.0) -> None:
        self.events.append(TraceEvent(kind=kind, data=data, elapsed_ms=elapsed_ms))

    def tool_calls(self) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == "tool_use"]

    def tool_latencies(self) -> dict[str, list[float]]:
        """Map tool name -> list of per-call elapsed_ms."""
        out: dict[str, list[float]] = {}
        for ev in self.events:
            if ev.kind == "tool_result":
                name = ev.data.get("tool_name", "unknown")
                out.setdefault(name, []).append(ev.elapsed_ms)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "final_answer": self.final_answer,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": (self.ended_at or time.time()) - self.started_at,
            "events": [
                {
                    "kind": e.kind,
                    "elapsed_ms": e.elapsed_ms,
                    "timestamp": e.timestamp,
                    "data": e.data,
                }
                for e in self.events
            ],
        }

    def append_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.to_dict(), default=str) + "\n")
