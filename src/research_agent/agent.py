"""The multi-tool agent loop.

This implements Anthropic's tool-use protocol:

1. Send the conversation + tool schemas to Claude.
2. If Claude returns ``stop_reason == "tool_use"``:
   - Echo the assistant message back into history as-is.
   - For every ``tool_use`` block, run the matching local Python function
     and add a ``tool_result`` block to a new user message.
   - Loop.
3. When Claude returns ``stop_reason == "end_turn"`` (or we hit
   ``max_steps``), pull out the final text and finish.

We keep base64 plot bytes out of the conversation history -- they would
balloon the context for free -- but record them on the trace so the
caller can display them.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from .llm import SYSTEM_PROMPT, TOOL_SCHEMAS, AnthropicLLM
from .tools import calculator_tool, code_exec_tool, search_tool
from .tracing import Trace

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))


ToolFn = Callable[..., dict[str, Any]]


def _result_for_llm(name: str, raw: dict[str, Any]) -> str:
    """Trim large fields (image base64) before sending back to Claude.

    The model only needs to know *how many* figures were produced and
    that they exist; it does not need the bytes."""
    if name == "code_exec_tool" and "figures" in raw:
        compact = dict(raw)
        compact["figures"] = [
            {"filename": f["filename"], "image_base64_len": len(f["image_base64"])}
            for f in raw.get("figures", [])
        ]
        return json.dumps(compact, default=str)
    return json.dumps(raw, default=str)

def _default_tool_registry() -> dict[str, ToolFn]:
    return {
        "search_tool": search_tool,
        "calculator_tool": calculator_tool,
        "code_exec_tool": code_exec_tool,
    }

class Agent:
    """Runs a query through Claude's tool-use loop and returns a Trace."""

    def __init__(
        self,
        llm: AnthropicLLM | None = None,
        tools: dict[str, ToolFn] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.llm = llm or AnthropicLLM()
        self.tools = tools or _default_tool_registry()
        self.tool_schemas = tool_schemas or TOOL_SCHEMAS
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_query: str) -> Trace:
        trace = Trace(query=user_query)
        trace.add("user", {"text": user_query})

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_query}
        ]

        for step in range(self.max_steps):
            t0 = time.time()
            try:
                response = self.llm.create_message(
                    messages=messages,
                    tools=self.tool_schemas,
                    system=self.system_prompt,
                )
            except Exception as exc:
                trace.add("error", {"phase": "llm_call", "error": str(exc)},
                          elapsed_ms=(time.time() - t0) * 1000)
                trace.final_answer = f"[error] LLM call failed: {exc}"
                trace.ended_at = time.time()
                return trace

            llm_ms = (time.time() - t0) * 1000
            trace.total_input_tokens += response.input_tokens
            trace.total_output_tokens += response.output_tokens
            trace.add(
                "model",
                {
                    "step": step,
                    "stop_reason": response.stop_reason,
                    "content": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
                elapsed_ms=llm_ms,
            )

            if response.stop_reason != "tool_use":
                trace.final_answer = _extract_text(response.content)
                trace.ended_at = time.time()
                trace.add("final", {"text": trace.final_answer})
                return trace

            messages.append({"role": "assistant", "content": response.content})

            tool_results_content: list[dict[str, Any]] = []
            for block in response.content:
                if block.get("type") != "tool_use":
                    continue
                tool_name = block["name"]
                tool_input = block.get("input", {}) or {}
                tool_use_id = block["id"]

                trace.add("tool_use", {"name": tool_name, "input": tool_input,
                                       "tool_use_id": tool_use_id})

                fn = self.tools.get(tool_name)
                t0 = time.time()
                if fn is None:
                    raw_result = {"ok": False, "error": f"unknown tool: {tool_name}"}
                else:
                    try:
                        raw_result = fn(**tool_input)
                    except TypeError as exc:
                        raw_result = {"ok": False, "error": f"bad tool args: {exc}"}
                    except Exception as exc:
                        raw_result = {"ok": False, "error": f"tool crashed: {exc}"}
                tool_ms = (time.time() - t0) * 1000

                trace.add(
                    "tool_result",
                    {
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "result": raw_result,
                    },
                    elapsed_ms=tool_ms,
                )

                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": _result_for_llm(tool_name, raw_result),
                        "is_error": not raw_result.get("ok", False),
                    }
                )

            messages.append({"role": "user", "content": tool_results_content})

        trace.add("error", {"phase": "loop", "error": f"max_steps={self.max_steps} reached"})
        trace.final_answer = "[error] agent exceeded max_steps without finishing"
        trace.ended_at = time.time()
        return trace


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Concatenate ``text`` blocks from a Claude response."""
    parts = [b.get("text", "") for b in content if b.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()