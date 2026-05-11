"""Thin wrapper around the Anthropic SDK.

We isolate the SDK behind ``AnthropicLLM`` so the agent loop only depends
on a tiny interface: ``create_message(messages, tools)``. This keeps the
agent code easy to test (a fake LLM is one class away) and makes it
trivial to swap in another provider later if needed.
"""

from __future__ import annotations

import os
from typing import Any
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
DEFAULT_MAX_TOKENS = 1024


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_tool",
        "description": (
            "Search the public web and return the top results (title, url, snippet) "
            "plus a short summary answer. Use for factual lookups, current events, "
            "and anything where the user expects up-to-date information. "
            "Do NOT use for math the calculator can handle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise web search query, ideally 3-10 words.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-10).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator_tool",
        "description": (
            "Evaluate a single arithmetic expression with full precision. "
            "Supports +, -, *, /, //, %, **, parentheses, and the math "
            "functions sqrt, log, log2, log10, exp, sin, cos, tan, asin, "
            "acos, atan, abs, round, min, max, pow, ceil, floor, factorial. "
            "Constants pi, e, tau, inf are available. Use this whenever "
            "the user asks for a numeric answer; do NOT do mental arithmetic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression, e.g. '(3.2e12) / 30000000'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "code_exec_tool",
        "description": (
            "Run a self-contained snippet of Python 3 in a subprocess with a "
            "short timeout. Useful for plotting (matplotlib is preinstalled "
            "and configured), simulations, list/string processing, or anything "
            "easier to express as code than as an arithmetic expression. "
            "Print results you want to see; matplotlib figures are captured "
            "automatically and returned as base64 PNG."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "Standalone Python 3 source code. Use print() for textual results.",
                }
            },
            "required": ["python_code"],
        },
    },
]


SYSTEM_PROMPT = """\
You are a careful research assistant with access to three tools: \
search_tool (web search), calculator_tool (exact arithmetic), and \
code_exec_tool (Python execution, including matplotlib).

Guidelines:
- For factual or current-events questions, call search_tool first and cite the URLs in your answer.
- For any numeric calculation, call calculator_tool rather than computing in your head.
- For plots, simulations, or anything algorithmic, call code_exec_tool.
- You may chain tools: e.g. search to get a number, then calculator to combine it.
- After tool calls, give a concise final answer in plain prose. If you cited search results, list the URLs.
- If a tool fails, decide whether to retry with different arguments or explain to the user that the lookup failed.
"""


@dataclass
class LLMResponse:
    """Normalised response from a single Anthropic API call."""

    stop_reason: str
    content: list[dict[str, Any]]
    raw: Any
    input_tokens: int
    output_tokens: int


class AnthropicLLM:
    """Tiny wrapper around ``anthropic.Anthropic`` for tool-use calls."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key)

    def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str = SYSTEM_PROMPT,
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        content = [block.model_dump() for block in response.content]
        usage = response.usage
        return LLMResponse(
            stop_reason=response.stop_reason or "",
            content=content,
            raw=response,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )
