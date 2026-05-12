"""Test the agent loop with a fake LLM that scripts a known dialogue."""

from __future__ import annotations

from research_agent.agent import Agent
from research_agent.llm import LLMResponse


class ScriptedLLM:
    """Returns canned responses in order. Mimics ``AnthropicLLM.create_message``."""

    def __init__(self, scripted: list[LLMResponse]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    def create_message(self, messages, tools, system=None):  # noqa: ARG002
        self.calls += 1
        if not self._scripted:
            raise AssertionError("LLM was called more times than scripted")
        return self._scripted.pop(0)


def _tool_use_response(name: str, args: dict, tool_use_id: str = "tu_1") -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        content=[
            {"type": "tool_use", "id": tool_use_id, "name": name, "input": args}
        ],
        raw=None,
        input_tokens=10,
        output_tokens=5,
    )


def _final_response(text: str) -> LLMResponse:
    return LLMResponse(
        stop_reason="end_turn",
        content=[{"type": "text", "text": text}],
        raw=None,
        input_tokens=20,
        output_tokens=10,
    )


def test_single_tool_call_then_finish():
    llm = ScriptedLLM(
        [
            _tool_use_response("calculator_tool", {"expression": "2 + 2"}),
            _final_response("The answer is 4."),
        ]
    )
    agent = Agent(llm=llm)  # type: ignore[arg-type]
    trace = agent.run("What is 2 + 2?")

    assert trace.final_answer == "The answer is 4."
    assert len(trace.tool_calls()) == 1
    tu = trace.tool_calls()[0]
    assert tu.data["name"] == "calculator_tool"
    results = [e for e in trace.events if e.kind == "tool_result"]
    assert results[0].data["result"]["result"] == 4


def test_unknown_tool_returns_error():
    llm = ScriptedLLM(
        [
            _tool_use_response("does_not_exist", {}),
            _final_response("I could not run that tool."),
        ]
    )
    agent = Agent(llm=llm)  # type: ignore[arg-type]
    trace = agent.run("test")

    results = [e for e in trace.events if e.kind == "tool_result"]
    assert results[0].data["result"]["ok"] is False
    assert "unknown tool" in results[0].data["result"]["error"]


def test_max_steps_enforced():
    llm = ScriptedLLM(
        [_tool_use_response("calculator_tool", {"expression": "1 + 1"}, f"tu_{i}")
         for i in range(20)]
    )
    agent = Agent(llm=llm, max_steps=2)  # type: ignore[arg-type]
    trace = agent.run("loop forever")
    assert trace.final_answer is not None
    assert "max_steps" in trace.final_answer
