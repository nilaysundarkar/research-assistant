# Multi-Tool LLM Agent (CSCI E-222 Final Project)

A modular LLM agent that can call three external tools — **search**, **calculator**, and **Python code execution** — to answer multi-step queries that exceed pure text generation.

Built with Anthropic Claude (native tool-use) and Tavily for web search. MCP integration is intentionally deferred to a future iteration (see "Stretch goal" in the proposal).

## Setup

These steps create a fresh virtual environment and install the project. Required Python: **3.10+**.

```bash
git clone https://github.com/nilaysundarkar/research-assistant

# or unzip the project from the submitted zip

cd research-assistant

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                     # REQUIRED: registers the research_agent package + `research-agent` CLI

cp .env.example .env                 # only needed for live runs — fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
```

> The project uses a `src/` layout, so `pip install -e .` is mandatory.
> Without it, `python -m research_agent.cli` fails with `ModuleNotFoundError: No module named 'research_agent'`.

If you have `[uv](https://github.com/astral-sh/uv)` installed, you can use the faster alternative:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Pre-recorded eval data

To let graders verify the project without spending on API calls, **the full evaluation run is pre-recorded and committed to this repo**. I produced the artifacts under `eval/results/` by running `python eval/run_eval.py` against the live Anthropic Claude and Tavily APIs (using my own API keys). What's committed:

- `eval/results/<timestamp>/traces.jsonl` — one full reasoning trace per query (model messages, tool calls, tool results, latencies, token counts)
- `eval/results/<timestamp>/summary.json` — aggregate metrics (per-tool latency stats, success rate, by-category breakdown)
- `viz/output/*.png` — the three required figures, rendered from those traces

The "How to run — verifying without API keys" section below walks through replaying these artifacts. To regenerate them yourself against the live APIs (requires your own API keys), see the [Evaluation](#evaluation) section.

## How to run — verifying without API keys

Running the agent live requires Anthropic and Tavily API keys. Because LLM calls cost money, **a complete eval run has been pre-recorded** and committed to this repo (see "Pre-recorded eval data" above). You can verify the project end-to-end without spending anything. (Complete the Setup section above first.)

```bash
# 1. Run the test suite (no API keys needed; uses a scripted fake LLM for the agent loop)
pytest tests/ -q
#    Expected: 30 passed

# 2. Inspect the canonical eval artifacts (40 queries across 4 categories)
ls eval/results/                           # one or more timestamped run dirs
cat eval/results/<timestamp>/summary.json  # aggregate metrics
head -1 eval/results/<timestamp>/traces.jsonl | python -m json.tool   # one full trace

# 3. Re-render the visualizations from the committed traces (no API key)
python viz/make_plots.py eval/results/<timestamp>/traces.jsonl
ls viz/output/                             # tool_usage_bar.png, tool_latency_box.png, sample_code_plot.png
```

What each step proves:


| Step                                       | What you've verified                                                                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `pytest tests/ -q`                         | Calculator's AST whitelist, code-exec subprocess timeout, agent's tool-use loop (with a scripted fake LLM)                 |
| Reading `eval/results/<ts>/summary.json`   | Agent solves real queries across 4 categories with measurable success rate                                                 |
| Reading individual lines in `traces.jsonl` | The full reasoning trace for any query: which tools were called, what they returned, how long each step took, token counts |
| Re-running `viz/make_plots.py`             | The trace data is reproducible — figures regenerate from the committed JSONL                                               |


If you do have an Anthropic and Tavily key and want to run the agent live, see "Usage" below (Setup is covered above).

## Project layout

```
research-assistant/
├── src/research_agent/
│   ├── agent.py           # multi-turn tool-use loop
│   ├── llm.py             # Anthropic wrapper + tool schemas + system prompt
│   ├── tracing.py         # structured trace events
│   ├── cli.py             # CLI entry point
│   └── tools/
│       ├── search.py      # Tavily web search
│       ├── calculator.py  # AST-safe arithmetic
│       └── code_exec.py   # subprocess sandbox (matplotlib-aware)
├── eval/
│   ├── queries.jsonl      # 40 curated test queries (10 per category)
│   └── run_eval.py        # batch runner -> traces.jsonl + summary.json
├── viz/
│   └── make_plots.py      # tool usage bar / latency box / sample plot
├── tests/                 # pytest suite (no API keys required)
├── traces/                # JSONL traces written at runtime
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Usage

### Single query

```bash
research-agent "What is the GDP of Texas divided by its population?"
```

### REPL

```bash
research-agent --repl
```

### Save matplotlib figures produced during a run

```bash
research-agent "Plot sin(x) from 0 to 2*pi" --save-figures ./out
```

### Programmatic

```python
from research_agent.agent import Agent

agent = Agent()
trace = agent.run("How many seconds are in 30 days?")
print(trace.final_answer)
print(trace.tool_latencies())
```

## How the agent loop works

```
user query ──► Anthropic Messages API (with tool schemas)
                      │
                      ▼
       ┌─────────────────────────────┐
       │ stop_reason == "tool_use"?  │── no ──► extract text → final answer
       └─────────────────────────────┘
                      │ yes
                      ▼
            for each tool_use block:
              run local Python tool
              attach tool_result to next user message
                      │
                      ▼
              loop (max_steps)
```

Each iteration is recorded on a `Trace` object: model call, tool call, tool result, errors, token usage, latencies. Traces are appended to `traces/runs.jsonl` and consumed by `eval/run_eval.py` and `viz/make_plots.py`.

### The three tools


| Tool              | Implementation                                                    | Returns                                         |
| ----------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| `search_tool`     | Tavily API (`TavilyClient.search`, basic depth, top-k)            | `{answer, results: [{title, url, snippet}]}`    |
| `calculator_tool` | `ast.parse` + recursive evaluator with whitelisted ops/functions  | `{result, expression}`                          |
| `code_exec_tool`  | Fresh `python` subprocess, configurable timeout, matplotlib `Agg` | `{stdout, stderr, exit_code, figures (base64)}` |


Image bytes are stripped before being sent back to Claude (only `image_base64_len` is reported) so the conversation context stays compact; the full bytes are kept on the trace for the CLI/viz layer.

## Evaluation

> **Requires API keys.** This section is for re-running the evaluation live against the Anthropic and Tavily APIs. If you only need to verify the project without spending money, the pre-recorded artifacts under `eval/results/` are enough — see "How to run — verifying without API keys" above.

A curated set of **40 queries** across 4 categories (10 each):

- `factual` — single-fact lookup, expects `search_tool`
- `math` — pure arithmetic, expects `calculator_tool`
- `code` — plotting / simulation / algorithms, expects `code_exec_tool`
- `mixed` — multi-tool chains (search + calc, search + plot, etc.)

```bash
python eval/run_eval.py                      # full sweep (~5-10 min)
python eval/run_eval.py --category math      # one category
python eval/run_eval.py --limit 5            # smoke test
python eval/run_eval.py --dry-run            # validate queries.jsonl (no API calls)
```

> The `--dry-run` flag only validates the queries file and does not hit the APIs, so it works without keys.

Outputs land in `eval/results/<timestamp>/`:

- `traces.jsonl` — one full trace per query (model messages, tool calls, latencies, token counts)
- `summary.json` — aggregate metrics: success rate, per-tool latency stats (min/median/mean/max), tool-usage histogram, by-category breakdown

## Visualizations

```bash
python viz/make_plots.py eval/results/<timestamp>/traces.jsonl
```

Writes three required figures into `viz/output/`:

1. `**tool_usage_bar.png**` — frequency of each tool across all queries
2. `**tool_latency_box.png**` — latency distribution per tool (boxplot, log-scaled y)
3. `**sample_code_plot.png**` — a representative figure produced by `code_exec_tool`, decoded from a trace

## Testing

```bash
pytest tests/ -q
```

The test suite (30 tests, no API keys required) covers:

- AST whitelist on `calculator_tool` (rejects imports, lambdas, attribute access, list literals, `__import__`, file I/O, etc.)
- Subprocess timeout, runtime-error reporting, and matplotlib figure capture in `code_exec_tool`
- Agent tool-use protocol via a scripted fake LLM (single tool call, unknown tool, max-steps cap)

## Configuration

Environment variables (see `.env.example`):


| Variable                | Default                 | Effect                          |
| ----------------------- | ----------------------- | ------------------------------- |
| `ANTHROPIC_API_KEY`     | *(required)*            | Anthropic API auth              |
| `TAVILY_API_KEY`        | *(required for search)* | Tavily API auth                 |
| `ANTHROPIC_MODEL`       | `claude-sonnet-4-5`     | Override the model id           |
| `AGENT_MAX_STEPS`       | `8`                     | Hard cap on tool-use iterations |
| `CODE_EXEC_TIMEOUT_SEC` | `8`                     | Per-call subprocess timeout     |


## Limitations and notes for graders

- The code-execution tool is a *light* sandbox (subprocess + timeout + restricted output size). It is **not** secure against an actively malicious model — that would require a Docker- or seccomp-backed runner. This is documented in the source.
- Search quality is bounded by Tavily's free-tier ranking; results are summarised per the agent's system prompt to reduce hallucination.
- The agent does not maintain a multi-turn conversation across CLI invocations — each query starts a fresh history. The REPL preserves nothing between turns by design (each query is independent).

## Future work (stretch goal)

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) integration is deferred. The cleanest path is:

1. Wrap each tool function in a small MCP server (`mcp.server.stdio`).
2. Replace the local `_default_tool_registry()` in `agent.py` with an `MCPToolClient` that lists tools from each server and dispatches by name.
3. The `Agent.run` loop is unchanged because the tool *interface* (`name`, `input_schema`, `dict` result) is identical.

A side-by-side latency/modularity comparison would be the natural extension.