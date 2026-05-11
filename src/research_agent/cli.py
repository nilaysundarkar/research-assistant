"""Command-line interface for the agent.

Examples:

    llm-agent "What is the GDP of Texas divided by its population?"
    llm-agent --repl
    llm-agent --query "Plot sin(x) from 0 to 2*pi" --save-figures ./out
"""

def main(argv: list[str] | None = None) -> int:
    print("Hello, world!")
    return 0