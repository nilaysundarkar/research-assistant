import math

import pytest

from research_agent.tools.calculator import calculator_tool


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2", 3),
        ("2 * (3 + 4)", 14),
        ("2 ** 10", 1024),
        ("10 / 4", 2.5),
        ("17 // 5", 3),
        ("17 % 5", 2),
        ("-3 + 5", 2),
        ("sqrt(2) ** 2", pytest.approx(2.0)),
        ("sin(pi / 2)", pytest.approx(1.0)),
        ("log(e)", pytest.approx(1.0)),
        ("factorial(6)", 720),
        ("pow(2, 8)", 256),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
    ],
)
def test_valid_expressions(expr, expected):
    res = calculator_tool(expr)
    assert res["ok"] is True
    assert res["result"] == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "x + 1",
        "lambda x: x",
        "[1, 2, 3]",
        "1 if True else 2",
    ],
)
def test_disallowed_expressions(expr):
    res = calculator_tool(expr)
    assert res["ok"] is False
    assert "error" in res


def test_division_by_zero():
    assert calculator_tool("1 / 0")["ok"] is False


def test_empty_string():
    assert calculator_tool("")["ok"] is False
