"""Safe arithmetic calculator.

Parses an expression with the ``ast`` module and walks the tree, only
permitting numeric literals, arithmetic operators, and a small whitelist
of math functions and constants. ``eval`` is never called on user input.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


class CalculatorError(ValueError):
    """Raised when an expression contains disallowed syntax or fails to evaluate."""


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise CalculatorError(f"unknown identifier: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise CalculatorError(f"operator not allowed: {type(node.op).__name__}")
        return op_fn(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise CalculatorError(f"unary operator not allowed: {type(node.op).__name__}")
        return op_fn(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalculatorError("only whitelisted math functions may be called")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        args = [_eval(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise CalculatorError(f"unsupported expression node: {type(node).__name__}")


def calculator_tool(expression: str) -> dict[str, Any]:
    """Evaluate an arithmetic ``expression`` safely.

    Returns a structured dict with the result or an error message. The
    return shape is intentionally JSON-serialisable so it can be passed
    straight back to the LLM as a tool result.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"ok": False, "error": "expression must be a non-empty string"}
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval(tree)
    except CalculatorError as exc:
        return {"ok": False, "error": str(exc), "expression": expression}
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax error: {exc.msg}", "expression": expression}
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        return {"ok": False, "error": str(exc), "expression": expression}
    return {"ok": True, "expression": expression, "result": value}
