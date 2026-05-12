import base64

import pytest

from research_agent.tools.code_exec import code_exec_tool


def test_basic_print():
    res = code_exec_tool("print(2 + 2)")
    assert res["ok"] is True
    assert res["exit_code"] == 0
    assert res["stdout"].strip() == "4"
    assert res["n_figures"] == 0


def test_runtime_error_reports_nonzero_exit():
    res = code_exec_tool("raise ValueError('boom')")
    assert res["ok"] is False
    assert res["exit_code"] != 0
    assert "ValueError" in res["stderr"]


def test_timeout():
    res = code_exec_tool("import time; time.sleep(10)", timeout_sec=1)
    assert res["ok"] is False
    assert "timed out" in res["error"]


def test_matplotlib_figure_captured():
    code = (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "x = np.linspace(0, 6.28, 100)\n"
        "plt.plot(x, np.sin(x))\n"
        "plt.title('sine')\n"
    )
    res = code_exec_tool(code, timeout_sec=15)
    if not res["ok"]:
        pytest.skip(f"matplotlib/numpy not available: {res.get('stderr', '')}")
    assert res["n_figures"] >= 1
    fig = res["figures"][0]
    raw = base64.b64decode(fig["image_base64"])
    assert raw.startswith(b"\x89PNG\r\n")


def test_empty_code():
    assert code_exec_tool("")["ok"] is False
