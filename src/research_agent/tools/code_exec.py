"""Restricted Python code execution tool.

The user's code is written to a temp file and run with the *current*
Python interpreter as a subprocess with a configurable timeout. A small
prelude forces matplotlib into the non-interactive ``Agg`` backend and,
after the user code finishes, saves every open figure to a temp dir.
The tool then returns stdout, stderr, and any saved figures encoded as
base64 PNG strings so the agent can describe (or display) them.

This is *not* a true sandbox -- the subprocess can read the filesystem
and import any installed module. It is good enough for an academic
project where the user controls the queries. For a public deployment
swap this for a Docker- or seccomp-backed runner.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SEC = int(os.environ.get("CODE_EXEC_TIMEOUT_SEC", "30"))
MAX_OUTPUT_CHARS = 8000
MAX_FIGURES = 4

# Shared matplotlib config directory. Reused across calls so matplotlib's
# font cache is built once per process instead of once per tool call.
# Contains only computed-from-system data, so sharing is safe.
_SHARED_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "llm_agent_mplconfig"
_SHARED_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)


_PRELUDE = """\
import os, sys, base64, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

_FIG_DIR = os.environ["LLM_AGENT_FIG_DIR"]

def _save_open_figures():
    for i, num in enumerate(_plt.get_fignums()):
        fig = _plt.figure(num)
        fig.savefig(os.path.join(_FIG_DIR, f"figure_{i:02d}.png"),
                    dpi=120, bbox_inches="tight")
"""

_POSTLUDE = """
try:
    _save_open_figures()
except Exception as _e:
    sys.stderr.write(f"[plot-save error] {_e}\\n")
"""


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


def code_exec_tool(
    python_code: str,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Execute ``python_code`` in an isolated subprocess.

    Returns a JSON-serialisable dict with stdout, stderr, exit code,
    elapsed wall time, and any matplotlib figures encoded as base64
    PNG. Capped at ``MAX_OUTPUT_CHARS`` and ``MAX_FIGURES``.
    """
    if not isinstance(python_code, str) or not python_code.strip():
        return {"ok": False, "error": "python_code must be a non-empty string"}

    workdir = Path(tempfile.mkdtemp(prefix="llm_agent_exec_"))
    fig_dir = workdir / "figures"
    fig_dir.mkdir()
    script = workdir / "user_script.py"

    full_source = (
        _PRELUDE
        + "\n# ---- user code ----\n"
        + textwrap.dedent(python_code)
        + "\n# ---- end user code ----\n"
        + _POSTLUDE
    )
    script.write_text(full_source, encoding="utf-8")

    env = {
        **os.environ,
        "LLM_AGENT_FIG_DIR": str(fig_dir),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(_SHARED_MPLCONFIGDIR),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "ok": False,
            "error": f"execution timed out after {timeout_sec}s",
            "stdout": _truncate(exc.stdout or ""),
            "stderr": _truncate(exc.stderr or ""),
        }

    figures: list[dict[str, str]] = []
    for path in sorted(fig_dir.glob("figure_*.png"))[:MAX_FIGURES]:
        figures.append(
            {
                "filename": path.name,
                "image_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )

    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "figures": figures,
        "n_figures": len(figures),
    }
