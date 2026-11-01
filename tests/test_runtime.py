"""archive_notebook from an exported notebook file (the Colab CLI route), off Colab.

    pip install -e . nbformat nbconvert pytest && pytest tests/test_runtime.py
"""

import json
import re
from html import unescape

import pytest

import petinterp.runtime as rt

pytest.importorskip("nbconvert")

# shaped like `colab log -o x.ipynb`: sources and outputs as lists of lines
SESSION = {
    "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    "cells": [
        {"cell_type": "markdown", "id": "m0", "metadata": {}, "source": ["# Colab Session: train\n"]},
        {"cell_type": "code", "id": "c0", "metadata": {}, "execution_count": 1,
         "source": ["for c in CONFIGS:\n", "    train(rt.start(c))\n"],
         "outputs": [{"output_type": "stream", "name": "stdout",
                      "text": ["epoch 49/49 train 0.0029 val 0.0045 *best (65s)\n"]}]},
    ],
}


def test_archive_from_exported_ipynb(tmp_path):
    src = tmp_path / "session.ipynb"
    src.write_text(json.dumps(SESSION), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rt.archive_notebook(run_dir, ipynb=src)

    html = (run_dir / "notebook.html").read_text(encoding="utf-8")
    text = unescape(re.sub(r"<[^>]+>", "", html))  # code cells are split into highlighting spans
    assert "epoch 49/49 train 0.0029" in text
    assert "train(rt.start(c))" in text
    assert not (run_dir / "notebook.html.tmp").exists()  # atomic write left nothing behind


def test_deferred_archive_then_pending(tmp_path, monkeypatch):
    # the 03_train loop under the CLI: each run's archive call is only recorded ...
    pending = tmp_path / "pending.txt"
    monkeypatch.setenv("PETINTERP_DEFER_ARCHIVE", str(pending))
    runs = [tmp_path / "runs" / e for e in ("p4_a", "p4_b")]
    for d in runs + runs[:1]:  # a resumed queue passes a finished run again
        d.mkdir(parents=True, exist_ok=True)
        rt.archive_notebook(d)
    assert not any((d / "notebook.html").exists() for d in runs)

    # ... and written for every run, once, from the executed notebook after the queue
    src = tmp_path / "executed.ipynb"
    src.write_text(json.dumps(SESSION), encoding="utf-8")
    assert rt.archive_pending(pending, src) == runs
    for d in runs:
        assert "epoch 49/49" in (d / "notebook.html").read_text(encoding="utf-8")
