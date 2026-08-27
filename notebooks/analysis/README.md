# Analysis notebooks

The paper assets. One notebook per paper section or figure, reading from `results/` and calling
the generators in `scripts/`. Expected over time: `main_results`, `suvmax_bland_altman`,
`lesion_subgroups`, `failure_cases`, `cross_tracer`.

Each notebook is created when the result it describes exists. Empty placeholders are not
committed ahead of time.

These notebooks are the place to *shape* a table or a figure interactively — but the finished
artefact is still produced by `scripts/make_tables.py` / `scripts/make_figures.py`, and metric
code still lives only in `src/petinterp/evaluation/`. The notebook calls those; it never grows
its own copy (CLAUDE.md rules 2, 3, 6).
