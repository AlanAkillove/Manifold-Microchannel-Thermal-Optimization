# Reproducibility

This repository is a reproducible research artifact for a manifold microchannel thermal-management modelling task. The reported manuscript, figures, tables, and JSON summary are generated from the raw task inputs and the code in this repository.

## Environment

- Python 3.10 or later
- Runtime dependencies declared in `pyproject.toml`
- Development dependencies: pytest and Ruff
- XeLaTeX and BibTeX for compiling the manuscript

The Python package uses a `src/` layout and can be installed in editable mode with `python -m pip install -e ".[dev]"`.

## Commands

Fast smoke check:

    python scripts/run_analysis.py --quick

The quick mode reads and validates the raw workbook, fits the cubic response-surface model, runs the interlaced five-fold check, and evaluates a small grid. It does not overwrite published outputs and normally completes within 15 seconds.

Full analysis:

    python scripts/run_analysis.py --full
    python scripts/make_figures.py
    python -m pytest -q

The full analysis writes `data/processed/attachment2_tidy.csv`, the 17 CSV files under `outputs/tables/`, and `outputs/results.json`. Figure generation writes the paper figures under `outputs/figures/`. On a desktop CPU, the full analysis typically takes about 10–15 minutes; Q4 and Q5 account for most of the runtime.

Compile the manuscript from `docs/paper/tex/`:

    xelatex -interaction=nonstopmode main.tex
    bibtex main
    xelatex -interaction=nonstopmode main.tex
    xelatex -interaction=nonstopmode main.tex

## Determinism

Randomized optimization calls use fixed seeds recorded in `outputs/results.json`. The model and validation stages are deterministic for the supplied data. The generated artifacts are intended to be reviewed together with the manuscript rather than treated as independent raw measurements.

## Verification record

Verification performed on 2026-09-12:

- [x] `python scripts/run_analysis.py --quick` completed successfully.
- [x] `python scripts/run_analysis.py --full` completed successfully and regenerated the result files.
- [x] `python scripts/make_figures.py` completed successfully.
- [x] `python -m pytest -q` — 13 tests passed.
- [x] XeLaTeX/BibTeX compilation completed successfully; the final manuscript has 44 pages.
- [x] Static LaTeX release check reported `PASS` for `docs/paper/tex/main.tex`.
- [x] All manuscript figure and code-listing references resolve to files in the repository.
- [x] Key values in `outputs/results.json` agree with the manuscript summary.

## Generated artifacts

The repository keeps final PDFs, figures, tables, and JSON results. LaTeX auxiliary files, Python caches, temporary renders, old versions, reference-paper full texts, and other drafting materials are kept outside the repository in the dated local archive. They are excluded from Git where applicable by `.gitignore`.
