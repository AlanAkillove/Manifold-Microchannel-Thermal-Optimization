# Release checklist

## Repository

- [x] README documents the project scope, directory layout, reproduction commands, and license boundary.
- [x] Source code, tests, raw attachments, processed data, figures, tables, and final results are kept in the repository.
- [x] Temporary files, old versions, reference PDFs, and generated build files are stored outside the repository archive.
- [x] A local Git repository has been initialized on the `main` branch.

## Manuscript and artifacts

- [x] The paper source, bibliography, document class, bibliography style, fonts, and compiled PDF are present.
- [x] The AI tool usage details source and PDF are present.
- [x] Final figures, tables, and `results.json` are present under `outputs/`.
- [x] The original problem materials are present under `data/raw/`; processed data are under `data/processed/`.

## Verification

- [x] `py -3.10 -m pytest -q` — 13 tests passed.
- [x] `py -3.10 scripts/make_figures.py` completed successfully.
- [x] `latex_release_check.py` reported `PASS` for `docs/paper/tex/main.tex`.
- [x] The paper compiled successfully with XeLaTeX and BibTeX.
- [ ] The complete `scripts/run_analysis.py` run was not completed in this environment because the Q4/Q5 differential-evolution search is long-running; rerun it before final release if a full clean recomputation is required.
- [ ] Initial commit, remote configuration, and push remain to be performed by the repository owner.

## Release status

**Ready for review.** The repository is organized for Git upload, but it should not be treated as a final release until the full analysis is rerun and its outputs are reviewed.

Verification date: 2026-09-11
