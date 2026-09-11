# Paper artifacts

The manuscript source and compiled documents are kept under `tex/` so that the relative figure and source-code references used by the LaTeX build remain stable.

- [`main.pdf`](tex/main.pdf): compiled manuscript.
- [`main.tex`](tex/main.tex): manuscript source.
- [`references.bib`](tex/references.bib): bibliography database.
- [`AI工具使用详情.pdf`](tex/AI工具使用详情.pdf): AI tool usage details.
- [`AI工具使用详情.tex`](tex/AI工具使用详情.tex): source for the AI tool usage details.
- `cumcmthesis.cls` and `gbt7714-numerical.bst`: LaTeX class and bibliography style.
- `fonts/` and the bundled Fira Code font: fonts used to reproduce the included PDFs.

To compile the manuscript, open a terminal in `docs/paper/tex/` and run:

    xelatex -interaction=nonstopmode main.tex
    bibtex main
    xelatex -interaction=nonstopmode main.tex
    xelatex -interaction=nonstopmode main.tex

Build artifacts such as `.aux`, `.bbl`, `.blg`, `.log`, and `.out` are intentionally not kept in the repository. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for the separate terms of the bundled template, bibliography style, and fonts.
