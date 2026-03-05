# CCP Paper (LaTeX)

Minimal LaTeX scaffold for writing the CCP methodology paper.

## Build

Option A (recommended): `latexmk`

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Option B: `pdflatex + bibtex`

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

