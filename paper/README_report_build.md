# Building the LaTeX-style report PDF

This folder contains a prettier Pandoc/XeLaTeX report version.

## Files

- `paper/final_report.md` — source Markdown report.
- `paper/latex/preamble.tex` — custom LaTeX styling: colors, headers, callout boxes, section styling.
- `paper/final_report.tex` — generated standalone LaTeX file, useful for inspection/debugging.
- `paper/final_report.pdf` — rendered PDF version.
- `paper/build_report_pdf.sh` — reproducible build script.

## Build from repository root

```bash
bash paper/build_report_pdf.sh
```

Equivalent command:

```bash
pandoc paper/final_report.md \
  -o paper/final_report.pdf \
  --pdf-engine=xelatex \
  --resource-path=.:paper:results/plots \
  --toc \
  --number-sections \
  -H paper/latex/preamble.tex \
  -V mainfont="DejaVu Sans" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono"
```

If the build environment does not have the dependencies:

```bash
sudo apt update
sudo apt install -y pandoc texlive-xetex texlive-latex-extra texlive-lang-cyrillic fonts-dejavu
```
