#!/usr/bin/env bash
set -euo pipefail

pandoc paper/final_report.md \
  -o paper/final_report.pdf \
  --pdf-engine=xelatex \
  --resource-path=.:paper:results/plots \
  --toc \
  --number-sections \
  --citeproc \
  --bibliography=paper/references.bib \
  -H paper/latex/preamble.tex \
  -V mainfont="DejaVu Sans" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono"

pandoc paper/final_report.md \
  -o paper/final_report.tex \
  --pdf-engine=xelatex \
  --resource-path=.:paper:results/plots \
  --toc \
  --number-sections \
  --citeproc \
  --bibliography=paper/references.bib \
  -H paper/latex/preamble.tex \
  -V mainfont="DejaVu Sans" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono"
