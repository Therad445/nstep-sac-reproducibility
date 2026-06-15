#!/usr/bin/env bash
set -euo pipefail

# Run from repository root.
# Requires pandoc + xelatex.

mkdir -p paper/build

pandoc paper/final_report.md \
  -o paper/build/final_report.tex \
  --standalone \
  --resource-path=.:paper:results/plots \
  --toc \
  --number-sections \
  -H paper/latex/preamble.tex \
  -V mainfont="DejaVu Sans" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono"

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
