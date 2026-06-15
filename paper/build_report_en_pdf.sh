#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pandoc paper/final_report_en.md   -o paper/final_report_en.pdf   --pdf-engine=xelatex   --resource-path=.:paper:results/plots   --toc   --number-sections   -H paper/latex/preamble.tex   -V mainfont="DejaVu Sans"   -V sansfont="DejaVu Sans"   -V monofont="DejaVu Sans Mono"
