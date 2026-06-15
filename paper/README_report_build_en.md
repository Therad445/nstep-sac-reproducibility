# Building the English LaTeX-style report

From the repository root:

```bash
bash paper/build_report_en_pdf.sh
```

Or run the command directly:

```bash
pandoc paper/final_report_en.md \
  -o paper/final_report_en.pdf \
  --pdf-engine=xelatex \
  --resource-path=.:paper:results/plots \
  --toc \
  --number-sections \
  -H paper/latex/preamble.tex \
  -V mainfont="DejaVu Sans" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono"
```

If dependencies are missing on Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y pandoc texlive-xetex texlive-latex-extra fonts-dejavu
```
