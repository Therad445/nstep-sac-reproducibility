# Building the final reports

The reports are written in Markdown and compiled to PDF with Pandoc + XeLaTeX.

## Dependencies

On Ubuntu / WSL:

```bash
sudo apt update
sudo apt install -y pandoc texlive-xetex texlive-latex-extra texlive-lang-cyrillic fonts-dejavu
```

## Build both reports

From the repository root:

```bash
bash paper/build_reports_pdf.sh
```

## Build English report only

```bash
bash paper/build_report_en_pdf.sh
```

## Build Russian report only

```bash
bash paper/build_report_ru_pdf.sh
```

## Direct command example

```bash
pandoc paper/final_report_en.md \
  -o paper/final_report_en.pdf \
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
```
