# Report v2 changes

This update expands the previous short reports into stronger paper-like reports.

## What was weak before

1. The literature context was too short.
2. References were mostly missing or mixed with project artifacts.
3. The method section was correct but too schematic.
4. Hyperparameters were not shown in a dedicated table.
5. The claim that n=3 learns faster was supported visually, but not with sample-efficiency markers.
6. Limitations were listed, but not analyzed as threats to validity.
7. The reports looked more like short handouts than a mini research report.

## What was added

- Extended related work:
  - SAC baseline;
  - n-step returns and credit assignment;
  - off-policy multi-step correction;
  - SACn and T-SAC.
- Real bibliography in `paper/references.bib`.
- Hyperparameter table.
- Environment details with Gymnasium references.
- Sample-efficiency table:
  - steps to return >= 3000;
  - steps to return >= 4000.
- Expanded discussion of the n=5 failure.
- Threats to validity:
  - statistical validity;
  - algorithmic validity;
  - hyperparameter fairness;
  - environment coverage;
  - compute budget.
- Reproducibility checklist.
- English and Russian v2 reports generated through Pandoc + XeLaTeX.

## Recommended use

Use `paper/final_report_en.pdf` as the main polished report if English is acceptable.
Use `paper/final_report.pdf` as the Russian-facing report or backup.

The reports are intentionally still moderate in size. The goal is a strong final project report, not an oversized fake paper.
