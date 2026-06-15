#!/usr/bin/env bash
set -euo pipefail

python -m src.plot_results --input-dir results/raw --out results/plots/learning_curves.png
