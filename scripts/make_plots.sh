#!/usr/bin/env bash
set -euo pipefail

python3 -m src.plot_results --input results/raw --out results/plots
