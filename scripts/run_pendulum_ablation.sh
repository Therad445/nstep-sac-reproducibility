#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/pendulum_sac_n1.yaml
python -m src.train --config configs/pendulum_sac_n3.yaml
python -m src.train --config configs/pendulum_sac_n5.yaml
python -m src.plot_results --input-dir results/raw --out results/plots/pendulum_learning_curves.png
