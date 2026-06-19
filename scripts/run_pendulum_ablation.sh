#!/usr/bin/env bash
set -euo pipefail

python3 -m src.train --config configs/pendulum_sac_n1.yaml
python3 -m src.train --config configs/pendulum_sac_n3.yaml
python3 -m src.train --config configs/pendulum_sac_n5.yaml
python3 -m src.plot_results --input results/raw --out results/plots
