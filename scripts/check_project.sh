#!/usr/bin/env bash
set -euo pipefail

echo "Checking Python dependencies..."
python3 - <<'PY'
import importlib.util
import sys

missing = []
for package in ["gymnasium", "torch", "numpy", "yaml", "matplotlib", "pandas"]:
    if importlib.util.find_spec(package) is None:
        missing.append(package)

if missing:
    print("Missing Python dependencies:", ", ".join(missing))
    print("Run:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install --upgrade pip")
    print("  pip install -r requirements.txt")
    sys.exit(1)
PY

echo "Checking Python syntax..."
python3 -m compileall -q src

echo "Checking shell script syntax..."
bash -n scripts/*.sh

echo "Checking environment smoke test..."
python3 - <<'PY'
import gymnasium as gym
import torch

env = gym.make("Pendulum-v1")
obs, info = env.reset(seed=0)

print("env:", env.spec.id)
print("obs shape:", obs.shape)
print("action space:", env.action_space)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

env.close()
PY

echo "Checking plot generation CLI..."
python3 -m src.plot_results --input results/raw --out /tmp/nstep_sac_plot_check

echo "Project check passed."
