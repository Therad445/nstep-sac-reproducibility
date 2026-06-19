#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall -q src
python3 - <<'PY'
import importlib

required = ["gymnasium", "torch", "yaml", "numpy"]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        missing.append(module_name)

if missing:
    print("Missing Python dependencies:", ", ".join(missing))
    print("Run:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install --upgrade pip")
    print("  pip install -r requirements.txt")
    raise SystemExit(1)

import gymnasium as gym
import torch
import yaml

with open('configs/pendulum_sac_n1.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

env = gym.make(cfg['env_id'])
obs, _ = env.reset(seed=cfg['seed'])
print('env:', cfg['env_id'])
print('obs shape:', obs.shape)
print('action space:', env.action_space)
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
env.close()
PY
