# Stress-Testing n-step Soft Actor-Critic under Limited Compute

Final RL project on n-step variants of Soft Actor-Critic.

The project studies whether n-step returns improve SAC training in continuous-control tasks when the compute budget is limited. The main comparison is between standard SAC with 1-step targets and SAC variants with 3-step and 5-step returns.

This is not a claim of a new state-of-the-art algorithm. The goal is more modest and, in practice, more useful for a student research project: understand the mechanism, reproduce a simplified version, run ablations, and honestly discuss where the method becomes unstable.

## Main comparison

```text
SAC n=1  -> baseline
SAC n=3  -> moderate n-step target
SAC n=5  -> more aggressive n-step target
```

## Environments

- `Pendulum-v1` as a cheap sanity check.
- `HalfCheetah-v5` as the main MuJoCo continuous-control environment.
- `Hopper-v5` or `Walker2d-v5` can be added later if there is enough time.

## Repository structure

```text
configs/      experiment configs
src/          training code and SAC implementation
scripts/      launch and plotting scripts
results/      raw logs, metrics and plots
docs/         notes, reproduction card and limitations
slides/       presentation materials
notebooks/    optional notebooks for quick analysis
```

## Quick start in WSL

```bash
cd ~/projects
unzip nstep-sac-reproducibility-starter.zip
cd nstep-sac-reproducibility

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

bash scripts/check_project.sh
python -m src.train --config configs/pendulum_sac_n1.yaml
```

## Current status

Early scaffold. No final experimental claims are made yet.

## GitHub workflow

The repository uses larger logical commits with Conventional Commit style. See `docs/github_workflow.md`.
