# Stress-Testing n-step Soft Actor-Critic under Limited Compute

This repository contains a final reinforcement learning project on n-step variants of Soft Actor-Critic (SAC).

The project studies a simple but important question: can n-step returns improve SAC training in continuous-control tasks when the compute budget is limited? The main comparison is between standard SAC with 1-step targets and SAC variants with 3-step and 5-step returns.

The work is framed as a bounded reproduction and ablation study. It does not claim a new state-of-the-art method. The goal is to understand the mechanism, implement the main idea, run controlled experiments, and analyze where n-step targets help or fail.

## Main idea

SAC is an off-policy actor-critic algorithm with entropy regularization. In the standard version, the critic is trained with a 1-step temporal-difference target. N-step returns can propagate reward information faster, which may improve sample efficiency. At the same time, in off-policy learning, longer targets may introduce more bias and variance because the sampled trajectory was collected by an older policy.

This project tests the trade-off directly:

```text
SAC n=1  -> standard 1-step baseline
SAC n=3  -> moderate n-step target
SAC n=5  -> longer n-step target
```

## Papers and context

The project is motivated by recent work on n-step SAC and sequence-aware critic training:

- **Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns** — a recent paper that uses trajectory chunks, a transformer-based critic, and aggregated n-step targets.
- **SACn: Soft Actor-Critic with n-step Returns** — a recent paper focused on the difficulty of combining SAC with n-step returns in an off-policy setting.

The project uses these papers as motivation and literature context, but the implementation is intentionally smaller. Instead of reproducing full paper-scale results, it focuses on a compact ablation under limited compute.

## Experiments

Two environments are used:

- `Pendulum-v1` as a cheap sanity check;
- `HalfCheetah-v5` as the main continuous-control benchmark.

The main completed ablations compare:

```text
SAC n=1
SAC n=3
SAC n=5
```

All runs use seed 0. This is a limitation, but it is also part of the project setting: the goal is a small-compute, reproducible student experiment rather than a large-scale benchmark.

## Results summary

### Pendulum-v1

On `Pendulum-v1`, all three variants train successfully and reach a similar final return. This confirms that the implementation works and that the n-step target mechanism does not immediately break SAC on a simple continuous-control task.

### HalfCheetah-v5

The main result is observed on `HalfCheetah-v5`:

| Method | Final eval return | Best eval return |
|---|---:|---:|
| SAC n=1 | 4354.49 | 4354.49 |
| SAC n=3 | 4578.91 | 4578.91 |
| SAC n=5 | 764.49 | 833.37 |

In this single-seed run, the 3-step variant performs best and reaches strong performance earlier than the 1-step baseline. The 5-step variant fails to learn a strong running policy.

The result supports the main project hypothesis: n-step credit assignment can help, but the horizon is sensitive. A moderate horizon may improve sample efficiency, while a longer naive n-step target can become unstable or ineffective in off-policy SAC.

## Repository structure

```text
configs/      experiment configs
src/          SAC implementation, training and plotting code
scripts/      launch and check scripts
results/      raw logs, summary tables and plots
docs/         notes, literature map, algorithm explanation and limitations
slides/       final presentation materials
```

## How to run

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check the project:

```bash
bash scripts/check_project.sh
```

Run Pendulum ablation:

```bash
bash scripts/run_pendulum_ablation.sh
```

Run HalfCheetah runs manually:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.train --config configs/halfcheetah_sac_n1.yaml
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.train --config configs/halfcheetah_sac_n3.yaml
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.train --config configs/halfcheetah_sac_n5.yaml
```

Build plots:

```bash
python -m src.plot_results --input results/raw --out results/plots
```

## Limitations

The current results are single-seed experiments. They should not be interpreted as a final claim that 3-step SAC is generally better than 1-step SAC across all environments and seeds. The stronger conclusion is narrower:

> In this limited-compute setting, a moderate 3-step target improved HalfCheetah learning, while a longer 5-step target was much less stable.

A larger study should add more seeds, more environments, and possibly compare against SACn-style corrections or a transformer-based critic.
