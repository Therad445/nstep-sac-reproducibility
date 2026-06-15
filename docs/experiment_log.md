# Experiment Log

Use this file as a simple lab notebook. Write down every run that matters.

## Template

```text
Date:
Machine:
Branch/commit:
Config:
Command:
Total steps:
Seed:
Result:
Problem:
Decision:
```

## Runs

### Run 001 — Pendulum single-seed sanity ablation

Date: 2026-06-15
Machine: Ryzen 5700 laptop, WSL Ubuntu, CPU training
Branch/commit: main, initial scaffold commit
Environment: `Pendulum-v1`
Total steps: 30,000
Evaluation: every 2,000 steps, 5 deterministic episodes
Seed: 0

Commands:

```bash
python -m src.train --config configs/pendulum_sac_n1.yaml
python -m src.train --config configs/pendulum_sac_n3.yaml
python -m src.train --config configs/pendulum_sac_n5.yaml
python -m src.plot_results --input results/raw --out results/plots
```

Compared runs:

| Run                     | n-step horizon | Final eval return | Best eval return | Runtime |
| ----------------------- | -------------: | ----------------: | ---------------: | ------: |
| `pendulum_sac_n1_seed0` |              1 |            -98.57 |           -98.55 |    352s |
| `pendulum_sac_n3_seed0` |              3 |            -98.09 |           -98.06 |    353s |
| `pendulum_sac_n5_seed0` |              5 |            -98.67 |           -98.67 |    347s |

Observation:

All three variants reached a similar final return on `Pendulum-v1`. The 3-step version improved faster in the early part of training and reached a good deterministic return by around 6k environment steps. The 1-step baseline started worse but caught up later. The 5-step version also converged, but its early curve was less smooth.

The run should be treated as a sanity check, not as a final statistical result. It uses only one seed and a simple environment. Still, it confirms that the implementation trains and that the n-step target mechanism does not immediately break SAC.

Additional note:

The critic loss was larger for longer n-step horizons. This is expected because the target contains more accumulated reward and can become noisier in an off-policy replay setting. This supports the main motivation of the project: n-step targets can help credit assignment, but they are not a free improvement and need stability analysis.

