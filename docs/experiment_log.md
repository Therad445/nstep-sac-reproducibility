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

### Run 002 — HalfCheetah single-seed n-step ablation

Date: 2026-06-15  
Machine: Ryzen 5700 laptop, WSL Ubuntu, CPU training  
Environment: `HalfCheetah-v5`  
Total steps: 100,000  
Evaluation: every 5,000 steps, 5 deterministic episodes  
Seed: 0  

Compared runs:

| Run | n-step horizon | Final eval return | Best eval return | Best step | Runtime |
|---|---:|---:|---:|---:|---:|
| `halfcheetah_sac_n1_seed0` | 1 | 4354.49 | 4354.49 | 100000 | 1320s |
| `halfcheetah_sac_n3_seed0` | 3 | 4578.91 | 4578.91 | 100000 | 1256s |
| `halfcheetah_sac_n5_seed0` | 5 | 764.49 | 833.37 | 90000 | 1286s |

Additional sample-efficiency markers:

| Run | Steps to return ≥ 3000 | Steps to return ≥ 4000 |
|---|---:|---:|
| `halfcheetah_sac_n1_seed0` | 65000 | 100000 |
| `halfcheetah_sac_n3_seed0` | 45000 | 70000 |
| `halfcheetah_sac_n5_seed0` | not reached | not reached |

Observation:

The 3-step SAC variant performed best in this single-seed HalfCheetah run. It reached return ≥3000 earlier than the 1-step baseline and crossed return ≥4000 around 70k environment steps, while the 1-step baseline reached this level only by the final evaluation point.

The 5-step variant did not learn a strong running policy in this run. Its return stayed below 1000, which suggests that a longer naive n-step target can be harmful in this off-policy SAC setting.

This result supports the main project hypothesis: n-step credit assignment can improve sample efficiency, but the horizon is sensitive. A moderate horizon may help, while a longer horizon can introduce too much noise, bias, or instability.
