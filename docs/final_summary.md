# Final Project Summary

## Title

**Stress-Testing n-step Soft Actor-Critic under Limited Compute**

## Project type

This project is a bounded reproduction and ablation study of n-step Soft Actor-Critic ideas. The goal is not to propose a new reinforcement learning algorithm, but to test one central mechanism from recent n-step SAC work: replacing the standard 1-step SAC critic target with a longer n-step target.

## Motivation

Soft Actor-Critic is a strong off-policy actor-critic algorithm for continuous-control tasks. In standard SAC, the critic usually learns from a 1-step temporal-difference target. This is stable, but reward information is propagated only one step at a time.

N-step returns can propagate rewards faster and may improve sample efficiency. However, in off-policy SAC, longer n-step targets are also risky: they use longer trajectory fragments from the replay buffer, which may have been collected by older policies. This can increase off-policy bias, target variance and sensitivity to the chosen horizon.

The project studies this trade-off directly.

## Research question

> Do n-step returns improve SAC under limited compute, and how sensitive is the result to the return horizon?

## Compared methods

The project compares three SAC variants:

| Method | Description |
|---|---|
| SAC n=1 | Standard 1-step SAC baseline |
| SAC n=3 | SAC with moderate 3-step target |
| SAC n=5 | SAC with longer 5-step target |

## Implementation

The implementation contains a compact SAC training pipeline with:

- squashed Gaussian actor;
- twin Q-critics;
- target critic networks;
- replay buffer;
- configurable n-step transition buffer;
- automatic entropy tuning;
- periodic deterministic evaluation;
- CSV logging and plot generation.

The main implementation files are:

```text
src/train.py
src/rl/sac.py
src/rl/nstep_buffer.py
src/rl/replay_buffer.py
src/rl/actor.py
src/rl/critic.py
```

## Experimental setup

Two environments were used:

| Environment | Role | Steps | Seed |
|---|---|---:|---:|
| `Pendulum-v1` | sanity check | 30,000 | 0 |
| `HalfCheetah-v5` | main benchmark | 100,000 | 0 |

Training was done on a Ryzen 5700 laptop through WSL Ubuntu, using CPU.

## Pendulum results

On `Pendulum-v1`, all three variants trained successfully and reached similar final quality. This environment was mainly used as a sanity check.

Interpretation:

> The n-step target implementation works and does not immediately break SAC on a simple continuous-control task.

## HalfCheetah results

The main result is from `HalfCheetah-v5`.

| Method | Final eval return | Best eval return |
|---|---:|---:|
| SAC n=1 | 4354.49 | 4354.49 |
| SAC n=3 | 4578.91 | 4578.91 |
| SAC n=5 | 764.49 | 833.37 |

The 3-step variant achieved the best final return and reached strong performance earlier than the 1-step baseline. The 5-step variant failed to learn a strong running policy under the same settings.

## Main interpretation

The experiment supports the central hypothesis:

> Moderate n-step credit assignment can improve SAC sample efficiency, but longer naive n-step targets may become unstable or ineffective in off-policy SAC.

The result is especially visible on `HalfCheetah-v5`: `n=3` improves over the baseline, while `n=5` performs much worse.

This also matches the motivation behind SACn and T-SAC-style work: n-step returns are useful, but the horizon and stability mechanism matter.

## Limitations

The main limitations are:

- only one random seed;
- only two environments;
- no full SACn correction mechanism;
- no transformer or sequence-aware critic;
- no separate hyperparameter tuning for different n-step horizons;
- limited compute budget.

Because of these limitations, the project should not claim that `n=3` is universally better than `n=1`, or that `n=5` always fails.

The correct claim is narrower:

> In this limited-compute single-seed ablation, 3-step SAC improved HalfCheetah learning compared with the 1-step baseline, while 5-step SAC failed under the same hyperparameters.

## Future work

The most useful extensions are:

1. run 3–5 seeds;
2. add more MuJoCo environments such as Hopper or Walker2d;
3. compare naive n-step SAC with SACn-style corrections;
4. tune hyperparameters separately for each horizon;
5. add a sequence-aware critic inspired by T-SAC;
6. report confidence intervals and failure rates.

## Final conclusion

This project shows that n-step SAC is not simply a bigger-is-better modification. The return horizon behaves like a stability parameter. A moderate horizon can help, but a longer naive target can break training.

The main value of the project is the implementation-level ablation and the failure analysis, not a state-of-the-art claim.
