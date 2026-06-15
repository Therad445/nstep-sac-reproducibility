# Reproduction Card

## Goal

Test whether n-step returns improve Soft Actor-Critic under a limited compute budget.

## Compared methods

- SAC with 1-step targets;
- SAC with 3-step targets;
- SAC with 5-step targets.

## Environments

- `Pendulum-v1`;
- `HalfCheetah-v5`.

## Seeds

Planned seeds:

```text
0, 1, 2
```

If compute is limited, the first complete run will use one seed per method, and then additional seeds will be added for the most important comparisons.

## Metrics

- evaluation return;
- best evaluation return;
- final evaluation return;
- actor loss;
- critic loss;
- entropy coefficient;
- wall-clock time;
- steps per second;
- failed or unstable runs.

## Completed runs so far

### Pendulum-v1 sanity ablation

The first completed experiment compares SAC with 1-step, 3-step and 5-step targets on `Pendulum-v1`.

| Run     | Seed | Total steps | Final eval return | Best eval return |
| ------- | ---: | ----------: | ----------------: | ---------------: |
| SAC n=1 |    0 |      30,000 |            -98.57 |           -98.55 |
| SAC n=3 |    0 |      30,000 |            -98.09 |           -98.06 |
| SAC n=5 |    0 |      30,000 |            -98.67 |           -98.67 |

Current interpretation:

This run is a sanity check. It shows that all three variants can train successfully on a small continuous-control environment. The 3-step variant improves faster early in training, while all methods converge to a similar final return. Since this is a single-seed result, it is not enough to claim a robust advantage of any horizon.

## Deviations from full paper-scale experiments

- smaller number of environments;
- fewer random seeds;
- lower number of environment steps;
- simplified implementation;
- no full-scale Transformer critic training in the main experiment.

## Main limitation

The project can show whether the n-step mechanism is useful or unstable in a small-compute setting, but it cannot make strong claims about state-of-the-art performance.
