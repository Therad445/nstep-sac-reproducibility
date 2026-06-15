# Weaknesses and Limitations

## Paper-level weaknesses

The general idea of n-step SAC is attractive, but it has a fragile point: SAC is off-policy.

A longer n-step target uses a longer fragment of a trajectory from the replay buffer. This trajectory was not necessarily collected by the current policy. Because of this, longer targets may increase off-policy bias. They may also increase target variance because several rewards are accumulated before bootstrapping.

This means that n-step returns are not a free improvement. They can speed up learning, but they can also make critic training less stable.

## Implementation limitations

The implementation in this project is intentionally compact. It is useful for an ablation study, but it is not a full reproduction of SACn or T-SAC.

Main limitations:

- no importance-sampling correction;
- no transformer critic;
- no separate hyperparameter tuning for each n-step horizon;
- no large-scale distributed experiments;
- no full paper-scale benchmark suite.

The same SAC hyperparameters are used for `n=1`, `n=3` and `n=5`. This makes the comparison clean, but it may also be unfair to `n=5`, which could require different learning rates, target smoothing or entropy settings.

## Experimental limitations

The experiments are single-seed runs. This is the most important limitation.

The current results are enough for a course project and for showing an implementation-level trend, but they are not enough to make a strong statistical claim.

Current setup:

- `Pendulum-v1`, seed 0, 30k steps;
- `HalfCheetah-v5`, seed 0, 100k steps;
- CPU training on a Ryzen 5700 laptop through WSL Ubuntu.

A stronger experiment should use:

- at least 3–5 seeds;
- more MuJoCo environments;
- longer training;
- confidence intervals;
- separately tuned hyperparameters;
- possibly SACn-style corrections or a sequence-aware critic.

## Observed weakness in results

The `HalfCheetah-v5` ablation shows a clear failure mode.

The 3-step variant performs best:

```text
SAC n=3 -> final eval return around 4579
```

The 1-step baseline is also strong:

```text
SAC n=1 -> final eval return around 4354
```

The 5-step variant performs much worse:

```text
SAC n=5 -> final eval return around 764
```

This suggests that increasing the horizon from 3 to 5 made the naive n-step target less useful or unstable in this setting.

The safest interpretation is:

> A moderate n-step horizon can improve sample efficiency, but a longer naive n-step target may introduce too much bias or variance in off-policy SAC.

## What cannot be claimed

The project should not claim:

- that `n=3` is always better than `n=1`;
- that `n=5` always fails;
- that the implementation reproduces full SACn or T-SAC;
- that the result is state of the art.

The correct claim is narrower:

> In a limited-compute single-seed ablation, 3-step SAC improved HalfCheetah learning compared with the 1-step baseline, while 5-step SAC failed under the same hyperparameters.

This is still a useful result because it directly shows the trade-off behind recent n-step SAC work.
