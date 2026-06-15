# Presentation Outline

## Slide 1 — Title

**Stress-Testing n-step Soft Actor-Critic under Limited Compute**

Subtitle: A bounded reproduction and ablation study of n-step SAC ideas.

Main message:

> I study whether n-step returns help SAC in a small-compute continuous-control setting.

## Slide 2 — Motivation

Long-horizon reinforcement learning has a credit assignment problem. A reward may arrive several steps after the useful action.

Standard 1-step TD learning is stable, but it propagates reward information slowly. N-step returns can speed this up, but in off-policy SAC they may also introduce bias and instability.

## Slide 3 — Background: SAC

Explain briefly:

- off-policy actor-critic;
- replay buffer;
- twin critics;
- entropy regularization;
- target networks;
- automatic entropy tuning.

One sentence:

> SAC is a strong continuous-control baseline, so if n-step returns help here, the effect is meaningful.

## Slide 4 — Why n-step returns

Show the difference:

```text
1-step: r_t + gamma V(s_{t+1})
n-step: r_t + gamma r_{t+1} + ... + gamma^n V(s_{t+n})
```

Main intuition:

> N-step returns propagate rewards faster, but use longer off-policy trajectory fragments.

## Slide 5 — Place in literature

Map:

```text
SAC -> n-step SAC -> SACn -> T-SAC
```

Short explanation:

- SAC: stable off-policy baseline;
- SACn: studies how to use n-step returns in SAC more carefully;
- T-SAC: uses trajectory chunks and a sequence-aware critic;
- this project: small-compute ablation of the central mechanism.

## Slide 6 — Project question

Research question:

> Do n-step returns improve SAC under limited compute, and how sensitive is the result to the horizon?

Compared variants:

```text
SAC n=1
SAC n=3
SAC n=5
```

## Slide 7 — Code / algorithm

Show pipeline:

```text
environment transition
-> n-step buffer
-> replay buffer
-> critic update
-> actor update
-> target update
-> evaluation log
```

Explain that for `n=1` the code reduces to standard SAC.

## Slide 8 — Experimental setup

Environments:

- `Pendulum-v1` for sanity check;
- `HalfCheetah-v5` as main MuJoCo benchmark.

Setup:

- seed 0;
- 30k steps for Pendulum;
- 100k steps for HalfCheetah;
- deterministic evaluation;
- CPU training on Ryzen 5700 laptop through WSL Ubuntu.

Important honesty point:

> This is a limited-compute single-seed study, not a full-scale benchmark.

## Slide 9 — Pendulum results

Show `pendulum_learning_curves.png`.

Main interpretation:

- all variants converge;
- n-step mechanism does not break SAC;
- the task is too simple to make a strong claim.

## Slide 10 — HalfCheetah results

Show `halfcheetah_learning_curves.png`.

Main table:

| Method | Final return | Best return |
|---|---:|---:|
| SAC n=1 | 4354 | 4354 |
| SAC n=3 | 4579 | 4579 |
| SAC n=5 | 764 | 833 |

Main interpretation:

> n=3 improves learning, n=5 fails under the same settings.

## Slide 11 — Why n=5 may fail

Possible reasons:

- higher off-policy bias;
- larger target variance;
- same learning rate may be unsuitable;
- entropy dynamics may change;
- one seed is not enough to separate robust failure from unlucky run.

Main message:

> Longer horizon is not automatically better.

## Slide 12 — Weaknesses

Limitations:

- single seed;
- only two environments;
- no full T-SAC reproduction;
- no SACn correction;
- hyperparameters not tuned separately for each horizon;
- results are preliminary.

But:

> Even with these limitations, the experiment shows the core trade-off.

## Slide 13 — Conclusion

Conclusion:

- implementation works;
- Pendulum confirms sanity;
- HalfCheetah shows a meaningful difference;
- moderate n-step target helped;
- longer naive n-step target was unstable;
- future work: more seeds, more environments, SACn-style corrections, sequence-aware critic.

## Slide 14 — Backup

Possible backup material:

- config table;
- exact commands;
- reproduction card;
- n-step buffer code fragment;
- critic target formula.
