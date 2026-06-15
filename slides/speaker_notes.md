# Speaker Notes

## Opening

Today I will present a small reproduction and ablation study of n-step Soft Actor-Critic.

The project is about one practical question: if n-step returns should help credit assignment, do they actually help SAC in a limited-compute continuous-control setup? And if they help, how sensitive is the result to the chosen horizon?

I compare SAC with `n=1`, `n=3` and `n=5`.

## Slide 1

The title is “Stress-Testing n-step Soft Actor-Critic under Limited Compute”.

I frame this as a bounded reproduction, not as a new algorithm paper. The goal is to understand the mechanism from recent n-step SAC papers and test it in a smaller setting.

## Slide 2

The motivation is credit assignment.

In reinforcement learning, useful actions may influence rewards much later. Standard 1-step temporal-difference learning is stable, but it moves reward information backward slowly.

N-step returns can make this faster because the critic target includes several rewards before bootstrapping.

But SAC is off-policy, and this makes the idea less trivial.

## Slide 3

SAC is the baseline.

It has a stochastic actor, two critics, target critics, replay buffer, and entropy regularization.

This makes SAC a good baseline because it is already strong on continuous-control tasks. If an n-step modification improves it or breaks it, the result is meaningful.

## Slide 4

The main algorithmic idea is replacing a 1-step target with an n-step target.

A 1-step target uses one reward and then bootstraps.

An n-step target accumulates several rewards and then bootstraps from a later state.

This can speed up learning, but it may increase off-policy bias and target variance.

## Slide 5

In the literature, this project sits between standard SAC, SACn, and T-SAC.

SACn is important because it discusses the difficulty of combining SAC with n-step returns.

T-SAC is important because it uses trajectory chunks and sequence-aware critics with n-step targets.

My project does not reproduce the full T-SAC pipeline. Instead, it tests the central mechanism: what happens when the SAC critic receives n-step targets.

## Slide 6

The research question is simple:

Do n-step returns improve SAC under limited compute, and how sensitive is the result to the return horizon?

I compare three variants: standard 1-step SAC, 3-step SAC, and 5-step SAC.

## Slide 7

The implementation is a compact SAC pipeline.

The key files are the training script, SAC update, replay buffer, n-step buffer, actor and critic.

The loop is standard: interact with environment, store transition, sample replay batch, update critic, update actor, update target networks, and periodically evaluate the deterministic policy.

## Slide 8

The main change is the n-step buffer.

Instead of storing only a raw transition, it keeps a short queue of transitions and produces an aggregated transition.

For example, for `n=3`, the reward part is:

`r_t + gamma r_{t+1} + gamma^2 r_{t+2}`.

Then the critic bootstraps from `s_{t+3}`.

For `n=1`, this reduces to standard SAC.

## Slide 9

I used two environments.

Pendulum is a cheap sanity check. It is useful to verify that the code works.

HalfCheetah is the main environment because it is a standard MuJoCo continuous-control benchmark.

All runs use seed 0. This is a limitation, but it keeps the project feasible on a laptop.

## Slide 10

The Pendulum result is mostly a sanity check.

All three variants converge to similar final quality.

This tells me that the implementation is not broken. It does not prove that n-step returns are better, because Pendulum is too simple for a strong comparison.

## Slide 11

The main result is on HalfCheetah.

The 1-step baseline reaches around 4354 final return.

The 3-step variant reaches around 4579 and learns faster.

The 5-step variant performs much worse and stays below 1000.

This is the most important plot in the presentation.

## Slide 12

The interpretation is that the horizon matters.

A moderate n-step horizon helped. It improved sample efficiency and final return.

But increasing the horizon to 5 did not help. Under the same hyperparameters, it made learning much worse.

This supports the main idea from the literature: n-step SAC can be useful, but it needs careful stabilization.

## Slide 13

The weaknesses are important.

This is a single-seed study. I do not claim that 3-step SAC is always better than 1-step SAC.

I also do not claim to reproduce full SACn or T-SAC.

The correct claim is narrower: in this limited-compute setting, 3-step SAC improved HalfCheetah learning, while 5-step SAC failed under the same settings.

## Slide 14

To conclude, I implemented SAC with configurable n-step targets and tested the central trade-off.

Pendulum confirmed the implementation.

HalfCheetah gave a meaningful ablation: n=3 helped, n=5 failed.

Future work would include more seeds, more environments, SACn-style correction, and possibly a sequence-aware critic.

The main message is that n-step horizon is a stability parameter, not just a bigger-is-better knob.
