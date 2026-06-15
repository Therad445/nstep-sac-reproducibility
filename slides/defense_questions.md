# Possible Defense Questions

## Why did you choose SAC?

SAC is a strong off-policy baseline for continuous-control tasks. Since the project is about n-step returns in off-policy actor-critic learning, SAC is a natural choice.

## Why not PPO?

PPO is on-policy. The main difficulty in this project is the off-policy nature of n-step returns. SAC makes this issue more visible.

## Why does n=3 help?

A moderate n-step target can propagate reward information faster than a 1-step target. In HalfCheetah, this likely helps the critic learn useful value estimates earlier.

## Why does n=5 fail?

The most likely explanation is that the longer target increases off-policy bias or target variance. It may also require different hyperparameters. Since the comparison uses the same settings for all horizons, n=5 may be too aggressive.

## Can you claim that n=3 is best?

No. The current experiment is single-seed and limited-compute. I can claim that n=3 worked best in this setting, not that it is universally best.

## Why only one seed?

Compute budget. The project was designed as a bounded student reproduction. More seeds are the first item in future work.

## What is the difference between this and T-SAC?

T-SAC uses a sequence-aware critic, often with trajectory chunks and a transformer-like architecture. This project does not reproduce that full model. It studies the simpler core mechanism: n-step targets in SAC.

## What would you improve first?

I would run 3–5 seeds on HalfCheetah, add Hopper or Walker2d, and compare naive n-step SAC with a SACn-style correction.
