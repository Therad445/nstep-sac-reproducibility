# Project Scope

## Working title

**Stress-Testing n-step Soft Actor-Critic under Limited Compute**

## Project type

This project is a bounded reproduction and ablation study of recent n-step Soft Actor-Critic ideas. The goal is not to claim a new state-of-the-art algorithm, but to test the central mechanism of n-step credit assignment in SAC under a limited student compute budget.

## Main papers

- **Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns**
- **SACn: Soft Actor-Critic with n-step Returns**

## Research question

Do n-step returns improve SAC training in a small-compute continuous-control setting, and how sensitive is the effect to the return horizon?

## Main experiment

The main comparison is:

- SAC with 1-step targets;
- SAC with 3-step targets;
- SAC with 5-step targets.

The first environment is `Pendulum-v1`, used as a cheap sanity check. The main environment is `HalfCheetah-v5`, because it is a standard MuJoCo continuous-control benchmark and also connects naturally to my previous PPO HalfCheetah project.

## Expected contribution

1. explanation of SAC and n-step targets;
2. literature bridge from SAC to SACn and T-SAC;
3. implementation or adaptation of n-step target computation;
4. ablation over the horizon `n`;
5. learning curves and summary tables;
6. failure and limitation analysis;
7. reproducibility notes with configs, seeds and exact commands.

## Non-goals

This is not a full-scale reproduction of all T-SAC paper results. The full paper setting requires larger compute, more environments and more seeds. Instead, this project focuses on the core mechanism and checks whether the idea remains useful in a smaller, reproducible setup.
